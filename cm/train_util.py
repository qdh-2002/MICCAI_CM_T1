import copy
import functools
import os
import torch
from PIL import Image
from .utils import polar2cartesian
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam
from .karras_diffusion import karras_sample
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from datetime import datetime
from torch.amp import autocast, GradScaler
from .fp16_util import (
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)
import numpy as np
from torchvision.utils import save_image
from torchvision.utils import save_image, make_grid
import torchvision.transforms.functional as TF


INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = RAdam(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        # decay_params = []
        # no_decay_params = []

        # for name, param in self.mp_trainer.master_params:
        #     if param.requires_grad:
        #         if "norm" in name or "bias" in name:  # Exclude normalization layers and biases
        #             no_decay_params.append(param)
        #         else:
        #             decay_params.append(param)

        # self.opt = RAdam(
        #     [
        #         {"params": decay_params, "weight_decay": self.weight_decay},   # Apply weight decay
        #         {"params": no_decay_params, "weight_decay": 0.0},  # No weight decay
        #     ],
        #     lr=self.lr
        # )
        # print('Excluding from weight decay: norm layer')

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        self.step = self.resume_step

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    ),
                )

        dist_util.sync_params(self.model.parameters())
        dist_util.sync_params(self.model.buffers())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while not self.lr_anneal_steps or self.step < self.lr_anneal_steps:
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self.step += 1
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # if dist.get_rank() == 0:
        #     with bf.BlobFile(
        #         bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
        #         "wb",
        #     ) as f:
        #         th.save(self.opt.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainer.master_params)
        dist.barrier()


class CMTrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        epoch,
        val_data,
        target_model,
        teacher_model,
        teacher_diffusion,
        training_mode,
        ema_scale_fn,
        total_training_steps,
        save_intermediates=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epoch = epoch
        self.val_data = val_data
        self.target_model = target_model
        self.teacher_model = teacher_model
        self.teacher_diffusion = teacher_diffusion
        self.training_mode = training_mode
        self.ema_scale_fn = ema_scale_fn
        self.total_training_steps = total_training_steps
        self.global_step = self.step

        # Intermediate saving
        self.save_intermediates = save_intermediates
        self.intermediate_dir = os.path.join(get_blob_logdir(), "intermediates")
        os.makedirs(self.intermediate_dir, exist_ok=True)

        # Load and freeze target
        if self.target_model is not None:
            self._load_and_sync_target_parameters()
            self.target_model.requires_grad_(False)
            self.target_model.train()
            self.target_model_param_groups_and_shapes = get_param_groups_and_shapes(
                self.target_model.named_parameters()
            )
            self.target_model_master_params = make_master_params(
                self.target_model_param_groups_and_shapes
            )

        # Load and freeze teacher
        if self.teacher_model is not None:
            self._load_and_sync_teacher_parameters()
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()

    def run_loop(self):
        print("Checkpoints & intermediates saved to:", get_blob_logdir())
        saved = False
        for epoch_id in range(self.epoch):
            logger.log(f"Epoch {epoch_id+1}/{self.epoch}...")
            for batch in self.data:
                self.run_step(batch)
                saved = False
                # checkpoint & validate
                if (
                    self.global_step
                    and self.save_interval != -1
                    and self.global_step % self.save_interval == 0
                ):
                    self.save()
                    saved = True
                    torch.cuda.empty_cache()
                    if os.environ.get("DIFFUSION_TRAINING_TEST") and self.step > 0:
                        return
                    self.val()
                # periodic log
                if self.global_step % self.log_interval == 0:
                    logger.log(datetime.now())
                    logger.dumpkvs()
        if not saved:
            self.save()

    def forward_backward(self, batch):
        self.mp_trainer.zero_grad()
        total = batch['lr_img'].shape[0]
        for i in range(0, total, self.microbatch):
            micro = {k: v[i : i + self.microbatch].to(dist_util.dev()) for k, v in batch.items()}
            micro["sampling_vec"] = micro["sampling_vec"].to(dist_util.dev())

            last = (i + self.microbatch) >= total
            t, weights = self.schedule_sampler.sample(micro['lr_img'].shape[0], dist_util.dev())
            ema, num_scales = self.ema_scale_fn(self.global_step)

            if self.training_mode != "consistency_training":
                raise ValueError(f"Unsupported mode: {self.training_mode}")

            compute_losses = functools.partial(
                self.diffusion.consistency_losses,
                self.ddp_model,
                micro,
                num_scales,
                model_kwargs={
                    "hr_inte":      micro["hr_inte"],
                    "sampling_vec": micro["sampling_vec"],
                },
                target_model=self.target_model,
                teacher_model=self.teacher_model,
                teacher_diffusion=self.teacher_diffusion,
            )
            if last or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

             # Extract intermediates
            xt      = losses.pop("xt")
            x0_pred = losses.pop("x0_pred")
            xs      = losses.pop("xs")  # list


            #####
            print(f"x0_pred.shape = {x0_pred.shape}; "
            f"hr_img.shape = {micro['hr_img'].shape}; "
            f"lr_img.shape = {micro['lr_img'].shape}"
            f"xs[0].shape = {xs[0].shape}; "
            f"xt.shape = {xt.shape}; "
            f"hr_inte.shape = {micro['hr_inte'].shape}")

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            self.mp_trainer.backward(loss)

            # save examples
            if self.save_intermediates and self.global_step % 100 == 0 and dist.get_rank() == 0:
                hr_inte = micro["hr_inte"].detach()
                h, w = hr_inte.shape[-2:]
                to_dump = {
                    "xt":      xt[:1],                              # take only the first example
                    "x0_pred": x0_pred[:1],
                    "hr_inte": hr_inte[:1],                        # take only the first example
                }

                if "hr_img" in micro:
                    gt = micro["hr_img"].detach().cpu()
                    to_dump["gt"] = gt

                hr_inte = micro["hr_inte"].detach()
                h, w = hr_inte.shape[-2:]
                with torch.no_grad():
                    sample = karras_sample(
                        self.diffusion,
                        self.target_model,
                        (1, 1, h, w),
                        steps=1281,
                        hr_inte=hr_inte[:1],  # take first sample in batch
                        model_kwargs={
                            "hr_inte": hr_inte[:1],
                            "sampling_vec": micro["sampling_vec"][:1],
                        },
                        device=dist_util.dev(),
                        clip_denoised=True,
                        sampler='onestep',
                    ).detach().cpu()
                to_dump["sample"] = sample

                # now save everything in one loop
                for name, tensor in to_dump.items():
                    # tensor is in [-1,1], bring to [0,1]
                    img = ((tensor + 1) / 2).clamp(0, 1)
                    path = os.path.join(
                        self.intermediate_dir,
                        f"{self.global_step:06d}_{name}.png"
                    )
                    save_image(img, path)


    def run_step(self, batch):
        self.forward_backward(batch)
        took = self.mp_trainer.optimize(self.opt)
        if took:
            self._update_ema()
            if self.target_model:
                self._update_target_ema()
            self.step += 1
            self.global_step += 1
        self._anneal_lr()
        self.log_step()

    def _update_target_ema(self):
        rate, _ = self.ema_scale_fn(self.global_step)
        with th.no_grad():
            update_ema(
                self.target_model_master_params,
                self.mp_trainer.master_params,
                rate=rate,
            )
            master_params_to_model_params(
                self.target_model_param_groups_and_shapes,
                self.target_model_master_params,
            )

    def val(self):
        self.target_model.eval()
        val_dir = os.path.join(get_blob_logdir(), 'val_imgs')
        os.makedirs(val_dir, exist_ok=True)
        sub = os.path.join(val_dir, str(self.global_step))
        os.makedirs(sub, exist_ok=True)
        with torch.no_grad():
            for i, v in enumerate(self.val_data):
                hr_img = v['hr_img'].to(dist_util.dev())
                h, w = hr_img.shape[-2:]
                hr_inte = v['hr_inte'].to(dist_util.dev())
                sample = karras_sample(
                    self.diffusion,
                    self.target_model,
                    (1,1,h,w),
                    steps=1281,
                    hr_inte=hr_inte,
                    model_kwargs={
                        "hr_inte": hr_inte,
                        "sampling_vec": v["sampling_vec"].to(dist_util.dev()),
                    },
                    device=dist_util.dev(),
                    clip_denoised=True,
                    sampler='onestep',
                )
                sample = ((sample + 1)/2).clamp(0,1).squeeze().cpu().numpy()*255
                hr_img = ((hr_img+1)/2).clamp(0,1).squeeze().cpu().numpy()*255
                hr_inte = ((hr_inte+1)/2).clamp(0,1).squeeze().cpu().numpy()*255
                imgs = [Image.fromarray(arr.astype(np.uint8), mode='L') for arr in (hr_inte, hr_img, sample)]
                out = Image.new('L', (w*3+20, h))
                out.paste(imgs[0], (0,0)); out.paste(imgs[1], (w+10,0)); out.paste(imgs[2], (2*w+20,0))
                out.save(os.path.join(sub, f"{self.global_step:06d}_{i+1}.png"))
        self.target_model.train()

    def save(self):
        step = self.global_step
        # main model
        with bf.BlobFile(bf.join(get_blob_logdir(), f"model{step:06d}.pt"), 'wb') as f:
            th.save(self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params), f)
        # target
        if self.target_model:
            with bf.BlobFile(bf.join(get_blob_logdir(), f"target_model{step:06d}.pt"), 'wb') as f:
                th.save(self.target_model.state_dict(), f)
        dist.barrier()

    def _no_sync_wrapper(self, fn):
        with self.ddp_model.no_sync():
            return fn()

    def _load_and_sync_target_parameters(self):
        resume = find_resume_checkpoint() or self.resume_checkpoint
        if resume:
            path, name = os.path.split(resume)
            tgt = name.replace("model", "target_model")
            ckpt = os.path.join(path, tgt)
            if bf.exists(ckpt) and dist.get_rank()==0:
                logger.log(f"loading target from {ckpt}...")
                self.target_model.load_state_dict(
                    dist_util.load_state_dict(ckpt, map_location=dist_util.dev())
                )
        dist_util.sync_params(self.target_model.parameters())
        dist_util.sync_params(self.target_model.buffers())

    def _load_and_sync_teacher_parameters(self):
        resume = find_resume_checkpoint() or self.resume_checkpoint
        if resume:
            path, name = os.path.split(resume)
            thr = name.replace("model", "teacher_model")
            ckpt = os.path.join(path, thr)
            if bf.exists(ckpt) and dist.get_rank()==0:
                logger.log(f"loading teacher from {ckpt}...")
                self.teacher_model.load_state_dict(
                    dist_util.load_state_dict(ckpt, map_location=dist_util.dev())
                )
        dist_util.sync_params(self.teacher_model.parameters())
        dist_util.sync_params(self.teacher_model.buffers())

    def log_step(self):
        step = self.global_step
        logger.logkv("step", step)
        logger.logkv("samples", (step+1)*self.global_batch)



def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
