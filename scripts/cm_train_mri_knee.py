#!/usr/bin/env python3
"""
Train a diffusion model on knee MRI slices with a 2D‐Gaussian undersampling mask.
"""
import argparse
import copy
import random
import warnings
from pathlib import Path
import sys
import os

import torch
import torch.distributed as dist
import numpy as np
import ml_collections

# allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cm import dist_util, logger
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util import CMTrainLoop

from cm.MRI_datasets_knee_kspace import create_dataloader

warnings.filterwarnings("ignore")
torch.manual_seed(42)
random.seed(0)
np.random.seed(0)


def main():
    args = create_argparser().parse_args()

    # setup distributed & logging
    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)
    logger.log("Command used: " + " ".join(sys.argv))

    # build model & diffusion
    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )

    if args.training_mode == "progdist":
        distillation = False
    elif "consistency" in args.training_mode:
        distillation = True
    else:
        raise ValueError(f"unknown training mode {args.training_mode}")

    m_and_d_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
    m_and_d_kwargs["distillation"] = distillation

    model, diffusion = create_model_and_diffusion(**m_and_d_kwargs)
    model.to(dist_util.dev())
    model.train()
    if args.use_fp16:
        model.convert_to_fp16()

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # optional teacher
    teacher_model = teacher_diffusion = None
    if args.teacher_model_path:
        logger.log(f"loading the teacher model from {args.teacher_model_path}")
        tm_kwargs = copy.deepcopy(m_and_d_kwargs)
        tm_kwargs["dropout"] = args.teacher_dropout
        tm_kwargs["distillation"] = False
        teacher_model, teacher_diffusion = create_model_and_diffusion(**tm_kwargs)
        teacher_model.load_state_dict(
            dist_util.load_state_dict(args.teacher_model_path, map_location="cpu")
        )
        teacher_model.to(dist_util.dev())
        teacher_model.eval()
        for dst, src in zip(model.parameters(), teacher_model.parameters()):
            dst.data.copy_(src.data)
        if args.use_fp16:
            teacher_model.convert_to_fp16()

    # target model for consistency
    target_model, _ = create_model_and_diffusion(**m_and_d_kwargs)
    target_model.to(dist_util.dev())
    target_model.train()
    dist_util.sync_params(target_model.parameters())
    dist_util.sync_params(target_model.buffers())
    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)
    if args.use_fp16:
        target_model.convert_to_fp16()

    # build config for our HDF5 loader
    configs = ml_collections.ConfigDict()
    configs.data = ml_collections.ConfigDict({
        "root":        args.data_dir,
        "mask_type":   args.mask_type,
        "gauss_accel": args.gauss_accel,
        "gauss_sigma": args.gauss_sigma,
        "skip":        args.skip if hasattr(args, 'skip') else 0,
    })
    configs.data.image_size = args.image_size
    #configs.data.is_multi   = False
    configs.data.is_complex = True
    configs.data.magpha     = False
    configs.data.h5_key = 'kspace' 

    effective_batch = (
        args.global_batch_size // dist.get_world_size()
        if args.batch_size == -1 else args.batch_size
    )
    configs.training = ml_collections.ConfigDict({
        "batch_size":  effective_batch,
        "num_workers": args.num_workers,
    })

    # data loaders
    logger.log("creating data loaders...")
    train_loader, val_loader = create_dataloader(configs, data_dir=args.data_dir, evaluation=False, sort=True)


    # run training loop
    logger.log("training...")
    CMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        teacher_diffusion=teacher_diffusion,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
        diffusion=diffusion,
        data=train_loader,
        val_data=val_loader,
        batch_size=effective_batch,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        epoch=args.epoch
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        mask_type="gauss",
        gauss_accel=8,
        gauss_sigma=0.5,
        num_workers=4,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        save_dir="",
        epoch=100,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    print("Visible GPUs:", torch.cuda.device_count(),
          "– Using:", torch.cuda.current_device(),
          torch.cuda.get_device_name(torch.cuda.current_device()))
    main()
