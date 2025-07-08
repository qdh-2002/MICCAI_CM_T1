import torch
import argparse
import logging
from tensorboardX import SummaryWriter
import os
from torch.amp import autocast, GradScaler
import numpy as np
from cm.MRI_datasets import *
from torch.utils.data.dataloader import DataLoader
import random, itertools
from PIL import Image
import numpy as np
import pydicom
from datetime import datetime
from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample
from cm.utils import *
import argparse
import warnings
warnings.filterwarnings("ignore")
# torch.manual_seed(42)
# random.seed(0)
# np.random.seed(0)
torch.cuda.set_device(3)

def create_dicom(template, img, patient_name, patient_id, new_study_uid, new_series_uid, instance_number):
    ds = template.copy()
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = new_series_uid
    ds.StudyInstanceUID = new_study_uid
    ds.SliceLocation = float(instance_number)
    ds.SliceThickness = 1 
    ds.ImagePositionPatient = [0, 0, float(instance_number)] 
    ds.ImageOrientationPatient = [1,0,0,0,1,0]
    ds.InstanceNumber = instance_number
    ds.PixelData = img.tobytes()
    ds.Rows, ds.Columns = img.shape
    return ds

def main():
    # parse configs
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir = args.save_dir)    
    # Initialize WandbLogger
    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model_and_diffusion_kwargs = args_to_dict(
        args, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = distillation
    model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    axis_distance = 2 # 15
    # datadict = {'/raid/kaifengpang/MicroUS_ToBeReformatted/087': [343, 686, 1029],#[674, 1050],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/089': [343, 686, 1029],#[63, 671],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/091': [343, 686, 1029],#[638, 908],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/092': [291, 582, 873],#[795, 858],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/094': [343, 686, 1029],#[675, 947],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/096': [291, 582, 873],#[600, 746],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/097': [291, 582, 873],#[609, 868],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/098': [343, 686, 1029],#[664, 892],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/100': [291, 582, 873],#[541, 724],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/102': [343, 686, 1029],#[595, 840],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/103': [291, 582, 873],#[460, 810],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/104': [343, 686, 1029],#[621, 1114],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/108': [291, 582, 873],#[400, 793],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/109': [291, 582, 873],#[582, 862],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/112': [343, 686, 1029],#[649, 859],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/117': [343, 686, 1029],#[574, 794],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/118': [291, 582, 873],#[500, 778],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/120': [343, 686, 1029],#[677, 1068],
    #             '/raid/kaifengpang/MicroUS_ToBeReformatted/121': [343, 686, 1029]#[523, 1145]
    #             }
    datadict = {'/raid/kaifengpang/MicroUS_ToBeReformatted/106': [343, 686, 1029],#[674, 1050],
                '/raid/kaifengpang/MicroUS_ToBeReformatted/107': [868, 873, 879],#[63, 671],
                '/raid/kaifengpang/MicroUS_ToBeReformatted/111': [291, 582, 873],#[638, 908],
                '/raid/kaifengpang/MicroUS_ToBeReformatted/113': [640, 646, 650],#[795, 858],
                '/raid/kaifengpang/MicroUS_ToBeReformatted/114': [635, 639, 644]#[675, 947]
                }

    

    logger.info('Begin Model Evaluation.')
    with torch.no_grad():
        for k in datadict.keys():
            case_id = k[-3:]
            dicom_template = pydicom.dcmread(glob.glob(os.path.join(k, '*.dcm'))[0])
            new_study_uid_ref = pydicom.uid.generate_uid()
            new_series_uid_ref = pydicom.uid.generate_uid()
            new_study_uid_sr = pydicom.uid.generate_uid()
            new_series_uid_sr = pydicom.uid.generate_uid()
            
            val_dataset = MicroUSAxialImageFolder(root_path = k, axis_distance = axis_distance, scale = 8)
            val_loader = DataLoader(dataset = val_dataset, batch_size = 1, shuffle = False, num_workers = 8)
            sample_slice = [4 * i + 1 for i in range(len(val_loader) // 4)]
            
            case_dir = os.path.join(logger.get_dir(), case_id)
            os.makedirs(case_dir, exist_ok = True)
            ref_dir = os.path.join(case_dir, 'ref')
            os.makedirs(ref_dir, exist_ok = True)
            ref_img_dir = os.path.join(ref_dir, 'imgs')
            os.makedirs(ref_img_dir, exist_ok = True)
            ref_dicom_dir = os.path.join(ref_dir, 'dicoms')
            os.makedirs(ref_dicom_dir, exist_ok = True)
            sr_dir = os.path.join(case_dir, 'sr')
            os.makedirs(sr_dir, exist_ok = True)
            sr_img_dir = os.path.join(sr_dir, 'imgs')
            os.makedirs(sr_img_dir, exist_ok = True)
            sr_dicom_dir = os.path.join(sr_dir, 'dicoms')
            os.makedirs(sr_dicom_dir, exist_ok = True)
            for slice_id in sample_slice:
                logger.info('Inference on case', case_id, 'slice', slice_id)
                logger.log(datetime.now())
                data_iter = iter(val_loader)
                v = next(itertools.islice(data_iter, slice_id - 1, None))
                lr_grids = v['lr_grids'].to(dist_util.dev())
                hr_grids = v['hr_grids'].to(dist_util.dev())
                hr_inte = v['hr_inte'].to(dist_util.dev())
                h, w = hr_inte.shape[-2 : ]
                meta_info = v['meta_info']
                # with autocast(device_type='cuda'): 
                sample = karras_sample(
                        diffusion,
                        model,
                        (1,1,h,w),
                        steps=args.steps,
                        hr_inte = hr_inte, 
                        lr_grids = lr_grids,
                        model_kwargs={},
                        device=dist_util.dev(),
                        clip_denoised=True,
                        sampler=args.sampler,
                        generator=None,
                        ts=ts,
                    )
                sr_img = ((sample + 1) / 2).clamp(0, 1).contiguous()
                ref_img = ((hr_inte + 1) / 2).clamp(0, 1)
                polar_coords = hr_grids
                h_expand = meta_info['h_expand'].to(dist_util.dev())
                ref_img = polar2cartesian(ref_img, polar_coords, h_expand)
                sr_img = polar2cartesian(sr_img, polar_coords, h_expand)
                Image.fromarray(ref_img[0]).convert('L').save('{}/case_{}_slice_{}_ref.png'.format(ref_img_dir, case_id, slice_id))
                Image.fromarray(sr_img[0]).convert('L').save('{}/case_{}_slice_{}_sr.png'.format(sr_img_dir, case_id, slice_id))
                ds_ref = create_dicom(dicom_template, 
                                      ref_img[0], 
                                      case_id + 'ref', 
                                      case_id + 'ref', 
                                      new_study_uid_ref,
                                      new_series_uid_ref, 
                                      slice_id)
                ds_ref.save_as(os.path.join(ref_dicom_dir, f'case_{case_id:03}_slice_{slice_id:04}_ref.dcm'))
                ds_sr = create_dicom(dicom_template, 
                                     sr_img[0], 
                                     case_id + '_sr_' + args.case_name,#  case_id + 'sr', 
                                     case_id + '_sr_' + args.case_name,#  case_id + 'sr', 
                                     new_study_uid_sr,
                                     new_series_uid_sr, 
                                     slice_id)
                ds_sr.save_as(os.path.join(sr_dicom_dir, f'case_{case_id:03}_slice_{slice_id:04}_sr.dcm'))

def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=151,
        model_path="",
        seed=42,
        ts="",
        save_dir="",
        case_name = "",
        channel_mult = (1, 2, 4, 8, 16)
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()