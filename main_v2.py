import argparse
import datetime
import json
import random
import time
import shutil
from pathlib import Path
import os, sys

import numpy as np
import torch
import torchvision.transforms as transforms
import util.misc as utils
from detrsmpl.data.datasets import build_dataloader
from mmcv.parallel import MMDistributedDataParallel

from datasets.dataset import MultipleDatasets
from engine import evaluate, train_one_epoch, inference
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.config import DictAction, cfg
from util.utils import ModelEma

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# 限制内存增长
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector',
                                     add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    
    # training parameters
    parser.add_argument('--output_dir',
                        default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device',
                        default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path',
                        help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--to_vid', action='store_true')
    parser.add_argument('--inference', action='store_true')
    # distributed training parameters

    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank',
                        type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--amp',
                        action='store_true',
                        help='Train with mixed precision')

    parser.add_argument('--inference_input', default=None, type=str)
    
    # Add new argument for JSON dataset processing
    parser.add_argument('--json_dataset', default=None, type=str,
                       help='Path to JSON dataset file')
    parser.add_argument('--process_json', action='store_true',
                       help='Process JSON dataset for pose extraction')
    
    return parser


def build_model_main(args, cfg):
    print(args.modelname)
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors, _ = build_func(
        args, cfg)
    return model, criterion, postprocessors, _


def process_json_dataset(args, model, criterion, postprocessors, device, cfg):
    """Process JSON dataset to extract pose data for each sample"""
    
    # Load JSON dataset
    with open(args.json_dataset, 'r') as f:
        dataset = json.load(f)
    
    # Create temporary directory for processing
    temp_dir = os.path.join(args.output_dir, 'temp_processing')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process each sample
    for idx, sample in enumerate(dataset):
        print(f"Processing sample {idx+1}/{len(dataset)}: {sample['id']}")
        
        # Find the human image (type=5)
        human_img_idx = None
        for i, img_type in enumerate(sample['type']):
            if img_type == 5:
                human_img_idx = i
                break
        
        if human_img_idx is None:
            print(f"No human image found in sample {sample['id']}")
            sample['pose'] = []
            sample['camera'] = []
            continue
            
        # Get the human image path
        human_img_path = sample['image_withhuman'][human_img_idx]
        if not human_img_path:  # Fallback to regular image if withhuman is empty
            human_img_path = sample['image'][human_img_idx]
        
        # Full path to the image (assuming images are in same directory as JSON)
        json_dir = os.path.dirname(args.json_dataset)
        full_img_path = os.path.join(json_dir, human_img_path)
        
        if not os.path.exists(full_img_path):
            print(f"Image not found: {full_img_path}")
            sample['pose'] = []
            sample['camera'] = []
            continue
        
        # Create temporary directory for this image
        sample_temp_dir = os.path.join(temp_dir, f"sample_{idx}")
        os.makedirs(sample_temp_dir, exist_ok=True)
        
        # Copy image to temporary directory
        temp_img_path = os.path.join(sample_temp_dir, os.path.basename(human_img_path))
        shutil.copy2(full_img_path, temp_img_path)
        
        # Update config with human_num for this sample
        human_num = sample.get('human_num', 1)
        cfg.num_person = human_num
        
        # Create dataset for this single image
        exec('from datasets.' + cfg.testset + ' import ' + cfg.testset)
        dataset_val = eval(cfg.testset)(sample_temp_dir, args.output_dir, 
                                        json_mode=True, human_num=human_num)
        
        data_loader_val = build_dataloader(
            dataset_val,
            args.batch_size,
            0 if 'workers_per_gpu' in args else 2,
            dist=args.distributed,
            shuffle=False)
        
        # Run inference
        from engine import inference_json
        pose_data, camera_data = inference_json(model, criterion, postprocessors,
                                               data_loader_val, device, args.output_dir,
                                               wo_class_error=False, args=args)
        
        # Add pose and camera data to sample
        sample['pose'] = pose_data
        sample['camera'] = camera_data
        
        # Clean up temporary directory for this sample
        shutil.rmtree(sample_temp_dir)
    
    # Save updated dataset
    output_json_path = args.json_dataset.replace('.json', '_with_pose.json')
    with open(output_json_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Saved updated dataset to: {output_json_path}")
    
    # Clean up main temporary directory
    shutil.rmtree(temp_dir)


def main(args):
    utils.init_distributed_mode_ssc(args)
    print('Loading config file from {}'.format(args.config_file))
    shutil.copy2(args.config_file,'config/aios_smplx.py')
    from config.config import cfg
    
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, 'config_cfg.py')
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, 'config_args_raw.json')
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            continue

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'),
                          distributed_rank=args.rank,
                          color=False,
                          name='detr')
    logger.info('git:\n  {}\n'.format(utils.get_sha()))
    logger.info('Command: ' + ' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, 'config_args_all.json')
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info('Full config saved to {}'.format(save_json_path))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info('args: ' + str(args) + '\n')

    if args.frozen_weights is not None:
        assert args.masks, 'Frozen training is meant for segmentation only'

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors, _ = build_model_main(args, cfg)

    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:
        model = MMDistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    logger.info('number of params:' + str(n_parameters))
    logger.info('params:\n' + json.dumps(
        {n: p.numel()
         for n, p in model.named_parameters() if p.requires_grad},
        indent=2))

    # Load checkpoint
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume,
                                                            map_location='cpu',
                                                            check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(
                    utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)

    # Process JSON dataset if specified
    if args.process_json and args.json_dataset:
        os.environ['EVAL_FLAG'] = 'TRUE'
        model.eval()
        process_json_dataset(args, model, criterion, postprocessors, device, cfg)
        return

    # Original inference and training code continues here...
    param_dicts = get_param_dict(args, model_without_ddp)
    optimizer = torch.optim.AdamW(param_dicts,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    if not args.eval:
        trainset= []
        for trainset_i,v in cfg.trainset_partition.items():
            exec('from datasets.' + trainset_i +
                ' import ' + trainset_i)
            trainset.append(
                eval(trainset_i)(transforms.ToTensor(), 'train'))
        trainset_loader = MultipleDatasets(trainset, make_same_len=False,partition=cfg.trainset_partition)
    
        data_loader_train = build_dataloader(
            trainset_loader,
            args.batch_size,
        0  if 'workers_per_gpu' in args else 1,
            dist=args.distributed)
    
    exec('from datasets.' + cfg.testset + ' import ' + cfg.testset)
    
    if not args.inference:
        dataset_val = eval(cfg.testset)(transforms.ToTensor(), "test")
    else:
        dataset_val = eval(cfg.testset)(args.inference_input, args.output_dir)
        
    data_loader_val = build_dataloader(
        dataset_val,
        args.batch_size,
        0  if 'workers_per_gpu' in args else 2,
        dist=args.distributed,
        shuffle=False)
    
    # Rest of the original main function continues...
    # [Include the rest of your original main function here]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script',
                                     parents=[get_args_parser()])
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)