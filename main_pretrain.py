# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import wandb
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# import timm.optim.optim_factory as optim_factory
from timm.optim import create_optimizer_v2
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae
import sys
from engine_pretrain import train_one_epoch

from timm.optim import optim_factory
from distributed_shampoo import (
    DefaultEigenvalueCorrectedShampooConfig,
    AdamGraftingConfig,
    DDPShampooConfig,
    DistributedShampoo,
)
def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')


    parser.add_argument('--skipless', action='store_true',default=False,
                        help='activate skipless training')
    parser.add_argument('--mimetic', default=None, type=float, nargs=2, help='mimetic init')
    parser.add_argument('--unet_style', action='store_true', default=False,
                        help='use U-Net style skip connections')
    parser.add_argument('--W_v', default=1.0, type=float, help='coefficient for Value')
    parser.add_argument('--W_p', default=1.0, type=float, help='coefficient for Projection')
    return parser


def main(args):

    misc.init_distributed_mode(args)
    
    if misc.is_main_process():
        # Define the paths for the log and args files
        log_file_path = os.path.join(args.output_dir, "log.txt") # <-- Changed to .txt
        args_file_path = os.path.join(args.output_dir, "args.json") # <-- Changed to .json

        # Create a logger that writes to both the console and the log file
        class Logger(object):
            def __init__(self, filename, stream=sys.stdout):
                self.terminal = stream
                self.log = open(filename, 'w')

            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
                self.flush()

            def flush(self):
                self.terminal.flush()
                self.log.flush()

        # Redirect stdout and stderr to the logger
        sys.stdout = Logger(log_file_path, sys.stdout)
        sys.stderr = Logger(log_file_path, sys.stderr)

        # Save the script arguments to a JSON file
        with open(args_file_path, "w") as f:
            json.dump(vars(args), f, indent=4)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)


        # ✨ Initialize wandb with the custom name
        wandb.init(
            project="mae-pretraining", 
            config=args, 
            name=args.output_dir, # Set the run name here
            sync_tensorboard=True
        )
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                            skipless=args.skipless, 
                                            mimetic=args.mimetic,
                                            unet_style=args.unet_style,
                                            W_v=args.W_v, W_p=args.W_p)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if log_writer is not None:
        wandb.watch(model, log_freq=100)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    if args.opt == 'soap':
        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = DistributedShampoo(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.95),
        epsilon=1e-8,
        weight_decay=args.weight_decay,
        max_preconditioner_dim=8192,
        precondition_frequency=100,
        use_decoupled_weight_decay=True,
        preconditioner_config=DefaultEigenvalueCorrectedShampooConfig,
                    distributed_config=DDPShampooConfig(
                communication_dtype=torch.float32,
                num_trainers_per_group=4,
                communicate_params=False,
            ),
        )
    else:
        optimizer = create_optimizer_v2(
            model_without_ddp,
            opt= getattr(args, 'opt', 'adamw'), 
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95) # For AdamW

        )
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, 
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                 epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "training_log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.mimetic is not None:
        args.output_dir =  f"output/pretrain_{args.model}_ep{args.epochs}_opt{args.opt}_blr{args.blr}_skipless{args.skipless}_svdortho_mimetic{args.mimetic[0]}_{args.mimetic[1]}_unet{args.unet_style}_Wv{args.W_v}_Wp{args.W_p}"
    else:
        args.output_dir =  f"output/pretrain_{args.model}_ep{args.epochs}_opt{args.opt}_blr{args.blr}_skipless{args.skipless}_unet{args.unet_style}"
    args.log_dir = args.output_dir
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
