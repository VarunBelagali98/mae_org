# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn.functional as F
import wandb 

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _, _ = model(samples, mask_ratio=args.mask_ratio)
        

        '''
        with torch.no_grad():
            _, _, _, latent = model(samples, mask_ratio=0)
            if latent != None:
                latent = latent.detach()
                std_cls, cov_cls = misc.vic_reg(latent[:, 0])
                koleo_cls = misc.KoLeoLoss().forward(torch.mean(latent, dim=1))
                std_gap, cov_gap = misc.vic_reg(latent[:, 0])
                koleo_gap = misc.KoLeoLoss().forward(torch.mean(latent, dim=1))'''

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)


        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        if  args.log_to_wandb:
            niters = epoch * len(data_loader) + data_iter_step
            wandb.log(
                        {
                        "lr": lr,
                        "Loss": loss.item(),
                        #"std_cls": std_cls,
                        #"cov_cls": cov_cls,
                        #"koleo_cls": koleo_cls,
                        #"std_gap": std_gap,
                        #"cov_gap": cov_gap,
                        #"koleo_gap": koleo_gap,
                        },
                        step=niters,
                    )



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(model: torch.nn.Module,
                    val_data_loader: Iterable,
                    device: torch.device, epoch: int, niters: int,
                    log_writer=None,
                    args=None):
    
    if (epoch+1) % 10 != 0 and epoch + 1 != args.epochs:
        return None

    features = []
    indexes = []
    model.eval()

    for i, data_item in enumerate(val_data_loader):
        samples, index = data_item
        samples = samples.cuda(device, non_blocking=True)
        with torch.no_grad():
            _, _, _, latent = model(samples, mask_ratio=0)
        f = latent[:, 0]
        f = f.cpu().detach()
        features.append(f)
        indexes.append(index)
    features = torch.cat(features, dim=0)
    indexes = torch.cat(indexes, dim=0)
    #print(features.shape, indexes.shape)

    inv_dist_entropy_norm = misc.evaluate_kmeans_entropy(features, 100, -1, True)
    dist_entropy_norm = misc.evaluate_kmeans_entropy(features, 100, 1, True)
    inv_dist_entropy = misc.evaluate_kmeans_entropy(features, 100, -1, False)
    dist_entropy = misc.evaluate_kmeans_entropy(features, 100, 1, False)


    if  args.log_to_wandb:
        #niters = epoch * len(val_data_loader)
        wandb.log(
            {
            "inv_dist_entropy": inv_dist_entropy,
            "dist_entropy": dist_entropy,
            "inv_dist_entropy_norm": inv_dist_entropy_norm,
            "dist_entropy_norm": dist_entropy_norm
            },
            step=niters,
            )

    return inv_dist_entropy, dist_entropy

