import os
import sys

import numpy as np

import argparse
from omegaconf import OmegaConf
import torch

def get_policy_conf(fname):
    # First of all, parse config file
    conf_file = fname
    assert os.path.exists(conf_file), f"Policy Config file {conf_file} does not exist!"
    conf = OmegaConf.load(conf_file)

    # build argparser from config file in order to overwrite with cli options
    parser = argparse.ArgumentParser(description='SSL pre-training')
    parser.add_argument('conf_file', type=str, help='path to config file')
    for key in conf:
        if conf[key] is None:
            parser.add_argument(f'--{key}', default=None)
        else:
            if key == 'gpu':
                parser.add_argument('--gpu', type=int, default=conf[key])
            elif key == 'multiprocessing_distributed':
                parser.add_argument('--multiprocessing_distributed', type=bool, default=conf[key])
            else:
                parser.add_argument(f'--{key}', type=type(conf[key]), default=conf[key])
    
    args, unknown = parser.parse_known_args()
    print("failed reading", unknown)
    print(args)
    return args

def load_policy_model(model, policy_args):
    if policy_args.backbone_pretrain is not None and os.path.exists(policy_args.backbone_pretrain):
        print(f"=> Loading backbone pretrained weights from {policy_args.backbone_pretrain}")
        checkpoint = torch.load(policy_args.backbone_pretrain, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if key.startswith('encoder.'):
                new_state_dict[key] = state_dict[key]
                del state_dict[key]
                # self.model.load(state_dict)
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f'Loading messages: \n {msg}')
        print("backbone cls token", new_state_dict["encoder.cls_token"])
    else:
        print("backbone paramters for policy network missing")

    if policy_args.policy_network_parameters is not None:
        model.initialize_policy(policy_args.policy_network_parameters)
    else:
        print("main paramters for policy network missing")

    return model