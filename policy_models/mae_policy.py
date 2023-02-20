from jinja2 import TemplateError
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import trunc_normal_

import math
import numpy as np

from policy_models.mae_encoder import build_2d_sincos_position_embedding, MAEViTEncoder
from policy_models.Policy import Policy
from policy_models.PatchEmbed2D import PatchEmbed2D

def patchify_image(x: Tensor, patch_size: int = 16):
	# patchify input, [B,C,H,W] --> [B,C,gh,ph,gw,pw] --> [B,gh*gw,C*ph*pw]
	B, C, H, _ = x.shape
	grid_size = H // patch_size

	x = x.reshape(B, C, grid_size, patch_size, grid_size, patch_size) # [B,C,gh,ph,gw,pw]
	x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, grid_size ** 2, C * patch_size ** 2) # [B,gh*gw,C*ph*pw]

	return x

def batched_shuffle_indices(batch_size, length, device):
	"""
	Generate random permutations of specified length for batch_size times
	Motivated by https://discuss.pytorch.org/t/batched-shuffling-of-feature-vectors/30188/4
	"""
	rand = torch.rand(size=(batch_size, length), device=device)
	batch_perm = rand.argsort(dim=1)
	return batch_perm

class MAE_Policy(nn.Module):
	""" Vision Transformer with support for patch or hybrid CNN input stage
	"""
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.grid_size = grid_size = args.input_size // args.patch_size

		# build positional encoding for encoder and decoder
		with torch.no_grad():
			self.encoder_pos_embed = build_2d_sincos_position_embedding(grid_size, 
																		args.encoder_embed_dim, 
																		num_tokens=1)

		embed_layer = PatchEmbed2D
		self.encoder = MAEViTEncoder(patch_size=args.patch_size,
							   in_chans=args.in_chans,
							   embed_dim=args.encoder_embed_dim,
							   depth=args.encoder_depth,
							   num_heads=args.encoder_num_heads,
							   embed_layer=embed_layer)

		self.mask_policy_network = Policy(args)


	def forward(self, x):
		args = self.args
		batch_size = x.size(0)
		in_chans = x.size(1)
		assert in_chans == args.in_chans
		out_chans = in_chans * args.patch_size ** 2
		assert x.size(2) == x.size(3) == args.patch_size * self.grid_size, "Unmatched patch size and grid size"
		x = patchify_image(x, args.patch_size) # [B,gh*gw,C*ph*pw]

		pos_embed = self.encoder_pos_embed.expand(x.size(0), -1, -1)
		x = self.encoder(x, pos_embed)[:, 1:] # discard CLS
		mask_scores = self.mask_policy_network(x)
		return mask_scores.detach()

	def initialize_policy(self, policy_network_parameters):
		self.mask_policy_network.initialize_policy(policy_network_parameters)
