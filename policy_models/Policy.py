import torch
import torch.nn as nn
import os
import numpy as np

class Policy(nn.Module):
	""" Policy Network for masking
	"""
	def __init__(self, args):
		super().__init__()

		kernel_size = args.encoder_embed_dim//args.policy_network_dim
		self.avg_pool = nn.AvgPool1d(kernel_size, stride=kernel_size)

		self.mask_policy_layer = nn.Linear(in_features=args.policy_network_dim, out_features=1, bias=False)
		print("args.policy_network_paramters", args.policy_network_parameters)
		if args.policy_network_parameters != None:
			self.initialize_policy(args.policy_network_parameters)

	def forward(self, x):
		x_downsample = self.avg_pool(x)
		mask_policy_output = self.mask_policy_layer(x_downsample)
		mask_policy_output = mask_policy_output.squeeze(dim=-1)
		nn_softmax = nn.Softmax(dim=1)
		mask_scores = nn_softmax(mask_policy_output)
		return mask_scores

	def initialize_policy(self, policy_network_parameters):
		print("Initialzing model")
		print(policy_network_parameters)
		policy_network_parameters = np.array(policy_network_parameters)
		policy_network_parameters = torch.FloatTensor(policy_network_parameters)
		policy_network_parameters = torch.unsqueeze(policy_network_parameters, dim=0)
		self.mask_policy_layer.weight = torch.nn.parameter.Parameter(policy_network_parameters, requires_grad=False)