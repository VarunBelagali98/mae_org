import torch
import torch.nn as nn
import os
import numpy as np

import torch
import torch.nn as nn
import os
import numpy as np

__all__ = [
	'Policy'
]

class Policy(nn.Module):
	""" Policy Network for masking
	"""
	def __init__(self, args):
		super().__init__()

		self.args = args
		kernel_size = args.encoder_embed_dim//args.policy_network_dim
		self.avg_pool = nn.AvgPool1d(kernel_size, stride=kernel_size)

		self.mask_policy_layer = nn.Linear(in_features=args.policy_network_dim, out_features=1, bias=False)
		self.gpu = args.gpu
		self.policy_network_dim = args.policy_network_dim

		print("args.policy_network_paramters", args.policy_network_parameters)
		print("masking ratio", args.mask_ratio)

		self.pca_basis = nn.Linear(in_features=args.encoder_embed_dim, out_features=args.policy_network_dim, bias=False)
  
		if args.policy_network_parameters != None:
			self.initialize_policy(args.policy_network_parameters)

		if args.pca_param_path != None:
			self.initialize_pca()

	def forward(self, x):
		#x_downsample = self.avg_pool(x)
		x_downsample = self.pca_reduce_dim(x)
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

	def forward_sigmoid(self, x):
		x_downsample = self.avg_pool(x)
		mask_policy_output = self.mask_policy_layer(x_downsample)
		#mask_policy_output = mask_policy_output.squeeze(dim=-1)
		mask_scores = torch.sigmoid(mask_policy_output)
		return mask_scores

	def pca_reduce_dim_deprecated(self, x):
		x = x.to(torch.float64)
		x_mean = torch.mean(x, dim=-2, keepdim=True)
		x_var = torch.var(x, dim=-2, keepdim=True)
		z = (x - x_mean) / (x_var + 1e-10)  #whiteing
		#print(z.shape)
		[u, s, v] = torch.pca_lowrank(z, q=self.policy_network_dim, center=False)
		x_downsample = torch.matmul(z, v)
		x_downsample = x_downsample.to(torch.float16)
		return x_downsample

	def pca_reduce_dim(self, x):
		x = (x - self.global_mean) / self.global_std
		x_downsample = self.pca_basis(x)
		return x_downsample

	def initialize_pca(self):
		print("Initialzing pca prams")
		print(self.args.pca_param_path)
		self.global_mean = torch.load(self.args.pca_param_path+"/global_mean.pt")
		self.global_std = torch.load(self.args.pca_param_path+"/global_std.pt")
		eigenvalues = torch.load(self.args.pca_param_path+"/eigenvalues.pt")
		eigenvectors = torch.load(self.args.pca_param_path+"/eigenvectors.pt")

		eigenvectors = eigenvectors[:, :self.args.policy_network_dim]
		eigenvectors = torch.t(eigenvectors)

		#print("eigen vectors")
		#print(eigenvectors)
  
		self.global_mean = torch.unsqueeze(self.global_mean, 0)
		self.global_std = torch.unsqueeze(self.global_std, 0)
  
		self.global_mean = self.global_mean.to(self.args.gpu)
		self.global_std = self.global_std.to(self.args.gpu)

		#print("eigen values")
		#print(eigenvalues.real)
		eigenvectors = eigenvectors.real

		self.global_mean = self.global_mean.to(torch.float16)
		self.global_std = self.global_std.to(torch.float16)
		eigenvectors = eigenvectors.to(torch.float16)
		#eigenvalues = eigenvalues.to(torch.float16)


		print("mean", self.global_mean.shape)
		print("std", self.global_std.shape)


		print("pca_basis", eigenvectors.shape)
		print("pca eigen vectos", eigenvectors)

		self.pca_basis.weight = torch.nn.parameter.Parameter(eigenvectors, requires_grad=False)
		
		print("pca_basics layer", self.pca_basis.weight)


class Policy_dep(nn.Module):
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