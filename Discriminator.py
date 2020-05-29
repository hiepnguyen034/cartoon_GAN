import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np


class Discriminator_Net(nn.Module):
	def __init__(self,nc, ndf, num_extra_layer=0, use_fc=False):
		"""
		@param nc: number of channels
		@param ndf: num of features in the Discriminator
		"""
		super().__init__()

		self.nc = nc
		self.ndf = ndf
		self.network = nn.Sequential(
			nn.Conv2d(in_channels = self.nc,
					  out_channels= self.ndf, 
					  kernel_size= 4,
					  stride=2,
					  padding = 1,
					  bias= False),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(self.ndf,
					self.ndf*2,
					4,2,1,bias=False),
			nn.BatchNorm2d(self.ndf*2),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf*4),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf*8),
			nn.LeakyReLU(0.2,inplace=True)
			)
		if use_fc:
			self.network.add_module('flatten', nn.Flatten())
			self.network.add_module('fc1',self.ndf*8*4*4, 1024)
			self.network.add_module('fc_out',1024,1)
		else:
			self.network.add_module('last-conv2d',nn.Conv2d(self.ndf*8, 1, 4, 1, 0, bias=False))
		self.network.add_module('sigmoid',nn.Sigmoid())
		#output is of shape (batch_size, 1, 1, 1)

	def forward(self,input):
		return self.network(input)