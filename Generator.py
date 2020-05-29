import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import numpy as np


class Generator_Net(nn.Module):
	def __init__(self, nz, ngf, nc):
		'''
		@param nz: num of features for latent vector z
		@param ngf: num of features for the Generator
		@param nc: number of channels
		'''
		super().__init__()
		self.nz = nz
		self.ngf = ngf
		self.nc = nc

		self.network = nn.Sequential(
			nn.ConvTranspose2d(in_channels = self.nz, 
								out_channels= self.ngf*8, 
								kernel_size = 4, 
								stride = 1,
								 padding = 0, 
								 bias = False
								 ),
			nn.BatchNorm2d(num_features= self.ngf*8),
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels= self.ngf*8, 
								out_channels=self.ngf*4,
								kernel_size=4,
								stride=2,
								padding =1,
								bias = False
								),
			nn.BatchNorm2d(num_features = self.ngf*4),
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=self.ngf*4,
								out_channels=self.ngf*2,
								kernel_size=4,
								stride=2,
								padding =1,
								bias = False
								),
			nn.BatchNorm2d(num_features= self.ngf*2),
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels = self.ngf*2,
								out_channels=self.ngf,
								kernel_size=4,
								stride=2,
								padding=1,
								bias=False),
			nn.BatchNorm2d(num_features=self.ngf),
			nn.ReLU(True),
			nn.ConvTranspose2d(self.ngf,self.nc,4,2,1,bias=False),
			nn.Tanh()
			)

	def forward(self,x):
		return self.network(x)