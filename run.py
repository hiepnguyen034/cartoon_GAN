import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Generator import Generator_Net 
from Discriminator import Discriminator_Net
import imageio

import math
def get_args():

	parser = argparse.ArgumentParser('Training DCGAN')

	parser.add_argument('--img_path',
		type=str,
		default ='./cartoon',
		help = 'path to img')

	parser.add_argument('--img_size',
		type=int,
		default = 64,
		help = 'size of img')

	parser.add_argument('--workers',
		type=int,
		default =2,
		help = 'number of workers')

	parser.add_argument('--cuda',
		type=bool,
		default= True,
		help= 'use GPU or not')

	parser.add_argument('--nc',
		type=int,
		default =3,
		help='number of channels')

	parser.add_argument('--nz',
		type=int,
		default= 100,
		help=' num features of latent vector')

	parser.add_argument('--ngf',
		type= int,
		default=64,
		help ='num features of generator')

	parser.add_argument('--ndf',
		type=int,
		default = 64,
		help = 'num of features for discriminator')

	parser.add_argument('--num_epochs',
		type = int,
		default = 20,
		help = 'number of epoches')

	parser.add_argument('--lr',
		type=float,
		default = .0002,
		help = 'learning rate',
		)

	parser.add_argument('--beta1',
		type= float,
		default = 0.5,
		help = 'beta 1')

	parser.add_argument('--ngpu',
		type = int,
		default = 1,
		help = 'num of gpu'
		)

	parser.add_argument('--batch_size',
		type=int,
		default = 64,
		help='num batch_size')

	parser.add_argument('--generate_img', 
		type = bool, 
		default =False, 
		help = 'generate img')

	parser.add_argument('--begin_train',
		type = bool,
		default = False,
		help = 'start training')

	parser.add_argument('--model_path',
		type = str,
		default = 'models/netG.pth',
		help = 'path to generator model')

	parser.add_argument('--make_grid',
		type = bool,
		default = True,
		help = 'output grid or single img?')

	parser.add_argument('--num_img',
		type=int,
		default = 64,
		help = 'how many images you want the model to generate?')

	args= parser.parse_args()

	return args

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

def get_device(cuda):
	if not torch.cuda.is_available():
		return "cpu"

	return "cuda:0" if cuda else "cpu"


def get_imgloader(img_path):
	data = dset.ImageFolder(root = img_path,
							transform= transforms.Compose([
							transforms.Resize(args.img_size),
							transforms.CenterCrop(args.img_size),
							transforms.ToTensor(),
							transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
							]))
	img_loader = torch.utils.data.DataLoader(data, batch_size = args.batch_size,
											shuffle=True, num_workers=args.workers)

	return img_loader	


def train_model(args):
	device = get_device(args.cuda)
	netG = Generator_Net(args.nz,args.ngf,args.nc).to(device)
	netD = Discriminator_Net(args.nc,args.ndf).to(device)
	netG.apply(weights_init)
	netD.apply(weights_init)

	criterion = nn.BCELoss()

	input_noise = torch.randn(64,args.nz, 1, 1, device = device)

	real_label = 1
	fake_label = 0

	opt_G = optim.Adam(netG.parameters(), lr =args.lr, betas=(args.beta1, 0.999))
	opt_D = optim.Adam(netD.parameters(), lr =args.lr, betas=(args.beta1, 0.999))

	img_list = []
	G_losses = []
	D_losses = []
	iters = 0

	img_loader = get_imgloader(args.img_path)
	print("Starting Training Loop on {}...".format(device))

	for epoch in range(args.num_epochs):

		for i, data in enumerate(img_loader, 0):
			#Calculate & backward loss D: maximize log(D(x)) + log(1 - D(G(z)))
			real_cpu = data[0].to(device)

			b_size = real_cpu.size(0)
			label = torch.full((b_size,), real_label, device= device)
			output = netD(real_cpu).view(-1)

			lossD_real = criterion(output,label)

			lossD_real.backward()
			D_x = output.mean().item()

			noise = torch.randn(b_size,args.nz, 1, 1, device= device)

			fake = netG(noise)
			label.fill_(fake_label)


			output = netD(fake.detach()).view(-1)

			lossD_fake = criterion(output,label)
			lossD_fake.backward()
			D_G_z1 = output.mean().item()

			lossD = lossD_real + lossD_fake

			opt_D.step()
			netD.zero_grad()

			#Calculate & backward loss G: maximize log(D(G(z)))
			label.fill_(real_label)
			output=netD(fake).view(-1)

			lossG= criterion(output,label)
			lossG.backward()

			D_G_z2 = output.mean().item()

			opt_G.step()
			netG.zero_grad()
		# Output training stats
				

			if i % 50 == 0:
				print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
					  % (epoch, args.num_epochs, i, len(img_loader),
						 lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))

			# Save Losses for plotting later
			G_losses.append(lossD.item())
			D_losses.append(lossG.item())

			# Check how the generator is doing by saving G's output on fixed_noise
			if (iters % 500 == 0) or ((epoch == args.num_epochs-1) and (i == len(img_loader)-1)):
				with torch.no_grad():
					fake = netG(input_noise).detach().cpu()
				img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

			if i % 50 == 0:
				print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
					  % (epoch, args.num_epochs, i, len(img_loader),
						 lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))

			# Save Losses for plotting later
			G_losses.append(lossG.item())
			D_losses.append(lossD.item())

			# Check how the generator is doing by saving G's output on fixed_noise
			if (iters % 500 == 0) or ((epoch == args.num_epochs-1) and (i == len(img_loader)-1)):
				with torch.no_grad():
					fake = netG(input_noise).detach().cpu()
				img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

			iters += 1

			fig = plt.figure(figsize=(8,8))
			plt.axis("off")

	for i, img in enumerate(img_list):
		pic = np.transpose(img,(1,2,0))
		plt.imshow(pic)
		imageio.imwrite('output_img/image_{}.jpg'.format(i), pic)

	torch.save(netG.state_dict(), 'netG.pth')
	torch.save(netD.state_dict(),'netD.pth')


def generate_img(args):
	netG = Generator_Net(args.nz,args.ngf,args.nc)
	netG.load_state_dict(torch.load(args.model_path,
	 					map_location =  lambda storage, loc: storage)
						)
	noise = torch.randn(args.num_img,args.nz, 1, 1, device = 'cpu')
	out = netG(noise).detach().cpu()
	if args.make_grid:
		img = vutils.make_grid(out, nrow=int(math.sqrt(args.num_img)), padding=2, normalize=True)
		img = np.transpose(img,(1,2,0))
		imageio.imwrite('output_img/single_output.jpg',img)

	else:
		print('generate single image')
		for i in range(out.shape[0]):
			img=np.transpose(out[i,:,:,:],(1,2,0))
			imageio.imwrite('output_img/single_outpu{}.jpg'.format(i),img)

if __name__ == '__main__':
	args = get_args()
	print(args)
	if args.begin_train:
		train_model(args)
	if args.generate_img:
		generate_img(args)	






