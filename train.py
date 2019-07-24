#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from maskdata import TissueDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from callbacks import *

from torch.utils.data import DataLoader
from tqdm import tqdm
from network.tiramisu import *
from utils import *

import os
import sys
import math
import timeit

import shutil

import setproctitle
import pdb
from torchvision.transforms import ToTensor, ToPILImage




def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--trainbatchSz', type=int, default=4)
	parser.add_argument('--validbatchSz', type=int, default=1)
	parser.add_argument('--nEpochs', type=int, default=300)
	parser.add_argument('--no-cuda', action='store_true')
	# parser.add_argument('--save',help="save folder path",default='/home/bmi/DP/src/densenet.pytorch/save')
	parser.add_argument('--save',help="save folder path",default='/home/uavws/DP/Tiramisu_DigestPath/checkpoints')
	# parser.add_argument('--data_root_dir',help="data root dir",default="/home/bmi/DP/tissue-train-pos")
	parser.add_argument('--train_root_dir',help="data root dir",default="/home/uavws/DP/tissue-train-pos")
	parser.add_argument('--valid_root_dir',help="data root dir",default="/home/uavws/DP/valid")
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--opt', type=str, default='adam',
						choices=('sgd', 'adam', 'rmsprop'))

	parser.add_argument('--lr',type=float,default= 1e-2)
	parser.add_argument('--resume_epoch',type=int,default=0)
	parser.add_argument('--save_epoch',type=int,default=50)
	parser.add_argument('--network',type=str,default='fcd103')
	parser.add_argument('--n_classes',type=int,default=1)
	parser.add_argument('--criterion',type=str,default='bce_dice')
	parser.add_argument('--patch_ratio',type = int,default =6)
	parser.add_argument('--valid_interval',type = int,default =100)

	args = parser.parse_args()

	args.cuda = not args.no_cuda and torch.cuda.is_available()
	args.save = args.save or 'work/densenet.base'
	setproctitle.setproctitle(args.save)

	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	if os.path.exists(args.save):
		shutil.rmtree(args.save)
	os.makedirs(args.save, exist_ok=True)

	normMean = [0.49139968, 0.48215827, 0.44653124]
	normStd = [0.24703233, 0.24348505, 0.26158768]
	unorm = UnNormalize(mean=normMean, std=normStd)

	trainTransform = copy_func(my_segmentation_transforms)
	validTransform = copy_func(my_segmentation_transforms)

	kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

	traindset = TissueDataset(root_dir=args.train_root_dir , transforms=trainTransform, train = True)
	validdset = TissueDataset(root_dir=args.valid_root_dir , transforms=validTransform, train = False)

	##############create train and vali loaders 

	trainLoader = DataLoader(
		traindset,
		batch_size=args.trainbatchSz, shuffle=True, **kwargs)
	validLoader = DataLoader(
		validdset,
		batch_size=args.validbatchSz, shuffle=False, **kwargs)


	if args.network == 'fcd103':
		net = FCDenseNet103(n_classes = args.n_classes)
	elif args.network == 'fcd67':
		net = FCDenseNet67(n_classes = args.n_classes)
	elif args.network == 'fcd57':
		net = FCDenseNet57(n_classes = args.n_classes)
	else:
		raise("Network not found !")

	print('  + Number of params: {}'.format(
		sum([p.data.nelement() for p in net.parameters()])))
	if args.cuda:
		net = net.cuda()

	if args.opt == 'sgd':
		optimizer = optim.SGD(net.parameters(), lr=1e-1,
							momentum=0.9, weight_decay=1e-4)
	elif args.opt == 'adam':
		optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
	elif args.opt == 'rmsprop':
		optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

	
	if args.criterion == 'bce_dice':
		criterion = copy_func(dice_coef_loss_bce)
	elif args.criterion == 'bce':
		criterion = copy_func(bce_loss)
	elif args.criterion == 'dice':
		criterion = copy_func(dice_loss)
	else:
		raise("Criterion not found!")

	logs = {'batch_tags': {} , 'epoch_tags': {} , 'datapt': 0}

	if args.resume_epoch != 0:
		print('Loading checkpoint from epoch {%d}'.format(args.resume_epoch))
		state_dict = torch.load(args.save + +"/Epoch_{}.tar".format(args.resume_epoch))
		net = state_dict['model_state_dict']
		optimizer = state_dict['optim_state_dict']
		start_epoch = state_dict['epoch']
		start_score = state_dict['score']
	else:
		start_epoch = 1
		start_score = -1.0
		logs['epoch_tags']['score'] = -1.0

	# Callbacks
	tbx_logger = Tensorboard(dir_ = './logs')
	tqdm_logger = Tqdm()
	#####################set parameters
	scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=6, 
				verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-12, eps=1e-08)
	scheduler_logger = Scheduler(scheduler)  
	##########arguments
	cpt_saver = ModelCheckpoint(dir_=args.save,model=net,optimizer=optimizer,interval=args.save_epoch,init_score=start_score)

	callbacks = [tbx_logger,tqdm_logger,cpt_saver]



	def train(args, epoch, net, trainLoader, optimizer,criterion):
		net.train()
		nTrain = len(trainLoader.dataset)
		pbar = tqdm(trainLoader)
		pbar.set_description('Epoch : {}'.format(epoch))
		for batch_idx, (data, target) in enumerate(pbar):
			start_time = timeit.default_timer()
			if args.cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)
			optimizer.zero_grad()
			output = net(data)
			loss = criterion(target,output)
			loss.backward()
			optimizer.step()
			logs['batch_tags']['logs'] = loss.item()
			logs['datapt'] += len(output)
			if logs['datapt']%10 == 0:
				logs['batch_tags']['image'] = ( unorm(data[0].cpu()),unorm(target[0].cpu()),unorm(output[0].cpu()) )
			for callback in callbacks:
				callback.on_batch_end(logs)
			stop_time = timeit.default_timer()
			exec_time = stop_time - start_time
			pbar.set_postfix(BatchLoss = round(loss.item(),3) , Exec_Time = str(round(exec_time,2))+'s/batch' ) 

	def validate(args, epoch, net, validLoader, optimizer,criterion):
		net.eval()
		tqdm.write("VALIDATING...")
		valid_loss = 0
		incorrect = 0
		logs['epoch_tags']['image'] = []
		pbar = tqdm(validLoader)
		with torch.no_grad():
			for data, targets in pbar:    
				for img_,target_ in zip(data,targets):
					img = ToPILImage()(img_)
					imwidth,imheight = img.size 
					plist = crop(img, imwidth//args.patch_ratio, imheight//args.patch_ratio)
					outlist = []
					for rlist in plist:
						out = []
						for i in range(len(rlist)):
							rlist[i] = ToTensor()(rlist[i])
							if args.cuda:
								rlist[i] = rlist[i].unsqueeze(0).cuda()
							image = rlist[i]
							output = net(image)
							output = ToPILImage()(output.cpu().squeeze(0))
							out.append(output)
							print(count)
						outlist.append(out)

					newimg = attach(outlist,imwidth//args.patch_ratio,imheight//args.patch_ratio,imwidth,imheight)
					logs['epoch_tags']['image'].append((unorm(img_.cpu()) ,unorm(target_.cpu()),unorm(ToTensor()(newimg))))  
					valid_loss += criterion(target_,ToTensor()(newimg))
			valid_loss /= len(validLoader) # loss function already averages over batch size
			logs['epoch_tags']['score'] = (1.0 -round(valid_loss.item(),3))
			for callback in callbacks:
				callback.on_val_end(logs)

	for epoch in range(start_epoch, args.nEpochs + 1):
		adjust_opt(args.opt, optimizer, epoch)
		train(args, epoch, net, trainLoader, optimizer,criterion)
		logs['epoch'] = epoch
		if epoch % args.valid_interval == 0:
			validate(args, epoch, net, validLoader, optimizer,criterion)
		for callback in callbacks:
			callback.on_epoch_end(logs)
		break

if __name__=='__main__':
	main()
