import torch
from PIL import Image
from torch.nn import functional as F
import types
import torchvision.transforms as transforms
import functools
import pdb
import numpy as np
import torchvision.transforms.functional as TF
import random




class UnNormalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, tensor):
		"""
		Args:
			tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
		Returns:
			Tensor: Normalized image.
		"""
		mean = self.mean
		std = self.std
		if tensor.shape[0] == 1:
			mean = [np.mean(self.mean),]
			std  = [np.mean(self.std),]
		for t, m, s in zip(tensor, mean, std):
			t.mul_(s).add_(m)  #unnormalize
			# t.sub_(m).div_(s)  #normalize
		return tensor
		
def crop(im, height, width):
	patchlist = []
	k = 0
	imgwidth, imgheight = im.size
	for i in range(0,imgheight,height):
		rlist = []
		for j in range(0,imgwidth,width):
			x = j+width
			y = i+height
			if x > imgwidth:
				x = imgwidth
			if y > imgheight:
				y = imgheight
			box = (j, i, x, y)
			a = im.crop(box)
			rlist.append(a)
		patchlist.append(rlist)
	return patchlist

def attach(p_list ,width,height,imgwidth,imgheight):
	new_im = Image.new('L', (imgwidth,imgheight))
	for i,r in enumerate(p_list):
		x = width*i
		for j,patch in enumerate(r):
			y = height*j 
			try:
				new_im.paste(patch, (y,x))
			except:
				pdb.set_trace()
	return new_im

def bce_loss(true, logits, pos_weight=None):
	"""Computes the weighted binary cross-entropy loss.
	Args:
		true: a tensor of shape [B, 1, H, W].
		logits: a tensor of shape [B, 1, H, W]. Corresponds to
			the raw output or logits of the model.
		pos_weight: a scalar representing the weight attributed
			to the positive class. This is especially useful for
			an imbalanced dataset.
	Returns:
		bce_loss: the weighted binary cross-entropy loss.
	"""
	bce_loss = F.binary_cross_entropy_with_logits(
		logits.float(),
		true.float(),
		pos_weight=pos_weight,
	)
	return bce_loss


def dice_loss(target,pred):
	"""This definition generalize to real valued pred and target vector.
This should be differentiable.
	pred: tensor with first dimension as batch
	target: tensor with first dimension as batch
	"""

	smooth = 1.

	# have to use contiguous since they may from a torch.view op
	iflat = pred.contiguous().view(-1)
	tflat = target.contiguous().view(-1)
	intersection = (iflat * tflat).sum()

	A_sum = torch.sum(iflat * iflat)
	B_sum = torch.sum(tflat * tflat)
	
	return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5):
	return bce_loss(y_true, y_pred) * bce + dice_loss(y_true, y_pred) * dice

def copy_func(f):
	"""Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
	g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
						   argdefs=f.__defaults__,
						   closure=f.__closure__)
	g = functools.update_wrapper(g, f)
	g.__kwdefaults__ = f.__kwdefaults__
	return g


def adjust_opt(optAlg, optimizer, epoch):
	if optAlg == 'sgd':
		if epoch < 150: lr = 1e-1
		elif epoch == 150: lr = 1e-2
		elif epoch == 225: lr = 1e-3
		else: return

		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

def resize2d(img):
	img_PIL = transforms.ToPILImage()(img)
	img_PIL = transforms.Resize([512,512])(img_PIL)
	img_PIL = transforms.ToTensor()(img_PIL)
	return img_PIL


def my_segmentation_transforms(image, segmentation,train):

	if train == True:
		image = TF.resize(image,(256))
		segmentation = TF.resize(segmentation,(256))

		# if random.random() > 5:
		angle = random.randint(-90, 90)
		image = TF.rotate(image, angle)
		segmentation = TF.rotate(segmentation, angle)
		
		# more transforms ...
		hT = random.sample([True,False],1)[0]
		if (hT):
			image = TF.hflip(image)
			segmentation = TF.hflip(segmentation)

		vT = random.sample([True,False],1)[0]

		if (vT):
			image = TF.vflip(image)
			segmentation = TF.vflip(segmentation)
		
	image = TF.to_tensor(image)
	segmentation = TF.to_tensor(segmentation)

	normMean = [0.49139968, 0.48215827, 0.44653124]
	normStd =  [0.24703233, 0.24348505, 0.26158768]
	image = TF.normalize(image,normMean,normStd)
	segmentation = TF.normalize(segmentation,[np.mean(normMean),],[np.mean(normStd),])

	return image, segmentation