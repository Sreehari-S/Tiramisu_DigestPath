import glob
import random
import os
import os.path
import numpy as np
import cv2
import torch
import imutils
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import pdb


import torch
import torchvision
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

from skimage.transform import resize

import sys



IMG_EXTENSIONS = ['.tif','.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.npz']


def is_image_file_and_exists(filename):
	"""Checks if a file is an image.

	Args:
		filename (string): path to a file

	Returns:
		bool: True if the filename ends with a known image extension
	"""
	filename_lower = filename.lower()
	return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS) and os.path.exists(filename)


def make_image_dataset(dir):
	images = []
	os.urandom(1)
	d = os.path.expanduser(dir)
	for root, _, fnames in sorted(os.walk(d)):
		random.shuffle(fnames)
		for fname in sorted(fnames):
			path = os.path.join(root, fname)        
			if is_image_file_and_exists(path) and 'mask' not in path:
				images.append(path)
	return images
 

def make_mask_dataset(images):
	masks = []
	new_images = []
	count = 0
	paired_images = []
	for i, image_path in enumerate(images):
		mask_path = os.path.splitext(image_path)[0]+'_mask.jpg'
		if is_image_file_and_exists(mask_path):
			masks.append(mask_path)
			new_images.append(image_path)
			count += 1
	return new_images, masks, count



class TissueDataset(Dataset):
	"""docstring for TissueDataset"""
	def __init__(self, data_root_dir,transforms=None, 
				 train = True):
		super(TissueDataset, self).__init__()
		self.image_root_dir = data_root_dir
		self.train = train

		self.images = make_image_dataset(self.image_root_dir)
 
		self.images, self.masks, count = make_mask_dataset(self.images)

		if len(self.images) == 0:
		   raise(RuntimeError("Found 0 images in subfolders of: " + self.image_root_dir + "\n"
							   "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

		self.transforms = transforms
		print("Initialized dataset")
	
	def __getitem__(self, index):
		# with SummaryWriter('/home/bmi/DP/src/densenet.pytorch/logs') as writer:
				# t_image = ToTensor()(image)
				# t_mask = ToTensor()(t_mask)
				# writer.add_image("Image",t_image)
				# writer.add_image("Mask",t_mask)
		if self.train:
			image, mask, path = self.make_img_gt_pair(index)
			image = Image.fromarray(image)
			mask  = Image.fromarray(mask)
			if self.transforms is not None:
				image = self.transforms(image)
				mask  = self.transforms(mask)
			return image, mask
		else:
			image,mask,path = self.make_img_gt_pair(index)
			image = Image.fromarray(image)
			mask = Image.fromarray(mask)
			if self.transforms is not None:
				image  = self.transforms(image)
				mask  = self.transforms(mask)
			return image, mask
		
	def __len__(self):
		return len(self.images)  
	  

	def make_img_gt_pair(self, index):
		"""
		Make the image-ground-truth pair
		"""
		path = self.images[index]

		image = cv2.imread(path)

		if self.train:
			h, w, _ = image.shape
			if self.masks[index] is None:
				mask = np.zeros(image.shape[:2], dtype=np.uint8)
				print("Mask not found!!!")
			else:
				mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
			mask = cv2.resize(mask, (w, h))

			while(1):
				white = np.where(mask > 0)
				white_ind = random.randint(0,len(white[0])-1)
				off_x = random.randint(-256,256)
				off_y = random.randint(-256,256)
				p_w = p_h = 512
				p_center = (white[0][white_ind] + off_x ,white[1][white_ind] + off_y)
				if p_center[0] >= (h - p_h//2) or p_center[0]<= p_h//2 or p_center[1] >= (w - p_w//2) or p_center[1] <= p_w//2:
					continue
				
				p_x  = int(p_center[0] - p_h/2.)
				p_y  = int(p_center[1] - p_w/2.)

				if p_x < 0:
					p_x = 0
					p_h = int(p_center[0]*2)
				if p_y < 0:
					p_y = 0
					p_w = int(p_center[1]*2)
				
				if p_x + p_h > h:
					p_h = h - p_x - 1
				if p_y + p_w > w:
					p_w = w - p_y - 1
				
				patch_image = image[p_x: p_x + p_h , p_y : p_y + p_w]
				patch_mask  = mask[p_x: p_x + p_h , p_y : p_y + p_w]

			return patch_image,patch_mask
		else:
			h, w, _ = image.shape
			if self.masks[index] is None:
				mask = np.zeros(image.shape[:2], dtype=np.uint8)
				print("Mask not found!!!")
			else:
				mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
			mask = cv2.resize(mask, (w, h))
			return image,mask


if __name__ == '__main__':


	dataset = TissueDataset(data_root_dir=r"/media/mirl/mirlproject2/Datasets/Challenge_Datasets/Histopathology/DigestPath2019/DigestPath2019/Colonoscopy_tissue_segment_dataset/tissue-train-pos")
	print("Length of Dataset:{}".format(len(dataset)))
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
	for i, batch in enumerate(dataloader):
		print(batch.size())
	#     grid_img = torchvision.utils.make_grid(batch, nrow=1)
	#     plt.figure()
	#     plt.imshow(grid_img.permute(1, 2, 0))
		break


