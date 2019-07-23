import numpy as np 
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import torch
import pdb
from tqdm import tqdm
import socket
import timeit



class Callback(object):
	"""Abstract base class used to build new callbacks"""
	def __init__(self):
		super(Callback, self).__init__()

	def on_batch_end(self,Logs):
		pass

	def on_epoch_end(self,logs):
		pass



class Tensorboard(Callback):
	""" Class to log all the required details during training/validation.
	Input is a string which will be the folder name (Under Logs) where logs are saved into."""	

	def __init__(self,dir_):	
		super(Tensorboard,self).__init__()	
		# Create a SummaryWriter object
		self.dir = dir_
		self.writer = SummaryWriter(log_dir=self.dir)

	# Function to log data
	def on_epoch_end(self,logs):
		# Logs data at the epoch end - mostly validation results
		for tag in logs['epoch_tags']:
			if 'logs' in tag:
				self.writer.add_scalar(tag,logs['epoch_tags'][tag],logs['epoch'])
			elif 'image' in tag:
				for img in logs['epoch_tags'][tag]:
					self.writer.add_image('Image',img[0],logs['epoch'])
					self.writer.add_image('Target',img[1],logs['epoch'])
					self.writer.add_image('Output',img[0],logs['epoch'])
			else:
				pass

	def on_batch_end(self,logs):
		# Logs data during training
		for tag in logs['batch_tags']:
			if 'logs' in tag:
				self.writer.add_scalar(tag,logs['batch_tags'][tag],logs['datapt'])
			elif 'image' in tag:
				self.writer.add_image('Image',img[0],logs['datapt'])
				self.writer.add_image('Target',img[1],logs['datapt'])
				self.writer.add_image('Output',img[0],logs['datapt'])
			else:
				pass


class ModelCheckpoint(Callback):
	""" Class to save the checkpoints. It takes in a model and corresponding optimizer
	and saves all the details along with epoch number. Two modes can be specified - "all"
	or "best". In "all" mode, a checkpoint is created in every epoch. Whereas in "best" 
	mode only if the passed parameter has an improved score or value, the checkpoint is
	saved."""
	def __init__(self,dir_,model,optimizer,interval,init_score):
		super(ModelCheckpoint,self).__init__()
		self.model = model
		self.optimizer = optimizer
		self.cpt_dir = dir_
		self.interval = interval
		os.makedirs(self.cpt_dir,exist_ok=True)
		self.best_score = init_score

	def on_epoch_end(self,logs):
		# Create a state dictionary for saving
		self.state_dict = {'epoch': logs['epoch'], 'model_state_dict':self.model.state_dict(), 
									 'optim_state_dict':self.optimizer.state_dict(), 'score': logs['epoch_tags']['score']}

		if logs['epoch_tags']['score'] > self.best_score:
			self.best_score=logs['epoch_tags']['score']
			torch.save(self.state_dict,self.cpt_dir+"/best_score.tar")	
		# Always save the last model
		if logs['epoch'] % self.interval == 0:
			torch.save(self.state_dict,self.cpt_dir+"/{Epoch_%d}.tar".format(logs['epoch']))



class Tqdm(Callback):
	""" Class used to print the validation results in the command line"""

	def __init__(self):
		super(Tqdm,self).__init__()

	def on_epoch_end(self,logs):
		score = logs['epoch_tags']['score']
		tqdm.write('Validation_avg_dice_score = {%.3f}',score )

class Scheduler(Callback):
	""" Class updates the scheduler after each epoch """
	def __init__(self,scheduler):
		super(Scheduler,self).__init__()
		self.scheduler = scheduler

	def on_epoch_end(self,logs):
		if self.scheduler.__class__.__name__=="ReduceLROnPlateau":
			# self.scheduler.step(logs['epoch_tags']['Validation/Accuracy'])
			self.scheduler.step((1 - ['epoch_tags']['score']))
		elif self.scheduler.__class__.__name__=="StepLR":
			self.scheduler.step()
		else:
			tqdm.write("Scheduler type mismatch")