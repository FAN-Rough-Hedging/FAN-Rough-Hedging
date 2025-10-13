import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional
from tqdm import tqdm
import os
import time

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
ABS_FATHER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ModelController(object):
	def __init__(self,
				 option_type:str='euro',	
				 device:str='cuda',
				 dataset_name:Optional[str]=None,
				 train_test_validation_split:Optional[list]=None):
		if dataset_name is None:
			train_dataset = {'St':torch.load(f'{DATASET_PATH}\\train_St_series_list.pt', weights_only=False),
						 	'delta_dict':torch.load(f'{DATASET_PATH}\\train_delta_series_list.pt', weights_only=False)}
			test_dataset = {'St':torch.load(f'{DATASET_PATH}\\test_St_series_list.pt', weights_only=False),
							'delta_dict':torch.load(f'{DATASET_PATH}\\test_delta_series_list.pt', weights_only=False)}
			validation_dataset =  {'St':torch.load(f'{DATASET_PATH}\\validation_St_series_list.pt', weights_only=False),
							   	'delta_dict':torch.load(f'{DATASET_PATH}\\validation_delta_series_list.pt', weights_only=False)}
			mode = 'stimulation'
		else:
			assert sum(train_test_validation_split) == 1
			try:
				total_dataset = pd.read_csv(f'{DATASET_PATH}\\{dataset_name}')['St'].tolist()
				len_dataset = len(total_dataset)

				train_end = len_dataset*train_test_validation_split[0]
				test_end = len_dataset*(train_test_validation_split[0]+train_test_validation_split[1])

				train_dataset = total_dataset[:train_end]
				test_dataset = total_dataset[train_end:test_end]
				validation_dataset = total_dataset[test_end:]
				mode = 'real'
			except:
				raise ValueError(f'未适配真实数据集相关代码')

		self.mode = mode
		self.device = device
		self.train_dataloader = DataLoader(train_dataset, mode=self.mode, option_type=option_type)
		self.test_dataloader = DataLoader(test_dataset, mode=self.mode, option_type=option_type)
		self.validation_dataloader = DataLoader(validation_dataset, mode=self.mode, option_type=option_type)

	def trainingModel(
		self, model:nn.Module, epochs:int=1,  batch_size:int=1000, 
		Optimizer:torch.optim.Optimizer=None, lr_scheduler:torch.optim.lr_scheduler._LRScheduler=None
	) -> (list, list, torch.nn.Module):

		time_start = time.time()
		self.train_dataloader.setBatchSize(batch_size)
		self.test_dataloader.setBatchSize(batch_size)

		model = model.to(self.device)
		if Optimizer is None:
			Optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.99))
		criterion = MSPE().to(self.device)

		training_loss = []
		testing_loss = []

		tmp_losses = torch.zeros(batch_size).to(self.device)
		if self.mode == 'stimulation':
			bar = tqdm(range(epochs), desc='Model_FAN is training...')
			for epoch in bar:

				model.train()
				with torch.no_grad():
					tmp_losses *= 0
				
				Optimizer.zero_grad()

				idx = 0
				for St, target_delta in self.train_dataloader:
					St:torch.Tensor
					target_delta:torch.Tensor
					St = St.to(self.device)
					pred_delta = model(St)
					batch_loss:torch.Tensor = criterion(
							pred_delta=pred_delta, 
							St=St, 
							target_delta=target_delta.to(self.device)
						)
					tmp_losses[idx] = batch_loss.detach()
					idx += 1
					(batch_loss/batch_size).backward(retain_graph=False)
					
				Optimizer.step()
				if lr_scheduler is not None:
					lr_scheduler.step()
				self.train_dataloader.setBatchSize(
					int(
						batch_size * (1/(epoch+1)) ** 0.35
					)
				)
				training_loss.append(torch.mean(tmp_losses).detach().cpu().numpy())

				model.eval()
				idx = 0
				with torch.no_grad():
					tmp_losses *= 0
					for St, target_delta in self.test_dataloader:
						St = St.to(self.device)
						pred_delta = model(St)
						tmp_losses[idx] = criterion(
							pred_delta=pred_delta, 
							St=St, 
							target_delta=target_delta.to(self.device)
						)
						idx += 1
					testing_loss.append(torch.mean(tmp_losses).detach().cpu().numpy())

				bar.set_description(
					f'epoch:[{epoch+1}/{epochs}], train loss:[{training_loss[-1]}], test loss:[{testing_loss[-1]}]')
				
				# checkpoint
				if epoch + 1 > epochs*0.5 and(epoch+1)%100==0:
					torch.save(model.state_dict(), f'{ABS_FATHER_PATH}\\Trained_Model\\Checkpoint\\FAN{epoch+1}Epoch{int(training_loss[-1])}TrainLoss{int(testing_loss[-1])}TestLoss_time{time_start}.pth')


		elif self.mode == 'real':
			bar = tqdm(range(epochs), desc='Model_FAN is training...')
			for epoch in bar:

				model.train()
				with torch.no_grad():
					tmp_losses *= 0
				idx = 0

				Optimizer.zero_grad()
				for St in self.train_dataloader:
					St = St.to(self.device)
					pred_delta = model(St)
					batch_loss = criterion(pred_delta=pred_delta, St=St)
					tmp_losses[idx] = batch_loss.detach()
					idx += 1
					(batch_loss/batch_size).backward(retain_graph=False)
				
				Optimizer.step()
				if lr_scheduler is not None:
					lr_scheduler.step()
				training_loss.append(torch.mean(tmp_losses).detach().cpu().numpy())


				model.eval()
				idx = 0
				with torch.no_grad():
					tmp_losses *= 0
					for St in self.test_dataloader:
						St = St.to(self.device)
						pred_delta = model(St)
						tmp_losses[idx] = criterion(pred_delta=pred_delta, St=St)
						idx += 1
					testing_loss.append(torch.mean(tmp_losses))

				bar.set_description(
					f'epoch:[{epoch + 1}/{epochs}], train loss:[{training_loss[-1]}], test loss:[{testing_loss[-1]}]')

		return (training_loss, testing_loss, model.cpu())

	def validationModel(self, model:nn.Module):
		model = model.to(self.device)
		criterion = MSPE().to(self.device)
		model.eval()
		print(self.validation_dataloader.batch_size)
		validation_losses = torch.zeros(self.validation_dataloader.__len__()).to(self.device)
		with torch.no_grad():
			idx = 0
			delta_pair = []
			if self.mode == 'stimulation':
				for St, target_delta in self.validation_dataloader:
					St:torch.Tensor
					target_delta:torch.Tensor
					St = St.to(self.device)
					pred_delta = model(St)
					validation_losses[idx] = criterion(
							pred_delta=pred_delta, 
							St=St, 
							target_delta=target_delta.to(self.device)
						)
					idx += 1
					delta_pair.append((pred_delta, target_delta))
				return (torch.mean(validation_losses).detach().cpu().numpy(), torch.std(validation_losses).detach().cpu().numpy(), delta_pair)

			elif self.mode == 'real':
				for St in self.validation_dataloader:
					St:torch.Tensor
					St = St.to(self.device)
					pred_delta = model(St)
					validation_losses[idx] = criterion(pred_delta=pred_delta, St=St)
					idx += 1
				return (torch.mean(validation_losses).detach().cpu().numpy(), torch.std(validation_losses).detach().cpu().numpy())
		


class MSPE(nn.Module):
	def __init__(self):
		super().__init__()
	
	def forward(self,
				pred_delta:torch.Tensor,
				target_delta:torch.Tensor,
				St:torch.Tensor) -> torch.Tensor:

		device = pred_delta.device
		dSt = torch.concat(
				[torch.zeros(1).to(device), St[1:].detach()-St[:-1].detach()],
				dim=0
			)
		loss1 = torch.mean(
				torch.cumsum(
					(pred_delta - target_delta) * dSt, dim=0)**2
		)
		loss2 = (pred_delta/torch.norm(pred_delta)) @ (target_delta/torch.norm(target_delta+1e-8)).view(-1,1)
		return loss1 + loss2

		

class DataLoader(object):
	def __init__(self, dataset:dict[str, list], mode:str, option_type:str='euro'):
		assert option_type in ['euro', 'asia'], 'option_type must be "euro" or "asia"'
		if mode == 'stimulation':
			self.dataset = {'St':dataset['St'].to('cuda'), 'delta':dataset['delta_dict'][option_type].to('cuda')}
			length = len(self.dataset['St'])
		else:
			self.dataset = {'St':dataset['St']}.to('cuda')
			length = len(self.dataset['St'])
		index = np.arange(length)
		np.random.shuffle(index)

		self.index = index.tolist()
		self.mode = mode
		self.batch_size = length
		self.current_batch = 0

	def __iter__(self):
		self.current_batch = 0
		return self

	def __next__(self):
		if self.current_batch < self.batch_size:
			self.current_batch += 1
			idx = self.index.pop(0)
			self.index.append(idx)
			if self.mode == 'stimulation':
				return self.dataset['St'][idx], self.dataset['delta'][idx]
			else:
				return self.dataset['St'][idx]
		else:
			self.current_batch = 0
			raise StopIteration

	def __len__(self):
		return self.batch_size

	def setBatchSize(self, batch_size):
		if self.batch_size > batch_size:
			self.batch_size = batch_size

__all__ = ['ModelController']


if __name__ == '__main__':
	modelController = ModelController()
	modelController.train_dataloader.setBatchSize(8)
	for i in modelController.train_dataloader:
		print(i)


