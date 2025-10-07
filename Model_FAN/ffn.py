import torch
import torch.nn as nn

class LinearGELU(nn.Module):
	def __init__(self, in_features, out_features, bias:bool=False, multi_head:int=1, layerize:bool=False):
		'''
		:param in_feature: 输入矩阵的行数量，即最后一个维度的大小
		:param out_feature: 输出矩阵的行数量，即最后一个维度的大小
		:param bias: 是否启用偏置
		:param multi_head: 多头数量，输入矩阵的第0个维度
		:param layerize: 是否启用层次化？当启用层次化的时候，可将该模块当作 nn.Act(nn.Linear(x))的组合，此时multi_head的数量固定为1
		'''
		super().__init__()
		self.gelu = nn.ModuleList([nn.GELU() for _ in range(multi_head)])
		self.linear = nn.ModuleList(
			[nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
				 for _ in range(multi_head)]
		)
		self.multi_head = multi_head
		self.in_features = in_features
		self.layerize = layerize
		if layerize:
			self.multi_head = 1
		
		for linear in self.linear:
			linear.weight = torch.nn.init.xavier_normal_(linear.weight)

	def forward(self, x):
		shape = x.shape
		x = x.transpose(-1,-2)
		if self.layerize:
			x = x.unsqueeze(0)
		if x.shape[0] != self.multi_head:
			raise ValueError(f"x.shape[0] != multi head num! x shape is {x.shape} instead!")
		if x.shape[-1] != self.in_features:
			raise ValueError(f"x.shape[-1] != {self.in_features}! x shape is {x.shape} instead!")

		_x = torch.empty(size=x.shape,device=x.device.type).float()

		for i,head_func in enumerate(self.linear):
			_x[i,...] = self.gelu[i](head_func(x[i,...]))

		return _x.reshape(shape)

	def weight_dict(self):
		dct = {}
		for i, linear in enumerate(self.linear):
			dct[f'linear{i}'] = linear.weight
		return dct

	def weightInit(self, InitWeight_dict: dict):
		if len(InitWeight_dict.keys()) != len(self.linear):
			raise ValueError(f"Init weight dict must have {len(self.linear)} elements!got {len(InitWeight_dict.keys())} instead!")

		for i, linear in InitWeight_dict.items():
			self.linear[i].weight = linear

class LinearSigmoid(nn.Module):
	def __init__(self, in_features, out_features, bias:bool=False, multi_head:int=1, layerize:bool=False):
		'''
		:param in_feature: 输入矩阵的行数量，即最后一个维度的大小
		:param out_feature: 输出矩阵的行数量，即最后一个维度的大小
		:param bias: 是否启用偏置
		:param multi_head: 多头数量，输入矩阵的第0个维度
		:param layerize: 是否启用层次化？当启用层次化的时候，可将该模块当作 nn.Act(nn.Linear(x))的组合，此时multi_head的数量固定为1
		'''
		super(LinearSigmoid, self).__init__()
		self.sigmoid = nn.ModuleList([nn.Sigmoid() for _ in range(multi_head)])
		self.linear = nn.ModuleList(
			[nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
			 for _ in range(multi_head)]
		)
		self.multi_head = multi_head
		self.in_features = in_features
		self.layerize = layerize
		if layerize:
			self.multi_head = 1

	def forward(self, x):
		shape = x.shape
		x = x.transpose(-1, -2)
		if self.layerize:
			x = x.unsqueeze(0)
		if x.shape[0] != self.multi_head:
			raise ValueError(f"x.shape[0] != multi head num! x shape is {x.shape} instead!")
		if x.shape[-1] != self.in_features:
			raise ValueError(f"x.shape[-1] != {self.in_features}! x shape is {x.shape} instead!")

		_x = torch.empty(size=x.shape, device=x.device.type).float()

		for i, head_func in enumerate(self.linear):
			_x[i, ...] = self.sigmoid[i](head_func(x[i,...]))

		return _x.reshape(shape)

	def weight_dict(self):
		dct = {}
		for i, linear in enumerate(self.linear):
			dct[f'linear{i}'] = linear.weight
		return dct

	def weightInit(self, InitWeight_dict: dict):
		if len(InitWeight_dict.keys()) != len(self.linear):
			raise ValueError(
				f"Init weight dict must have {len(self.linear)} elements!got {len(InitWeight_dict.keys())} instead!")

		for i, linear in InitWeight_dict.items():
			self.linear[i].weight = linear

class LinearSwish(nn.Module):
	def __init__(self, in_features, out_features, bias:bool=False, beta:float=1.0, multi_head:int=1, layerize:bool=False):
		'''
		:param in_feature: 输入矩阵的行数量，即最后一个维度的大小
		:param out_feature: 输出矩阵的行数量，即最后一个维度的大小
		:param bias: 是否启用偏置
		:param multi_head: 多头数量，输入矩阵的第0个维度
		:param layerize: 是否启用层次化？当启用层次化的时候，可将该模块当作 nn.Act(nn.Linear(x))的组合，此时multi_head的数量固定为1
		'''
		super(LinearSwish, self).__init__()
		self.sigmoid = nn.ModuleList([nn.Sigmoid() for _ in range(multi_head)])
		self.beta = beta
		self.linear = nn.ModuleList(
			[nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
			 for _ in range(multi_head)]
		)
		self.multi_head = multi_head
		self.in_features = in_features
		self.layerize = layerize
		if layerize:
			self.multi_head = 1

	def forward(self, x):
		shape = x.shape
		x = x.transpose(-1, -2)
		if self.layerize:
			x = x.unsqueeze(0)
		if x.shape[0] != self.multi_head:
			raise ValueError(f"x.shape[0] != multi head num! x shape is {x.shape} instead!")
		if x.shape[-1] != self.in_features:
			raise ValueError(f"x.shape[-1] != {self.in_features}! x shape is {x.shape} instead!")

		_x = torch.empty(size=x.shape, device=x.device.type).float()

		for i, head_func in enumerate(self.linear):
			tmp = head_func(x[i,...])
			_x[i, ...] = tmp*self.sigmoid[i](tmp*self.beta)

		return _x.reshape(shape)

	def weight_dict(self):
		dct = {}
		for i, linear in enumerate(self.linear):
			dct[f'linear{i}'] = linear.weight
		return dct

	def weightInit(self, InitWeight_dict: dict):
		if len(InitWeight_dict.keys()) != len(self.linear):
			raise ValueError(
				f"Init weight dict must have {len(self.linear)} elements!got {len(InitWeight_dict.keys())} instead!")

		for i, linear in InitWeight_dict.items():
			self.linear[i].weight = linear