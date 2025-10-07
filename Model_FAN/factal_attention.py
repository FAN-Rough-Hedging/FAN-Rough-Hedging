import torch
import torch.nn as nn

class FactualAttention(nn.Module):
	def __init__(self,
				 series_len:int,
				 hurst:float,
				 d_v:int,
				 d_emb:int,
				 bias:bool=False,
				 device:str='cuda',
				 dtype:torch.dtype=torch.float32):
		'''
		:param series_len: the length of the hedge series
		:param hurst: Hurst Exponent
		:param d_v: Value matrix dimension
		:param d_emb: Embedding dimension
		:param bias: whether to use learnable bias
		:param device: torch.Tensor on cpu or cuda
		:param dtype: dtype of torch.Tensor
		'''
		super().__init__()
		assert d_v > 0
		assert series_len > 0

		self.d_v = d_v
		self.series_len = series_len
		self.hurst = hurst - 0.5
		self.dt = 1 / series_len
		self.Wv = nn.Linear(in_features=d_emb, out_features=d_v, bias=bias)
		self.device = device
		self.dtype = dtype
		self.A = self.__factual_attention_matrix__()

	def __factual_attention_matrix__(self) -> torch.Tensor:
		A = torch.zeros(size=(self.series_len, self.series_len), dtype=self.dtype)
		for i in range(self.series_len):
			for j in range(i):
				A[i, j] = ((i-j)*self.dt + 1e-8) ** self.hurst
		A += torch.eye(n=self.series_len, dtype=self.dtype) * (1e-8) ** self.hurst
		A = A.reshape(self.series_len, 1, self.series_len).repeat(1, self.d_v, 1)
		return A.detach().to(self.device) 

	def forward(self, embed_series:torch.Tensor) -> torch.Tensor:
		V = self.Wv(embed_series.transpose(-1,-2)).transpose(-1,-2).unsqueeze(0).repeat(self.series_len, 1, 1)
		return torch.sum(self.A * V, dim=-1, keepdim=False)

