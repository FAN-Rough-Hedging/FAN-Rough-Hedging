from .ffn import LinearSwish, LinearGELU, LinearSigmoid
from .factal_attention import FactualAttention
import torch
import torch.nn as nn

class FAN(nn.Module):
	def __init__(self,
				 series_len:int,
				 hurst:float,
				 d_v:int,
				 d_emb:int,
				 activation:str='swish',
				 bias:bool=False,
				 device:str='cuda',
				 dtype:torch.dtype=torch.float32):
		'''
		:param series_len: the length of the hedge series
		:param hurst: Hurst Exponent
		:param d_v: Value matrix dimension
		:param d_emb: Embedding dimension
		:param activation: activation function
		:param bias: whether to use learnable bias
		:param device: torch.Tensor on cpu or cuda
		:param dtype: dtype of torch.Tensor
		'''
		super().__init__()

		self.series_len = series_len
		self.d_v = d_v
		self.d_emb = d_emb
		self.bias = bias
		self.device = device
		self.dtype = dtype

		self.Emb = nn.Linear(d_emb, series_len, bias=False).to(self.device)
		self.eye = torch.eye(d_emb, d_emb).to(self.device)
		self.zero = torch.zeros(1).to(self.device)

		self.factual_attention_block = FactualAttention(
				 series_len=series_len,
				 hurst=hurst,
				 d_v=d_v,
				 d_emb=d_emb,
				 bias=bias,
				 device=device,
				 dtype=dtype
		)

		self.FFN = LinearSwish(
			in_features=(self.d_v+self.d_emb),
			out_features=(self.d_v+self.d_emb),
			bias=bias,
			layerize=True
		)

		if activation == 'gelu':
			self.FFN = LinearGELU(
				in_features=(self.d_v + self.d_emb),
				out_features=(self.d_v + self.d_emb),
				bias=bias,
				layerize=True
			)

		elif activation == 'sigmoid':
			self.FFN = LinearSigmoid(
				in_features=(self.d_v + self.d_emb),
				out_features=(self.d_v + self.d_emb),
				bias=bias,
				layerize=True
			)

		self.Wo = nn.Linear(in_features=(self.d_v+self.d_emb), out_features=self.series_len, bias=True)

		self.Emb.weight = torch.nn.init.xavier_normal_(self.Emb.weight)
		self.Wo.weight = torch.nn.init.xavier_normal_(self.Wo.weight)

	def forward(self, hedge_series:torch.Tensor) -> (torch.Tensor,torch.Tensor):
		delta_x = torch.concat(
			tensors=[self.zero, hedge_series[1:].detach()-hedge_series[:-1].detach()],
			dim=0
		)
		emb_series = delta_x.unsqueeze(0).repeat(self.d_emb,1) * self.Emb(self.eye)
		C = self.factual_attention_block(emb_series)
		ffn = self.FFN(torch.concat([emb_series, C.transpose(-1,-2)], dim=-2)).transpose(-1,-2)
		return torch.mean(self.Wo(ffn),dim=0)


