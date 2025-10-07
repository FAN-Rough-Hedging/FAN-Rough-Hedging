"""
High-Frequency Option Hedging Simulation - Data Processing Module

This module provides data generation and processing capabilities for high-frequency option hedging simulations.
It includes classes for generating synthetic financial time series data with fractional Brownian motion,
calculating optimal delta hedging strategies for European and Asian options, and managing datasets.

Key Features:
- Fractional Brownian motion simulation with configurable Hurst parameter
- European and Asian option delta calculation using Monte Carlo methods
- Memory-efficient data generation and storage
- GPU acceleration support via PyTorch
- Visualization capabilities for generated data

Classes:
    DataGenerator: Main class for generating synthetic financial time series and option hedging data
    DataProcessor: Utility class for processing and analyzing generated data

Author: [Your Name]
Date: [Current Date]
Version: 1.0
"""

from pandas.core.missing import F
from sympy import series
import torch
import torch.nn as nn
import psutil
import itertools
from torch.nn.modules import L1Loss
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
import os
import time

# Global constant for dataset storage path
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')

class DataGenerator(object):
	"""
	Data Generator for High-Frequency Option Hedging Simulation
	
	This class generates synthetic financial time series data using fractional Brownian motion
	and calculates optimal delta hedging strategies for European and Asian options.
	The generated data is used for training machine learning models in option hedging strategies.
	
	Key Methods:
	- __init__: Initialize the data generator with market parameters
	- _InnerFunc_DataGenerator_: Main data generation pipeline
	- _InnerFunc_StGenerator: Generate underlying asset price series and delta hedging strategies
	- __InnerFunc_Wperp_Wv_Gen_Euro__: Calculate European option delta using Monte Carlo simulation
	- __InnerFunc_Wperp_Wv_Gen_Asia__: Calculate Asian option delta using Monte Carlo simulation
	- _InnerFunc_GenerateCombinedSeries_: Generate correlated Brownian motion paths
	- _InnerFunc_L_Operator_Generator_: Generate Cholesky decomposition for fractional Brownian motion
	
	Workflow:
	1. __init__ -> _InnerFunc_L_Operator_Generator_ -> _InnerFunc_GenerateCombinedSeries_ -> _InnerFunc_DataGenerator_
	2. _InnerFunc_DataGenerator_ -> _InnerFunc_StGenerator
	3. _InnerFunc_StGenerator -> __InnerFunc_Wperp_Wv_Gen_Euro__ / __InnerFunc_Wperp_Wv_Gen_Asia__
	"""
	
	def __init__(self,
				 series_len: int,
				 rou: float = None,
				 train_series_amount: int = 16,
				 test_series_amount: int = 8,
				 validation_series_amount: int = 8,
				 mcmc_num: int = 10,
				 series_dtype: torch.dtype = torch.float32,
				 hurst: float = 0.1,
				 S_0: float = 100,
				 sigma_0: float = 0.2,
				 r: float = 0.03,
				 K:float = 100,
				 device: str = 'cuda',
				 is_visualize: bool = False,
				 num_workers: int = 1):
		"""
		Initialize the DataGenerator with market parameters and simulation settings.
		
		This constructor sets up all necessary parameters for generating synthetic financial data,
		validates system resources, and initializes the fractional Brownian motion infrastructure.
		
		Args:
			series_len (int): Length of each time series (number of time steps)
			rou (float, optional): Correlation coefficient between two Brownian motions. 
								  If None, defaults to sqrt(0.5)
			train_series_amount (int): Number of training time series to generate
			test_series_amount (int): Number of testing time series to generate
			validation_series_amount (int): Number of validation time series to generate
			mcmc_num (int): Number of Monte Carlo simulations for delta calculation
			series_dtype (torch.dtype): Data type for tensor operations
			hurst (float): Hurst parameter for fractional Brownian motion (0 < H < 1)
			S_0 (float): Initial underlying asset price
			sigma_0 (float): Initial volatility
			r (float): Risk-free interest rate (annualized)
			K (float): Strike price for options
			device (str): Computing device ('cuda' or 'cpu')
			is_visualize (bool): Whether to generate visualization plots
			num_workers (int): Number of CPU workers for parallel processing
		
		Raises:
			ValueError: If insufficient memory is available for data generation
			AssertionError: If invalid parameter values are provided
		"""

		# Parameter validation - ensure all inputs are within valid ranges
		assert series_len > 0, "Series length must be positive"
		assert mcmc_num > 0, "Monte Carlo simulation count must be positive"
		assert train_series_amount > 0, "Training series amount must be positive"
		assert test_series_amount > 0, "Test series amount must be positive"
		assert validation_series_amount > 0, "Validation series amount must be positive"
		assert is_visualize in [True, False], 'is_visualize should be True or False'
		assert device in ['cuda', 'cpu'], "Device must be 'cuda' or 'cpu'"
		
		# GPU availability check - fallback to CPU if CUDA is not available
		if device == 'cuda':
			if not torch.cuda.is_available():
				print(f'\033[33m[WARNING] CUDA is not available, falling back to CPU \033[0m')
				device = 'cpu'

		# Memory requirement calculation and validation
		# Estimate memory needed for storing all generated time series data
		test_tensor = torch.zeros([], dtype=series_dtype)
		total_number = (train_series_amount + test_series_amount + validation_series_amount)
		require_memory_size = total_number * series_len * test_tensor.element_size() * 2  # Factor of 2 for safety margin

		# Check system memory availability
		mem = psutil.virtual_memory()
		available_mem = mem.available
		if available_mem <= require_memory_size:
			raise ValueError(f"""Insufficient memory: required {require_memory_size/(1024**3):.2f} GB, but only {available_mem/(1024**3):.2f} GB available.""")
		else:
			print(f'Memory check passed: {available_mem/(1024**3):.2f} GB available, {require_memory_size/(1024**3):.2f} GB required')

		# Store simulation parameters
		self.mcmc_num = mcmc_num  # Number of Monte Carlo simulations for delta calculation
		self.is_visualize = is_visualize  # Flag for generating visualization plots
		self.series_len = series_len  # Length of each time series
		self.dt = torch.tensor([1 / series_len])  # Time step size (normalized to unit interval)
		self.device = device  # Computing device (CPU or GPU)
		
		# Store dataset size parameters
		self.train_series_amount = int(train_series_amount)
		self.test_series_amount = int(test_series_amount)
		self.validation_series_amount = int(validation_series_amount)
		
		# Store market parameters
		self.hurst = hurst  # Hurst parameter for fractional Brownian motion
		self.sigma_0 = sigma_0  # Initial volatility
		self.S_0 = torch.tensor([S_0])  # Initial underlying asset price
		self.r = r/series_len  # Risk-free rate (adjusted for time step)
		self.K = torch.tensor([K])  # Strike price for options
		
		# Handle correlation parameter
		if rou is None:
			self.rou = torch.sqrt(torch.tensor(0.5))  # Default correlation
		else:
			self.rou = torch.tensor(rou)  # User-specified correlation
			
		# Store technical parameters
		self.series_dtype = series_dtype  # Data type for tensor operations
		self.num_workers = num_workers  # Number of CPU workers

		# Initialize fractional Brownian motion infrastructure
		self.L = self._InnerFunc_L_Operator_Generator_()  # Cholesky decomposition matrix
		
		# Calculate number of Brownian motion pairs needed
		# Using combinatorial formula: C(n,2) >= total_series_needed
		n = int(torch.sqrt(
			2 * torch.tensor(
				[self.train_series_amount + self.test_series_amount + self.validation_series_amount]
			)
		).item()) + 2
		
		# Generate correlated Brownian motion paths
		self.series_dic_lst = self._InnerFunc_GenerateCombinedSeries_(n)
		
		# Start the main data generation process
		self._InnerFunc_DataGenerator_()


	def _InnerFunc_DataGenerator_(self) -> (list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]):
		train_series_list, train_delta = self._InnerFunc_StGenerator(amount=self.train_series_amount, hedge_type=('euro'))

		test_series_list, test_delta = self._InnerFunc_StGenerator(amount=self.test_series_amount, hedge_type=('euro'))
		val_series_list, val_delta = self._InnerFunc_StGenerator(amount=self.validation_series_amount, hedge_type=('euro'))

		torch.save(train_series_list, f'{DATASET_PATH}/train_St_series_list.pt')
		torch.save(test_series_list, f'{DATASET_PATH}/test_St_series_list.pt')
		torch.save(val_series_list, f'{DATASET_PATH}/validation_St_series_list.pt')

		torch.save(train_delta, f'{DATASET_PATH}/train_delta_series_list.pt')
		torch.save(test_delta, f'{DATASET_PATH}/test_delta_series_list.pt')
		torch.save(val_delta, f'{DATASET_PATH}/validation_delta_series_list.pt')

		if self.is_visualize:
			for i in range(self.train_series_amount):
				plt.plot(train_series_list[i])
			plt.title(f'train series')
			plt.show()

			for i in range(self.train_series_amount):
				plt.plot(train_delta['euro'][i])
			plt.title(f'train delta series(euro)')
			plt.show()

			for i in range(self.train_series_amount):
				plt.plot(train_delta['asia'][i])
			plt.title(f'train delta series(asia)')
			plt.show()


	def _InnerFunc_StGenerator(self, amount:int, hedge_type:tuple=('euro','asia')) -> (torch.Tensor, dict[torch.Tensor]):
		'''
		return: 
		 St, best_delta_dict = { euro | asia : torch.Tensor(size=(amount, series_len))}
		'''
		assert 'euro' in hedge_type or 'asia' in hedge_type

		series_dic_lst = self.series_dic_lst
		St_series_dataset = []
		sigma_t_dataset = []

		# St (Underlying Asset) series generation
		with torch.no_grad():
			for idx in tqdm(range(amount), desc='St series generating...'):
				Wv = series_dic_lst[idx][0]
				Ws = series_dic_lst[idx][1]
				Vt = (self.L @ Wv.reshape(-1, 1)).reshape(-1)
				EVt2 = torch.mean(Vt * Vt)
				sigma_t = self.sigma_0 * torch.exp((Vt - 0.5 * EVt2) * 0.5)
				Zt = self.rou * Wv + torch.sqrt(1 - self.rou ** 2) * Ws
				dZt = Zt - torch.concat([Zt[0].unsqueeze(0).detach(), Zt[:-1].detach()], dim=0)

				St = torch.cat([ 
					self.S_0,
					self.S_0 * torch.cumprod(1 + sigma_t[:-1].detach() * dZt[:-1].detach(), dim=0)]
				)

				St_series_dataset.append(St)
				sigma_t_dataset.append(sigma_t**2)

			best_delta_dataset = {}
			exponent = torch.tensor([self.hurst - 0.5])

			# generate Lt
			L_t = []
			for t in tqdm(range(self.series_len), desc='L_t series generating...'):
				L = _InnerFunc_L_Operator_Generator_at_time_(
					u=t, series_len=self.series_len, hurst=self.hurst, dt=self.dt
				)
				L_t.append(L)
			sqrt_rho_square = torch.sqrt(1 - self.rou ** 2)
			sigma_eps = torch.tensor([5e-3])
			if 'euro' in hedge_type:
				# european buy call option
				sigma_t_dataset = torch.stack(sigma_t_dataset)
				best_delta_dataset['euro'] = torch.zeros(size=(amount, self.series_len))

				for i in tqdm(range(amount), desc='euro delta* series generating...'):
					St = St_series_dataset[i]
					sigma_t = sigma_t_dataset[i]
					best_delta = torch.zeros(size=(1,self.series_len))
					for t in range(self.series_len):
						EWperp_t, EWv_t = self.__InnerFunc_Wperp_Wv_Gen_Euro__(
							S_t=St, K=self.K, sigma_t=sigma_t,
							rho=self.rou, sqrt_rho=sqrt_rho_square, 
							dt=self.dt, sqrt_dt=torch.sqrt(self.dt), 
							mcmc_num=self.mcmc_num,
							series_len= self.series_len, t=t,
							H=exponent, init_seed = int(time.time()),
							L_t=L_t[t]
						)
						best_delta[:,t] = (self.rou * EWv_t + torch.sqrt(1-self.rou ** 2) * EWperp_t) / (St[t] * torch.max(sigma_t[t], sigma_eps))
						
					best_delta_dataset['euro'][i] = best_delta

			if 'asia' in hedge_type:
				# asian average buy call option
				if isinstance(sigma_t_dataset, list):
					sigma_t_dataset = torch.stack(sigma_t_dataset)
				best_delta_dataset['asia'] = torch.zeros(size=(amount, self.series_len))

				for i in tqdm(range(amount), desc='asia delta* series generating...'):
					St = St_series_dataset[i]
					sigma_t = sigma_t_dataset[i]
					best_delta = torch.zeros(size=(1,self.series_len))
					for t in range(self.series_len):
						EWperp_t, EWv_t = self.__InnerFunc_Wperp_Wv_Gen_Asia__(
							S_t=St, K=self.K, sigma_t=sigma_t,
							rho=self.rou, sqrt_rho=sqrt_rho_square,
							dt=self.dt, sqrt_dt=torch.sqrt(self.dt),
							mcmc_num=self.mcmc_num,
							series_len=self.series_len, t=t,
							H=exponent, init_seed = int(time.time()),
							L_t=L_t[t]
						)
						best_delta[:,t] = (self.rou * EWv_t + torch.sqrt(1-self.rou ** 2) * EWperp_t) / (St[t] * torch.max(sigma_t[t], sigma_eps))
					best_delta_dataset['asia'][i] = best_delta

		return torch.stack(St_series_dataset), best_delta_dataset

	@torch.jit.script
	def __InnerFunc_Wperp_Wv_Gen_Euro__(
								S_t: torch.Tensor,
								K: torch.Tensor,
								sigma_t: torch.Tensor,
								rho: torch.Tensor,
								sqrt_rho: torch.Tensor,
								dt: torch.Tensor,
								sqrt_dt: torch.Tensor,
								mcmc_num: int,
								series_len: int,
								t: int,
								H: torch.Tensor,
								init_seed: int,
								L_t: torch.Tensor):
		'''
		S_t: [ series_len ]
		K: [ 1 ]
		sigma_t: [ series_len ]
		rho: [ 1 ]
		dt: [ 1 ]
		Wperp_container: [ mcmc_num ]
		Wv_container: [ mcmc_num ]
		Wperp_motion: [ series_len ]
		Wv_motion: [ series_len ]
		H: [ 1 ]
		'''

		Wperp_container = torch.jit.annotate(torch.Tensor, torch.zeros(mcmc_num, dtype=torch.float32))
		Wv_container = torch.jit.annotate(torch.Tensor, torch.zeros(mcmc_num, dtype=torch.float32))
		
		fenzi = torch.zeros(1)
		fenmu = torch.zeros(1)
		part21 = torch.zeros(1)
		part22 = torch.zeros(1)

		for k in range(mcmc_num):
			# generate the Wperp & Wv path
			torch.manual_seed(init_seed+k)
			dWperp = torch.randn(series_len) * sqrt_dt
			torch.manual_seed(init_seed+k+1)
			dWv = torch.randn(series_len) * sqrt_dt
			fenzi *= 0
			fenmu *= 0
			part21 *= 0
			part22 *= 0
			for u in range(t, series_len):
				factor = rho * dWperp[u] + sqrt_rho * dWv[u]
				fenzi += sigma_t[u] * factor
				fenmu += sigma_t[u] ** 2 * dt
				if u != t:
					# part2 += ((u - t)/series_len) ** H * (factor - sigma_t[u]** 2 *dt)
					part22 += ((u - t)/series_len) ** H * (- sigma_t[u]** 2 *dt)

			torch.manual_seed(init_seed+k*t+2)
			# gen the dZ from t to T: [series_len-t]

			dZ = torch.randn(series_len-t, dtype=torch.float32) * sqrt_dt         
			# gen the integral L_t @ dZ.reshape(-1, 1) = [series_len-t, series_len-t] * [series_len-t,1] = [series_len-t,1]
			ser = (L_t @ dZ.reshape(-1, 1)).reshape(-1)
			for u in range(series_len-t):
				part21 += ser[u]

			Wperp_container[k] = sqrt_rho * sigma_t[t] * fenzi[0] / fenmu[0]
			Wv_container[k] =  (1-t/series_len) + part21[0] + part22[0]

		ST = S_t[-1].detach()
		Wperp_t = torch.zeros(1, dtype=torch.float32)
		Wv_t = torch.zeros(1, dtype=torch.float32)
		for k in range(mcmc_num):
			Wperp_t += Wperp_container[k]
			Wv_t += Wv_container[k]

		EWperp_t = torch.max(ST-K, torch.zeros(1)) * Wperp_t / mcmc_num
		EWv_t = torch.max(ST-K, torch.zeros(1)) * Wv_t / mcmc_num
		return EWperp_t, EWv_t
	

	@torch.jit.script
	def __InnerFunc_Wperp_Wv_Gen_Asia__(
								S_t: torch.Tensor,
								K: torch.Tensor,
								sigma_t: torch.Tensor,
								rho: torch.Tensor,
								sqrt_rho: torch.Tensor,
								dt: torch.Tensor,
								sqrt_dt: torch.Tensor,
								mcmc_num: int,
								series_len: int,
								t: int,
								H: torch.Tensor,
								init_seed: int,
								L_t: torch.Tensor):
		'''
		S_t: [ series_len ]
		K: [ 1 ]
		sigma_t: [ series_len ]
		rho: [ 1 ]
		dt: [ 1 ]
		Wperp_container: [ mcmc_num ]
		Wv_container: [ mcmc_num ]
		Wperp_motion: [ series_len ]
		Wv_motion: [ series_len ]
		H: [ 1 ]
		'''

		Wperp_container = torch.zeros(mcmc_num, dtype=torch.float32)
		Wv_container = torch.zeros(mcmc_num, dtype=torch.float32)

		fenzi = torch.zeros(1)
		fenmu = torch.zeros(1)
		part21 = torch.zeros(1)
		part22 = torch.zeros(1)

		for k in range(mcmc_num):
			# generate the Wperp & Wv path
			torch.manual_seed(init_seed+k)
			dWperp = torch.randn(series_len) * sqrt_dt
			torch.manual_seed(init_seed+k+1)
			dWv = torch.randn(series_len) * sqrt_dt
			fenzi *= 0
			fenmu *= 0
			part21 *= 0
			part22 *= 0
			for u in range(t, series_len):
				factor = rho * dWperp[u] + sqrt_rho * dWv[u]
				fenzi += sigma_t[u] * factor
				fenmu += sigma_t[u] ** 2 * dt
				if u != t:
					# part2 += ((u - t)/series_len) ** H * (factor - sigma_t[u]** 2 *dt)
					part22 += ((u - t)/series_len) ** H * (- sigma_t[u]** 2 *dt)

			                                                # [series_len-t, series_len-t]
			torch.manual_seed(init_seed+k*t+2)
			# gen the dZ from t to T: [series_len-t]
			dZ = torch.randn(series_len-t, dtype=torch.float32) * sqrt_dt         
			# gen the integral L_t @ dZ.reshape(-1, 1) = [series_len-t, series_len-t] * [series_len-t,1] = [series_len-t,1]
			ser = (L_t @ dZ.reshape(-1, 1)).reshape(-1)
			for u in range(series_len-t):
				part21 += ser[u]

			Wperp_container[k] = sqrt_rho * sigma_t[t] * fenzi[0] / fenmu[0]
			Wv_container[k] =  (1-t/series_len) + part21[0] + part22[0]

		mean_S = torch.zeros(1, dtype=torch.float32)
		for t in range(series_len):
			mean_S += S_t[t]
		mean_S /= series_len
		Wperp_t = torch.zeros(1, dtype=torch.float32)
		Wv_t = torch.zeros(1, dtype=torch.float32)
		for k in range(mcmc_num):
			Wperp_t += Wperp_container[k]
			Wv_t += Wv_container[k]
		EWperp_t = torch.max(mean_S-K, torch.zeros(1)) * Wperp_t / mcmc_num
		EWv_t = torch.max(mean_S-K, torch.zeros(1)) * Wv_t / mcmc_num
		return EWperp_t, EWv_t

	def _InnerFunc_GenerateCombinedSeries_(self, n: int) -> torch.Tensor:
		"""
		when it comes to large number of series creation
		we need to use different seed to make sure it would not be the same series
		create an empty tensor to allocate the memory first
		the torch.Tensor is row-major, different to pandas.Dataframe
		because we need to use index to locate the Brown Series having the 1/sqrt(2) variance,
		the max row index should be equal to n
		:param n: the number of series to generate
		:return:
		"""
		seeds = list(range(n))
		brown = []
		factor = torch.sqrt(torch.tensor([1/self.series_len]))

		# standard brownian path generation
		for idx in range(n):
			torch.manual_seed(seeds[idx])
			base_series = torch.cumsum(
				torch.randn(self.series_len,dtype=self.series_dtype),
				dim = 0
			) * factor
			brown.append(base_series)
		combinations = list(itertools.combinations(brown,2))

		count = 0
		for _ in combinations:
			count += 1

		# couple the independent standard brownian path
		combined_series = torch.zeros(size=(count,2,self.series_len))
		count = 0
		for s1, s2 in combinations:
			combined_series[count,0,:] = s1.unsqueeze(0)
			combined_series[count,1,:] = s2.unsqueeze(0)
			count += 1
		return combined_series

	def _InnerFunc_L_Operator_Generator_(self) -> torch.Tensor:
		'''
		calculate the self-covariance matrix
		and decompose the self-covariance matrix into L^TL to get the L Operator
		:return:
		'''
		cov = torch.empty(size=(self.series_len,self.series_len), dtype=torch.float64)
		H = 2 * self.hurst
		for i in range(self.series_len):
			for j in range(self.series_len):
				cov[i][j] = ((i + 1)**H + (j + 1)**H - abs(i-j)**H)
		cov = cov * self.dt**H * 0.5
		eps = torch.tensor(1e-7, dtype=torch.float64)
		cov.diagonal().add_(eps)
		L = torch.linalg.cholesky(cov)
		return L.to(torch.float32)
	
	
@torch.jit.script
def _InnerFunc_L_Operator_Generator_at_time_(u: int, series_len: int, hurst: float, dt: torch.Tensor) -> torch.Tensor:
	cov = torch.zeros(size=(series_len-u, series_len-u), dtype=torch.float64)
	H = 2 * hurst
	for i in range(series_len - u):
		for j in range(series_len - u):
			cov[i][j] = ((i + 1)**H + (j + 1)**H - abs(i-j)**H)
	cov = cov * dt**H * 0.5
	eps = torch.tensor(1e-7, dtype=torch.float64)
	cov.diagonal().add_(eps)
	L = torch.linalg.cholesky(cov)
	return L.to(torch.float32)


if __name__ == '__main__':
	a = DataGenerator(
		series_len=128, 
		rou=-0.7, 
		train_series_amount=4000, 
		test_series_amount=500, 
		validation_series_amount=500,
		mcmc_num=10,
		is_visualize=True
	)



