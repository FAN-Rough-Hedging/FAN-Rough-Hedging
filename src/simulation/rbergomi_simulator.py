# src/simulation/rbergomi_simulator.py

import numpy as np

def get_fbm_cov_matrix(T, M, H):
    """
    Calculates the covariance matrix for increments of a fractional Brownian motion.
    This is a key component for simulating the fractional noise driving the volatility.
    """
    h = T / M
    gamma = 2 * H
    
    # Vectorized calculation for speed
    i, j = np.mgrid[1:M+1, 1:M+1]
    cov = 0.5 * ( (np.abs(i*h)**gamma) + (np.abs(j*h)**gamma) - (np.abs((i-j)*h)**gamma) )
    
    # Convert covariance of fBM to covariance of its increments
    cov_inc = np.zeros((M, M))
    cov_inc[0,0] = cov[0,0]
    cov_inc[1:, 0] = cov[1:, 0] - cov[:-1, 0]
    cov_inc[0, 1:] = cov[0, 1:] - cov[0, :-1]
    cov_inc[1:, 1:] = cov[1:, 1:] + cov[:-1, :-1] - cov[1:, :-1] - cov[:-1, 1:]
    
    return cov_inc

class PathSimulator:
    """
    A complete path simulator for the Rough Bergomi model.
    It handles both the simulation of historical paths (to create F_t) and
    the conditional simulation of future paths.
    """
    def __init__(self, H, rho, v, sigma0_sq, T):
        self.H = H
        self.rho = rho
        self.v = v
        self.sigma0_sq = sigma0_sq
        self.T = T
        self.sqrt_one_minus_rho_sq = np.sqrt(1 - rho**2)
        self._chol_cache = {}

    def get_cholesky(self, M):
        """Gets or caches the Cholesky decomposition of the fractional noise covariance matrix."""
        if M in self._chol_cache:
            return self._chol_cache[M]
        
        cov_matrix = get_fbm_cov_matrix(1.0, M, self.H) # Time-normalized
        try:
            chol = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            cov_matrix += np.eye(M) * 1e-10 # Add jitter for positive definiteness
            chol = np.linalg.cholesky(cov_matrix)
        self._chol_cache[M] = chol
        return chol

    def simulate_history(self, t_start, M_hist, S0):
        """Simulates paths from time 0 to t_start."""
        # ... (code from your simulate_history function) ...
        # Same as your original code, just ensure S_path[0] = S0
        h = t_start / M_hist
        times = np.linspace(0, t_start, M_hist + 1)
        
        Z_v_hist = np.random.randn(M_hist)
        Z_s_hist = np.random.randn(M_hist)
        
        chol = self.get_cholesky(M_hist)
        dW_v_hist_correlated = chol @ Z_v_hist * np.sqrt(h)
        
        V_path_hist = np.zeros(M_hist + 1)
        for i in range(M_hist):
            kernel_vals = ((times[i+1] - times[:i+1])**(self.H - 0.5))
            V_path_hist[i+1] = np.sum(kernel_vals * dW_v_hist_correlated[:i+1])

        sigma_sq_path = np.zeros(M_hist + 1)
        S_path = np.zeros(M_hist + 1)
        
        sigma_sq_path[0] = self.sigma0_sq
        S_path[0] = S0
        
        for i in range(M_hist):
            # G_t from paper (here as sigma_sq_path)
            sigma_sq_path[i+1] = self.sigma0_sq * np.exp(
                self.v * V_path_hist[i+1] - 0.5 * self.v**2 * times[i+1]**(2*self.H)
            )
            sigma_sq_path[i] = np.maximum(sigma_sq_path[i], 0)
            
            dW_s = Z_s_hist[i] * np.sqrt(h)
            # dZ_t = rho*dW_v + sqrt(1-rho^2)*dW_s
            dZ = self.rho * Z_v_hist[i] * np.sqrt(h) + self.sqrt_one_minus_rho_sq * dW_s

            S_path[i+1] = S_path[i] * np.exp(
                -0.5 * sigma_sq_path[i] * h + np.sqrt(sigma_sq_path[i]) * dZ
            )

        return {
            "S_t": S_path[-1],
            "sigma_sq_t": sigma_sq_path[-1],
            "V_hist": V_path_hist,
            "t_start": t_start,
            "dW_v_hist": dW_v_hist_correlated,
            "hist_times": times
        }


    def simulate_future(self, history, M, Z_v, Z_s):
        """Simulates future paths conditional on the provided history."""
        # ... (code from your simulate_future function) ...
        # This function is well-defined and can be copied directly.
        # Just ensure all parameters (rho, v, etc.) are accessed via `self.`
        t_start = history["t_start"]
        h = (self.T - t_start) / M
        future_times = np.linspace(t_start, self.T, M + 1)
        
        kernel_matrix = (future_times[1:, np.newaxis] - history["hist_times"][:-1])**(self.H - 0.5)
        V_from_hist = kernel_matrix @ history["dW_v_hist"]
        
        chol = self.get_cholesky(M)
        dW_v_future_correlated = chol @ Z_v * np.sqrt(h)
        
        V_from_future = np.zeros(M)
        for i in range(M):
            kernel_vals = ((future_times[i+1] - future_times[1:i+2])**(self.H - 0.5))
            V_from_future[i] = np.sum(kernel_vals * dW_v_future_correlated[:i+1])

        V_future_path = V_from_hist + V_from_future

        sigma_sq_path = np.zeros(M + 1)
        S_path = np.zeros(M + 1)
        sigma_sq_path[0] = history["sigma_sq_t"]
        S_path[0] = history["S_t"]
        
        dZ = self.rho * Z_v * np.sqrt(h) + self.sqrt_one_minus_rho_sq * Z_s * np.sqrt(h)
        
        for i in range(M):
            sigma_sq_path[i+1] = self.sigma0_sq * np.exp(
                self.v * V_future_path[i] - 0.5 * self.v**2 * future_times[i+1]**(2*self.H)
            )
            sigma_sq_path[i] = np.maximum(sigma_sq_path[i], 0)

            S_path[i+1] = S_path[i] * np.exp(
                -0.5 * sigma_sq_path[i] * h + np.sqrt(sigma_sq_path[i]) * dZ[i]
            )

        return {
            "S_path": S_path,
            "sigma_sq_path": sigma_sq_path,
            "Z_v": Z_v,
            "Z_s": Z_s,
            "h": h
        }