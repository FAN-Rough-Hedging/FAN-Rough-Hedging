# src/benchmark/malliavin_delta.py

import numpy as np
from src.simulation.rbergomi_simulator import PathSimulator

def _calculate_payload_components(path_data, K, sigma_t):
    """
    Calculates the components P*W^s and P*W^v for a single path.
    This corresponds to calculating the terms inside the expectation of Eq. (37) and (38)
    in the mathematical document, which are then used in the final Delta formula Eq. (40).
    W^s is approximated using the "pathwise" method (see mathematical doc Appendix A).
    W^v is approximated by the first-step method, a simplification of the Volterra kernel.
    """
    S_T = path_data["S_path"][-1]
    payoff = max(S_T - K, 0)
    
    if payoff < 1e-12:
        return 0.0, 0.0

    h = path_data["h"]
    
    # Calculate W^s component (g_perp in paper)
    sigma_path = np.sqrt(np.maximum(path_data["sigma_sq_path"][:-1], 0))
    dW_s = np.sqrt(h) * path_data["Z_s"]
    
    numerator_s = np.sum(sigma_path * dW_s)
    denominator_s = np.sum(sigma_path**2 * h)
    
    # g_perp = E[P * W_perp | F_t]
    g_perp_component = 0.0
    if denominator_s > 1e-12:
        W_s = sigma_t * numerator_s / denominator_s
        g_perp_component = payoff * W_s

    # Calculate W^v component (g_v in paper)
    # Using a first-step approximation for the complex Volterra kernel integral
    Z_v_1 = path_data["Z_v"][0]
    W_v = Z_v_1 / np.sqrt(h)
    g_v_component = payoff * W_v
    
    return g_perp_component, g_v_component

def calculate_optimal_delta_mlqmc(history, simulator, K, rho, L, M0, B, N_SAMPLES_PER_LEVEL):
    """
    Estimates the optimal hedging delta Delta_t using a Multilevel Quasi-Monte Carlo method.
    This function numerically computes E[P*W | F_t] to find the g^v and g^perp terms
    needed for the final delta formula (Eq. 40).
    """
    g_s_total, g_v_total = 0.0, 0.0
    S_t, sigma_sq_t = history["S_t"], history["sigma_sq_t"]
    sigma_t = np.sqrt(max(sigma_sq_t, 0))

    if S_t * sigma_t < 1e-9:
        return 0.0

    # Loop over each MLQMC level
    for l in range(L + 1):
        N_l = N_SAMPLES_PER_LEVEL[l]
        M_l = M0 * (B**l)
        sum_diff_s, sum_diff_v = 0.0, 0.0

        for _ in range(N_l):
            # Generate random numbers for the fine path
            Z_v_fine = np.random.randn(M_l)
            Z_s_fine = np.random.randn(M_l)

            # Simulate fine path and calculate payload components
            path_fine = simulator.simulate_future(history, M_l, Z_v_fine, Z_s_fine)
            payload_s_fine, payload_v_fine = _calculate_payload_components(path_fine, K, sigma_t)
            
            diff_s, diff_v = payload_s_fine, payload_v_fine

            if l > 0:
                # Coupled coarse path from fine path random numbers
                M_c = M0 * (B**(l-1))
                Z_v_coarse = np.sum(Z_v_fine.reshape(M_c, B), axis=1) / np.sqrt(B)
                Z_s_coarse = np.sum(Z_s_fine.reshape(M_c, B), axis=1) / np.sqrt(B)
                
                path_coarse = simulator.simulate_future(history, M_c, Z_v_coarse, Z_s_coarse)
                payload_s_coarse, payload_v_coarse = _calculate_payload_components(path_coarse, K, sigma_t)
                
                diff_s -= payload_s_coarse
                diff_v -= payload_v_coarse

            sum_diff_s += diff_s
            sum_diff_v += diff_v
            
        g_s_total += sum_diff_s / N_l
        g_v_total += sum_diff_v / N_l

    # Assemble final delta according to Eq. (40):
    # Delta_t = (1 / S_t*sigma_t) * (sqrt(1-rho^2)*g_perp + rho*g_v)
    delta_star = (np.sqrt(1 - rho**2) * g_s_total + rho * g_v_total) / (S_t * sigma_t)
    
    # Clip extreme values to prevent numerical instability from polluting the dataset
    return np.clip(delta_star, -2.0, 2.0)