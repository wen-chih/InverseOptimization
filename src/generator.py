# src/generator.py

import numpy as np
from .utils import solve_LP

class DataGenerator:
    def __init__(self, config):
        self.cfg = config
        self.p_true = None 
        
    def set_ground_truth(self):

        # np.random.seed(self.cfg.scenario_id + self.n_vars + self.n_constrs)
        np.random.seed(self.cfg.scenario_id)
        
        lb, ub = self.cfg.p_range
        self.p_true = np.random.uniform(lb, ub, self.cfg.n_vars)
        return self.p_true

    def generate_dataset(self, n_samples, noise_level=0.0):
        """
        Create samples (List of Dicts)
        noise_level: higher level is more noisy (std in standard normal). 0.0 is optimal
        """

        dataset = []
        n, m = self.cfg.n_vars, self.cfg.n_constrs
        m_eq = self.cfg.n_eq_constrs
        
        for _ in range(n_samples):
            # ... (Random generation of A, b, etc. remains the same) ...
            # generate x0 first and then create b accordingly to ensure the feasibility

            if self.cfg.use_x_ub:
                lb, ub = self.cfg.x_ub_range
                x_ub = np.random.uniform(lb, ub, n)
                x0 = np.random.uniform(0, x_ub, n)
            else:
                x_ub = None
                x0 = np.random.uniform(0, 10, n)
                
            lb, ub = self.cfg.A_range
            A = np.random.uniform(lb, ub, (m, n))
            b = A @ x0 + np.random.uniform(0.1, 2.0, m) # Slack > 0
            
            if m_eq > 0:
                lb, ub = self.cfg.A_eq_range
                A_eq = np.random.uniform(lb, ub, (m_eq, n))
                b_eq = A_eq @ x0
            else:
                A_eq, b_eq = None, None
                
            x_clean = solve_LP(self.p_true, A, b, A_eq, b_eq, x_ub)
            
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, size=self.p_true.shape)
                if np.random.uniform(0, 1) > 0.8: # outlier   
                    _zeros = np.zeros(self.p_true.shape)
                    _zeros[0] = 30
                    noise = noise + _zeros 

                p_noisy = np.maximum(self.p_true + noise, 0.0001) # create a noisy p
                x_obs = solve_LP(p_noisy, A, b, A_eq, b_eq, x_ub) # x_obs is based on p_noisy
                
            else:
                x_obs = x_clean
            
            dataset.append({
                'x_opt': x_clean, # optimal solution
                'x': x_obs,       # observed solution
                'A': A, 'b': b,
                'A_eq': A_eq, 'b_eq': b_eq,
                'x_ub': x_ub
            })
            
        return dataset

