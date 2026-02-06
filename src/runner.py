# src/runner.py

import numpy as np
from .solvers.strict import StrictIOSolver
from .solvers.robust import RobustIOSolver 


class ExperimentRunner:
    def run_experiment(self, train_data, test_data, p_true, config, noise_level, solver_type):
        """
        Modified to accept datasets directly for accumulated training data experiments.
        """
        
        # select IO Solver
        if solver_type == "Strict":
            # optimal solutions
            solver = StrictIOSolver()
        elif solver_type == "Robust":
            # noisy data
            solver = RobustIOSolver()
          
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
            
        # Training (data fitting based on training data)
        solver.fit(train_data)
        
        # Fail to train
        if solver.p_hat is None:
            return {
                "train_noise": noise_level,
                "solver_type": solver_type,
                "n_samples_train": len(train_data),
                "A_shape": train_data[0]['A'].shape if train_data else None,
                "p_true": p_true,
                "success": False,
                "p_hat": None,
                "param_error": np.nan,
                "avg. decision_error": np.nan,
                "avg. relative objVal gap": np.nan, 
                "P10 decision_error": np.nan, # P10, Lower Bound
                "P90 decision_error": np.nan, # P10, Upper Bound
                "P10 relative objVal gap": np.nan, # P90, Lower Bound                
                "P90 relative objVal gap": np.nan # P90, Upper Bound                
            }

        # Performance evaulation based on testing data
        decision_errors = []
        relative_objVal_gaps = []
        
        # Calculate Parameter Error
        p_hat_norm = solver.p_hat / np.linalg.norm(solver.p_hat)
        p_true_norm = p_true / np.linalg.norm(p_true)
        param_error = np.linalg.norm(p_true_norm - p_hat_norm)

        for d in test_data:
            # Predict x using learned p_hat
            x_pred = solver.predict(d['A'], d['b'], d['A_eq'], d['b_eq'], d['x_ub'])
            
            # Decision Error: ||x_true - x_pred||
            decision_errors.append(np.linalg.norm(d['x_opt'] - x_pred))
            
            # Relative Objective Value Gap
            # Gap = (True_Obj - Pred_Obj) / |True_Obj|
            # Note: evaluate both x vectors using p_true (the ground truth objective)
            obj_true = p_true @ d['x_opt']
            obj_pred = p_true @ x_pred
            
            if abs(obj_true) > 1e-9:
                gap = (obj_true - obj_pred) / abs(obj_true)
            else:
                gap = 0.0
            relative_objVal_gaps.append(gap)
            
        return {
            "train_noise": noise_level,
            "solver_type": solver_type,
            "n_samples_train": len(train_data),
            "A_shape": train_data[0]['A'].shape,        
            "p_true": p_true,
            "success": True,
            "p_hat": solver.p_hat,
            "param_error": param_error,
            "avg. decision_error": np.mean(decision_errors),
            "avg. relative objVal gap": np.mean(relative_objVal_gaps),
            "P10 decision_error": np.percentile(decision_errors, 10), # P10, Lower Bound
            "P90 decision_error": np.percentile(decision_errors, 90), # P90, Upper Bound
            "P10 relative objVal gap": np.percentile(relative_objVal_gaps, 10), # P10, Lower Bound
            "P90 relative objVal gap": np.percentile(relative_objVal_gaps, 90) # P90, Upper Bound
        }

