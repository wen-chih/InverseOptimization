# src/solvers/strict.py

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .base import BaseInverseSolver

# Inherit from BaseInverseSolver
class StrictIOSolver(BaseInverseSolver):
    def __init__(self):
            # initialize the parent class, 
            super().__init__() 
            self.p_hat = None

    def fit(self, dataset, p0=None):
        """
        Inverse Optimization (without noise)
        """
        if not dataset: return
        
        n_vars = dataset[0]['x'].shape[0]
        model = gp.Model("StrictIO")
        model.setParam("OutputFlag", 0)

        if p0 is None:
            # initial guess
            # p0 = np.zeros(v_size)  # type 1
            p0 = np.mean(dataset[0]['A'], axis=0) # type 2   


        # define variable p
        p = model.addMVar(n_vars, name="p")
         # Normalization
        model.addConstr(p.sum() == 1) # avoid p=0

        # Create Dual Constraints
        for k, data in enumerate(dataset):
            x_obs = data['x']
            A, b = data['A'], data['b']
            A_eq, b_eq = data['A_eq'], data['b_eq']
            x_ub = data['x_ub']
            
            # --- Dual Variables ---
            lam = model.addMVar(A.shape[0], lb=0.0)
            
            nu = None
            if A_eq is not None:
                nu = model.addMVar(A_eq.shape[0], lb=-GRB.INFINITY)
            
            mu = None
            if x_ub is not None:
                mu = model.addMVar(n_vars, lb=0.0)

            # --- Constraint A: Dual Feasibility ---
            # A^T lam + A_eq^T nu + mu >= p
            lhs = A.T @ lam
            if nu is not None: lhs += A_eq.T @ nu
            if mu is not None: lhs += mu
            model.addConstr(lhs >= p)

            # --- Constraint B: Strict Strong Duality ---
            # Dual Obj - Primal Obj == 0
            dual_obj = b @ lam
            if nu is not None: dual_obj += b_eq @ nu
            if mu is not None: dual_obj += x_ub @ mu
            
            primal_obj = p @ x_obs
            
            # Strong Duality 
            model.addConstr(dual_obj - primal_obj == 0, name=f"strict_gap_{k}")


        # Objective function: minimize L1 norm to the initial guess
        _d_plus = model.addMVar(n_vars, name="gap +") # wrt 1-norm
        _d_minus = model.addMVar(n_vars, name="gap -") # wrt 1-norm
        model.addConstr(p - p0 == _d_plus - _d_minus) # 1-norm constraint (for obj function)
        
        # Define objective function
        ones = np.ones(n_vars)
        model.setObjective(_d_plus @ ones + _d_minus @ ones, sense=GRB.MINIMIZE)          

        try:
            model.optimize()
            if model.Status == GRB.OPTIMAL:
                self.p_hat = p.X
            elif model.status == GRB.INFEASIBLE:
                print("No feasible solution exists. Infeasible")  
                self.p_hat = None
            elif model.status == GRB.UNBOUNDED:
                print("No feasible solution exists. Unbounded")
                self.p_hat = None
            else:
                print("No solution found")
                self.p_hat = None

        except Exception as e:
                    print(f"Optimization error: {e}")
                    self.p_hat = None        


            
    
