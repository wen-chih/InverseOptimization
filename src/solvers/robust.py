# src/solvers/robust.py
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .base import BaseInverseSolver

# Inherit from BaseInverseSolver
class RobustIOSolver(BaseInverseSolver):
    def __init__(self):
        # initialize the parent class, 
        super().__init__() 
        self.p_hat = None

    def fit(self, dataset):
        """
        Inverse Optimization for noisy data
        Min sum(|Dual Obj - Primal Obj|)
        """
        if not dataset: return
        
        n_vars = dataset[0]['x'].shape[0]
        model = gp.Model("RobustIO")
        model.setParam("OutputFlag", 0)

        # p (obj coef to be estimated)
        p = model.addMVar(n_vars, lb=0.0, name="p")
        # Normalization
        model.addConstr(p.sum() == 1) # avoid p=0

        total_error = 0

        # Create Dual Constraints for each x_obs

        for k, data in enumerate(dataset): 
            x_obs = data['x']
            A, b = data['A'], data['b']
            A_eq, b_eq = data['A_eq'], data['b_eq']
            x_ub = data['x_ub']
            
            # --- Dual Variables ---
            lam = model.addMVar(A.shape[0], lb=0.0)

            nu = None # wrt A_eq, b_eq
            if A_eq is not None: nu = model.addMVar(A_eq.shape[0], lb=-GRB.INFINITY)

            mu = None # wrt x_ub
            if x_ub is not None: mu = model.addMVar(n_vars, lb=0.0)

            # --- Constraint A: Dual Feasibility ---
            lhs = A.T @ lam
            if nu is not None: lhs += A_eq.T @ nu # wrt A_eq, b_eq
            if mu is not None: lhs += mu # wrt x_ub
            model.addConstr(lhs >= p)

            # --- Constraint B: Soft Strong Duality ---
            # Dual Obj - Primal Obj == Error_Pos - Error_Neg
            dual_obj = b @ lam
            if nu is not None: dual_obj += b_eq @ nu # wrt A_eq, b_eq
            if mu is not None: dual_obj += x_ub @ mu # wrt x_ub
            
            # following is for the objective function
            primal_obj = p @ x_obs
            
            # ======= 1-norm error =======
            gap_pos = model.addMVar(1, lb=0.0)
            gap_neg = model.addMVar(1, lb=0.0)
            
            model.addConstr(dual_obj - primal_obj == gap_pos - gap_neg)
            
            total_error += (gap_pos + gap_neg)

        # Objective
        model.setObjective(total_error, GRB.MINIMIZE)

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


    def get_p_hat(self):    
        return self.p_hat

    