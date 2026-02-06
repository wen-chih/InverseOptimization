# src/evaluator.py

import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v

def evaluate_metrics(c_true, c_pred, X_true_test, X_pred_test):
    """
     Parameter Error (wrt obj_coeff) 和 Decision Error (wrt x)
    """

    
    # 1. Parameter Error (Cosine Similarity or Norm diff after normalization)
    c_t = normalize(c_true)
    c_p = normalize(c_pred)
    param_error = np.linalg.norm(c_t - c_p) # L2 norm
    

    diff = X_true_test - X_pred_test
    decision_error = np.linalg.norm(diff) # L2 norm

    obj_gap = (c_true@X_true_test - c_true@X_pred_test)/(c_true@X_true_test)
    
    return {
        "param_error": param_error,
        "decision_error": decision_error, 
        "relative objVal gap": obj_gap
    }