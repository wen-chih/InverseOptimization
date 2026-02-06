# src/solvers/base.py

from abc import ABC, abstractmethod
import numpy as np
from ..utils import solve_LP 

class BaseInverseSolver(ABC):
    def __init__(self):
        super().__init__() 
        self.p_hat = None


    @abstractmethod
    def fit(self, dataset):
        pass

    def predict(self, A, b, A_eq=None, b_eq=None, x_ub=None):
        """
        Return optimal x based on p_hat
        """
        return solve_LP(self.p_hat, A, b, A_eq, b_eq, x_ub)
        

    