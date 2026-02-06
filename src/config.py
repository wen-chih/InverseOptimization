# src/config.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    # --- dimension ---
    n_vars: int             # x 的維度
    n_constrs: int          # Ax <= b 的數量
    n_eq_constrs: int = 0   # A_eq x = b_eq 的數量 (Optional)
    use_x_ub: bool = False  # 是否有 x <= x_ub (Optional)
    
    n_samples_train: int = 50   # sample size for training
    n_samples_test: int = 100    # sample size for testing
    scenario_id: int = 0        # Random Seed

    # variable: type = (min, max)
    p_range: tuple[float, float] = (1.0, 10.0)
    A_range: tuple[float, float] = (0.0, 10.0)
    x_ub_range: tuple[float, float] = (5.0, 15.0)
    A_eq_range: tuple[float, float] = (0.0, 10.0)

    