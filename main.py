# main.py

import numpy as np
from src.config import ExperimentConfig
from src.runner import ExperimentRunner
from src.generator import DataGenerator



def main():

    runner = ExperimentRunner()
                    
    # Initialize Config 
    cfg = ExperimentConfig()
    
    # Generate Data 
    gen = DataGenerator(cfg)
    p_true = gen.set_ground_truth()
    
    # Generate Training Data
    train_data = gen.generate_dataset(cfg.n_samples_train, noise_level=0.1)
    # Generate Testing Data 
    test_data = gen.generate_dataset(cfg.n_samples_test, noise_level=0.0) 

        
    # run IO                        
    res = runner.run_experiment(
        train_data=train_data,
        test_data=test_data,
        p_true=p_true,
        config=cfg,
        noise_level=noise,
        solver_type="Strict" # "Robust"
    )

    return res
    




# ----------------------------------------------------
if __name__ == "__main__":
    res = main()



