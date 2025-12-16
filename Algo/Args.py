
from dataclasses import dataclass

@dataclass
class Args:
    seed: int = 42
    
    training_dataset_filepath : str = 'Navigation_Mujoco_dataset_pink_4X.h5'
    target_file_path : str = 'Navigation_Mujoco_dataset_target_expert_12.1.h5'
    add_expert_data_filepath: str = None
    expert_file_path: str = 'Navigation_Mujoco_dataset_expert_4x.h5'
    
    exp_name: str = "experiment"
    
    # wandb
    track: bool = True
    wandb_project_name: str = 'offline_navigation_simulated_environment'
    wandb_entity: str = 'bolt-um' 

    env_id: str = 'TD3_Navigation_129' 

    # Algorithm specific arguments
    total_offline_steps: int = 800_000
    '''the discount factor gamma'''
    gamma: float = 0.95
    '''the target network update rate'''
    tau: float = 0.005

    batch_size: int = 1024
    '''the frequency of training policy(actor)'''
    policy_frequency: int = 2
    '''the learning rate'''
    learning_rate: float = 3e-4

    '''the noise added to target policy during critic update'''
    policy_noise: float = 0.2
    noise_clip: float = 0.5

    '''balance BC'''
    alpha: float = 0.001

    done_threshold: float = 0.90

    reward_scale: float = 1

    bonus: float = 1

    load_model_path: str = None

    feature_dim: int = 512









    # Value sample ratio (Critic): [Current(S_t), Geometric(future), Uniform(future), Random]
    value_sample_ratios: tuple = (0.1, 0.2, 0.3, 0.4)
    # Policy sample ratio (Actor): [Current(S_t), Geometric(future), Uniform(future), Random]
    policy_sample_ratios: tuple = (0.0, 0.0, 0.0, 1.0)

