
from dataclasses import dataclass

@dataclass
class Args:
    seed: int = 42
    
    training_dataset_filepath = ['bolt_nav_01.h5', 'bolt_nav_02.h5', 'bolt_nav_05.h5','bolt_nav_06.h5']


    target_file_path : str = 'evaluation_goals.h5'
    add_expert_data_filepath: str = None
    expert_file_path: str = None
    
    exp_name: str = "experiment"
    
    # wandb
    track: bool = True
    wandb_project_name: str = 'Navigation Training(OfflineRL) for Real Robot'
    wandb_entity: str = 'bolt-um' 

    env_id: str = 'TD3_Navigation_129' 

    # Algorithm specific arguments
    total_offline_steps: int = 500_000
    '''the discount factor gamma'''
    gamma: float = 0.99
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

    done_threshold: float = 0.80

    reward_scale: float = 1

    bonus: float = 1

    load_model_path: str = None
    #"/home/xiaoming/Research/Offline RL/12_offline_RL/saved_models2Hz-2history+filter_Sparse+Red+0.99Gamma+HER_0.01bc/TD3_step179999"

    feature_dim: int = 512

    history_length: int = 2

    # 14*14 patches' std in observation encoder, used for uncertainty estimation
    # Any image'std value below this threshold will be considered as a bad goal!
    std_threshold: float = 0.020



    # Value sample ratio (Critic): [Current(S_t), Geometric(future), Uniform(future), Random]
    value_sample_ratios: tuple = (0.1, 0.4, 0.1, 0.4)
    # Policy sample ratio (Actor): [Current(S_t), Geometric(future), Uniform(future), Random]
    policy_sample_ratios: tuple = (0.0, 0.0, 0.0, 1.0)

