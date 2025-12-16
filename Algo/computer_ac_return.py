import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import h5py
import numpy as np
import flashbax as fbx
import jax.numpy as jnp
import jax
import time
import gc
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def cosine_similarity(X, Y):
    dot = jnp.sum(X*Y, axis=-1)
    norm_x = jnp.linalg.norm(X, axis=-1)
    norm_y = jnp.linalg.norm(Y, axis=-1)
  
    L2 = norm_x * norm_y + 1e-8
    return dot / L2

def generate_approach_trajectory(dim=4, steps=100):
    """
    Generates a trajectory moving FROM Opposite (-1) TO Goal (+1).
    This direction allows us to get a POSITIVE max return of 2.0.
    """
    s_g = np.random.randn(dim)
    s_g /= np.linalg.norm(s_g)

    # Start roughly opposite to the goal (Similarity ~ -1.0)
    s_start = -s_g + np.random.normal(0, 0.1, dim)
    
    # End roughly at the goal (Similarity ~ 1.0)
    s_end = s_g + np.random.normal(0, 0.1, dim)

    trajectory = []
    for t in range(steps + 1):
        alpha = t / steps
        # Linear interpolation
        s_t = (1 - alpha) * s_start + alpha * s_end
        # Add noise
        s_t += np.random.normal(0, 0.05, dim)
        trajectory.append(s_t)
        
    return s_g, np.array(trajectory)

# --- Simulation ---
DIM = 4
STEPS = 200

#s_g, trajectory = generate_approach_trajectory(dim=DIM, steps=STEPS)




filepath = '/home/xiaoming/Research/Offline RL/12_offline_RL/Navigation_Mujoco_dataset_expert_4x.h5'
try:
            with h5py.File(filepath, "r") as f:
                keys = list(f.keys()) 
     
                grp = f['episode_2']

                raw_feats = grp["features"][:]
                # Simple Normalization
                norms = np.linalg.norm(raw_feats, axis=-1, keepdims=True)
                raw_feats = raw_feats / (norms + 1e-8)

                N = raw_feats.shape[0]
                trajectory = raw_feats
                s_g = raw_feats[N-1]  # Middle state as goal

             




except Exception as e:
            print(f"Error loading {filepath}: {e}")


















similarities = []
rewards = []

# Initial state similarity
sim_start = cosine_similarity(trajectory[0], s_g)

for t in range(len(trajectory) - 1):
    s_curr = trajectory[t]
    s_next = trajectory[t+1]
    
    # Using SIMILARITY now (Range -1 to 1)
    sim_curr = cosine_similarity(s_curr, s_g)
    sim_next = cosine_similarity(s_next, s_g)
    
    # Reward is improvement in Similarity
    r = sim_next - sim_curr
    
    similarities.append(sim_curr)
    rewards.append(r)


# Last step for plotting
similarities.append(cosine_similarity(trajectory[-1], s_g))

cumulative_rewards = np.cumsum(rewards)
sim_final = similarities[-1]

# --- Plotting ---
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# 1. Cosine Similarity (Now -1 to 1)
axs[0].plot(similarities, color='blue', label='Cosine Similarity')
axs[0].axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='At Goal (1.0)')
axs[0].axhline(y=-1.0, color='r', linestyle='--', alpha=0.5, label='Opposite (-1.0)')
axs[0].set_ylabel('Similarity')
axs[0].set_ylim(0.0, 1.1)  # Set bounds to show -1 to 1 clearly
axs[0].set_title(f'Cosine Similarity (Range -1 to 1)')
axs[0].legend(loc='lower right')
axs[0].grid(True, alpha=0.3)

# 2. Step Reward
axs[1].plot(rewards, color='orange', alpha=0.7, label='Reward (Sim_next - Sim_curr)')
axs[1].axhline(y=0, color='black', linewidth=0.8)
axs[1].set_ylabel('Reward')
axs[1].set_title('Instantaneous Reward')
axs[1].legend(loc='upper right')
axs[1].grid(True, alpha=0.3)

# 3. Cumulative Return
expected_max = sim_final - sim_start
axs[2].plot(cumulative_rewards, color='green', linewidth=2, label='Cumulative Return')
axs[2].axhline(y=expected_max, color='purple', linestyle='--', label=f'Limit ({expected_max:.2f})')
axs[2].set_ylabel('Total Return')
axs[2].set_xlabel('Step')
axs[2].set_title(f'Cumulative Return (Max possible is 2.0)')
axs[2].legend(loc='upper left')
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Verification Print
print(f"{'Metric':<20} | {'Value':<10}")
print("-" * 35)
print(f"{'Start Similarity':<20} | {sim_start:.4f}")
print(f"{'Final Similarity':<20} | {sim_final:.4f}")
print(f"{'Calculated Sum':<20} | {cumulative_rewards[-1]:.4f}")
print(f"{'Diff (Final-Start)':<20} | {(sim_final - sim_start):.4f}")
print(f"{'Theoretical Max':<20} | 2.0000")