import h5py

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def cosine_similarity(X, Y):
    dot = jnp.sum(X*Y, axis=-1)
    norm_x = jnp.linalg.norm(X, axis=-1)
    norm_y = jnp.linalg.norm(Y, axis=-1)
  
    L2 = norm_x * norm_y + 1e-8
    return dot / L2



# open bolt_nav_01.h5
def open_bolt_nav_h5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            #observations = f['observations'][:]
            features = f['features'][:]
            stds = f['stds'][:]
            actions = f['actions'][:]
            total_steps = f.attrs.get('total_steps', len(features))
        return features, stds

    except Exception as e:
        print(f"Error opening H5 file {file_path}: {e}")
        return None 
file_path = 'bolt_nav_06.h5'

features, stds =  open_bolt_nav_h5(file_path)



N = features.shape[0]
trajectory = features
s_g = features[1606] 

#for 01
target_index = [704, 1264, 1435, 1700]
#for 02
target_index = [243, 300, 414, 951, 964, 1424, 1537]
#for 05
target_index = [316, 1645, 1606 ]
#for 06
target_index = [256,1182]


similarities = []
rewards = []
index = []
# Initial state similarity
sim_start = cosine_similarity(trajectory[0], s_g)

if True:


    with h5py.File(file_path, 'r') as f:
            index = list(range(1600, 1800))
            observations = f['observations'][index]
            # plot 
            n = 15
            fig, ax = plt.subplots(n, 2*n, figsize=(80, 80))
            for i, idx in enumerate(index):
                ax[i//(2*n), i%(2*n)].imshow(observations[i])
                ax[i//(2*n), i%(2*n)].set_title(f'Step {index[i]:.4f}')
                ax[i//(2*n), i%(2*n)].axis('off')
            plt.suptitle('High Similarity Observations')
            plt.show()  


for t in range(len(trajectory) - 1):
    s_curr = trajectory[t]
    s_next = trajectory[t+1]
    
    # Using SIMILARITY now (Range -1 to 1)
    sim_curr = cosine_similarity(s_curr, s_g)
    sim_next = cosine_similarity(s_next, s_g)
    if sim_curr > 0.8:
        print(f"High similarity at step {t}: {sim_curr:.4f}")
        print(f"Corresponding std: {stds[t]:.4f}")
        index.append(t)
     
    # Reward is improvement in Similarity
    r = sim_next - sim_curr
    
    similarities.append(sim_curr)
    rewards.append(r)


if True:
    sim_index = np.array(index)
    with h5py.File(file_path, 'r') as f:
            observations = f['observations'][index]
            # plot 
            n = 8
            fig, ax = plt.subplots(n, 2*n, figsize=(80, 80))
            for i, idx in enumerate(index):
                ax[i//(2*n), i%(2*n)].imshow(observations[i])
                ax[i//(2*n), i%(2*n)].set_title(f'Step {stds[idx]:.4f}')
                ax[i//(2*n), i%(2*n)].axis('off')
            plt.suptitle('High Similarity Observations')
            plt.show()  



if False:
    # Plot high similarity observations
    these_stds = stds[index]
    print("Corresponding stds for high similarity observations:", these_stds)
    ord_std = np.argsort(stds)
    new_std = stds[ord_std]
    low_std_indices = ord_std[1600:]
    index = low_std_indices.tolist()
    index.sort()

    with h5py.File('bolt_nav_01.h5', 'r') as f:
            observations = f['observations'][index]
            # plot 
            n = 12
            fig, ax = plt.subplots(n, 2*n, figsize=(80, 80))
            for i, idx in enumerate(index):
                ax[i//(2*n), i%(2*n)].imshow(observations[i])
                ax[i//(2*n), i%(2*n)].set_title(f'Step {stds[idx]:.4f}')
                ax[i//(2*n), i%(2*n)].axis('off')
            plt.suptitle('High Similarity Observations')
            plt.show()  



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

