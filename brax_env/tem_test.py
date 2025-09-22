
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import jax.numpy as jnp
import h5py


from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import gc
import matplotlib.patches as patches
import matplotlib.pyplot as plt

act1min = 1e9-1
act1max = -1e9+1
act2min = 1e9-1
act2max = -1e9+1
file_path = "Navigation_Mujoco_dataset_full.h5"
obstacles_array = jnp.array([
            [-6,  6, 1],
            [-3,  6, 1],
            [ 2,  6, 1],
            [ 6,  6, 1],
            [-6,  3, 1],
            [-3,  3, 1],
            [ 2,  3, 1],
            [ 6,  3, 1],
            [-6,  0, 1],
            [-3,  0, 1.5],
            [ 2,  0, 1],
            [ 6,  0, 1],
            [-6, -3, 1.7],
            [-3, -3, 2],
            [ 2, -3, 1.7],
            [ 6, -3, 1.7],
            [-6, -6, 0.5],
            [-3, -6, 0.5],
            [ 2, -6, 0.5],
            [ 6, -6, 0.5],
        ])
    
with h5py.File(file_path, "r") as f:
    keys = list(f.keys())
    fig, ax = plt.subplots(figsize=(6,6))
    for key in keys:
        grp = f[key]

        # latents size (2000X, 1, 768), the first element shoule be abandoned
        #images = grp["images"][1:]
      #  features = grp["features"][1:]
        actions = grp["actions"][1:]
        positions = grp["positions"][1:]
        quaternions = grp["quaternions"][1:]
        act1min = min(actions[:, 0].min(), act1min)
        act1max = max(actions[:, 0].max(), act1max)
        act2min = min(actions[:, 1].min(), act2min)
        act2max = max(actions[:, 1].max(), act2max)

        '''should check this carefully, ideal form is: [add_batch_size, [data_size]]'''
        #images_array = jnp.asarray(images, dtype=jnp.int8)[None, ...]
      #  feature_array = jnp.array(features).reshape(1,features.shape[0],768).astype(jnp.float32)
      #  action_array = jnp.array(actions).reshape(1,features.shape[0],2).astype(jnp.float32)
        position_array = jnp.asarray(positions, dtype=jnp.float32)[None, ...].reshape([1999,2])
        quaternions_array = jnp.asarray(quaternions, dtype=jnp.float32)[None, ...]



        ax.plot(position_array[:, 0], position_array[:, 1], color='blue', label='Trajectory')
        for x, y, r in obstacles_array:
            circle = patches.Circle((x, y), r, facecolor="gray", alpha=0.5, edgecolor="black")
            ax.add_patch(circle)
            ax.plot(x, y, "ko")   
       

        betta = 30
        for i in range(len(position_array)//betta-1):
            j= betta*i
            plt.arrow(
                position_array[j, 0], position_array[j, 1],
                position_array[j+betta, 0] - position_array[j, 0],
                position_array[j+betta, 1] - position_array[j, 1],
                shape='full', head_width=0.2, length_includes_head=True, color='blue'
            )
     
    ax.axis('equal')
    ax.legend()
    fig.canvas.draw()
    plt.show()
