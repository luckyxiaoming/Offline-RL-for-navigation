import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List
import mediapy as media
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx
import jax
from jax import numpy as jnp
import numpy as np
import os
from brax.io import mjcf
from random_explore import random_explore_policy

import h5py



'''
frames extraction function cannot work in brax env(MJX), so mujoco is used here.
'''

def Navigation_environment(seeds:int, episode_size:int):
  path  = os.getcwd() + '/fake_go2.xml'
  sys = mjcf.load(path)

  # Make model, data, and renderer
  mj_model = mujoco.MjModel.from_xml_path(path)
  mj_data = mujoco.MjData(mj_model)
  renderer = mujoco.Renderer(mj_model)

  frame_number = episode_size

  frames_list = []
  actions_list = []
  key = jax.random.PRNGKey(seeds)
  mujoco.mj_resetData(mj_model, mj_data)
  current_pos = jnp.array([0, 0]).reshape([2,1])
  current_quat = jnp.array([1, 0, 0, 0]).reshape([4,1])
  j = 0
  while j < frame_number:

    key, pos_list, quat_list, distances_list, dir_list = random_explore_policy(key, current_pos, current_quat)
    for i in zip(pos_list, quat_list, distances_list, dir_list):

      renderer.update_scene(mj_data, 'came')
      pixels = renderer.render()
      frames_list.append(pixels)

      next_pos = i[0]
      next_quat = i[1]
      mj_data.mocap_pos = np.array([next_pos[0],next_pos[1], 0.3]).reshape([3])
      mj_data.mocap_quat = np.array([next_quat]).reshape([4])
      current_pos = next_pos
      current_quat = next_quat
      action = np.array([i[2], i[3]])
      actions_list.append(action)
      mujoco.mj_step(mj_model, mj_data)
      j += 1

  images = np.stack(frames_list, axis=0)   # (N, H, W, C)
  actions = np.stack(actions_list, axis=0) 
  return images, actions



def collect_mujoco_data(seeds, episode_size=1000, episode_num=2):



  f = h5py.File("Navigation_Mujoco_datase.h5", "w")
  for ep in range(episode_num): 
    grp = f.create_group(f"episode_{ep}")
    images, actions = Navigation_environment(seeds+ep, episode_size)

    grp.create_dataset("images", data=images, compression="gzip")
    grp.create_dataset("actions", data=actions)
    print(f"The {ep} episodes clllection is done!")

  
  f.close()
  print("Data collection is done!")
    


collect_mujoco_data(seeds=1, episode_size=40000, episode_num=5)
      
    




