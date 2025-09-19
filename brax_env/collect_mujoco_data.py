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
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import h5py
import torch



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

  ###########################################################
  ####################  prepare Dinov3#######################
  ###########################################################
 

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print("Device:", device)
  model_id = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
  processor = AutoImageProcessor.from_pretrained(model_id)
  model =   AutoModel.from_pretrained(model_id).to(device)

  def cal_latent(obs):
      image = Image.fromarray(obs)
      inputs = processor(images=image, return_tensors="pt").to(device)
      with torch.no_grad(): #
          outputs = model(**inputs)
      #last_hidden_state = outputs.last_hidden_state
      pooler_output = outputs.pooler_output
      return pooler_output.cpu().numpy()




  actions_list = []
  position_list = []
  quaterion_list = []
 
  key = jax.random.PRNGKey(seeds)
  mujoco.mj_resetData(mj_model, mj_data)
  current_pos = jnp.array([0, 0]).reshape([2,1])
  rng = np.random.default_rng(seeds)  
  random_angle = rng.uniform(-3.14, 3.14)

  current_quat = jnp.array([jnp.cos(random_angle/2), 0, 0, jnp.sin(random_angle/2)]).reshape([4,1])

  mj_data.mocap_pos = np.array([current_pos[0,1],current_pos[1,1], 0.3]).reshape([3])
  mj_data.mocap_quat = np.array([current_quat]).reshape([4])
  renderer.update_scene(mj_data, 'came')

  pixels = renderer.render()
  h, w, c= pixels.shape
  images = np.empty((frame_number, h, w, c), dtype=int)
  feature = cal_latent(pixels)
  w, c= feature.shape
  features = np.empty((frame_number, w, c))
  
  



  j = 0
  while j < frame_number:

    key, pos_list, quat_list, distances_list, dir_list = random_explore_policy(key, current_pos, current_quat)
    for i in zip(pos_list, quat_list, distances_list, dir_list):

      renderer.update_scene(mj_data, 'came')
      pixels = renderer.render()
      images[j] = pixels
      feature = cal_latent(pixels)
      features[j] = feature


      next_pos = i[0]
      next_quat = i[1]
      mj_data.mocap_pos = np.array([next_pos[0],next_pos[1], 0.3]).reshape([3])
      mj_data.mocap_quat = np.array([next_quat]).reshape([4])
      current_pos = next_pos
      current_quat = next_quat
      action = np.array([i[2], i[3]])
      actions_list.append(action)
      position_list.append(current_pos)
      quaterion_list.append(current_quat)
      mujoco.mj_step(mj_model, mj_data)
      j += 1
      if j % 2000 == 0:
        print(f"{j} frames have been collected!")
      if j == frame_number:
         break


  actions = np.stack(actions_list, axis=0) 
  positions = np.stack(position_list, axis=0)
  quaternions = np.stack(quaterion_list, axis=0)
  return images, features, actions, positions, quaternions





def collect_mujoco_data(seeds, episode_size=1000, episode_num=1):


  for i in range(episode_num):
    t1 = time.time()
    newseeds = seeds+i
    images, features, actions, positions, quaternions = Navigation_environment(newseeds, episode_size)

    print("One episode time:", time.time() - t1)

    with h5py.File("Navigation_Mujoco_dataset_full.h5", "a") as f:
        a = list(f.keys())

        grp = f.create_group(f"episode_{newseeds}")
        grp.create_dataset("images", data=images, compression="gzip")
        grp.create_dataset("features", data=features)
        grp.create_dataset("actions", data=actions)
        grp.create_dataset("positions", data=positions)
        grp.create_dataset("quaternions", data=quaternions)
        print(f"The seeds of {newseeds} episodes collection is done!")


  
  
  
  print("Data collection is done!")
    



collect_mujoco_data(seeds=1, episode_size=10000, episode_num=8)
      
    




