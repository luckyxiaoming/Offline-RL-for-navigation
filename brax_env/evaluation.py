
import os 
import time

import numpy as np

import mediapy as media
import matplotlib.pyplot as plt
import mujoco

import jax
from jax import numpy as jnp
import numpy as np
import os
from brax.io import mjcf
from random_explore import random_explore_policy
import torch

from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import gc
import matplotlib.patches as patches
'''
frames extraction function cannot work in brax env(MJX), so mujoco is used here.
'''


def cal_z_from_quat(Q):
   '''Only for form of [q0, 0, 0, q3] '''
   Q = np.array([Q]).reshape([4,])
   q0 = Q[0] # cos(thet/2)
   q3 = Q[3] # sin(thet/2)
   sinthet = 2*q0*q3
   costhet = q0**2 - q3**2
   thet = np.atan2(sinthet,costhet)
   return thet



class evaluation():
    def __init__(self):
        path  = os.getcwd() + '/fake_go2.xml'
        sys = mjcf.load(path)
    

        # Make model, data, and renderer
        self.mj_model = mujoco.MjModel.from_xml_path(path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.renderer = mujoco.Renderer(self.mj_model)

        # Make feature-extractor: Dinov3
        self.prepare_Dinov3()

        self.obstacles = [
            {"pos": jnp.array([-6, 6]), "radius": 1},
            {"pos": jnp.array([-3, 6]), "radius": 1},
            {"pos": jnp.array([2, 6]), "radius": 1},
            {"pos": jnp.array([6, 6]), "radius": 1},
            {"pos": jnp.array([-6, 3]), "radius": 1},
            {"pos": jnp.array([-3, 3]), "radius": 1},
            {"pos": jnp.array([2, 3]), "radius": 1},
            {"pos": jnp.array([6, 3]), "radius": 1},
            {"pos": jnp.array([-6, 0]), "radius": 1},
            {"pos": jnp.array([-3, 0]), "radius": 1.5},
            {"pos": jnp.array([2, 0]), "radius": 1},
            {"pos": jnp.array([6, 0]), "radius": 1},
            {"pos": jnp.array([-6, -3]), "radius": 1.7},
            {"pos": jnp.array([-3, -3]), "radius": 2},
            {"pos": jnp.array([2, -3]), "radius": 1.7},
            {"pos": jnp.array([6, -3]), "radius": 1.7},
            {"pos": jnp.array([-6, -6]), "radius": 0.5},
            {"pos": jnp.array([-3, -6]), "radius": 0.5},
            {"pos": jnp.array([2, -6]), "radius": 0.5},
            {"pos": jnp.array([6, -6]), "radius": 0.5},

        ]

        self.obstacles_array = jnp.array([
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



        self.feature_list= []
        self.pos_list= []
        self.distance_list=[]


    def is_in_obstacle(self, point, ):
        point = point.reshape([2,1])
        for obs in self.obstacles:
            distance = np.square(np.array(point).squeeze() - obs["pos"]).sum()**0.5
            if distance <  obs["radius"]:
                return True
            if jnp.abs(point[0]) > 9 or jnp.abs(point[1]) > 9:
                return True
        return False





    def prepare_Dinov3(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", self.device)
        model_id = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model =   AutoModel.from_pretrained(model_id).to(self.device)

    def cal_latent(self, obs):
        image = Image.fromarray(obs)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad(): #
            outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        return pooler_output.cpu().numpy()


            
        


    def reset(self, goal_obs, goal_pos, initial_pos=[0,0],intial_quat=[1,0,0,0]):
        self.steps = 0
        self.goal= goal_obs
        self.goal_pos = goal_pos

        self.feature_list= []
        self.pos_list= []
        self.distance_list=[]
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        current_pos = jnp.array(initial_pos).reshape([2])
        current_quat = jnp.array(intial_quat).reshape([4,1])
        self.mj_data.mocap_pos = np.array([current_pos[0],current_pos[1], 0.3]).reshape([3])
        self.mj_data.mocap_quat = np.array([current_quat]).reshape([4])
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.renderer.update_scene(self.mj_data, 'came')
        initial_obs = self.renderer.render()
        initial_features = self.cal_latent(initial_obs) 
        done = False
        return initial_features, done


    def cal_command(self, action):
        action = action.reshape([2])
        current_pos = self.mj_data.mocap_pos.reshape([3])
        current_quat = self.mj_data.mocap_quat
        current_angle = cal_z_from_quat(current_quat)
        distance = action[0]
        add_angle = action[1]
        next_angle = current_angle + add_angle
        action = jnp.array([distance*jnp.cos(next_angle), distance*jnp.sin(next_angle)]).squeeze()

        next_pos = np.array([current_pos[0] + action[0], current_pos[1] + action[1]])

        new_angle = jnp.arctan2(action[1], action[0]).reshape([1])


        next_quat = jnp.array([jnp.cos((new_angle)/2), [0], [0], jnp.sin((new_angle)/2)])
        return next_pos, next_quat

    def step(self, action):
        self.steps += 1
        next_pos, next_quat = self.cal_command(action)
        
        collision = self.is_in_obstacle(next_pos)
        if collision:
           action = action.reshape([2])
           newaction = jnp.array([0.000001, action[1]])
           next_pos, next_quat = self.cal_command(newaction)


        self.mj_data.mocap_pos = np.array([next_pos[0],next_pos[1], 0.3]).reshape([3])
        self.mj_data.mocap_quat = np.array([next_quat]).reshape([4])
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.renderer.update_scene(self.mj_data, 'came')
        obs = self.renderer.render()
        feature = self.cal_latent(obs).squeeze()

        L2 = -jnp.square(feature - self.goal).mean()
        reward = jnp.tanh(0.5*(L2))
        distance = jnp.square(self.goal_pos - next_pos).sum()**0.5

        self.feature_list.append(feature)
        self.pos_list.append(next_pos)
        self.distance_list.append(distance)

        if distance < 0.1 or self.steps > 1000:
           done = True
        else:
           done = False

        return feature, reward, done
    

    def evaluation_tracjectory(self):


        trajectory = jnp.array(self.pos_list)
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(trajectory[:, 0], trajectory[:, 1], color='blue', label='Trajectory')
        for x, y, r in self.obstacles_array:
            circle = patches.Circle((x, y), r, facecolor="gray", alpha=0.5, edgecolor="black")
            ax.add_patch(circle)
            ax.plot(x, y, "ko")   
       

        betta = 30
        for i in range(len(trajectory)//betta-1):
            j= betta*i
            plt.arrow(
                trajectory[j, 0], trajectory[j, 1],
                trajectory[j+betta, 0] - trajectory[j, 0],
                trajectory[j+betta, 1] - trajectory[j, 1],
                shape='full', head_width=0.8, length_includes_head=True, color='blue'
            )
        ax.plot(self.goal_pos[0], self.goal_pos[1], marker='*', color='red', markersize=15, label='Goal')
        ax.axis('equal')
        ax.legend()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # (H, W, 4)

        # ARGB → RGB
        img_array = buf[:, :, 1:]   #
        plt.close(fig)
        gc.collect()

        return img_array
       
       

       



        
    

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
      if j % 2000 == 0:
        print(f"{j} frames have been collected!")

  images = np.stack(frames_list, axis=0)   # (N, H, W, C)
  actions = np.stack(actions_list, axis=0) 
  return images, actions
