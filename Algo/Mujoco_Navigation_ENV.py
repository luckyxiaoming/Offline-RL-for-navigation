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
from random_explore import random_explore_policy4
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import h5py
import torch
from memory_profiler import profile
import gc



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



class Navigation_sim_environment():
  def __init__(self):

   

    path  = os.getcwd() + '/fake_go2flat.xml'

    # Make model, data, and renderer
    self.mj_model = mujoco.MjModel.from_xml_path(path)
    self.mj_data = mujoco.MjData(self.mj_model)
    self.renderer = mujoco.Renderer(self.mj_model)
    

    # prepare Dinov3
    self.prepare_Dinov3()

    # define max_action parameters
    self.action_min = np.array([0.001, -0.17])
    self.action_max = np.array([0.1, 0.17])

    self.sample_interal = 6



  def select_action(self):
    
    # random select a binary value
    relative_position = self.target_pos - self.current_pos[0:2]
    relative_angle = self.target_angle - self.current_angle
    if relative_angle > np.pi:
        relative_angle = relative_angle - 2*np.pi
    if relative_angle < -np.pi:
        relative_angle = relative_angle + 2*np.pi
    #print('relative position and angle:', relative_position, relative_angle)

    if np.abs(relative_position).sum() < 1.2:
        self.target_pos = np.random.uniform(low=-5.5, high=5.5, size=2)
        self.target_angle = np.random.uniform(low=-3.14, high=3.14, size=1).squeeze()
        print('new target:', self.target_pos, self.target_angle)
       
    


   

    X =  np.random.uniform(low=-0.2, high=0.2)
    Y =  np.random.uniform(low=-0.15, high=0.15)
    R = np.random.uniform(low=-0.5, high=0.5)
    gaussion_noise =  np.array([X, Y, R])
    theta = np.concatenate((relative_position, np.array([relative_angle])), axis=0)
    mu = np.array([0.01, 0.01, 0.01]) 
    command = theta * mu  + gaussion_noise

    command = np.clip(command, a_min=np.array([-0.3, -0.2, -0.5]), a_max=np.array([0.3, 0.2, 0.5])) 

    position_command = command[0:2]
    position_command_in_robot = self.Rz.T @ position_command
    rotation_command = command[2]
    rotation_command_in_robot = rotation_command 

    action_copmmand = np.array([position_command_in_robot[0], position_command_in_robot[1], rotation_command_in_robot])






  
    return action_copmmand
       

  def update_desired_transition(self, desired_transition, transition, max_steps):
    full = False
    
    if desired_transition is None:
        # first
        desired_transition = {k: v for k, v in transition.items()}
        return desired_transition, full
     
    updated = {}
    for k in desired_transition.keys():
        concat = np.concatenate([desired_transition[k], transition[k]], axis=0)
        # stop at max_steps
        if concat.shape[0] > max_steps:
            concat = concat[0:max_steps]
            full =True
        updated[k] = concat
    return updated, full

  def collect_one_episod_data(self):
     
     full = False
     desired_transitions = None
     initial_pos = np.random.uniform(low=-5.5, high=5.5, size=2)
     initial_angle = np.random.uniform(low=-3.14, high=3.14, size=1).squeeze()
     initial_quat = np.array([np.cos((initial_angle)/2), 0, 0, np.sin((initial_angle)/2)])
     print('initial position and angle:', initial_pos, initial_angle)
     self.reset(initial_pos=initial_pos, initial_quat=initial_quat)
     self.target_pos = np.random.uniform(low=-5.5, high=5.5, size=2)
     self.target_angle = np.random.uniform(low=-3.14, high=3.14, size=1).squeeze()
     print('target position and angle:', self.target_pos, self.target_angle)
     while not full:
        

        command = self.select_action()
        transitions = self.controller(command)
        done = transitions['dones'][0,0]
        if done:
           initial_pos = np.random.uniform(low=-5.5, high=5.5, size=2)
           initial_angle = np.random.uniform(low=-3.14, high=3.14, size=1).squeeze()
           initial_quat = np.array([np.cos((initial_angle)/2), 0, 0, np.sin((initial_angle)/2)])
           print('collision, reset to:', initial_pos, initial_angle)
           self.reset(initial_pos=initial_pos, initial_quat=initial_quat)
        desired_transitions, full = self.update_desired_transition(desired_transitions, transitions, max_steps= self.frame_number)

     return desired_transitions
        
 

  def make_h5py_file(self, seeds:int, episode_size:int, episode_num: int):
    self.seeds = seeds
    self.key = jax.random.PRNGKey(seeds)
    self.frame_number = episode_size

    for i in range(episode_num):

    # generate a random point
      t1 = time.time()
      newseeds = self.seeds + i
      np.random.seed(newseeds)
      desired_transitions = self.collect_one_episod_data()

      

      with h5py.File("Navigation_Mujoco_dataset_S1.h5", "a") as f:

          grp = f.create_group(f"episode_{newseeds}")

          for k in desired_transitions.keys():
            if k == 'images':
              grp.create_dataset(k, data=desired_transitions[k], compression="gzip")
            else:
              grp.create_dataset(k, data=desired_transitions[k])
            
          print("One episode time:", time.time() - t1)  
          print(f"The seeds of {newseeds} episodes collection is done!")
      
      gc.collect()    

  def prepare_Dinov3(self):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", self.device)

    model_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"
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
  
  def reset(self, initial_pos=[0,0],initial_quat=[1,0,0,0]):
    self.steps = 0
    self.image_list = []
    self.feature_list = []
    self.next_image_list = []
    self.next_feature_list = []
    self.action_list = []
    self.done_list = []
    self.position_list = []
    self.quaterion_list = []
    self.frame_list =  []

    mujoco.mj_resetData(self.mj_model, self.mj_data)
    current_pos = jnp.array(initial_pos).reshape([2])
    current_quat = initial_quat
    self.current_pos = current_pos
    self.current_angle = cal_z_from_quat(current_quat).squeeze()
    self.mj_data.mocap_pos = np.array([current_pos[0],current_pos[1], 0.3]).reshape([3])
    self.mj_data.mocap_quat = np.array([current_quat]).reshape([4])
    mujoco.mj_step(self.mj_model, self.mj_data)
    self.renderer.update_scene(self.mj_data, 'came')
    self.update_Translation()


  def is_in_obstacle(self, point):
    point = point.reshape([2,1])
    if jnp.abs(point[0]) > 6.5 or jnp.abs(point[1]) > 6.5:
            return True
    return False


  def update_Translation(self):
    rz = self.current_angle
    self.Rz = np.array([[np.cos(rz), -np.sin(rz)],
                   [np.sin(rz), np.cos(rz)]]).reshape([2,2])
     
  
  def cal_command(self, action):
    '''transfer action to mujoco_position command'''
    action = np.array(action).reshape([3])
    current_pos = self.mj_data.mocap_pos.reshape([3])
    current_quat = self.mj_data.mocap_quat
    current_angle = cal_z_from_quat(current_quat)
    self.update_Translation()
    self.current_pos = current_pos
    self.current_angle = current_angle
    add_angle = action[2]

    translation_action_in_world = self.Rz @ action[0:2].reshape([2,1])
    rotation_action_in_world = add_angle 

    next_pos = np.array([current_pos[0] + translation_action_in_world[0], current_pos[1] + translation_action_in_world[1]])
    new_angle = current_angle + rotation_action_in_world
    next_quat = jnp.array([jnp.cos((new_angle)/2), 0, 0, jnp.sin((new_angle)/2)])
    return next_pos.reshape([2]), next_quat.reshape([4])
  
  def step(self, action):
    self.steps += 1
    next_pos, next_quat = self.cal_command(action)
    collision = False
    collision = self.is_in_obstacle(next_pos)
    self.mj_data.mocap_pos = np.array([next_pos[0],next_pos[1], 0.3]).reshape([3])
    self.mj_data.mocap_quat = np.array([next_quat]).reshape([4])
    mujoco.mj_step(self.mj_model, self.mj_data)
    self.renderer.update_scene(self.mj_data, 'came')
    image = self.renderer.render()
    feature = self.cal_latent(image).squeeze()
    done = collision

    return image, feature, done, next_pos, next_quat
  
  def step_without_Dinov3(self, action):
    self.steps += 1
    next_pos, next_quat = self.cal_command(action)
    collision = False
    collision = self.is_in_obstacle(next_pos)
    self.mj_data.mocap_pos = np.array([next_pos[0],next_pos[1], 0.3]).reshape([3])
    self.mj_data.mocap_quat = np.array([next_quat]).reshape([4])
    mujoco.mj_step(self.mj_model, self.mj_data)
    image = []
    feature =[]




    done = collision

    return image, feature, done, next_pos, next_quat
  
  def eval_controller(self, command):
    command = np.array(command).reshape([3])
    if abs(command[0]) < 0.05:
      command[0] = 0.05 * np.sign(command[0])
    self.current_command = np.array(command).reshape([3])
    image, feature, done, next_pos, next_quat = self.step([0,0,0])


   # plt.imshow(image)

   # plt.show()

    self.image_list.append(image)
    self.feature_list.append(feature)
    self.position_list.append(next_pos)
    self.quaterion_list.append(next_quat)
    self.action_list.append(self.current_command.copy())
    action = self.current_command.copy()
    image, feature, done, next_pos, next_quat = self.step(action)
    done = jnp.array(done).reshape([1])
    self.next_image_list.append(image)
    self.next_feature_list.append(feature)
    self.done_list.append(done)



    

    return image, feature, done
    
  def controller(self, command):
    if abs(command[0]) < 0.05:
        command[0] = 0.05 * np.sign(command[0])

    self.current_command = np.array(command).reshape([3])
    image, feature, done, next_pos, next_quat = self.step([0,0,0])
    
    self.image_list = []
    self.feature_list = []
    self.next_image_list = []
    self.next_feature_list = []
    self.action_list = []
    self.done_list = []
    self.position_list = []
    self.quaterion_list = []
   # plt.imshow(image)

   # plt.show()


    sample_interal = self.sample_interal
    self.image_list.append(image)
    self.frame_list.append(image)
    self.feature_list.append(feature)
    self.position_list.append(next_pos)
    self.quaterion_list.append(next_quat)
    self.action_list.append(self.current_command.copy())
    action = self.current_command.copy()
    image, feature, done, next_pos, next_quat = self.step(action)
    done = jnp.array(done).reshape([1])
    self.next_image_list.append(image)
    self.next_feature_list.append(feature)
    self.done_list.append(done)
    images = np.stack(self.image_list, axis=0)
    features = np.stack(self.feature_list, axis=0)
    actions = np.stack(self.action_list, axis=0)

    positions = np.stack(self.position_list, axis=0)
    quaterions = np.stack(self.quaterion_list, axis=0)
    next_features = np.stack(self.next_feature_list, axis=0)
    dones = np.stack(self.done_list, axis=0)
    next_images = np.stack(self.next_image_list, axis=0)


    transition = {
    'images': images,
    'features': features,
    "actions": actions, 
    "next_images": next_images,
    'next_features': next_features,
    'positions': positions,
    'quaterions': quaterions,
    "dones": dones,
      } 
    return transition
  
  def create_expert_trajectory(self):

    
    desired_transitions = None
    self.reset(initial_pos=[4,-3])

  
    transitions = self.controller([-0.1, 0.1, 0.5])
    desired_transitions, full = self.update_desired_transition(desired_transitions, transitions, max_steps= 1000)
    transitions = self.controller([-0.1, 0.1, 0.5])
    desired_transitions, full = self.update_desired_transition(desired_transitions, transitions, max_steps= 1000)


    for i in range(20):
      transitions = self.controller([0.2, 0.0, 0])
      desired_transitions, full = self.update_desired_transition(desired_transitions, transitions, max_steps= 1000)

    media.show_video(self.frame_list, fps=10)

    media.write_video("indoorframetest222.mp4", self.frame_list, fps=10)

    

      

    with h5py.File("Navigation_Mujoco_dataset_expert_S1.h5", "a") as f:

          grp = f.create_group(f"episode_{1}")

          for k in desired_transitions.keys():
            if k == 'images':
              grp.create_dataset(k, data=desired_transitions[k], compression="gzip")
            else:
              grp.create_dataset(k, data=desired_transitions[k])
            
          print(f"The seeds of {1} episodes collection is done!")
      
    gc.collect()    
     
  def evaluation_tracjectory(self, goal_pos, goal_quat, final_reward):
    trajectory = np.stack(self.position_list, axis=0)
    goal_angle = cal_z_from_quat(goal_quat).squeeze()
    u = 0.5 * jnp.cos(goal_angle)*0.6
    v = 0.5 * jnp.sin(goal_angle)*0.6

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(trajectory[:, 0], trajectory[:, 1], color='blue', label='Trajectory')

    betta = 30
    for i in range(len(trajectory)//betta-1):
        j= betta*i
        plt.arrow(
                trajectory[j, 0], trajectory[j, 1],
                trajectory[j+10, 0] - trajectory[j, 0],
                trajectory[j+10, 1] - trajectory[j, 1],
                shape='full', head_width=0.05, length_includes_head=True, color='green'
            )
    ax.plot(goal_pos[0], goal_pos[1], marker='*', color='red', markersize=15, label='Goal')

    plt.arrow(goal_pos[0], goal_pos[1], u,v, shape='full', head_width=0.5, length_includes_head=True, color='red')
    if final_reward > 0.9:

      plt.text(goal_pos[0],goal_pos[1], 'Success!', fontsize=12, ha='center')
    ax.axis('equal')
    ax.legend()
    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # (H, W, 4)

    # ARGB → RGB
    img_array = buf[:, :, 1:].copy()   #
    plt.close(fig)
    gc.collect()

    return img_array

  def prepare_evaluation_state(self):
    x_points = jnp.arange(-6.4, 6.4, 0.8)  
    y_points = jnp.arange(-6.4, 6.4, 0.8)
    X, Y = jnp.meshgrid(x_points, y_points)
    coordinates = jnp.vstack([X.ravel(), Y.ravel()]).T
    angle = 0
    self.eval_inital_angle = angle
    initial_quat = np.array([np.cos((angle)/2), 0, 0, np.sin((angle)/2)])
    feature_list = []

    for i in coordinates:
       self.reset(initial_pos=i, initial_quat=initial_quat)
       image, feature, done, next_pos, next_quat = self.step([0.0, 0.0, 0.0])
       feature_list.append(feature)
    self.goal_number = 136
    self.eval_features = np.stack(feature_list, axis=0)
    self.eval_goal_features =  jnp.broadcast_to(self.eval_features[self.goal_number], self.eval_features.shape)
    self.eval_positions = coordinates


    action0 = jnp.array([0, 0])
    action1 = jnp.arange(-0.55, 0.55, 0.55/2.5)
    X,Y = jnp.meshgrid(action0, action1)
    self.eval_action_space = jnp.vstack([X.ravel(), Y.ravel()]).T


  def draw_action_vector(self, positions, actions):
    # draw rotation direction
    angles = self.eval_inital_angle + actions[:,2] 
    u = jnp.cos(angles)*0.005
    v =  jnp.sin(angles)*0.005
    fig, ax = plt.subplots(figsize=(6,6), dpi=100)
    ax.plot(self.eval_positions[self.goal_number,0], self.eval_positions[self.goal_number,1], marker='*', color='red', markersize=15, label='Goal')
    plt.arrow( self.eval_positions[self.goal_number,0], self.eval_positions[self.goal_number,1],1.5, 0,shape='full', head_width=0.2, length_includes_head=True, color='red')

    ax.quiver(positions[:, 0], positions[:, 1], u, v, scale=0.1, color='orange')

    ax.axis('equal')
    ax.legend()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    canvas_width, canvas_height = fig.canvas.get_width_height()
    buf = buf.reshape(canvas_width, canvas_height, 4)

        # ARGB → RGB
    img_rotation_vector = buf[:, :, 1:].copy()   #
    plt.close(fig)
    gc.collect()


    # Draw translation direction 
    action_in_robot = actions[:,0:2]
    Rz = jnp.array([[jnp.cos(self.eval_inital_angle), -jnp.sin(self.eval_inital_angle)],
                    [jnp.sin(self.eval_inital_angle), jnp.cos(self.eval_inital_angle)]])
    actions_in_world = action_in_robot
    u = np.sign(actions_in_world[:,0]) * np.exp(actions_in_world[:,0]) * 0.01
    v =  np.sign(actions_in_world[:,1]) * np.exp(actions_in_world[:,1]) * 0.01
    fig, ax = plt.subplots(figsize=(6,6), dpi=100)
    ax.plot(self.eval_positions[self.goal_number,0], self.eval_positions[self.goal_number,1], marker='*', color='red', markersize=15, label='Goal')
    plt.arrow( self.eval_positions[self.goal_number,0], self.eval_positions[self.goal_number,1],1.5, 0,shape='full', head_width=0.2, length_includes_head=True, color='red')

    ax.quiver(positions[:, 0], positions[:, 1], u, v, scale=0.1, color='blue')

    ax.axis('equal')
    ax.legend()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    canvas_width, canvas_height = fig.canvas.get_width_height()
    buf = buf.reshape(canvas_width, canvas_height, 4)

        # ARGB → RGB
    img_translation_vector = buf[:, :, 1:].copy()   #
    plt.close(fig)
    gc.collect()


    return img_rotation_vector, img_translation_vector

  def draw_Qvalue_vector(self, positions, eval_Q):
    N = eval_Q.shape[0]
    fig, ax = plt.subplots(figsize=(6,6), dpi=100)
 
    for i in range(N):
      angles = self.eval_inital_angle + self.eval_action_space[i][1]
      u = jnp.exp(eval_Q[i]) * jnp.cos(angles)*0.006
      v = jnp.exp(eval_Q[i]) * jnp.sin(angles)*0.006
      ax.quiver(positions[:, 0], positions[:, 1], u, v, scale=0.1, color='yellow')

    argmax_Q = eval_Q.argmax(axis=0)
    max_Q = eval_Q.max(axis=0)
    set_action_space = jnp.broadcast_to(self.eval_action_space[:,1], [len(max_Q), self.eval_action_space.shape[0]])

    angles = set_action_space[np.arange(len(argmax_Q)), argmax_Q] + self.eval_inital_angle

    u = jnp.exp(max_Q) * jnp.cos(angles)*0.006
    v = jnp.exp(max_Q) * jnp.sin(angles)*0.006
    ax.quiver(positions[:, 0], positions[:, 1], u, v, scale=0.1, color='blue')

    ax.plot(self.eval_positions[self.goal_number,0], self.eval_positions[self.goal_number,1], marker='*', color='red', markersize=15, label='Goal')
    plt.arrow( self.eval_positions[self.goal_number,0], self.eval_positions[self.goal_number,1],1.5, 0,shape='full', head_width=0.2, length_includes_head=True, color='red')


    ax.axis('equal')
    ax.legend()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    canvas_width, canvas_height = fig.canvas.get_width_height()
    buf = buf.reshape(canvas_width, canvas_height, 4)

        # ARGB → RGB
    img_array = buf[:, :, 1:].copy()   #
    plt.close(fig)
    gc.collect()
     
    return img_array

  def evaluation_vector(self):
     
     pass






#test = Navigation_sim_environment()
#test.create_expert_trajectory()
#test.make_h5py_file(seeds=1, episode_size=1000, episode_num = 20)


     
    
     




      
    




