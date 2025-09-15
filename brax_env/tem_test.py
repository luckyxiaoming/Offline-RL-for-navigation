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

from scipy.interpolate import CubicHermiteSpline



obstacles = [
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




def is_in_obstacle(point, obstacles):
    point = point.reshape([2,1])
    for obs in obstacles:
        distance = np.square(np.array(point).squeeze() - obs["pos"]).sum()**0.5
        if distance <  obs["radius"]:
            return True
        if jnp.abs(point[0]) > 9 or jnp.abs(point[1]) > 9:
            return True
    return False



def random_policy(key, current_quat=[1,0,0,0] ,max_distance=0.3, max_rotation=0.8, min_abs_rotation=False):
    current_angle = jnp.asin(current_quat[3]).reshape([1]) * 2
    rotation = jax.random.uniform(key[1], shape=(1,), minval=-max_rotation, maxval=max_rotation)
    if min_abs_rotation:
      rotation = jnp.sign(rotation)*jax.random.uniform(key[1], shape=(1,), minval=0.707, maxval=max_rotation)
    rotation = rotation + current_angle
    distance = jax.random.uniform(key[2], shape=(1,), minval=max_distance/2, maxval=max_distance)
    action = jnp.array([distance*jnp.cos(rotation), distance*jnp.sin(rotation)]).squeeze()
    return action


def step_controller(points, direction_values):
  actions = points[1:] - points[:-1]
  dir_acts = direction_values[1:] - direction_values[:-1]
  distances = jnp.sum(jnp.square(actions), axis=1)**0.5
  pos_list=[]
  quat_list=[]
  distances_list=[]
  dir_list=[]
  for i in range(len(dir_acts)):
    next_quat = jnp.array([jnp.cos((direction_values[i+1])/2), 0, 0, jnp.sin((direction_values[i+1])/2)])
    pos_list.append(points[i+1])
    quat_list.append(next_quat)
    distances_list.append(distances[i])
    dir_list.append(dir_acts[i])

  return pos_list, quat_list, distances_list, dir_list



   


def explore(key, current_pos, current_quat):
  current_pos = current_pos.squeeze()
  collision = True
  j = 0
  while collision:
    j += 1
    keys = jax.random.split(key, 5)
    key = keys[3]
    Rsign = jax.random.choice(keys[4], jnp.array([1, 2, 3, 4, 5]), shape=())
    if not Rsign == 1:
      action = random_policy(keys, max_rotation=0.02, current_quat=current_quat)
    else:
      action = random_policy(keys, max_distance=0.002 ,max_rotation=2, current_quat=current_quat, min_abs_rotation=True)
    next_pos = current_pos + action
    collision = is_in_obstacle(next_pos, obstacles)

  distance = jnp.sum(jnp.square(action))**0.5
  new_angle = jnp.arctan2(action[1], action[0]).reshape([1])
  next_quat = jnp.array([jnp.cos((new_angle)/2), [0], [0], jnp.sin((new_angle)/2)])



  add_angle = new_angle - jnp.asin(current_quat[3]).reshape([1]) * 2
  pos_list=[]
  quat_list=[]
  act_list=[]
  steps = max(distance//0.05, jnp.abs(add_angle//0.05))
  steps = max(np.array([1]), steps).reshape([1])
  points, direction_values = interpolate_path(current_pos, current_quat, next_pos, next_quat, int(steps[0]))
  pos_list, quat_list, distances_list, dir_list = step_controller(points, direction_values)

  return key, pos_list, quat_list, distances_list, dir_list
    

def interpolate_path(current_pos, current_quat, next_pos, next_quat, steps):
    
    
    A = np.array(current_pos).squeeze()
    B = np.array(next_pos).squeeze()
    C = np.arcsin(current_quat[3].squeeze()) * 2  
    D = np.arcsin(next_quat[3].squeeze()) * 2     
    distance = np.square(B - A).sum()**0.5
     
    # calculate tangent vectors
    tangent_A = np.array([np.cos(C), np.sin(C)])
    tangent_B = np.array([np.cos(D), np.sin(D)])


    t = np.array([0, distance])
    points = np.vstack((A, B))
    tangents = np.vstack((tangent_A, tangent_B))

    # create cubic Hermite splines for x and y coordinates
    path_x = CubicHermiteSpline(t, points[:, 0], tangents[:, 0])
    path_y = CubicHermiteSpline(t, points[:, 1], tangents[:, 1])

    t_values = np.linspace(0, distance, steps)

    x_values = path_x(t_values)
    y_values = path_y(t_values)
    points = np.column_stack((x_values, y_values))


    vel_x = path_x.derivative()
    vel_y = path_y.derivative()
    dx_values = vel_x(t_values)
    dy_values = vel_y(t_values)
    direction_values = np.arctan2(dy_values, dx_values)
    return points, direction_values






path  = os.getcwd() + '/fake_go2.xml'
sys = mjcf.load(path)

# Make model, data, and renderer
mj_model = mujoco.MjModel.from_xml_path(path)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)





# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

jit_step = jax.jit(mjx.step)

duration = 100# (seconds)
framerate = 10  # (Hz)

frames = []


key = jax.random.PRNGKey(1)
mujoco.mj_resetData(mj_model, mj_data)
mjx_data = mjx.put_data(mj_model, mj_data)

current_pos = jnp.array([0, 0]).reshape([2,1])
current_quat = jnp.array([1, 0, 0, 0]).reshape([4,1])
while mjx_data.time < duration:
  #a= 0.02*np.sin(np.array(mj_data.time))
  #print(a)
  key, pos_list, quat_list, distances_list, dir_list = explore(key, current_pos, current_quat)
  for i in zip(pos_list, quat_list):
    next_pos = i[0]
    next_quat = i[1]
    mj_data.mocap_pos = np.array([next_pos[0],next_pos[1], 0.3]).reshape([3])
    mj_data.mocap_quat = np.array([next_quat]).reshape([4])
    current_pos = next_pos
    current_quat = next_quat
    mjx_data = mjx.put_data(mj_model, mj_data)


    mjx_data = jit_step(mjx_model, mjx_data)
    mj_data = mjx.get_data(mj_model, mjx_data)
    if len(frames) < mjx_data.time * framerate:
      #renderer.update_scene(mj_data, scene_option=scene_option)
      
      renderer.update_scene(mj_data, 'godcame')
      pixels = renderer.render()
      frames.append(pixels)
  


# Simulate and display video.
media.show_video(frames, fps=framerate)

media.write_video("indoorframetest.mp4", frames, fps=framerate)
print('end')


