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




path  = os.getcwd() + '/fake_go2.xml'
sys = mjcf.load(path)

# Make model, data, and renderer
mj_model = mujoco.MjModel.from_xml_path(path)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)


# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 30# (seconds)
framerate = 10  # (Hz)

frames = []
action = []
key = jax.random.PRNGKey(1)
mujoco.mj_resetData(mj_model, mj_data)
current_pos = jnp.array([0, 0]).reshape([2,1])
current_quat = jnp.array([1, 0, 0, 0]).reshape([4,1])
while mj_data.time < duration:

  key, pos_list, quat_list, distances_list, dir_list = random_explore_policy(key, current_pos, current_quat)
  for i in zip(pos_list, quat_list):
    next_pos = i[0]
    next_quat = i[1]
    mj_data.mocap_pos = np.array([next_pos[0],next_pos[1], 0.3]).reshape([3])
    mj_data.mocap_quat = np.array([next_quat]).reshape([4])
    current_pos = next_pos
    current_quat = next_quat


    mujoco.mj_step(mj_model, mj_data)
    #if len(frames) < mj_data.time * framerate:
    #renderer.update_scene(mj_data, scene_option=scene_option)
    renderer.update_scene(mj_data, 'came')
    pixels = renderer.render()
    frames.append(pixels)
  


# Simulate and display video.
media.show_video(frames, fps=framerate)

media.write_video("indoorframetest.mp4", frames, fps=framerate)
print('end')


