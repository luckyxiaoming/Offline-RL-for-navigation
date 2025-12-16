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
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import h5py
import torch
import gc
from pynput import keyboard


'''
frames extraction function cannot work in brax env(MJX), so mujoco is used here.
'''

import numpy as np
from numpy.fft import irfft, rfftfreq
from scipy.stats import norm
@jax.jit
def calculate_cosine_similarity(X, Y):
    dot = jnp.sum(X*Y, axis=-1)
    norm_x = jnp.linalg.norm(X, axis=-1)
    norm_y = jnp.linalg.norm(Y, axis=-1)
  
    L2 = norm_x * norm_y + 1e-8
    return dot / L2

def powerlaw_psd(exponent, size, fmin=0, mode= 'uniform', rng=None):
    """Gaussian (1/f)**beta noise.

    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

    Normalised to unit variance

    Parameters:
    -----------

    exponent : float
        The power-spectrum of the generated noise is proportional to

        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2

        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.

    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.

    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper.

        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.

    rng : np.random.Generator, optional
        Random number generator (for reproducibility). If not passed, a new
        random number generator is created by calling
        `np.random.default_rng()`.


    Returns
    -------
    out : array
        The samples.


    Examples:
    ---------

    >>> # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """

    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)

    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1./samples)    # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    # Build scaling factors for all frequencies
    s_scale = f
    ix = np.sum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.    # correct f = +-0.5
    sigma = 2 * np.sqrt(np.sum(w**2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(None,) * dims_to_add + (Ellipsis,)]

    # Generate scaled random power + phase
    if rng is None:
        rng = np.random.default_rng()
    sr = rng.normal(scale=s_scale, size=size)
    si = rng.normal(scale=s_scale, size=size)

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si[..., -1] = 0
        sr[..., -1] *= np.sqrt(2)    # Fix magnitude

    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0
    sr[..., 0] *= np.sqrt(2)    # Fix magnitude

    # Combine power + corrected phase to Fourier components
    s = sr + 1J * si

    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma

    if mode == 'uniform':
        z = 2 * norm.cdf(y) - 1  # map to -1 to 1
        return z
    
    return y

    


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
  def __init__(self, args):

   

    path  = os.getcwd() + '/fake_go2flat.xml'
    print('Mujoco model path:', path)

    # Make model, data, and renderer
    self.mj_model = mujoco.MjModel.from_xml_path(path)
    self.mj_data = mujoco.MjData(self.mj_model)
    self.renderer = mujoco.Renderer(self.mj_model)
    


    # define max_action parameters
    self.action_min = args.action_min
    self.action_max = args.action_max

    self.training_dataset_name = args.train_dataset_name

    self.sample_interal = 6
    self.beta = args.beta
    self.initialize_colored_noise(beta1=self.beta, beta2=self.beta, beta3=self.beta)


    # prepare Dinov3
    self.prepare_Dinov3()
    gc.collect()



  
  def initialize_colored_noise(self, beta1, beta2, beta3):
         ## initialize colored noise for action exploration
    noise1 = powerlaw_psd(exponent=beta1, size =(int(60),int(200)), mode='uniform').ravel()
    noise2 = powerlaw_psd(exponent=beta2, size =(int(60),int(200)), mode='uniform').ravel()
    noise3 = powerlaw_psd(exponent=beta3, size =(int(60),int(200)), mode='uniform').ravel()

    x_action = (self.action_max[0] + self.action_min[0])/2 + (self.action_max[0] - self.action_min[0])/2 * noise1
    y_action = (self.action_max[1] + self.action_min[1])/2 + (self.action_max[1] - self.action_min[1])/2 * noise2
    z_action = (self.action_max[2] + self.action_min[2])/2 + (self.action_max[2] - self.action_min[2])/2 * noise3

    self.action_noise = np.stack((x_action, y_action, z_action))
    self.action_step = 0
     

  def select_action(self):
    
    self.action_step += 1
    i = self.action_step 
    action_command = np.array([self.action_noise[0, i], self.action_noise[1, i], self.action_noise[2, i]])
    action_command = np.round(action_command, 4)

  
    return action_command
       

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

  def collect_one_episod_data(self, seeds:int):
     
    full = False
    desired_transitions = None
    initial_pos = np.random.uniform(low=3.5, high=5.5, size=2) * np.random.choice([-1,1], size=2)
    initial_angle = np.random.uniform(low=-3.14, high=3.14, size=1).squeeze()
    initial_quat = np.array([np.cos((initial_angle)/2), 0, 0, np.sin((initial_angle)/2)])
    print('initial position and angle:', initial_pos, initial_angle)
    self.reset(initial_pos=initial_pos, initial_quat=initial_quat)
    try_number = 0
    while not full:
        

        
      ## detect collision
      collision = True
      


      command = self.select_action()

      if self.action_step >= 9990:
          self.initialize_colored_noise(beta1=self.beta, beta2=self.beta, beta3=self.beta)
          self.action_step  = 0


      next_pos1, next_quat = self.cal_command_without_change_env(command)
      next_pos = np.round(next_pos1, 4)
      next_pos2, next_quat = self.cal_command_without_change_env(command)

      collision = self.is_in_obstacle(next_pos)

      if collision:
          try_number += 1
          if (try_number+1)  % 100 == 0:
            print('stuck in obstacle for ', try_number, ' times')

      transitions = self.controller(command, collision=collision)

      desired_transitions, full = self.update_desired_transition(desired_transitions, transitions, max_steps= self.frame_number)

    if seeds % 10 ==0:
      media.show_video(self.frame_list, fps=10)
      media.write_video(f"indoorframe_{seeds}.mp4", self.frame_list, fps=10)


    return desired_transitions
  

          
  def online_evaluation(self, actor, goal_state, goal_pos, initial_pos, initial_quat, name, done_threshold):
    full = False
    desired_transitions = None
    self.frame_list = []
    self.reset(initial_pos=initial_pos, initial_quat=initial_quat)
    try_number = 0
    image, feature, done, next_pos, next_quat = self.step([0,0,0])
    while not full:
      ## detect collision
      collision = True
 
        # Simple Normalization
      
      norms = np.linalg.norm(feature, axis=-1, keepdims=True)
      feature = feature / (norms + 1e-8)

      command = actor.get_eval_action(feature.reshape([1024]), goal_state.reshape([1024]))

      next_pos1, next_quat = self.cal_command_without_change_env(command)
      next_pos = np.round(next_pos1, 4)
      next_pos2, next_quat = self.cal_command_without_change_env(command)

      collision = self.is_in_obstacle(next_pos)

      if collision:
          try_number += 1
          if (try_number+1)  % 100 == 0:
            print('stuck in obstacle for ', try_number, ' times')

      transitions = self.controller(command, collision=collision)
      feature = transitions['features'][-1]
      cos_sim =calculate_cosine_similarity(feature.reshape([1024]), goal_state.reshape([1024]))
      if cos_sim > done_threshold:
          print(f"Reach goal with cosine similarity {cos_sim:.4f}")
          full = True
      
      

      desired_transitions, full = self.update_desired_transition(desired_transitions, transitions, max_steps= 300)
    #media.show_video(self.frame_list, fps=10)
    #media.write_video(f"evaluation_{name}.mp4", self.frame_list, fps=10)

    return desired_transitions, self.frame_list

  def make_h5py_file(self, seeds:int, episode_size:int, episode_num: int, save_images: bool = False):
    self.seeds = seeds
    self.key = jax.random.PRNGKey(seeds)
    self.frame_number = episode_size

    for i in range(episode_num):

    # generate a random point
      t1 = time.time()
      newseeds = self.seeds + i
      np.random.seed(newseeds)
      desired_transitions = self.collect_one_episod_data(newseeds)

      

      with h5py.File(f"{self.training_dataset_name}", "a") as f:

          grp = f.create_group(f"episode_{newseeds}")

          for k in desired_transitions.keys():
            if k == 'images' or k == "next_images":
              if save_images:
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
    if jnp.abs(point[0]) < 3 and jnp.abs(point[1]) < 3:
            return True
    return False


  def update_Translation(self):
    rz = self.current_angle
    self.Rz = np.array([[np.cos(rz), -np.sin(rz)],
                   [np.sin(rz), np.cos(rz)]]).reshape([2,2])
     
  
  def cal_command_without_change_env(self, action):
      '''transfer action to mujoco_position command'''
      action = np.array(action).reshape([3])
      current_pos = self.mj_data.mocap_pos.reshape([3])
      current_quat = self.mj_data.mocap_quat
      current_angle = cal_z_from_quat(current_quat)

      rz = self.current_angle
      Rz = np.array([[np.cos(rz), -np.sin(rz)],
                   [np.sin(rz), np.cos(rz)]]).reshape([2,2])

      add_angle = action[2]

      translation_action_in_world = Rz @ action[0:2].reshape([2,1])
      rotation_action_in_world = add_angle 

      next_pos = np.array([current_pos[0] + translation_action_in_world[0], current_pos[1] + translation_action_in_world[1]])
      new_angle = current_angle + rotation_action_in_world
      next_quat = jnp.array([jnp.cos((new_angle)/2), 0, 0, jnp.sin((new_angle)/2)])
      return next_pos.reshape([2]), next_quat.reshape([4])


  def cal_command(self, action):
    '''transfer action to mujoco_position command'''
    action = np.array(action).reshape([3])
    current_pos = self.mj_data.mocap_pos.reshape([3])
    current_quat = self.mj_data.mocap_quat
    current_angle = cal_z_from_quat(current_quat)
    self.update_Translation()
    
    self.current_pos = current_pos
    self.current_angle = np.round(current_angle, 6)
    add_angle = action[2]

    translation_action_in_world = self.Rz @ action[0:2].reshape([2,1])
    rotation_action_in_world = add_angle 

    next_pos = np.array([current_pos[0] + translation_action_in_world[0], current_pos[1] + translation_action_in_world[1]])
    new_angle = current_angle + rotation_action_in_world
    next_quat = jnp.array([jnp.cos((new_angle)/2), 0, 0, jnp.sin((new_angle)/2)])
    return next_pos.reshape([2]), next_quat.reshape([4])
  
  def step(self, action, use_Dinov3: bool = True):
    self.steps += 1
    next_pos, next_quat = self.cal_command(action)
    next_pos= np.round(next_pos, 4)
    collision = False
    collision = self.is_in_obstacle(next_pos)
    self.mj_data.mocap_pos = np.array([next_pos[0],next_pos[1], 0.3]).reshape([3])
    self.mj_data.mocap_quat = np.array([next_quat]).reshape([4])
    mujoco.mj_step(self.mj_model, self.mj_data)
    self.renderer.update_scene(self.mj_data, 'came')
    image = self.renderer.render()
    if use_Dinov3:
      feature = self.cal_latent(image).squeeze()
    else:
       feature = jnp.zeros([1024,])
    done = collision

    return image, feature, done, next_pos, next_quat
  
  

    
  def controller(self, command, collision=False):
    if collision:
      col_command = command * np.array([0,0,3])
      self.current_command = np.array(col_command).reshape([3])
    else:


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

    self.image_list.append(image)
    self.frame_list.append(image)
    self.feature_list.append(feature)
    self.position_list.append(next_pos)
    self.quaterion_list.append(next_quat)
    self.action_list.append(np.array(command))
    action = self.current_command.copy()

  

    image, feature, done, next_pos, next_quat = self.step(action, use_Dinov3=False)
    image, feature, done, next_pos, next_quat = self.step([0,0,0])

    
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
  


  def create_expert_trajectory(self, name2):

    
    desired_transitions = None
    self.frame_list = []
    self.reset(initial_pos=[0,-4], initial_quat=[0.7071,0,0,0.7071])
    end_flag = False
    commands = [0.0, 0.0, 0.0]
    import mujoco.viewer
    key_states = {
      'y': False,
      't': False,
      'u': False,
      's': False,
      'o': False,
      'p': False,
    }
    def on_press(key):
      """callback when a key is pressed"""
      try:
          char = key.char.lower()
          if char in key_states:
              key_states[char] = True
      except AttributeError:
          # Handle non-character keys (e.g., Shift, Ctrl, but we only care about WASD)
          pass
    def on_release(key):
      """callback when a key is released"""
      try:
          char = key.char.lower()
          if char in key_states:
              key_states[char] = False
              # Ensure control signals are zeroed on key release; handled by main loop, but state updated here
      except AttributeError:
          pass
      
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    expert_number = 15




    with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
    
      for expert_i in range(expert_number):

        desired_transitions = None


            
        while viewer.is_running() and not key_states['s']:
            if key_states['y']:
              commands[0] = self.action_max[0]
            if key_states['t']:
              commands[1] = self.action_max[1]
            if key_states['u']:
              commands[1] = self.action_min[1]
            if key_states['o']:
              commands[2] = self.action_max[2]
            if key_states['p']:
              commands[2] = self.action_min[2]

            st = np.abs(np.array(commands)).sum()
            time.sleep(0.1)
      
            if st > 0.0:
              next_pos1, next_quat = self.cal_command_without_change_env(commands)
              next_pos = np.round(next_pos1, 4)
              collision = self.is_in_obstacle(next_pos)

              transitions = self.controller(commands, collision=collision)
              viewer.sync()
              time.sleep(0.1)
              commands = [0.0, 0.0, 0.0]
              desired_transitions, full = self.update_desired_transition(desired_transitions, transitions, max_steps= 1000)
              print('current expert position:',transitions['positions'][-1], "current quaterion:", transitions['quaterions'][-1])


        


        media.show_video(self.frame_list, fps=10)
        media.write_video(f"indoor_{name2}_Tra{expert_i}.mp4", self.frame_list, fps=10)
        with h5py.File(f"Navigation_Mujoco_dataset{name2}_expert_12.1.h5", "a") as f:
          grp = f.create_group(f"episode_{expert_i}")
          for k in desired_transitions.keys():
            if k == 'images':
                grp.create_dataset(k, data=desired_transitions[k], compression="gzip")
            else:
                grp.create_dataset(k, data=desired_transitions[k])
              
          print(f"The seeds of {expert_i} episodes collection is done!")
        desired_transitions = None
        self.frame_list = []
        gc.collect()    
     

  

  def create_test_data(self, name2):

    
    desired_transitions = None
    self.frame_list = []
    self.reset(initial_pos=[0,-4], initial_quat=[0.7071,0,0,0.7071])
    end_flag = False
    commands = [0.0, 0.0, 0.0]
    import mujoco.viewer
    key_states = {
      'y': False,
      't': False,
      'u': False,
      's': False,
      'o': False,
      'p': False,
    }
    def on_press(key):
      """callback when a key is pressed"""
      try:
          char = key.char.lower()
          if char in key_states:
              key_states[char] = True
      except AttributeError:
          # Handle non-character keys (e.g., Shift, Ctrl, but we only care about WASD)
          pass
    def on_release(key):
      """callback when a key is released"""
      try:
          char = key.char.lower()
          if char in key_states:
              key_states[char] = False
              # Ensure control signals are zeroed on key release; handled by main loop, but state updated here
      except AttributeError:
          pass
      
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    expert_number = 15




    with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
    
      for expert_i in range(expert_number):

        desired_transitions = None


            
        while viewer.is_running() and not key_states['s']:
            if key_states['y']:
              commands[0] = self.action_max[0]
            if key_states['t']:
              commands[1] = self.action_max[1]
            if key_states['u']:
              commands[1] = self.action_min[1]
            if key_states['o']:
              commands[2] = self.action_max[2]
            if key_states['p']:
              commands[2] = self.action_min[2]

            st = np.abs(np.array(commands)).sum()
            time.sleep(0.1)
      
            if st > 0.0:
              next_pos1, next_quat = self.cal_command_without_change_env(commands)
              next_pos = np.round(next_pos1, 4)
              collision = self.is_in_obstacle(next_pos)

              transitions = self.controller(commands, collision=collision)
              viewer.sync()
              time.sleep(0.1)
              commands = [0.0, 0.0, 0.0]
              #desired_transitions, full = self.update_desired_transition(desired_transitions, transitions, max_steps= 1000)
              print('current expert position:',transitions['positions'][-1], "current quaterion:", transitions['quaterions'][-1])


        


        #media.show_video(self.frame_list, fps=10)
        #media.write_video(f"indoor_{name2}_Tra{expert_i}.mp4", self.frame_list, fps=10)
        with h5py.File(f"Navigation_Mujoco_dataset{name2}_expert_12.1.h5", "a") as f:
          desired_transitions, full = self.update_desired_transition(desired_transitions, transitions, max_steps= 1000)
          grp = f.create_group(f"Target_{expert_i}")
          for k in desired_transitions.keys():
            if k == 'images':
                grp.create_dataset(k, data=desired_transitions[k], compression="gzip")
            else:
                grp.create_dataset(k, data=desired_transitions[k])
              
          print(f"The target {expert_i}'s information is saved!")
          time.sleep(1)
        desired_transitions = None
        self.frame_list = []
        gc.collect()    
     
  
  def creat_vector_field(self):
 
    
    
    pass



    
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm   


def quat2yaw(quat):
    """
    Converts a quaternion (w, x, y, z) to a yaw angle (in radians).
    Assumes MuJoCo convention: [w, x, y, z]
    """
    # If your data is [x, y, z, w], change indices accordingly.
    # MuJoCo standard is usually [w, x, y, z].
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Standard conversion formula
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw

def analyze_dataset(file_path):
    print(f"Loading dataset: {file_path}...")
    
    # 1. Load Data
    all_positions = []
    all_quaternions = []
    all_actions = []
    
    # Store trajectories separately for plotting lines
    trajectories = [] 

    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        print(f"Found {len(keys)} episodes.")
        
        for key in keys:
            grp = f[key]
            
            # Load raw data
            pos = grp['positions'][:]  # Shape (T, 3) usually (x, y, z)
            quat = grp['quaterions'][:] # Note: Using your spelling 'quaterions'
            act = grp['actions'][:]
            
            all_positions.append(pos)
            all_quaternions.append(quat)
            all_actions.append(act)
            trajectories.append(pos)

    # Concatenate for global stats
    global_pos = np.concatenate(all_positions, axis=0)
    global_quat = np.concatenate(all_quaternions, axis=0)
    global_act = np.concatenate(all_actions, axis=0)

    print(f"Total transitions: {len(global_pos)}")

    # ---------------------------------------------------------
    # 2. Plot 1: Trajectory Map (Top-Down View)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(18, 12))
    
    # Subplot 1: Raw Lines
    ax1 = fig.add_subplot(2, 2, 1)
    for traj in trajectories:
        # Assuming index 0 is X and index 1 is Y
        ax1.plot(traj[:, 0], traj[:, 1], alpha=0.3, linewidth=0.8)
    
    # Plot start and end points of the LAST trajectory to indicate direction
    ax1.scatter(trajectories[-1][0,0], trajectories[-1][0,1], c='green', label='Start (Ex)', s=50)
    ax1.scatter(trajectories[-1][-1,0], trajectories[-1][-1,1], c='red', label='End (Ex)', s=50)
    
    ax1.set_title("All Trajectories (Top-Down)")
    ax1.set_xlabel("Position X")
    ax1.set_ylabel("Position Y")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # 3. Plot 2: Position Coverage Heatmap
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    # 2D Histogram
    h = ax2.hist2d(global_pos[:, 0], global_pos[:, 1], bins=50, cmap='inferno', norm=LogNorm())
    fig.colorbar(h[3], ax=ax2)
    ax2.set_title("Position Density (Heatmap)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    # ---------------------------------------------------------
    # 4. Plot 3: Orientation Coverage (Polar Plot)
    # ---------------------------------------------------------
    # Convert Quaternion to Yaw
    yaws = quat2yaw(global_quat)
    
    ax3 = fig.add_subplot(2, 2, 3, projection='polar')
    # Create histogram bins for angles
    bins_number = 36
    n, bins, patches = ax3.hist(yaws, bins_number, color='skyblue', edgecolor='black')
    ax3.set_title("Heading (Yaw) Coverage")

    # ---------------------------------------------------------
    # 5. Plot 4 & 5: Action Distribution
    # ---------------------------------------------------------
    # Assuming action dim is 2 or 3. If more, we plot the first 2.
    act_dim = global_act.shape[1]
    
    # Plot Action Dim 0 vs Dim 1 (Scatter density)
    #ax4 = fig.add_subplot(2, 3, 4)
    #if act_dim >= 2:
    #    ax4.hist2d(global_act[:, 0], global_act[:, 2], bins=50, cmap='Blues')
    #    ax4.set_xlabel("Action[0]")
    #    ax4.set_ylabel("Action[2]")
    #    ax4.set_title("Action Space Coverage (Dim 0 vs 2)")
    #else:
    #    ax4.hist(global_act[:, 0], bins=50)
    #    ax4.set_title("Action Distribution (Dim 0)")

    # Plot Histogram for all action dimensions
    ax5 = fig.add_subplot(2, 2, 4)
    for i in range(act_dim):
        sns.kdeplot(global_act[:, i], ax=ax5, label=f"Dim {i}", fill=True, alpha=0.3)
    ax5.set_title("Action Value Distribution")
    ax5.legend()

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 6. Calculate Quantitative Metrics
    # ---------------------------------------------------------
    print("\n--- Coverage Metrics ---")
    
    # Metric 1: Position Bounds
    print(f"Position Range X: [{global_pos[:,0].min():.2f}, {global_pos[:,0].max():.2f}]")
    print(f"Position Range Y: [{global_pos[:,1].min():.2f}, {global_pos[:,1].max():.2f}]")
    
    # Metric 2: Estimated Occupied Area (Binning method)
    # We discretize the space into 0.2x0.2 grids and count occupied grids
    x_bins = np.arange(np.floor(global_pos[:,0].min()), np.ceil(global_pos[:,0].max()), 0.2)
    y_bins = np.arange(np.floor(global_pos[:,1].min()), np.ceil(global_pos[:,1].max()), 0.2)
    H, _, _ = np.histogram2d(global_pos[:,0], global_pos[:,1], bins=[x_bins, y_bins])
    
    occupied_bins = np.sum(H > 0)
    total_area_bins = H.size
    print(f"Occupied Spatial Bins (0.2 units): {occupied_bins}")
    print(f"Space Utilization Ratio: {occupied_bins / total_area_bins:.4f} (relative to bounding box)")

    # Metric 3: Action Saturation
    # Check how many actions are hitting the limits (e.g., -1 or 1)
    # Assuming normalized actions [-1, 1] or similar limits
    threshold = 0.99
    saturated_count = np.sum(np.abs(global_act) > threshold)
    print(f"Saturated Actions (>|{threshold}|): {saturated_count} / {global_act.size} ({saturated_count/global_act.size*100:.2f}%)")






from dataclasses import dataclass

@dataclass
class Args:
    action_min = np.array([0.0, -0.1, -0.1])
    action_max = np.array([0.2, 0.1, 0.1])
    beta = 1.0
    train_dataset_name = 'Mujoco_training_dataset_pink_2HZ.h5'

args = Args()

test = Navigation_sim_environment(args=args)
#test.create_expert_trajectory('_4X')
#test.create_test_data('_target')
test.make_h5py_file(seeds=100, episode_size=320, episode_num = 24 , save_images=False)


#file_path = f"/home/xiaoming/Research/Offline RL/12_offline_RL/{args.train_dataset_name}"
#analyze_dataset(file_path)  
    


#
