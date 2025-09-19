import h5py
import numpy as np
import flashbax as fbx
import jax.numpy as jnp
import jax
import time



def make_buffer():
    t1 = time.time()

    ################################################
    ########     prepare Buffer      ###############
    ################################################

    # First define hyper-parameters of the buffer.
    max_length_time_axis = 40000 # Maximum length of the buffer along the time axis. 
    min_length_time_axis = 10000 # Minimum length across the time axis before we can sample.
    sample_batch_size = 1 # Batch size of trajectories sampled from the buffer.
    add_batch_size = 1 # Batch size of trajectories added to the buffer.
    sample_sequence_length = 512 # Sequence length of trajectories sampled from the buffer.
    period = 1 # Period at which we sample trajectories from the buffer.

    # Instantiate the trajectory buffer, which is a NamedTuple of pure functions.
    buffer = fbx.make_trajectory_buffer(
        max_length_time_axis=max_length_time_axis,
        min_length_time_axis=min_length_time_axis,
        sample_batch_size=sample_batch_size,
        add_batch_size=add_batch_size,
        sample_sequence_length=sample_sequence_length,
        period=period
        )

    fake_timestep = {
            'feature': jnp.zeros(768, dtype=jnp.float32),
            'action': jnp.zeros(2, dtype=jnp.float32)
        }


    buffer_state = buffer.init(fake_timestep)

    ################################################
    ######## Load data from h5 files ###############
    ################################################

    file_path = ["Navigation_Mujoco_latent_1.h5",
                 "Navigation_Mujoco_latent_2.h5"
                 ]
 

    act1min = 1e9-1
    act1max = -1e9+1
    act2min = 1e9-1
    act2max = -1e9+1
   
    for path in file_path:
        with h5py.File(path, "r") as f:
            key = list(f.keys())
            grp = f[key[0]]
            # latents size (2000X, 1, 768), the first element shoule be abandoned
            latents = grp["latents"][1:]
            actions = grp["actions"][1:]
            act1min = min(actions[:, 0].min(), act1min)
            act1max = max(actions[:, 0].max(), act1max)
            act2min = min(actions[:, 1].min(), act2min)
            act2max = max(actions[:, 1].max(), act2max)
            
            feature_array = jnp.array(latents).reshape(1,latents.shape[0],768).astype(jnp.float32)
            action_array = jnp.array(actions).reshape(1,latents.shape[0],2).astype(jnp.float32)
            experiences = {
                    'feature':feature_array,
                    'action': action_array
                } 
            buffer_state = buffer.add(buffer_state, experiences)











    env_params = {
      "action_space_low": jnp.array([act1min, act2min]),
      "action_space_high": jnp.array([act1max, act2max]) ,
      "action_dimension": 2,
      "observation_size": 768,
      "action_discrete": False
  }
    print(f"The buffer is done! Cost time: {time.time() - t1} seconds")


    return buffer, buffer_state, env_params



class sample_buffer():
    def __init__(self,done_threshold=-0.4):
        self.sign_normal_obs = False
        self.sign_normal_reward = False
        self.obs_mean = 0 
        self.obs_std = 1
        self.scale_reward = 1
        
        self.done_threshold = done_threshold
        
        self.buffer, self.buffer_state, self.env_params = make_buffer() 
        self.key = jax.random.PRNGKey(3)

    def sample1(self, batch_size: int, sequence_length: int):
        '''sample a batch of data from a sequence'''
        if batch_size >= sequence_length:
            raise ValueError("batch size should be smaller than sequence length!")

        keys = jax.random.split(self.key, 5)
        self.key = keys[0]
        batch = self.buffer.sample(self.buffer_state, keys[1], batch_size=1, sequence_length=512)
        feature = batch.experience['feature'] 
        action = batch.experience['action']
        random_index = jax.random.randint(keys[2], shape=(batch_size,), minval=0, maxval=sequence_length-1)
        goal_index = jax.random.randint(keys[3], shape=(1,), minval=0, maxval=sequence_length-1)
        state = jnp.array([feature[1,random_index,:]])
        action = jnp.array([action[1,random_index,:]])
        next_state = jnp.array([feature[1,random_index+1,:]])
        goal_state = jnp.array([feature[1,goal_index,:]])
        reward, done = self.calculate_reward(state, goal_state)

        return state, goal_state, action, next_state, reward, done 
    
    def normalize_reward(self, scale_factor):

        if self.sign_normal_reward:
            raise NameError("This function was already called!")
        self.scale_reward = scale_factor
        print(f'The reward in Dataset has been norimalized!')
        self.sign_normal_reward = True

    def calculate_reward(self, state, goal_state):
        ''' calculate the reward and done signal
         the L2 range is about (-?, 0)
         the reward range is about (-1, 0)
         plese set _self.done_threshold_ to control when the episode is done'''

        L2 = -jnp.square(state - goal_state).mean(axis=-1)
        reward = jnp.tanh(0.5*(L2))  # normalize the reward to (-1, 1)
        done_reward = jnp.tanh(0.5*self.done_threshold)
        done = jnp.where(reward>done_reward, 1, 0)
        return reward, done
       
    
    def normalize_observations(self):


        if self.sign_normal_obs:
            raise NameError("This function was already called!")
        self.sign_normal_obs = True
        index = self.buffer_state.current_index
        obs_array = self.buffer_state.experience['observation']
        current_array = obs_array[:, 0:index, :]
        self.obs_mean = current_array.mean(axis = 1)
        self.obs_std = current_array.std(axis = 1) + 1e-6
        print(f'The observation in Dataset has been norimalized!')
        return self.obs_mean, self.obs_std


#buffer, buffer_state, env_params = make_buffer()

test = sample_buffer()
state, goal_state, action, next_state, reward, done = test.sample1(batch_size=256 ,sequence_length=512)
print(state.shape, action.shape, next_state.shape, reward.shape, done.shape)