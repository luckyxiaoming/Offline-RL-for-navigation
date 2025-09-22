import h5py
import numpy as np
import flashbax as fbx
import jax.numpy as jnp
import jax
import time
import gc


def make_buffer():
    t1 = time.time()

    ################################################
    ########     prepare Buffer      ###############
    ################################################

    # First define hyper-parameters of the buffer.
    max_length_time_axis = 80000 # Maximum length of the buffer along the time axis. 
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
         #   'image': jnp.zeros([240,320,3], dtype=jnp.int8),
            'feature': jnp.zeros(768, dtype=jnp.float32),
            'position': jnp.zeros(2, dtype=jnp.float32),
            'quaternion': jnp.zeros(4, dtype=jnp.float32),
            'action': jnp.zeros(2, dtype=jnp.float32)
        }


    buffer_state = buffer.init(fake_timestep)

    ################################################
    ######## Load data from h5 files ###############
    ################################################

    file_path = "Navigation_Mujoco_dataset_full.h5"

 

    act1min = 1e9-1
    act1max = -1e9+1
    act2min = 1e9-1
    act2max = -1e9+1

    
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        for key in keys:
            grp = f[key]

            # latents size (2000X, 1, 768), the first element shoule be abandoned
            #images = grp["images"][1:]
            features = grp["features"][1:]
            actions = grp["actions"][1:]
            positions = grp["positions"][1:]
            quaternions = grp["quaternions"][1:]
            act1min = min(actions[:, 0].min(), act1min)
            act1max = max(actions[:, 0].max(), act1max)
            act2min = min(actions[:, 1].min(), act2min)
            act2max = max(actions[:, 1].max(), act2max)

            '''should check this carefully, ideal form is: [add_batch_size, [data_size]]'''
            #images_array = jnp.asarray(images, dtype=jnp.int8)[None, ...]
            feature_array = jnp.array(features).reshape(1,features.shape[0],768).astype(jnp.float32)
            action_array = jnp.array(actions).reshape(1,features.shape[0],2).astype(jnp.float32)
            position_array = jnp.asarray(positions, dtype=jnp.float32)[None, ...]
            quaternions_array = jnp.asarray(quaternions, dtype=jnp.float32)[None, ...]



            experiences = {
                #'image': images_array,
                'feature':feature_array,
                'action': action_array,
                'position': position_array,
                'quaternion': quaternions_array,
                } 
            buffer_state = buffer.add(buffer_state, experiences)

            gc.collect()











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
    def __init__(self,done_threshold=-0.2):
        self.sign_normal_obs = False
        self.sign_normal_reward = False
        self.obs_mean = 0 
        self.obs_std = 1
        self.scale_reward = 1
        
        self.done_threshold = done_threshold
        
        self.buffer, self.buffer_state, self.env_params = make_buffer() 
        self.key = jax.random.PRNGKey(3)

    def update_dataset():
        pass


    def sample1(self, batch_size: int, distance: int):
        '''sample a batch of data from a sequence'''
        sequence_length = 2000
        if batch_size >= sequence_length:
            raise ValueError("batch size should be smaller than sequence length!")

        keys = jax.random.split(self.key, 5)
        self.key = keys[0]
        batch = self.buffer.sample(self.buffer_state, keys[1], batch_size=1, sequence_length=sequence_length)
        feature = batch.experience['feature'] 
        action = batch.experience['action']
        position = batch.experience['position']
        #quaternion = batch.experience['quaternion']
        random_index = jax.random.randint(keys[2], shape=(batch_size,), minval=0, maxval=sequence_length-distance-1)
      

        ###
        goal_add_index = jax.random.randint(keys[3], shape=(batch_size,), minval=1, maxval=distance)
        goal_index = random_index + goal_add_index


        state = jnp.array([feature[1,random_index,:]])
        action = jnp.array([action[1,random_index,:]])
        position = jnp.array([position[1,random_index,:]])
        #quaternion = jnp.array([quaternion[1,random_index,:]])
        next_state = jnp.array([feature[1,random_index+1,:]])
        goal_state = jnp.array([feature[1,goal_index,:]])

        reward, done = self.calculate_reward(state, goal_state)

        return state, goal_state, action, next_state, reward, done, position,


    def sampleforeval(self, batch_size: int, distance: int):
        '''only for evaluation'''
        sequence_length = 1200
        if batch_size >= sequence_length:
            raise ValueError("batch size should be smaller than sequence length!")

        keys = jax.random.split(self.key, 5)
        self.key = keys[0]
        batch = self.buffer.sample(self.buffer_state, keys[1], batch_size=1, sequence_length=sequence_length)
        state= batch.experience['feature'] 
        position = batch.experience['position']
        quaternion = batch.experience['quaternion']
        
      

        return state, position, quaternion
    
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
        reward = jnp.tanh(0.5*(L2)) # normalize the reward to (-1, 1)
        done_reward = jnp.tanh(0.5*self.done_threshold)
        done = jnp.where(reward>done_reward, 1, 0)
        reward = jnp.where(reward>done_reward, 1, reward)
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

#test = sample_buffer()
##state, goal_state, action, next_state, reward, done = test.sample1(batch_size=256 ,sequence_length=512)
#print(state.shape, action.shape, next_state.shape, reward.shape, done.shape)