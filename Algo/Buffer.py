import h5py
import numpy as np
import flashbax as fbx
import jax.numpy as jnp
import jax
import time
import gc
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def make_flat_buffer():
    t1 = time.time()

    ################################################
    ########     prepare Buffer      ###############
    ################################################

    # First define hyper-parameters of the buffer.
    max_length = 30000 # Maximum length of the buffer along the time axis. 
    min_length = 8000 # Minimum length across the time axis before we can sample.
    sample_batch_size = 256
    add_sequences = True
    add_batch_size = 1 

    # Instantiate the trajectory buffer, which is a NamedTuple of pure functions.
    buffer = fbx.make_flat_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        add_sequences = add_sequences,
        add_batch_size=add_batch_size,
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

    file_path = "Navigation_Mujoco_dataset_full3.h5"

 

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


def plot_dataset_coverage(file_path = 'Navigation_Mujoco_dataset_full2.h5'):
    t1 = time.time()
   # file_path = 'Navigation_Mujoco_dataset_full3.h5'
    plt.figure(figsize=(8, 8))
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        for key in keys:
            grp = f[key]

            # latents size (2000X, 1, 768), the first element shoule be abandoned
            #images = grp["images"][1:]
            #features = grp["features"][1:]
            actions = grp["actions"][1:]
            positions = grp["positions"][1:]
            #quaternions = grp["quaternions"][1:]


            '''should check this carefully, ideal form is: [add_batch_size, [data_size]]'''
         

            position_array = jnp.asarray(positions, dtype=jnp.float32)[None, ...]
              
            plt.plot(positions[:, 0], positions[:, 1], label="Exploration Path", color='blue')


            gc.collect()
        # plot obstacles

    plt.show()

    pass


class sample_flat_buffer():
    def __init__(self,done_threshold=-0.3):
        
        self.sign_normal_obs = False
        self.sign_normal_reward = False
        self.obs_mean = 0 
        self.obs_std = 1
        self.scale_reward = 1
        
        self.done_threshold = -0.3
        self.zero_threshold = -0.5
        
        self.buffer, self.buffer_state, self.env_params = make_flat_buffer() 
        self.prepare_expert_evaluation()
        self.key = jax.random.PRNGKey(3)

    def update_dataset():
        pass
    
    def prepare_expert_evaluation(self):
        file_path = "Navigation_Mujoco_dataset_expert1.h5"
        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
            for key in keys:
                grp = f[key]
                n = 177
                actions = grp["actions"][1:n] 
           
                features = grp["features"][1:n]
                positions = grp["positions"][1:n]
                quaternions = grp["quaternions"][1:n]

                features = features.reshape(features.shape[0],768) 
                '''should check this carefully, ideal form is: [add_batch_size, [data_size]]'''
                goal = features[-1]
                goat_state = jnp.broadcast_to(goal, features[0:-1].shape)
              
                self.expert1= {
                    'state' : jnp.array(features[0:-1]).astype(jnp.float32),
                    'action' : jnp.array(actions[0:-1]).astype(jnp.float32),
                    'next_state' : jnp.array(features[1:]).astype(jnp.float32),
                    'positions':  jnp.array(positions[0:-1]).astype(jnp.float32),
                    'quaternions':  jnp.array(quaternions[0:-1]).astype(jnp.float32),
                    'goal_state': goat_state.astype(jnp.float32)
                            }
                
                experiences = {
                
                    'feature':jnp.array(features).reshape(1,features.shape[0],768).astype(jnp.float32),
                    'action': jnp.array(actions).reshape(1,features.shape[0],2).astype(jnp.float32),
                    'position': jnp.array(positions).reshape(1,features.shape[0],2).astype(jnp.float32),
                    'quaternion': jnp.array(quaternions).reshape(1,features.shape[0],4).astype(jnp.float32)
                    } 
                self.buffer_state = self.buffer.add(self.buffer_state, experiences)

                    


    def sample1(self):
        '''sample a batch of data from a flatbuffer'''

        keys = jax.random.split(self.key, 3)
        self.key = keys[0]
        batch = self.buffer.sample(self.buffer_state, keys[1])

       

        ###
   

        state = batch.experience.first['feature'] 
        action = batch.experience.first['action']
        position = batch.experience.first['position']
        #quaternion = jnp.array([quaternion[1,random_index,:]])
        next_state = batch.experience.second['feature'] 
        batch2 = self.buffer.sample(self.buffer_state, keys[2])
        goal_state = batch2.experience.first['feature'] 

        reward, done = self.calculate_reward(next_state, goal_state, action)

        return state, goal_state, action, next_state, reward, done, position,


    def sampleforeval(self):
        '''only for evaluation'''

        keys = jax.random.split(self.key, 2)
        self.key = keys[0]
        batch = self.buffer.sample(self.buffer_state, keys[1])
        state = batch.experience.first['feature'] 
        position = batch.experience.first['position']
        quaternion = batch.experience.first['quaternion']
        
        return state, position, quaternion
    
    
    def calculate_reward(self, next_state, goal_state, action):
        ''' calculate the reward and done signal
         the L2 range is about (-?, 0)
         the reward range is about (0, 1)
         if L2 < zero_threshold, reward1 = 0;
         if L2 > done_threshold, reward = 1;
         if L2
         plese set _self.done_threshold_ to control when the episode is done'''

        L2 = -jnp.square(next_state - goal_state).mean(axis=-1)

        reward1 = jnp.tanh(0.5*(L2))+1 # normalize the reward to (0, 1)
        reward2 = action[:, 0] * 0.2 # action[0] refers to distance for each timestep. reward2 encourage robot to keep moving! 0 < action[0] < 0.05
        done_reward = jnp.tanh(0.5*self.done_threshold)+1
        zero_reward = jnp.tanh(0.5*self.zero_threshold)+1
        done = jnp.where(reward1>=done_reward, 1, 0)
        reward1 = jnp.where(reward1<=zero_reward, 0, reward1)
        reward = reward1 + reward2
        reward = jnp.where(reward1>=done_reward, 1, reward)
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




#plot_dataset_coverage('Navigation_Mujoco_dataset_full2.h5')
#plot_dataset_coverage('Navigation_Mujoco_dataset_expert1.h5')
#buffer, buffer_state, env_params = make_flat_buffer()


#test = sample_flat_buffer()
#test.sampleforeval()
##state, goal_state, action, next_state, reward, done = test.sample1(batch_size=256 ,sequence_length=512)
#print(state.shape, action.shape, next_state.shape, reward.shape, done.shape)