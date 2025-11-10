import h5py
import numpy as np
import flashbax as fbx
import jax.numpy as jnp
import jax
import time
import gc
import matplotlib.pyplot as plt
import matplotlib.patches as patches




def make_item_buffer(file_path):
    t1 = time.time()

    ################################################
    ########     prepare Buffer      ###############
    ################################################

    # First define hyper-parameters of the buffer.
    max_length = 15000 # Maximum length of the buffer along the time axis. 
    min_length = 6000 # Minimum length across the time axis before we can sample.
    sample_batch_size = 256
    add_batches = True

    # Instantiate the trajectory buffer, which is a NamedTuple of pure functions.
    buffer = fbx.make_item_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        add_batches=add_batches,
        )

    fake_timestep = {
        'feature': jnp.zeros(1024, dtype=jnp.float32),
        "action": jnp.zeros(3, dtype=jnp.float32),
        'next_feature':jnp.zeros(1024, dtype=jnp.float32),
        } 


    buffer_state = buffer.init(fake_timestep)

    ################################################
    ######## Load data from h5 files ###############
    ################################################


    act1min = 1e9-1
    act1max = -1e9+1
    act2min = 1e9-1
    act2max = -1e9+1
    act3min = 1e9-1
    act3max = -1e9+1

    
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        for key in keys:
            grp = f[key]
 
            features = grp["features"][:][0:-1, :]
            actions = grp["actions"][:][0:-1, :]
            next_features = grp["features"][:][1:, :]

            act1min = min(actions[:, 0].min(), act1min)
            act1max = max(actions[:, 0].max(), act1max)
            act2min = min(actions[:, 1].min(), act2min)
            act2max = max(actions[:, 1].max(), act2max)
            act3min = min(actions[:, 2].min(), act3min)
            act3max = max(actions[:, 2].max(), act3max)

            '''should check this carefully, ideal form is: [add_batch_size, [data_size]]'''
        
            feature_array = jnp.array(features).reshape(features.shape[0],1024).astype(jnp.float32)
            next_feature_array = jnp.array(next_features).reshape(next_features.shape[0],1024).astype(jnp.float32)
            action_array = jnp.array(actions).reshape(features.shape[0],3).astype(jnp.float32)
          


            experiences = {
                'feature':feature_array,
                'next_feature': next_feature_array,
                'action': action_array,

                } 
            buffer_state = buffer.add(buffer_state, experiences)

            gc.collect()











    env_params = {
      "action_space_low": jnp.array([act1min, act2min, act3min]),
      "action_space_high": jnp.array([act1max, act2max, act3max]),
      "action_dimension": 3,
      "observation_size": 1024,
      "action_discrete": False
  }
    print(f"The buffer is done! Cost time: {time.time() - t1} seconds")


    return buffer, buffer_state, env_params


class sample_item_buffer():
    def __init__(self, file_path, expert_file_path, done_threshold=0.7):
        
        self.sign_normal_obs = False
        self.sign_normal_reward = False
        self.obs_mean = 0 
        self.obs_std = 1
        self.scale_reward = 1
        
        self.done_threshold = done_threshold

        
        self.buffer, self.buffer_state, self.env_params = make_item_buffer(file_path) 

        self.prepare_expert_dataset(expert_file_path)
        #self.prepare_expert_evaluation()
        self.key = jax.random.PRNGKey(3)
    

    def prepare_expert_dataset(self, expert_file_path):
        # create a dictionary with 20 keys, each key contains expert data from one episode
        EXPERT = {}


        with h5py.File(expert_file_path, "r") as f:
            keys = list(f.keys())
            epsoide_count = 0
            for key in keys:
                grp = f[key]
                features = grp["features"][:][0:-1, :]
                actions = grp["actions"][:][0:-1, :]
                next_features = grp["features"][:][1:, :]

                feature_array = jnp.array(features).reshape(features.shape[0],1024).astype(jnp.float32)
                next_feature_array = jnp.array(next_features).reshape(next_features.shape[0],1024).astype(jnp.float32)
                action_array = jnp.array(actions).reshape(features.shape[0],3).astype(jnp.float32)
                goal = next_feature_array[-1]
                goal_feature = jnp.broadcast_to(goal, feature_array.shape)

                experiences = {
                    'feature':feature_array,
                    'next_feature': next_feature_array,
                    'action': action_array,
                    'goal_feature': goal_feature
                    } 
                EXPERT[epsoide_count] = experiences
                epsoide_count += 1


                gc.collect()
        self.EXPERT = EXPERT


    def update_dataset(self, seeds, num, expert_file_path):
        keys = jax.random.PRNGKey(seeds)


        
        fake_timestep = {
            'feature': jnp.zeros(1024, dtype=jnp.float32),
            "action": jnp.zeros(3, dtype=jnp.float32),
            'next_feature':jnp.zeros(1024, dtype=jnp.float32),
            } 
        

        pass



    def sample1(self):
        '''sample a batch of data from a flatbuffer'''

        keys = jax.random.split(self.key, 4)
        self.key = keys[0]
        batch = self.buffer.sample(self.buffer_state, keys[1])

       

        ###
   

        state = batch.experience['feature'] 
        action = batch.experience['action']

        next_state = batch.experience['next_feature'] 
        batch2 = self.buffer.sample(self.buffer_state, keys[2])
        goal_state = batch2.experience['feature'] 

        reward, done, log_reward = self.calculate_reward(state, next_state, goal_state)

        return state, goal_state, action, next_state, reward, done, log_reward


    def sampleforeval(self):
        '''only for evaluation'''
        key = jax.random.PRNGKey(3)
        batch = self.buffer.sample(self.buffer_state, key)
        state = batch.experience['feature'] 
        position = batch.experience['position']
        quaternion = batch.experience['quaternion']
        
        return state, position, quaternion
    
    
    def check_goal_state(self,  state):
        keys = jax.random.split(self.key, 2)
        self.key = keys[0]
        batch1 = self.buffer.sample(self.buffer_state, keys[1])
        goal_state = batch1.experience['feature'] 
        

        

        done_threshold = self.done_threshold
        current_done = 1
        t = 0
        while jnp.sum(current_done)>0:
            keys = jax.random.split(self.key, 2)
            self.key = keys[0]
            batch2 = self.buffer.sample(self.buffer_state, keys[1])
            goal_state2 = batch2.experience.first['feature'] 
            current_cos = self.calculate_cosine_similarity(state, goal_state)
            reward1 = jnp.exp((current_cos-1)*30)

            reward2 = jnp.repeat(reward1.reshape([256,1]), 1024, axis=1)
            goal_state = jnp.where(reward2>=done_threshold, goal_state2, goal_state)

            current_cos = self.calculate_cosine_similarity(state, goal_state)
            reward1 = jnp.exp((current_cos-1)*30)
            current_done = jnp.where(reward1>=done_threshold, 1, 0)
            t += 1
            if t > 5:
                print('error!!!!!')
                break

   
     
        if current_done.sum()>0:
            ttt= jnp.where(reward1>=done_threshold)
         
        return goal_state



    def calculate_reward(self, state, next_state, goal_state):
        ''' calculate the reward and done signal
         the L2 range is about (-?, 0)
         the reward range is about (0, 1)
         if L2 < zero_threshold, reward1 = 0;
         if L2 > done_threshold, reward = 1;
         if L2
         plese set _self.done_threshold_ to control when the episode is done'''
        

        current_cos = self.calculate_cosine_similarity(state, goal_state)
        next_cos = self.calculate_cosine_similarity(next_state, goal_state)

        reward1 = jnp.exp((current_cos-1)*5)
        reward2 = jnp.exp((next_cos-1)*5)
 

       
        reward = (reward2-reward1)*0.2 

        done_threshold = self.done_threshold


 

  

        done = jnp.where(reward2>=done_threshold, 1, 0)


        reward = jnp.where(reward2>=done_threshold, 1, reward)
        mean_reward = reward.mean()
    #    if done.sum() >= 1 and not len(done) == 1:
    #        print(f'current batch conclude Done = {done.sum()}')
        return reward, done, mean_reward

    def calculate_cosine_similarity(self, X, Y):
        dot = jnp.sum(X*Y, axis=-1)
        L2 = jnp.linalg.norm(X, axis=-1) * jnp.linalg.norm(Y, axis=-1) + 1e-8
        return dot / L2



       
    
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




#plot_dataset_coverage('Navigation_Mujoco_dataset_OU_S3.h5')
#plot_dataset_coverage('Navigation_Mujoco_dataset_expert_S1.h5')

#buffer, buffer_state, env_params = make_item_buffer(file_path='/home/xiaoming/Research/Coding/offline_RL/realworlddata/S11_pink_raw_01.h5')




#test = sample_item_buffer(file_path='/home/xiaoming/Research/Coding/offline_RL/realworlddata/S11_4Hz_training_01.h5',
#                        expert_file_path='/home/xiaoming/Research/Coding/offline_RL/realworlddata/S11_4Hz_expert_01.h5')


#obs,next_obs,goal_obs, actions, terminals= test.prepare_expert_evaluation()
#reward, done, _ = test.calculate_reward(obs,next_obs,goal_obs, terminals)
#N = np.where(done==1)[0][0]
#new_reward = reward[:N+1]
#print(new_reward)
#gamma_array = jnp.array([0.99**i for i in range(N+1)])
#Return = (new_reward * gamma_array).sum()
#print(Return)

#test.sample1()
##state, goal_state, action, next_state, reward, done = test.sample1(batch_size=256 ,sequence_length=512)
#print(state.shape, action.shape, next_state.shape, reward.shape, done.shape)