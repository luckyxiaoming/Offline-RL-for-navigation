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
    max_length = 60000 # Maximum length of the buffer along the time axis. 
    min_length = 2000 # Minimum length across the time axis before we can sample.
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
            'feature': jnp.zeros(1024, dtype=jnp.float32),
            'position': jnp.zeros(2, dtype=jnp.float32),
            'quaternion': jnp.zeros(4, dtype=jnp.float32),
            'action': jnp.zeros(2, dtype=jnp.float32)
        }


    buffer_state = buffer.init(fake_timestep)

    ################################################
    ######## Load data from h5 files ###############
    ################################################

    file_path = "Navigation_Mujoco_dataset_full2.h5"

 

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
            feature_array = jnp.array(features).reshape(1,features.shape[0],1024).astype(jnp.float32)
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
      "action_space_high": jnp.array([act1max, act2max]),
      "action_dimension": 2,
      "observation_size": 1024,
      "action_discrete": False
  }
    print(f"The buffer is done! Cost time: {time.time() - t1} seconds")


    return buffer, buffer_state, env_params



def make_item_buffer(file_path):
    t1 = time.time()

    ################################################
    ########     prepare Buffer      ###############
    ################################################

    # First define hyper-parameters of the buffer.
    max_length = 36000 # Maximum length of the buffer along the time axis. 
    min_length = 2000 # Minimum length across the time axis before we can sample.
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
        #'images': images,
        'feature': jnp.zeros(1024, dtype=jnp.float32),
        "action": jnp.zeros(3, dtype=jnp.float32),
        #"next_images": next_images,
        'next_feature':jnp.zeros(1024, dtype=jnp.float32),
        'position': jnp.zeros(2, dtype=jnp.float32),
        'quaternion': jnp.zeros(4, dtype=jnp.float32),
        "done": jnp.zeros(1, dtype=jnp.bool)
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

            features = grp["features"][:]
            actions = grp["actions"][:]
            actions = jnp.where(actions>0.85, 0.85, actions)
            actions = jnp.where(actions<-0.85, -0.85, actions)
            positions = grp["positions"][:]
            quaternions = grp["quaterions"][:]
            next_features = grp["next_features"][:]
            dones = grp["dones"][:]

            act1min = min(actions[:, 0].min(), act1min)
            act1max = max(actions[:, 0].max(), act1max)
            act2min = min(actions[:, 1].min(), act2min)
            act2max = max(actions[:, 1].max(), act2max)
            act3min = min(actions[:, 2].min(), act3min)
            act3max = max(actions[:, 2].max(), act3max)

            '''should check this carefully, ideal form is: [add_batch_size, [data_size]]'''
            #images_array = jnp.asarray(images, dtype=jnp.int8)[None, ...]
            feature_array = jnp.array(features).reshape(features.shape[0],1024).astype(jnp.float32)
            next_feature_array = jnp.array(next_features).reshape(next_features.shape[0],1024).astype(jnp.float32)
            action_array = jnp.array(actions).reshape(features.shape[0],3).astype(jnp.float32)
            position_array = jnp.asarray(positions, dtype=jnp.float32)[...]
            quaternions_array = jnp.asarray(quaternions, dtype=jnp.float32)[...]
            dones_array = jnp.asarray(dones, dtype=jnp.bool)[...]



            experiences = {
                #'image': images_array,
                'feature':feature_array,
                'next_feature': next_feature_array,
                'action': action_array,
                'position': position_array,
                'quaternion': quaternions_array,
                'done': dones_array,
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



def plot_dataset_coverage(file_path = 'Navigation_Mujoco_dataset_full2.h5'):
    t1 = time.time()
   # file_path = 'Navigation_Mujoco_dataset_full3.h5'
    plt.figure(figsize=(8, 8))
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        for key in keys[:]:
            grp = f[key]

            # latents size (2000X, 1, 768), the first element shoule be abandoned
            #images = grp["images"][1:]
            #features = grp["features"][1:]
            actions = grp["actions"][1:]
            positions = grp["positions"][1:]
            dones = grp["dones"][1:]
            #quaternions = grp["quaternions"][1:]


            '''should check this carefully, ideal form is: [add_batch_size, [data_size]]'''
         
            
            position_array = jnp.asarray(positions, dtype=jnp.float32)[None, ...]
            dert_pos = np.abs(position_array[0,1:,:]- position_array[0,:-1, :]).sum(axis=-1)
            index = np.where(dert_pos>0.5)[0]
            if np.size(index) >0:
                start_idx = 0  

                for i in range(len(index)):
                    end_idx = index[i] + 1 
                    plt.plot(positions[start_idx:end_idx, 0], positions[start_idx:end_idx, 1], label=f"Segment {i+1}", color='blue')
                    start_idx = end_idx 

                plt.plot(positions[start_idx:, 0], positions[start_idx:, 1], label=f"Segment {len(index)+1}", color='blue')
            else:
                plt.plot(positions[:, 0], positions[:, 1], label="Exploration Path", color='blue')


            gc.collect()
        # plot obstacles

    plt.show()

    pass


class sample_flat_buffer():
    def __init__(self,done_threshold=0.7):
        
        self.sign_normal_obs = False
        self.sign_normal_reward = False
        self.obs_mean = 0 
        self.obs_std = 1
        self.scale_reward = 1
        
        self.done_threshold = 0.7
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
                n = 83
                actions = grp["actions"][1:n] 
           
                features = grp["features"][1:n]
                positions = grp["positions"][1:n]
                quaternions = grp["quaternions"][1:n]

                features = features.reshape(features.shape[0],1024) 
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
                
                    'feature':jnp.array(features).reshape(1,features.shape[0],1024).astype(jnp.float32),
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
        #batch2 = self.buffer.sample(self.buffer_state, keys[2])
        goal_state = self.check_goal_state(state)

        reward, done, log_reward = self.calculate_reward(state, next_state, goal_state, action)

        return state, goal_state, action, next_state, reward, done, log_reward


    def sampleforeval(self):
        '''only for evaluation'''

        keys = jax.random.split(self.key, 3)
        self.key = keys[0]
        batch = self.buffer.sample(self.buffer_state, keys[1])
        state = batch.experience.first['feature'] 
        position = batch.experience.first['position']
        quaternion = batch.experience.first['quaternion']
        
        return state, position, quaternion
    
    
    def check_goal_state(self,  state):
        keys = jax.random.split(self.key, 2)
        self.key = keys[0]
        batch1 = self.buffer.sample(self.buffer_state, keys[1])
        goal_state = batch1.experience.first['feature'] 

        

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



    def calculate_reward(self, state, next_state, goal_state, action):
        ''' calculate the reward and done signal
         the L2 range is about (-?, 0)
         the reward range is about (0, 1)
         if L2 < zero_threshold, reward1 = 0;
         if L2 > done_threshold, reward = 1;
         if L2
         plese set _self.done_threshold_ to control when the episode is done'''
        

        current_cos = self.calculate_cosine_similarity(state, goal_state)
        next_cos = self.calculate_cosine_similarity(next_state, goal_state)

        reward1 = jnp.exp((current_cos-1)*30)
        reward2 = jnp.exp((next_cos-1)*30)

       
        reward = reward2-reward1

        done_threshold = self.done_threshold


        current_done = jnp.where(reward1>=done_threshold, 1, 0)

  

        done = jnp.where(reward2>=done_threshold, 1, 0)

        reward = jnp.where(reward2>=done_threshold, 1, reward)
        mean_reward = reward.mean()
        if current_done.sum() >= 1 and not len(current_done) == 1:
            print(f'current state batch conclude current_done = {current_done.sum()}')
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


class sample_item_buffer():
    def __init__(self, file_path, done_threshold=0.7):
        
        self.sign_normal_obs = False
        self.sign_normal_reward = False
        self.obs_mean = 0 
        self.obs_std = 1
        self.scale_reward = 1
        
        self.done_threshold = 0.7
        self.zero_threshold = -0.5
        
        self.buffer, self.buffer_state, self.env_params = make_item_buffer(file_path) 
        #self.prepare_expert_evaluation()
        self.key = jax.random.PRNGKey(3)

    def update_dataset():
        pass
    
    def prepare_expert_evaluation(self):
        file_path = "Navigation_Mujoco_dataset_expert_S1.h5"
        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
            for key in keys:
                grp = f[key]
                actions = grp["actions"][:] 
           
                features = grp["features"][:]
                next_features = grp["next_features"][:]
                positions = grp["positions"][:]
                terminals =  grp["dones"][:]
  
                features = features.reshape(features.shape[0],1024) 
                '''should check this carefully, ideal form is: [add_batch_size, [data_size]]'''
                goal = features[-1]
                goal_states = jnp.broadcast_to(goal, features.shape)

        return features, next_features, goal_states, actions, terminals
              

                

                    


    def sample1(self):
        '''sample a batch of data from a flatbuffer'''

        keys = jax.random.split(self.key, 4)
        self.key = keys[0]
        batch = self.buffer.sample(self.buffer_state, keys[1])

       

        ###
   

        state = batch.experience['feature'] 
        action = batch.experience['action']
        terminal = batch.experience['done']
        next_state = batch.experience['next_feature'] 
        batch2 = self.buffer.sample(self.buffer_state, keys[2])
        goal_state = batch2.experience['feature'] 

        reward, done, log_reward = self.calculate_reward(state, next_state, goal_state, terminal)

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



    def calculate_reward(self, state, next_state, goal_state, ternimal):
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
        reward3 = jnp.where(ternimal==1, -1, 0).reshape(reward1.shape)

       
        reward = (reward2-reward1)*0.2 + reward3

        done_threshold = self.done_threshold


 

  

        done = jnp.where(reward2>=done_threshold, 1, 0)
        ternimal = jnp.array(ternimal).squeeze()
        ternimal = ternimal.astype(jnp.int8)
        done = done | ternimal

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




#plot_dataset_coverage('Navigation_Mujoco_dataset_S1.h5')
#plot_dataset_coverage('Navigation_Mujoco_dataset_expert_S1.h5')

#buffer, buffer_state, env_params = make_item_buffer('Navigation_Mujoco_dataset_S1.h5')




#test = sample_item_buffer(file_path='Navigation_Mujoco_dataset_S1.h5')
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