import h5py
import numpy as np
import flashbax as fbx
import jax.numpy as jnp
import jax
import time
import gc
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from Args import Args 

@jax.jit
def calculate_cosine_similarity(X, Y):
    dot = jnp.sum(X*Y, axis=-1)
    norm_x = jnp.linalg.norm(X, axis=-1)
    norm_y = jnp.linalg.norm(Y, axis=-1)
  
    L2 = norm_x * norm_y + 1e-8
    return dot / L2

@jax.jit
def calculate_reward_batch(current_sim, next_sim, done_threshold, scale=1.0, bonus=1.0):
    #step_reward = scale * (next_sim - current_sim)
    #is_current_done = current_sim >= done_threshold
    is_next_done = next_sim >= done_threshold
    is_success = is_next_done
    #reward = step_reward
    
    # here we try sparse reward
    bonus = jnp.where(is_success, 1, 0)
    reward = bonus
    
    return reward, is_success


class DynamicHERBuffer:
    def __init__(self, args: Args):
        """
        Initialization uses the Args dataclass.
        """
        self.args = args
        self.batch_size = args.batch_size
        self.done_threshold = args.done_threshold
        self.key = jax.random.PRNGKey(args.seed)
        self.gamma = args.gamma
        

        # Temporary storage
        self.features = []
        self.next_features = []
        self.actions = []
        self.episode_info = [] 
        self.S0 = []
        self.Send = []


        print("loading (Dynamic HER)...")
        
        current_idx = 0
        # 1. Load Main Training Dataset
        if hasattr(args, 'training_dataset_filepath') and args.training_dataset_filepath:
            print(f"Loading Training File: {args.training_dataset_filepath}")
            steps_1 = self._load_file(args.training_dataset_filepath, current_idx)
            current_idx += steps_1
        else:
            print("Warning: No training dataset filepath provided in args.")

        # 2. Load Additional Expert Training Data (if exists)
        if hasattr(args, 'add_expert_data_filepath') and args.add_expert_data_filepath:
            print(f"Loading Additional Expert Training File: {args.add_expert_data_filepath}")
            steps_2 = self._load_file(args.add_expert_data_filepath, current_idx)
            current_idx += steps_2

        # Transform to JAX Array
        if len(self.features) > 0:
            self.features = jnp.array(np.concatenate(self.features), dtype=jnp.float32)
            self.next_features = jnp.array(np.concatenate(self.next_features), dtype=jnp.float32)
            self.actions = jnp.array(np.concatenate(self.actions), dtype=jnp.float32)
        else:
            raise ValueError("No data loaded! Check file paths.")

        # Process episode info
        episode_starts = [info[0] for info in self.episode_info]
        episode_lens = [info[1] for info in self.episode_info]
        
        self.episode_starts = jnp.array(episode_starts, dtype=jnp.int32)
        self.episode_lens = jnp.array(episode_lens, dtype=jnp.int32)
        self.episode_indices = jnp.arange(len(self.episode_info), dtype=jnp.int32)
        self.total_steps = len(self.features)

        print(f"Finish dataset load! Steps: {self.total_steps}, Trajectories: {len(self.episode_info)}")
        
        self.env_params = {
            'action_space_low': self.actions.min(axis=0),
            'action_space_high': self.actions.max(axis=0),
            'action_dimension': self.actions.shape[-1],
            'observation_size': self.features.shape[-1]
        }
        
        # Load Expert Dataset for Evaluation (if needed)
        self.prepare_expert_dataset(args.expert_file_path)
        self.prepare_goal_data(args.target_file_path)
        gc.collect()

    def _load_file(self, filepath, start_global_idx):
        steps_added = 0
        try:
            with h5py.File(filepath, "r") as f:
                keys = list(f.keys()) 
                for key in keys:
                    grp = f[key]

                    raw_feats = grp["features"][:]
                    # Simple Normalization
                    norms = np.linalg.norm(raw_feats, axis=-1, keepdims=True)
                    raw_feats = raw_feats / (norms + 1e-8)

                    acts = grp["actions"][:]
                    
                    length = min(len(raw_feats)-1, len(acts))
                    if length < 5: continue
                    
                    self.features.append(raw_feats[:length])
                    self.next_features.append(raw_feats[1:length+1])
                    self.actions.append(acts[:length])

                    self.S0.append(raw_feats[0])
                    self.Send.append(raw_feats[length-1])

                    self.episode_info.append((start_global_idx, length))
                    
                    start_global_idx += length
                    steps_added += length
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            
        return steps_added 
    
    def _sample_basic_transitions(self, key, n_samples):

        ep_idxs = jax.random.choice(key, self.episode_indices, shape=(n_samples,))
        starts = self.episode_starts[ep_idxs]
        lens = self.episode_lens[ep_idxs]
        t_offsets = jax.random.randint(key, (n_samples,), 0, lens)
        cur_idxs = starts + t_offsets
        
        s = self.features[cur_idxs]
        a = self.actions[cur_idxs]
        ns = self.next_features[cur_idxs]
        return s, a, ns, starts, t_offsets, lens, cur_idxs

    def _get_geometric_offsets(self, key, lens, t_offsets, p=0.05):
        raw_offsets = jax.random.geometric(key, p, shape=lens.shape)
        remaining_steps = lens - t_offsets - 1
        valid_remaining = jnp.maximum(remaining_steps, 1)
        return jnp.minimum(raw_offsets, valid_remaining)

    def _sample_batch_internal(self, ratios):
        r_cur, r_geom, r_uni, r_rand = ratios
        
        n_cur = int(self.batch_size * r_cur)
        n_geom = int(self.batch_size * r_geom)
        n_uni = int(self.batch_size * r_uni)
        # Ensure sum matches batch_size perfectly
        n_rand = self.batch_size - n_cur - n_geom - n_uni 
        
        keys = jax.random.split(self.key, 15)
        self.key = keys[0]
        
        s_list, g_list, a_list, ns_list, m_list = [], [], [], [], []

        # 1. Current State (p_cur): select current state as goal
        if n_cur > 0:
            s, a, ns, _, _, _, _ = self._sample_basic_transitions(keys[1], n_cur)
            g = s 
            m = jnp.ones(n_cur)
            s_list.append(s); g_list.append(g); a_list.append(a); ns_list.append(ns); m_list.append(m)

        # 2. Geometric Future (p_geom) goal from geometric distribution, 
        # if gamma is closer to 1, sample farther future
        # if gamma is closer to 0, sample nearer future
        if n_geom > 0:
            s, a, ns, starts, t, lens, _ = self._sample_basic_transitions(keys[2], n_geom)
            offsets = self._get_geometric_offsets(keys[3], lens, t, p=(1-self.gamma))
            g_idxs = starts + t + offsets
            g = self.features[g_idxs]
            m = jnp.ones(n_geom)
            s_list.append(s); g_list.append(g); a_list.append(a); ns_list.append(ns); m_list.append(m)

        # 3. Uniform Future (p_traj): 
        if n_uni > 0:
            s, a, ns, starts, t, lens, _ = self._sample_basic_transitions(keys[4], n_uni)
            rnd_floats = jax.random.uniform(keys[5], (n_uni,))
            offsets = ((lens - t - 1) * rnd_floats).astype(jnp.int32) + 1
            g_idxs = starts + t + offsets
            g = self.features[g_idxs]
            m = jnp.ones(n_uni)
            s_list.append(s); g_list.append(g); a_list.append(a); ns_list.append(ns); m_list.append(m)

        # 4. Random Goal (p_rand)
        if n_rand > 0:
            s, a, ns, _, _, _, _ = self._sample_basic_transitions(keys[6], n_rand)
            rnd_goal_idxs = jax.random.randint(keys[7], (n_rand,), 0, self.total_steps)
            g = self.features[rnd_goal_idxs]
            m = jnp.zeros(n_rand)
            s_list.append(s); g_list.append(g); a_list.append(a); ns_list.append(ns); m_list.append(m)

        # Concatenate
        state = jnp.concatenate(s_list)
        action = jnp.concatenate(a_list)
        next_state = jnp.concatenate(ns_list)
        goal_state = jnp.concatenate(g_list)
        mask = jnp.concatenate(m_list)
        
        cur_sim = calculate_cosine_similarity(state, goal_state)
        next_sim = calculate_cosine_similarity(next_state, goal_state)
        

        scale = self.args.reward_scale
        bonus = self.args.bonus
        
        reward, done = calculate_reward_batch(cur_sim, next_sim, self.done_threshold, scale=scale, bonus=bonus)
        done = done.astype(jnp.float32)

        return state, goal_state, action, next_state, reward, done, mask

    def sample_value_batch(self):
        # Uses args.value_sample_ratios (e.g., 0.2, 0.5, 0.1, 0.2)
        return self._sample_batch_internal(self.args.value_sample_ratios)
    

    def sample_policy_batch(self):
        # Uses args.policy_sample_ratios (e.g., 0.0, 0.0, 1.0, 0.0)
        return self._sample_batch_internal(self.args.policy_sample_ratios)
    
    def prepare_goal_data(self, target_file_path):
        GOALS = {}
        episode_count = 0
        POS = {}
        try:
            with h5py.File(target_file_path, "r") as f:
                for key in f.keys():
                    re_key = f'Target_{episode_count}'
                    grp = f[re_key]
                    
                    raw_feats = grp["features"][:]
                    norms = np.linalg.norm(raw_feats, axis=-1, keepdims=True)
                    norms_feats = raw_feats / (norms + 1e-8)
                                            
                    if "positions" in grp:
                        position = grp["positions"][:]
                    if "quaterions" in grp: 
                        quaternion = grp["quaterions"][:]
                    elif "quaternions" in grp:
                        quaternion = grp["quaternions"][:]
                    POS[episode_count] = position
                    
                    # JAX Conversion
                    goal = {
                        'goal_feature': jnp.array(norms_feats[0], dtype=jnp.float32),
                        'goal_position': jnp.array(position[0], dtype=jnp.float32),
                        'goal_quaternion': jnp.array(quaternion[0], dtype=jnp.float32),
                    }

                    GOALS[episode_count] = goal
                    episode_count += 1
        except Exception as e:
            print(f"Error loading expert file {file_path}: {e}")

        pos_array = np.array(list(POS.values()))
        for T in POS:
            cur_pos = pos_array[T]
            cur_dist = jnp.linalg.norm(cur_pos - pos_array, axis=-1)
            sorted_indices = int(np.argmax(cur_dist))
            GOALS[T]['initial_position'] = GOALS[sorted_indices]['goal_position']
            GOALS[T]['initial_quaternion'] = GOALS[sorted_indices]['goal_quaternion']
            GOALS[T]['initial_distance'] = jnp.linalg.norm(GOALS[T]['initial_position'] - GOALS[T]['goal_position'])  
            print(f"Goal {T}: Initial distance to goal is {GOALS[T]['initial_distance']:.4f}")          

        print(f"GOAL DATA: {episode_count} goal have been loaded.")

        self.GOALS = GOALS


    def prepare_expert_dataset(self, expert_file_path):
        # create a dictionary with 20 keys, each key contains expert data from one episode
        EXPERT = {}
        act1min = 1e9-1
        act1max = -1e9+1
        act2min = 1e9-1
        act2max = -1e9+1
        act3min = 1e9-1
        act3max = -1e9+1


        with h5py.File(expert_file_path, "r") as f:
            keys = list(f.keys())
            epsoide_count = 0
            for key in keys:
                grp = f[key]

                raw_feats = grp["features"][:]
                norms = np.linalg.norm(raw_feats, axis=-1, keepdims=True)
                raw_feats = raw_feats / (norms + 1e-8)


                features = raw_feats[0:-1, :]
                actions = grp["actions"][:][0:-1, :]
                next_features = raw_feats[1:, :]
                positions = grp["positions"][:][0:-1, :]
                quaternions = grp["quaterions"][:][0:-1, :]
                act1min = min(actions[:, 0].min(), act1min)
                act1max = max(actions[:, 0].max(), act1max)
                act2min = min(actions[:, 1].min(), act2min)
                act2max = max(actions[:, 1].max(), act2max)
                act3min = min(actions[:, 2].min(), act3min)
                act3max = max(actions[:, 2].max(), act3max)
                

                feature_array = jnp.array(features).reshape(features.shape[0],1024).astype(jnp.float32)
                next_feature_array = jnp.array(next_features).reshape(next_features.shape[0],1024).astype(jnp.float32)
                
                action_array = jnp.array(actions).reshape(features.shape[0],3).astype(jnp.float32)
                goal = next_feature_array[-1]
                goal_feature = jnp.broadcast_to(goal, feature_array.shape)

                experiences = {
                    'feature':feature_array,
                    'next_feature': next_feature_array,
                    'action': action_array,
                    'goal_feature': goal_feature,
                    'position': positions,
                    'quaternion': quaternions,

                    } 
                EXPERT[epsoide_count] = experiences
                epsoide_count += 1


                gc.collect()
            print(f"In expert dataset, action min:{act1min},{act2min},{act3min}, action max: {act1max},{act2max},{act3max},")
        self.EXPERT = EXPERT
    


    @staticmethod
    @jax.jit
    def calculate_cosine_similarity(X, Y):
        dot = jnp.sum(X*Y, axis=-1)
        norm_x = jnp.linalg.norm(X, axis=-1)
        norm_y = jnp.linalg.norm(Y, axis=-1)
    
        L2 = norm_x * norm_y + 1e-8
        return dot / L2
