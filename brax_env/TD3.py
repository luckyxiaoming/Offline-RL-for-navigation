
import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import optax
from jaxtyping import PRNGKeyArray, Array
import random

import tyro
from dataclasses import dataclass
import os
import time
import os
import pickle
import brax
import jax
import numpy as np
from brax import envs
from brax.io import model
from brax.io import html
import time

from IPython.display import HTML, clear_output
from brax.training.acme import types 
from flax import struct
from typing import Any, Optional, Tuple
import jax.numpy as jnp

from Buffer import sample_buffer
from evaluation import evaluation



@dataclass
class Args:
    seed: int = 1

    # dataset
    file_name = 'expert_batch256'  # 'all_expert';

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    
    # wandb
    '''if _True_, this experiment will be tracked on Wandb'''
    track: bool = True
    wandb_project_name: str = 'OfflineRL_Navigation_Mujoco'
    wandb_entiti: str = None

    env_id: str = 'TD3_Navigation_Mujoco' 

    # Algorithm specific arguments
    total_offline_steps: int = 300_000
    '''the discount factor gamma'''
    gamma: float = 0.99
    '''the target network update rate'''
    tau: float = 0.995
    '''the batch size of samples from the replay memory''' 
    batch_size: int = 256
    '''the frequency of training ploicy(actor)'''
    policy_frequency: int = 2
    '''learning rate of the optimizer'''
    learning_rate: float = 3e-4
    '''the scale of policy_noise'''  
    policy_noise: float = 0.2
    '''the noise of policy clip'''
    noise_clip: float = 0.5
    '''Behavior Clone factor'''
    alpha = 0.01
    '''Length of sample sequence'''
    distance = 5
 

class QNetwork(eqx.Module):
    """MLP"""
    action_dim: int
    trunk: nn.Sequential

    def __init__(self, key: PRNGKeyArray, obs_size: int, action_dim: int):

        keys = jax.random.split(key, 4)
        self.action_dim = action_dim

        self.trunk = nn.Sequential([
            nn.Linear(in_features=obs_size * 2 + action_dim, out_features=512, key=keys[1]),
            nn.LayerNorm(shape=512),
            nn.Lambda(jax.nn.leaky_relu),
            nn.Linear(in_features=512, out_features=512, key=keys[2]),
            nn.LayerNorm(shape=512),
            nn.Lambda(jax.nn.leaky_relu),
            nn.Linear(in_features=512, out_features= 1, key=keys[3])
        ])



    def __call__(self, state: Array, action: Array):
        # need to check for specific cas
        state = state.astype(jnp.float32)
        action = action.astype(jnp.float32)
        combined = jnp.concatenate([state, action], axis=1 )
        q_value = eqx.filter_vmap(self.trunk)(combined)

        return q_value.squeeze(-1)


class Actor(eqx.Module):
    '''final activation function is _tanh_, output range: [-1, 1]'''
    action_dim: int 
    action_scale: Any = eqx.static_field()
    action_bias: Any = eqx.static_field()
    trunk: nn.Sequential

    def __init__(self, key: PRNGKeyArray, obs_size: int, action_dim: int, action_scale, action_bias):
        keys = jax.random.split(key, 5)
        self.action_bias = action_bias.tolist()
        self.action_scale = action_scale.tolist()
        self.action_dim = action_dim

        last_layer = nn.Linear(in_features=512, out_features=action_dim, key=keys[3])
        last_layer = eqx.tree_at(
            where=lambda l: l.weight,
            pytree=last_layer,
            replace_fn=lambda w: jnp.zeros_like(w)
        )
        last_layer = eqx.tree_at(
            where=lambda l: l.bias,
            pytree=last_layer,
            replace_fn=lambda b: jnp.zeros_like(b)
        )

        self.trunk = nn.Sequential([
            nn.Linear(in_features=obs_size*2, out_features=512, key=keys[1]),
            nn.LayerNorm(shape=512),
            nn.Lambda(jax.nn.leaky_relu),
            nn.Linear(in_features=512, out_features=512, key=keys[2]),
            nn.LayerNorm(shape=512),
            nn.Lambda(jax.nn.leaky_relu),
            last_layer,
            nn.Lambda(jax.nn.tanh),
            ])
        
        

    def __call__(self, state: Array):
        action_scale = jnp.array(self.action_scale, dtype=jnp.float32)
        action_bias  = jnp.array(self.action_bias, dtype=jnp.float32)
        state = state.astype(jnp.float32)
        x = eqx.filter_vmap(self.trunk)(state)
        action = action_scale * x + action_bias

        return action
        


class TD3(object):
    def __init__(self, args, key):

        self.args = args
        self.total_it = 0
        self.policy_frequency = args.policy_frequency
        obs_size = args.env_params['observation_size']
        action_dim = args.env_params['action_dimension']
        action_discrete = args.env_params['action_discrete']
        action_high = args.env_params['action_space_high']
        action_low = args.env_params['action_space_low']
        self.loss_actor = []
        
        # this is modified for _tanh_ 
        action_scale = (action_high - action_low)/2
        action_bias =  (action_high + action_low)/2

        self.wandb_log_A = {

            "BC_value_pi_action":0,
            "grad_norm":0,
            'lambda*Q.mean': 0,
            'Q1_obs_pi':0,
            'lmbda':0,
            'action(s)[0]': 0,
            'action(s)[1]': 0,
            'bc':0,
            }
  


        keys = jax.random.split(key, 7)
        self.key = keys[6]
        
        # TRY NOT TO MODIFY
        # 1. initialize all networks
        self.actor = Actor(action_dim=action_dim, 
                           action_scale=action_scale, 
                           action_bias=action_bias,
                           obs_size=obs_size,
                           key=keys[0]
                           )
        self.qf1 = QNetwork(key=keys[1], obs_size=obs_size, action_dim=action_dim)
        self.qf2 = QNetwork(key=keys[2], obs_size=obs_size, action_dim=action_dim)
    
        self.actor_target = Actor(action_dim=action_dim, 
                           action_scale=action_scale, 
                           action_bias=action_bias,
                           obs_size=obs_size,
                           key=keys[3]
                           )
        self.qf1_target = QNetwork(key=keys[4], obs_size=obs_size, action_dim=action_dim)
        self.qf2_target = QNetwork(key=keys[5], obs_size=obs_size, action_dim=action_dim)

        # 2. initialize optax
        actor_scheduler = optax.cosine_decay_schedule(
                            init_value=args.learning_rate,
                            decay_steps=args.total_offline_steps/2,
                            alpha= args.learning_rate/10 )

        Q_scheduler = optax.cosine_decay_schedule(
                            init_value=args.learning_rate,
                            decay_steps=args.total_offline_steps/2,
                            alpha= args.learning_rate/10 )
        
        self.actor_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=actor_scheduler))

        self.qf1_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=Q_scheduler))
        self.qf2_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=Q_scheduler))

        self.actor_optimizer_state = self.actor_optimizer.init(eqx.filter(self.actor, eqx.is_array))
        self.qf1_optimizer_state = self.qf1_optimizer.init(eqx.filter(self.qf1, eqx.is_array))
        self.qf2_optimizer_state = self.qf2_optimizer.init(eqx.filter(self.qf2, eqx.is_array))

    def save_model(self):
        eqx.tree_serialise_leaves("q1_model.eqx", self.qf1)
        eqx.tree_serialise_leaves("q2_model.eqx", self.qf2)
        eqx.tree_serialise_leaves("q1_target_model.eqx", self.qf1_target)
        eqx.tree_serialise_leaves("q2_target_model.eqx", self.qf2_target)
        eqx.tree_serialise_leaves("actor_model.eqx", self.actor)
        eqx.tree_serialise_leaves("actor_target_model.eqx", self.actor_target)
        print('All moder have been saved!')

    def load_model(self):
        self.qf1 = eqx.tree_deserialise_leaves("q1_model.eqx", self.qf1)
        self.qf2 = eqx.tree_deserialise_leaves("q2_model.eqx", self.qf2)
        self.qf1_target = eqx.tree_deserialise_leaves("q1_target_model.eqx", self.qf1_target)
        self.qf2_target = eqx.tree_deserialise_leaves("q2_target_model.eqx", self.qf2_target)
        self.actor = eqx.tree_deserialise_leaves("actor_model.eqx", self.actor)
        self.actor_target = eqx.tree_deserialise_leaves("actor_target_model.eqx", self.actor_target)
        print('All moder have been loaded!')


    @staticmethod
    @eqx.filter_jit
    def update_q_function(  action_high, action_low, policy_noise, 
                            noise_clip, gamma,
                            actor_target: Actor,
                            qf1: QNetwork, qf1_target: QNetwork,
                            qf2: QNetwork, qf2_target: QNetwork,
                            qf1_optimizer, qf2_optimizer,
                            qf1_optimizer_state: optax.OptState,
                            qf2_optimizer_state: optax.OptState,
                            observation, action, reward, done,
                            next_observation, key: jax.random.PRNGKey
                            ):
        key, noise_key = jax.random.split(key)

        #  Target Policy Smoothing Regularization
        raw_noise = (jax.random.normal(noise_key, action.shape) * policy_noise)
        noise = raw_noise.clip(-noise_clip, noise_clip)
        
        action_scale = jnp.array(actor_target.action_scale, dtype=jnp.float32)
        next_actions = (actor_target(next_observation) + 
            noise * action_scale).clip(action_low, action_high)

        # q_function targets
        next_q1 = qf1_target(next_observation, next_actions)
        next_q2 = qf2_target(next_observation, next_actions)
        min_next_q = jnp.minimum(next_q1, next_q2)
        y = reward + (1 - jnp.squeeze(done)) * gamma * min_next_q

        #  Loss
        def loss_q_fn(qf: QNetwork):
            q_value = qf(observation, action)
            return jnp.square((q_value - jax.lax.stop_gradient(y))).mean()
        
        loss_qf1, grad_qf1 = eqx.filter_value_and_grad(loss_q_fn)(qf1)
        loss_qf2, grad_qf2 = eqx.filter_value_and_grad(loss_q_fn)(qf2)

        update_qf1, qf1_optimizer_state = qf1_optimizer.update(grad_qf1, qf1_optimizer_state, eqx.filter(qf1, eqx.is_array))
        qf1 = eqx.apply_updates(qf1, update_qf1)

        update_qf2, qf2_optimizer_state = qf2_optimizer.update(grad_qf2, qf2_optimizer_state, eqx.filter(qf2, eqx.is_array))
        qf2 = eqx.apply_updates(qf2, update_qf2)
        

        # wandb_log
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grad_qf1)))
        wandb_log_Q = {
            "next_q1":next_q1.mean(),
            "next_q2":next_q2.mean(),
            'grad_norm': grad_norm
    
        }

        return actor_target, qf1, qf1_optimizer_state, qf2, qf2_optimizer_state, loss_qf1, loss_qf2, key, wandb_log_Q

    @staticmethod
    @eqx.filter_jit
    def update_actor_and_targets(tau, alpha,
                                actor: Actor, actor_target: Actor,
                                qf1: QNetwork, qf2: QNetwork,
                                qf1_target: QNetwork, qf2_target: QNetwork,
                                actor_optimizer,
                                actor_optimizer_state,
                                observations,
                                action,
                                ):

        # Actor loss
        def loss_actor_fn(actor):
            pi = actor(observations)
            Q = qf1(observations, pi)
            bc = jnp.square(pi - action).mean()
     
            #bc = jnp.sum(bc2)
            loss = -(alpha * Q.mean() - bc)
            return loss
        

        loss_actor, grads_actor = eqx.filter_value_and_grad(loss_actor_fn)(actor)
        
        
        # update actor
        update_actor, actor_optimizer_state = actor_optimizer.update(
            grads_actor, 
            actor_optimizer_state, 
            eqx.filter(actor, eqx.is_array)
            )
        actor = eqx.apply_updates(actor, update_actor)

        # Soft updates of target networks

        def soft_update(A, B, tau):
            '''B = (1 - tau) * A + tau * B '''
            params_A, static_A = eqx.partition(A, eqx.is_array)
            params_B, static_B = eqx.partition(B, eqx.is_array)
            params_C = jax.tree.map(lambda A, B: (1 - tau) * A + tau * B, params_A, params_B)
            return eqx.combine(params_C, static_B)

        actor_target = soft_update(actor, actor_target, tau)
        qf1_target = soft_update(qf1, qf1_target, tau)
        qf2_target = soft_update(qf2, qf2_target, tau)


        # wandb_log
        pi = actor(observations)

        bc = jnp.square(pi - action).mean()
        Q1_obs_pi =  qf1(observations, pi).mean() 

        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads_actor)))
        wandb_log_A = {

            'grad_norm':grad_norm,
            'Q1_obs_pi':Q1_obs_pi,
            'action(s)[0]': pi[0][0],
            'action(s)[1]': pi[0][1],
            'bc':bc,



        }

        return actor, actor_optimizer_state, actor_target, qf1, qf2, qf1_target, qf2_target, loss_actor, wandb_log_A
    



    def train(self, observation, action, next_observation, reward, done):
        self.total_it += 1

        # update Q
        self.actor_target, self.qf1, self.qf1_optimizer_state, self.qf2, self.qf2_optimizer_state, \
            self.loss_qf1, self.loss_qf2, self.key, self.wandb_log_Q = self.update_q_function(
            action_high=self.args.env_params['action_space_high'],
            action_low=self.args.env_params['action_space_low'],
            policy_noise=self.args.policy_noise, noise_clip=self.args.noise_clip,
            gamma=self.args.gamma,
            actor_target=self.actor_target, 
            qf1=self.qf1, qf1_target=self.qf1_target,qf1_optimizer=self.qf1_optimizer,
            qf2=self.qf2, qf2_target=self.qf2_target,qf2_optimizer=self.qf2_optimizer,
            qf1_optimizer_state=self.qf1_optimizer_state, 
            qf2_optimizer_state=self.qf2_optimizer_state,
            observation=observation, action=action, reward=reward,
            done=done, next_observation=next_observation,key=self.key
        )

        # Delayed policy(actor) updates
        if self.total_it % self.policy_frequency == 0:
            self.actor, self.actor_optimizer_state, self.actor_target, \
                self.qf1, self.qf2, self.qf1_target, self.qf2_target, \
                    self.loss_actor, self.wandb_log_A = self.update_actor_and_targets(
                tau=self.args.tau, alpha=self.args.alpha, actor=self.actor, actor_target=self.actor_target, 
                qf1=self.qf1, qf1_target=self.qf1_target,
                qf2=self.qf2, qf2_target=self.qf2_target,
                actor_optimizer=self.actor_optimizer,
                actor_optimizer_state=self.actor_optimizer_state,
                observations=observation,
                action=action,
            )
        






        



        





def main(alpha, file_name):
    args = tyro.cli(Args)
    args.alpha = alpha
    args.file_name = file_name
    args.env_id = file_name
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"


    # 1. run wandb if required
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entiti,
            config=vars(args),
            name = run_name,

        )
    
    # 2. Seeding
    key = jax.random.PRNGKey(args.seed)
    key,RL_key = jax.random.split(key, 2)


    # 3. prepare ReplayBuffer
    ReplayBuffer = sample_buffer()
    args.env_params = ReplayBuffer.env_params


    # 4. prepare network    
    td3 = TD3(args, RL_key)


    # 5. prepare evaluation env
    eval_env = evaluation()

    # 6. training loop

    total_offline_steps = args.total_offline_steps
    
    for step in range(total_offline_steps):
        observation, goal_observation, action, next_observation, reward, done, _ = ReplayBuffer.sample1(batch_size=256, distance=args.distance)
        goal_tiled = goal_observation # b_broadcasted = jnp.broadcast_to(goal_observation, observation.shape)
        observation = jnp.concatenate([observation, goal_tiled], axis=-1).squeeze(axis=0)
        next_observation = jnp.concatenate([next_observation, goal_tiled], axis=-1).squeeze(axis=0)
        action = action.squeeze(axis=0)
        reward = reward.squeeze(axis=0)
        done = done.squeeze(axis=0)

        key, sample_key = jax.random.split(key, 2)
        td3.train(observation, action, next_observation, reward, done)
        if args.track:
            wandb.log({
                    'Q1_loss:': td3.loss_qf1,
                    'Q2_loss:': td3.loss_qf2,
                    'Policy_loss:': td3.loss_actor,
                    'next_q1': td3.wandb_log_Q["next_q1"],
                    'next_q2': td3.wandb_log_Q["next_q2"],
                    'Q_grad_norm': td3.wandb_log_Q["grad_norm"],
                    'Actor_grad_norm': td3.wandb_log_A["grad_norm"],
                    'Q1_obs_pi': td3.wandb_log_A['Q1_obs_pi'],
                    'action(s)[0]': td3.wandb_log_A['action(s)[0]'],
                    'action(s)[1]':td3.wandb_log_A['action(s)[1]'],
                    'bc':td3.wandb_log_A['bc'],
                    'distance_between G and obs': args.distance,

                    
            })
        
        if (step+2) % 1000 == 0:
            args.distance = round(args.distance*1.1)
            args.distance = min(args.distance, 1000)

        if (step+2) % 10000 == 0:
            td3.save_model()

        if (step+1) % 1000 ==0:
            
            avg_reward = 0
            n_test = 3
            for ci in range(n_test):
                observation, position, quaternion = ReplayBuffer.sampleforeval(batch_size=256, distance=args.distance)
                goal_obs=observation[0,100+args.distance,:]
                goal_pos=position[0,100+args.distance,:]
                intial_pos = position[0,100,:]
                intial_quet = quaternion[0,100,:]

                key, sample_key = jax.random.split(key, 2)
                obs, done= eval_env.reset(goal_obs, goal_pos, initial_pos=intial_pos, intial_quat=intial_quet)
                t = 0
                Return = 0
                rollout = []
                while (not done) and (t < 1001):
                    obs = jnp.concatenate([obs, goal_obs.reshape(obs.shape)], axis=-1).reshape([1,1536])
                    act = td3.actor(obs)
             
                    obs, reward, done  = eval_env.step(act)
                    Return += reward * 0.99 ** t
                    if done:
                        img = eval_env.evaluation_tracjectory()
                    t += 1
                
                avg_reward += Return
                    
            avg_reward /= n_test
            print(f'steps: {step}, avg_reward: {avg_reward}')
            
         
            wandb.log({
                    'Average_Return': avg_reward,
                    'visualization': wandb.Image(img)
                })
            










def evaluate_agent_Brax():
    env_name = 'walker2d'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    backend = 'generalized'  # @param ['generalized', 'positional', 'spring']

    env = envs.get_environment(env_name=env_name,
                            backend=backend)


    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)
    jit_env_step = jax.jit(env.step)
    return jit_env_step, jit_env_reset, state






if __name__ == "__main__":
   
    main(0, 'expert_batch256')


        


