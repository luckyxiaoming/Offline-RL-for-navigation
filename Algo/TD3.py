import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import optax
from jaxtyping import PRNGKeyArray, Array

import tyro
from dataclasses import dataclass
import os
import time
import numpy as np
import matplotlib.pyplot as plt 
from flax import struct
from typing import Any, Optional, Tuple

# Ensure Args and Buffer are in the same directory
from Args import Args
from Buffer import DynamicHERBuffer, calculate_cosine_similarity, calculate_reward_batch
from Mujoco_Navigation_ENV import Navigation_sim_environment

from FQE import FQE

import mediapy as media
# ==============================================================================
# Network Definitions
# ==============================================================================




class QNetwork(eqx.Module):
    """
    Critic: Q(s, g, a)
    Input: Concatenation of [State, Goal, Action_Norm]
    """

    proj_state: nn.Sequential
    backbone: nn.Sequential
    action_proj: nn.Sequential
    head: nn.Linear

    action_dim: int = eqx.field(static=True)
    history_length: int = eqx.field(static=True)
    single_obs_size: int = eqx.field(static=True)

    def __init__(self, key: PRNGKeyArray, obs_size: int, history_length: int, action_dim: int):
        keys = jax.random.split(key, 7)
        self.action_dim = action_dim
        self.history_length = history_length
        self.single_obs_size = obs_size // history_length

        self.proj_state = nn.Sequential([
            # Layer 1: Compress slightly (1024 -> 512)
            nn.Linear(in_features=self.single_obs_size, out_features=512, key=keys[0]),
            nn.LayerNorm(shape=512, elementwise_affine=True), 
            nn.Lambda(jax.nn.leaky_relu),
            # Layer 2: Project to latent dim (512 -> 256)
            nn.Linear(in_features=512, out_features=256, key=keys[1]), # Use a new key!
            nn.LayerNorm(shape=256, elementwise_affine=True), 
            nn.Lambda(jax.nn.leaky_relu),
        ])

        self.action_proj = nn.Sequential([
            nn.Linear(action_dim, 256, key=keys[2]),
            nn.LayerNorm(256, elementwise_affine=True),
            nn.Lambda(jax.nn.leaky_relu),
        ])
       
        # Input dim = 256(s_proj)*n + 256 (g_proj) + 256 (a_proj)
        self.backbone = nn.Sequential([
            nn.Linear(256 * history_length + 256 + 256, 512, key=keys[3]),
            nn.LayerNorm(512, elementwise_affine=True),
            nn.Lambda(jax.nn.leaky_relu),
            nn.Linear(512, 512, key=keys[4]),
            nn.LayerNorm(512, elementwise_affine=True),
            nn.Lambda(jax.nn.leaky_relu),
            nn.Linear(512, 512, key=keys[5]),
            nn.Lambda(jax.nn.leaky_relu),
        ])

        self.head = nn.Linear(512, 1, key=keys[6])

    def get_features(self, state: Array, goal_state: Array, action: Array):
        

        state = state.astype(jnp.float32)
        goal_state = goal_state.astype(jnp.float32)
        action = action.astype(jnp.float32)
        
        s_separated = jnp.reshape(state, (self.history_length, self.single_obs_size))

        s_proj = jax.vmap(self.proj_state)(s_separated) 
        s_emb = jnp.reshape(s_proj, (-1,))
        g_proj = self.proj_state(goal_state)
        a_proj = self.action_proj(action)

        x = jnp.concatenate([s_emb, g_proj, a_proj], axis=-1)
        features = self.backbone(x)
        return features

    def __call__(self, state: Array, goal_state: Array, action: Array):
        features = self.get_features(state, goal_state, action)
        q_value = self.head(features)
        return q_value.squeeze(-1)



class QPredictor(eqx.Module):
    net: nn.Sequential

    def __init__(self, key: PRNGKeyArray, feature_dim: int, obs_dim: int):
        """
        Args:
            feature_dim:  512
            obs_dim: 1024
        """
        keys = jax.random.split(key, 3)
        
        self.net = nn.Sequential([
            nn.Linear(feature_dim, feature_dim, key=keys[0]),
            nn.LayerNorm(feature_dim, elementwise_affine=True),
            nn.Lambda(jax.nn.leaky_relu), 
            
            nn.Linear(feature_dim, feature_dim, key=keys[1]),
            nn.LayerNorm(feature_dim, elementwise_affine=True),
            nn.Lambda(jax.nn.leaky_relu),

            nn.Linear(feature_dim, obs_dim, key=keys[2])
        ])

    def __call__(self, q_features: Array):

        raw_prediction = self.net(q_features)
        prediction = raw_prediction / (jnp.linalg.norm(raw_prediction, axis=-1, keepdims=True) + 1e-6)
        return prediction


class APredictor(eqx.Module):
    net: nn.Sequential
    action_proj: nn.Sequential

    def __init__(self, key: PRNGKeyArray, feature_dim: int, obs_dim: int, action_dim: int):
        """
        Args:
            feature_dim:  512 + 3
            obs_dim: 1024
        """
        keys = jax.random.split(key, 4)
        
        self.action_proj = nn.Sequential([
            nn.Linear(action_dim, 256, key=keys[0]),
            nn.LayerNorm(256, elementwise_affine=True),
            nn.Lambda(jax.nn.leaky_relu),
        ])
        
        self.net = nn.Sequential([
            nn.Linear(feature_dim + 256, feature_dim, key=keys[1]),
            nn.LayerNorm(feature_dim, elementwise_affine=True),
            nn.Lambda(jax.nn.leaky_relu), 
            
            nn.Linear(feature_dim, feature_dim, key=keys[2]),
            nn.LayerNorm(feature_dim, elementwise_affine=True),
            nn.Lambda(jax.nn.leaky_relu),
            nn.Linear(feature_dim, obs_dim, key=keys[3])
        ])

    def __call__(self, actor_features: Array, action: Array):
        
        a_emb = self.action_proj(action)
        combined = jnp.concatenate([actor_features, a_emb], axis=-1)
        raw_prediction = self.net(combined)
        prediction = raw_prediction / (jnp.linalg.norm(raw_prediction, axis=-1, keepdims=True) + 1e-6)
        return prediction


class Actor(eqx.Module):
    """
    Actor: pi(s, g) -> a
    Output range: [-1, 1] (via Tanh)
    """
    action_dim: int 
    action_scale: Any = eqx.field(static=True)
    action_bias: Any = eqx.field(static=True)
    history_length: int = eqx.field(static=True)
    single_obs_size: int = eqx.field(static=True)
    proj_state: nn.Sequential
    backbone: nn.Sequential
    head: nn.Sequential


    def __init__(self, key: PRNGKeyArray, obs_size: int, history_length: int, action_dim: int, action_scale, action_bias):
        keys = jax.random.split(key, 6)
        self.action_bias = action_bias.tolist()
        self.action_scale = action_scale.tolist()
        self.action_dim = action_dim
        self.history_length = history_length
        self.single_obs_size = obs_size // history_length


        
        self.proj_state = nn.Sequential([
            # Layer 1: Compress slightly (1024 -> 512)
            nn.Linear(in_features=self.single_obs_size, out_features=512, key=keys[0]),
            nn.LayerNorm(shape=512, elementwise_affine=True), 
            nn.Lambda(jax.nn.leaky_relu),
            # Layer 2: Project to latent dim (512 -> 256)
            nn.Linear(in_features=512, out_features=256, key=keys[1]), # Use a new key!
            nn.LayerNorm(shape=256, elementwise_affine=True), 
            nn.Lambda(jax.nn.leaky_relu),
        ])



        self.backbone = nn.Sequential([
            nn.Linear(in_features=256 * self.history_length + 256, out_features=512, key=keys[2]),
            nn.LayerNorm(shape=512, elementwise_affine=True),
            nn.Lambda(jax.nn.leaky_relu),
            nn.Linear(in_features=512, out_features=512, key=keys[3]),
            nn.LayerNorm(shape=512, elementwise_affine=True),
            nn.Lambda(jax.nn.leaky_relu),
            nn.Linear(in_features=512, out_features=512, key=keys[4]),
            nn.LayerNorm(shape=512, elementwise_affine=True),
            nn.Lambda(jax.nn.leaky_relu),
        ])


        # Initialize last layer weights to be small for stability
        last_layer = nn.Linear(in_features=512, out_features=action_dim, key=keys[5])
        last_layer = eqx.tree_at(
            where=lambda l: l.weight, pytree=last_layer, replace_fn=lambda w: w * 1e-4
        )
        last_layer = eqx.tree_at(
            where=lambda l: l.bias, pytree=last_layer, replace_fn=lambda b: b * 1e-4
        )

        self.head = nn.Sequential([
            last_layer,
            nn.Lambda(jax.nn.tanh),
        ])

    def get_features(self, state: Array, goal_state: Array):
        state = state.astype(jnp.float32)
        goal_state = goal_state.astype(jnp.float32)

        s_separated = jnp.reshape(state, (self.history_length, self.single_obs_size))

        
        s_proj = jax.vmap(self.proj_state)(s_separated) 
        s_emb = jnp.reshape(s_proj, (-1,)) 
        g_proj = self.proj_state(goal_state)

        x = jnp.concatenate([s_emb, g_proj], axis=-1)
        features = self.backbone(x)
        return features

    def __call__(self, state: Array, goal_state: Array):
        features = self.get_features(state, goal_state)
        y = self.head(features)
        return y
    
    def get_eval_action(self, state: Array, goal_state: Array):
        action_scale = jnp.array(self.action_scale, dtype=jnp.float32)
        action_bias  = jnp.array(self.action_bias, dtype=jnp.float32)
        raw_action = self.__call__(state, goal_state)
        action = action_scale * raw_action + action_bias
        return action


# ==============================================================================
# TD3 Agent Logic
# ==============================================================================

class TD3(object):
    def __init__(self, args, key=jax.random.PRNGKey(0)):
        self.args = args
        self.total_it = 0
        self.policy_frequency = args.policy_frequency
        obs_size = args.env_params['observation_size'] * args.history_length
        action_dim = args.env_params['action_dimension']
     
        action_high = jnp.array(args.env_params['action_space_high'])
        action_low = jnp.array(args.env_params['action_space_low'])
        
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0
        
        # Logging 
        self.loss_qf1 = 0.0
        self.loss_qf2 = 0.0
        self.loss_actor = 0.0 # Initial value
        self.wandb_log_Q = {"next_q1":0, "next_q2":0, 'grad_norm':0}
        self.wandb_log_P = {"loss_Apred":0, "loss_Qpred":0}
        self.wandb_log_A = {
            "grad_norm":0,
            'Q1_obs_pi':0,
            'lmbda':0,
            'loss_bc': 0,
            'loss_rl': self.args.alpha,
            'actor_loss': 0
        }

        keys = jax.random.split(key, 9)
        self.key = keys[6]
        
        # 1. Initialize Networks
        self.actor = Actor(key=keys[0], obs_size=obs_size, history_length=args.history_length, action_dim=action_dim, action_scale=self.action_scale, action_bias=self.action_bias)
        self.actor_target = Actor(key=keys[3], obs_size=obs_size, history_length=args.history_length, action_dim=action_dim, action_scale=self.action_scale, action_bias=self.action_bias)
        
        self.qf1 = QNetwork(key=keys[1], obs_size=obs_size, history_length=args.history_length, action_dim=action_dim)
        self.qf2 = QNetwork(key=keys[2], obs_size=obs_size, history_length=args.history_length, action_dim=action_dim)
        self.qf1_target = QNetwork(key=keys[4], obs_size=obs_size, history_length=args.history_length, action_dim=action_dim)
        self.qf2_target = QNetwork(key=keys[5], obs_size=obs_size, history_length=args.history_length, action_dim=action_dim)

        self.Qpredictor = QPredictor(key=keys[6], feature_dim=512, obs_dim=obs_size)    
        self.Apredictor = APredictor(key=keys[7], feature_dim=512, obs_dim=obs_size, action_dim=action_dim) 

        # 2. Initialize Optimizers
        # Using linear schedule from args
        actor_scheduler = optax.linear_schedule(
                            init_value=args.learning_rate,
                            end_value= 0.02 * args.learning_rate,
                            transition_begin= args.total_offline_steps/8,
                            transition_steps= args.total_offline_steps)

        Q_scheduler = optax.linear_schedule(
                            init_value=args.learning_rate,
                            end_value= 0.02 * args.learning_rate,
                            transition_begin= args.total_offline_steps/8,
                            transition_steps= args.total_offline_steps)
        
        self.noise_scheduler = optax.linear_schedule(
                    init_value=args.policy_noise,
                    end_value= 0.01 * args.policy_noise,
                    transition_begin= args.total_offline_steps/8,
                    transition_steps= args.total_offline_steps)



        # Optimizer ChainsW
        self.actor_optimizer = optax.chain(optax.clip_by_global_norm(10), optax.adamw(learning_rate=actor_scheduler))
        self.qf1_optimizer = optax.chain(optax.clip_by_global_norm(10), optax.adamw(learning_rate=Q_scheduler))
        self.qf2_optimizer = optax.chain(optax.clip_by_global_norm(10), optax.adamw(learning_rate=Q_scheduler))
        self.qpred_optimizer = optax.chain(optax.clip_by_global_norm(10), optax.adamw(learning_rate=Q_scheduler))
        self.apred_optimizer = optax.chain(optax.clip_by_global_norm(10), optax.adamw(learning_rate=Q_scheduler))

        # Optimizer States
        self.actor_optimizer_state = self.actor_optimizer.init(eqx.filter(self.actor, eqx.is_array))
        self.qf1_optimizer_state = self.qf1_optimizer.init(eqx.filter(self.qf1, eqx.is_array))
        self.qf2_optimizer_state = self.qf2_optimizer.init(eqx.filter(self.qf2, eqx.is_array))
        self.qpred_optimizer_state = self.qpred_optimizer.init(eqx.filter(self.Qpredictor, eqx.is_array))
        self.apred_optimizer_state = self.apred_optimizer.init(eqx.filter(self.Apredictor, eqx.is_array))



    def save_model(self, step):
        output_dir = "saved_models"+f"{self.args.env_id}"+f"/TD3_step{step}/"
        os.makedirs(output_dir, exist_ok=True)
        eqx.tree_serialise_leaves(os.path.join(output_dir, "q1_model.eqx"), self.qf1)
        eqx.tree_serialise_leaves(os.path.join(output_dir, "q2_model.eqx"), self.qf2)
        eqx.tree_serialise_leaves(os.path.join(output_dir, "actor_model.eqx"), self.actor)
        eqx.tree_serialise_leaves(os.path.join(output_dir, "q1_target_model.eqx"), self.qf1_target)
        eqx.tree_serialise_leaves(os.path.join(output_dir, "q2_target_model.eqx"), self.qf2_target)
        eqx.tree_serialise_leaves(os.path.join(output_dir, "actor_target_model.eqx"), self.actor_target)
        print('All models have been saved!')

    def load_actor_model(self, path):
        self.actor = eqx.tree_deserialise_leaves(os.path.join(path, "actor_model.eqx"), self.actor)
        self.actor_target = eqx.tree_deserialise_leaves(os.path.join(path, "actor_target_model.eqx"), self.actor_target)
        print('Actor models have been loaded!')


    def load_all_model(self, path):
        self.qf2 = eqx.tree_deserialise_leaves(os.path.join(path, "q2_model.eqx"), self.qf2)
        self.actor = eqx.tree_deserialise_leaves(os.path.join(path, "actor_model.eqx"), self.actor)
        self.qf1 = eqx.tree_deserialise_leaves(os.path.join(path, "q1_model.eqx"), self.qf1)
        self.qf2_target = eqx.tree_deserialise_leaves(os.path.join(path, "q2_model.eqx"), self.qf2_target)
        self.actor_target = eqx.tree_deserialise_leaves(os.path.join(path, "actor_model.eqx"), self.actor_target)
        self.qf1_target = eqx.tree_deserialise_leaves(os.path.join(path, "q1_model.eqx"), self.qf1_target)
        
        
        print('Actor, Q1, Q2 models have been loaded!')


    # Updated Functions
    @staticmethod
    @eqx.filter_jit
    def update_q_function(  policy_noise, 
                            noise_clip, gamma,
                            actor_target: Actor,
                            qf1: QNetwork, qf1_target: QNetwork,
                            qf2: QNetwork, qf2_target: QNetwork,
                            qf1_optimizer, qf2_optimizer,
                            qf1_optimizer_state: optax.OptState,
                            qf2_optimizer_state: optax.OptState,
                            observation, goal_observation, action, reward, done,
                            next_observation, key: jax.random.PRNGKey
                            ):
        key, noise_key = jax.random.split(key)

        # Target Policy Smoothing
        raw_noise = (jax.random.normal(noise_key, action.shape) * policy_noise)
        noise = raw_noise.clip(-noise_clip, noise_clip)
        
        # Get target action (Output is [-1, 1])
        next_action_pi = jax.vmap(actor_target)(next_observation, goal_observation)
        next_actions = (next_action_pi + noise).clip(-1.0, 1.0)

        # Q-function targets
        next_q1 = jax.vmap(qf1_target)(next_observation, goal_observation, next_actions)
        next_q2 = jax.vmap(qf2_target)(next_observation, goal_observation, next_actions)
        min_next_q = jnp.minimum(next_q1, next_q2)
        
        # Bellman Target
        y = reward + (1 - jnp.squeeze(done)) * gamma * min_next_q
        y = jax.lax.stop_gradient(y)

        # Loss function
        def loss_q_fn(qf: QNetwork):
            q_value = jax.vmap(qf)(observation, goal_observation, action)
            loss = jnp.square(q_value - y).mean()
            return loss

        loss_qf1, grad_qf1 = eqx.filter_value_and_grad(loss_q_fn)(qf1)
        loss_qf2, grad_qf2 = eqx.filter_value_and_grad(loss_q_fn)(qf2)

        # Update
        update_qf1, qf1_optimizer_state = qf1_optimizer.update(grad_qf1, qf1_optimizer_state, eqx.filter(qf1, eqx.is_array))
        qf1 = eqx.apply_updates(qf1, update_qf1)
        update_qf2, qf2_optimizer_state = qf2_optimizer.update(grad_qf2, qf2_optimizer_state, eqx.filter(qf2, eqx.is_array))
        qf2 = eqx.apply_updates(qf2, update_qf2)

        # Logging
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
                                goal_observation,
                                action,
                                mask,
                                ):

        def loss_actor_fn(actor):
            pi = jax.vmap(actor)(observations, goal_observation)
            Q = jax.vmap(qf1)(observations, goal_observation, pi)
            loss_rl = - Q

            # BC Loss (Minimize MSE)
            loss_bc = jnp.square(pi - action).sum(axis=-1)
            
            loss =loss_rl + alpha * loss_bc
            final_loss = loss.mean()
            # Now remove BC
            return final_loss, (loss_rl.mean(), loss_bc.mean(), Q.mean(), pi)
        
        (loss_actor, (loss_rl, loss_bc, q_val, pi_val)), grads_actor = \
            eqx.filter_value_and_grad(loss_actor_fn, has_aux=True)(actor)
        
        update_actor, actor_optimizer_state = actor_optimizer.update(grads_actor, actor_optimizer_state, eqx.filter(actor, eqx.is_array))
        actor = eqx.apply_updates(actor, update_actor)

        # Soft updates
        def soft_update(A, B, tau):
            params_A, static_A = eqx.partition(A, eqx.is_array)
            params_B, static_B = eqx.partition(B, eqx.is_array)
            params_C = jax.tree.map(lambda curr, targ: (1 - tau) * targ + tau * curr, params_A, params_B)
            return eqx.combine(params_C, static_B)

        actor_target = soft_update(actor, actor_target, tau)
        qf1_target = soft_update(qf1, qf1_target, tau)
        qf2_target = soft_update(qf2, qf2_target, tau)

        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads_actor)))
        wandb_log_A = {
            'grad_norm':grad_norm,
            'Q1_obs_pi':q_val,
            'loss_bc':loss_bc,
            'loss_rl':loss_rl,
            'actor_loss': loss_actor
        }

        return actor, actor_optimizer_state, actor_target, qf1, qf2, qf1_target, qf2_target, loss_actor, wandb_log_A
    

    @staticmethod
    @eqx.filter_jit
    def update_predictor(actor: Actor,
                                qf1: QNetwork, 
                                Qpredictor: QPredictor,
                                Apredictor: APredictor,
                                qpred_optimizer,
                                qpred_optimizer_state,
                                apred_optimizer,
                                apred_optimizer_state,
                                observations,
                                goal_observations,
                                next_observations,
                                actions,
                                ):

        
        Q_features= jax.vmap(qf1.get_features)(observations, goal_observations, actions)
        Q_features = jax.lax.stop_gradient(Q_features)

        def loss_Qpredictor_fn(Qpredictor):
            pre_next_s = jax.vmap(Qpredictor)(Q_features)
            loss = jnp.mean((pre_next_s - observations)**2)
            return loss
        
        loss_qpred, grads_qpred = eqx.filter_value_and_grad(loss_Qpredictor_fn)(Qpredictor)
        
        update_qpred, qpred_optimizer_state = qpred_optimizer.update(grads_qpred, qpred_optimizer_state, eqx.filter(Qpredictor, eqx.is_array))
        Qpredictor = eqx.apply_updates(Qpredictor, update_qpred)

        
        actor_features = jax.vmap(actor.get_features)(observations, goal_observations)
        actor_features = jax.lax.stop_gradient(actor_features)
        def loss_Apredictor_fn(Apredictor):
            
            pre_next_s = jax.vmap(Apredictor)(actor_features, actions)
            loss = jnp.mean((pre_next_s - observations)**2)
            return loss
        loss_apred, grads_apred = eqx.filter_value_and_grad(loss_Apredictor_fn)(Apredictor)
        update_apred, apred_optimizer_state = apred_optimizer.update(grads_apred, apred_optimizer_state, eqx.filter(Apredictor, eqx.is_array))
        Apredictor = eqx.apply_updates(Apredictor, update_apred)


        wandb_log_P = {
            'loss_Apred':loss_apred,
            'loss_Qpred':loss_qpred,
        }

        return  Qpredictor, qpred_optimizer_state, Apredictor, apred_optimizer_state, wandb_log_P
    

    def train(self, buffer):
        """
        Main Training Step: 
        1. Sample Value Batch -> Update Critic
        2. Sample Policy Batch -> Update Actor
        """
        self.total_it += 1
        # --- 1. Update Critic (Mixed Sampling) ---
        obs, g_obs, act, next_obs, rew, done, mask = buffer.sample_value_batch()
        
        # Normalize action
        act_norm = (act - self.action_bias) / self.action_scale

        self.actor_target, self.qf1, self.qf1_optimizer_state, self.qf2, self.qf2_optimizer_state, \
            self.loss_qf1, self.loss_qf2, self.key, self.wandb_log_Q = self.update_q_function(
            policy_noise=self.noise_scheduler(self.total_it), noise_clip=self.args.noise_clip,
            gamma=self.args.gamma,
            actor_target=self.actor_target, 
            qf1=self.qf1, qf1_target=self.qf1_target, qf1_optimizer=self.qf1_optimizer,
            qf2=self.qf2, qf2_target=self.qf2_target, qf2_optimizer=self.qf2_optimizer,
            qf1_optimizer_state=self.qf1_optimizer_state, 
            qf2_optimizer_state=self.qf2_optimizer_state,
            observation=obs, goal_observation=g_obs, action=act_norm, reward=rew,
            done=done, next_observation=next_obs, key=self.key
        )

        if False:
            self.Qpredictor, self.qpred_optimizer_state, self.Apredictor, self.apred_optimizer_state, \
                self.wandb_log_P = self.update_predictor(
                actor=self.actor,
                qf1=self.qf1,
                Qpredictor=self.Qpredictor, 
                qpred_optimizer=self.qpred_optimizer,
                qpred_optimizer_state=self.qpred_optimizer_state,
                Apredictor=self.Apredictor,
                apred_optimizer=self.apred_optimizer,
                apred_optimizer_state=self.apred_optimizer_state,
                observations=obs,
                goal_observations=g_obs,
                next_observations=next_obs,
                actions=act_norm
            )

        # --- 2. Update Actor (Policy Sampling: Future Only) ---
        if self.total_it % self.policy_frequency == 0:
            obs_p, g_obs_p, act_p, _, _, _, mask_p = buffer.sample_policy_batch()
            act_p_norm = (act_p - self.action_bias) / self.action_scale

            self.actor, self.actor_optimizer_state, self.actor_target, \
                self.qf1, self.qf2, self.qf1_target, self.qf2_target, \
                self.loss_actor, self.wandb_log_A = self.update_actor_and_targets(
                tau=self.args.tau, alpha=self.args.alpha, actor=self.actor, actor_target=self.actor_target, 
                qf1=self.qf1, qf1_target=self.qf1_target,
                qf2=self.qf2, qf2_target=self.qf2_target,
                actor_optimizer=self.actor_optimizer,
                actor_optimizer_state=self.actor_optimizer_state,
                observations=obs_p, goal_observation=g_obs_p,
                action=act_p_norm, mask=mask_p,
            )


# ==============================================================================
#  Main Function
# ==============================================================================

def main(args):
    
    run_name = f"{args.env_id}_{int(time.time())}"

    # 1. run wandb if required
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity, 
            config=vars(args),
            name=run_name,

        )
    name = f"{args.env_id}"
    # 2. Seeding
    key = jax.random.PRNGKey(args.seed)
    key, RL_key = jax.random.split(key, 2)

    # 3. prepare Buffer
    ReplayBuffer = DynamicHERBuffer(args) # Uses updated initialization
    args.env_params = ReplayBuffer.env_params

    # 4. prepare network        
    action_high = jnp.array(args.env_params['action_space_high'])
    action_low = jnp.array(args.env_params['action_space_low'])
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    td3 = TD3(args, RL_key)

    if args.load_model_path is not None:
        td3.load_all_model(args.load_model_path)
        print(f"Loaded Model from {args.load_model_path}")

    # 5. prepare evaluation functions
    eval_env = Navigation_sim_environment()


    # FQE
    fqe = FQE(args)

    

    @eqx.filter_jit
    def calculate_metrics_batch(q1, actor, s, g, a_norm, next_s):
        # Q(s_expert, a_expert)
        Q_i = jax.vmap(q1)(s, g, a_norm)
        # pi(next_s) -> output is [-1, 1]
        p_ii = jax.vmap(actor)(next_s, g)
        # Q(next_s, pi(next_s))
        Q_ii = jax.vmap(q1)(next_s, g, p_ii)
        # pi(s) -> output is [-1, 1]
        p_i = jax.vmap(actor)(s, g)
        # Q(s, pi(s))
        Q_pi_i = jax.vmap(q1)(s, g, p_i)
        return Q_i, Q_ii, Q_pi_i

    # 6. training loop
    total_offline_steps = args.total_offline_steps
    print("Start Training...")
    
    for step in range(total_offline_steps):
       
        td3.train(ReplayBuffer)
        
        if args.track:
            wandb.log({
                    'Q1_loss:': td3.loss_qf1,
                    'Q2_loss:': td3.loss_qf2,
                    'Policy_loss:': td3.wandb_log_A['actor_loss'],
                    'next_q1': td3.wandb_log_Q["next_q1"],
                    'next_q2': td3.wandb_log_Q["next_q2"],
                    'Q_grad_norm': td3.wandb_log_Q["grad_norm"],
                    'Actor_grad_norm': td3.wandb_log_A["grad_norm"],
                    'Q1_obs_pi': td3.wandb_log_A['Q1_obs_pi'],
                    'loss_rl':td3.wandb_log_A['loss_rl'],
                    'loss_bc':td3.wandb_log_A['loss_bc'],
                    'done_treshold': args.done_threshold,
                    'loss_Apred': td3.wandb_log_P['loss_Apred'],
                    'loss_Qpred': td3.wandb_log_P['loss_Qpred'],
                 })
        
        # Calculate Fitted Q Evaluation 
        if (step+1) % 30000 == 0:
            t1 = time.time()
            ReplayBuffer.batch_size = 512*20
            obs, g_obs, act, next_obs, rew, done, mask = ReplayBuffer.sample_value_batch()
            
            ReplayBuffer.batch_size = args.batch_size

            # Normalize action
            act_norm = (act - action_bias) / action_scale
            done =  np.expand_dims(done, axis=1)
            obs_Q_feature = jax.vmap(td3.qf1.get_features)(obs, g_obs, act_norm)
            next_pi_action_norm = jax.vmap(td3.actor)(next_obs, g_obs)
            next_obs_Q_feature = jax.vmap(td3.qf1.get_features)(next_obs, g_obs, next_pi_action_norm)
            fqe_loss = fqe.linear_regression(z=obs_Q_feature, next_z=next_obs_Q_feature, rewards=rew, done=done)
            
            # eva
            obs_s0 = jnp.array(ReplayBuffer.S0)
            obs_sg = jnp.array(ReplayBuffer.Send)
            pi_action_at_s_norm = jax.vmap(td3.actor)(obs_s0, obs_sg)
            S0_Q_feature = jax.vmap(td3.qf1.get_features)(obs_s0, obs_sg, pi_action_at_s_norm)

            fqe_value = fqe.evaluate(z=S0_Q_feature)

            if args.track:
                    wandb.log({
                       # 'Metrics/Mean of Metrics 1:E_expert(Q(s_expert,a,s_g) -(r+ r(Q(s,pi(s),s_g))))': EE1,
                       # 'Metrics/Mean of Metrics 2: E(Q(s_expert, pi(s), s_g)) ': EE2,
                      #  'Metrics/Mean of Metrics 3: E(Q(s_expert, pi(s), s_g)) - E_expert(Q(s_expert,a,s_g) -(r+ r(Q(s,pi(s),s_g)))) ': EE3,
                      #  'Metrics/Mean of Metrics 4: E(Q(s_expert, pi(s), s_g) - Q(s_expert,a,s_g)) ': EE4,
                      #  'Metrics/Expert Discounted Return mean': expert_returns_mean,
                        'Metrics/FQE Loss': fqe_loss,
                        'Metrics/FQE Estimated Value': fqe_value.mean(),
                    #    'The mean of Cosine Similarity to next_state from Q': cos_sim_qpred_mean,
                    #    'The mean of Cosine Similarity to next_state from policy': cos_sim_apred_mean
                    })
            print(f'spend time for calculating metrics: {time.time() - t1} seconds at step {step} ')



        # Calculate 2 metrics for detecing possible overfitting
        if (step+1) % 2000000000 == 0:
            t1 = time.time()
            # Evaluate on Expert Data

            expert_number = 0
            E1_list = []
            E2_list = []
            E4_list = []
            cos_sim_qpred_list = []
            cos_sim_apred_list = []
            expert_returns = []
            for i in ReplayBuffer.EXPERT:
                expert = ReplayBuffer.EXPERT[i]
                obs_full = expert['feature']
                next_obs_full = expert['next_feature']
                goal_full = expert['goal_feature']
                action_full = expert['action'] # Raw action [0, 0.2]


                # test predictor
                if False:
                    cur_obs = obs_full[0]
                    cur_goal = goal_full[0]
                    nex_obs = next_obs_full[0]
                    act = action_full[0]
                    Q_features= td3.qf1.get_features(cur_obs, cur_goal, ((act- action_bias) / action_scale))
                    qpred_next_s = td3.Qpredictor(Q_features)
                    
                    ap_features= td3.actor.get_features(cur_obs, cur_goal)
                    apred_next_s = td3.Apredictor(ap_features, ((act- action_bias) / action_scale))

                    cos_sim_qpred = calculate_cosine_similarity(qpred_next_s, nex_obs)
                    cos_sim_apred = calculate_cosine_similarity(apred_next_s, nex_obs)
                    cos_sim_apred_list.append(cos_sim_apred)
                    cos_sim_qpred_list.append(cos_sim_qpred)


                # Use Global Helper Function
                expert_reward, expert_done = calculate_reward_batch(
                    calculate_cosine_similarity(obs_full, goal_full),
                    calculate_cosine_similarity(next_obs_full, goal_full),
                    done_threshold=args.done_threshold,
                    scale=args.reward_scale, bonus=args.bonus
                )


               
                if jnp.sum(expert_done) > 0:
                    index = jnp.where(expert_done == 1)[0][0]
                else:
                    index = len(expert_done) - 1
                    # print('Warning: no done signal in this expert episode!')

                obs_batch = obs_full[0:index+1]
                next_obs_batch = next_obs_full[0:index+1]
                goal_batch = goal_full[0:index+1]
                raw_action_batch = action_full[0:index+1]
                expert_action_norm = (raw_action_batch - action_bias) / action_scale

                expert_gamma = jnp.array([args.gamma**k for k in range(index+1)])
                Discount_Return = jnp.sum(expert_reward[0:index+1] * expert_gamma)
                expert_returns.append(Discount_Return)

                Metric1 = 0
                Metric2 = 0
                E1 = 0
                E2 = 0
                Q_i, Q_ii, Q_pi_i = calculate_metrics_batch(
                    td3.qf1, td3.actor, 
                    obs_batch, goal_batch, expert_action_norm, next_obs_batch
                )
                
                target = expert_reward[0:index+1] + (1 - expert_done[0:index+1]) * expert_gamma * Q_ii
                Metric1 = Q_i - target
                
                E1 = Metric1.mean()
                E1_list.append(E1)

                Metric2 = Q_ii * (1 - expert_done[0:index+1])
                E2 = Metric2.mean()
                E2_list.append(E2)
                
                Metric4 = Q_pi_i - Q_i
                E4 = Metric4.mean()
                E4_list.append(E4)
                
                str_number = str(expert_number)
                expert_number +=1
                if args.track:
                    wandb.log({
                        'sub_metrics/Metrics 1 for'+str_number+':E_expert(Q(s_expert,a,s_g) -(r+ r(Q(s,pi(s),s_g))))': E1,
                        'sub_metrics/Metrics 2 for'+str_number+': E(Q(s_expert, pi(s), s_g)) ': E2,
                        'sub_metrics/Metrics 4 for'+str_number+': E(Q(s_expert, pi(s), s_g) - Q(s_expert,a,s_g)) ': E4,
                    })

            EE1 = jnp.array(E1_list).mean()
            EE2 = jnp.array(E2_list).mean()
            EE3 = EE2 - EE1
            EE4 = jnp.array(E4_list).mean()

            cos_sim_qpred_mean = jnp.array(cos_sim_qpred_list).mean()
            cos_sim_apred_mean = jnp.array(cos_sim_apred_list).mean()
            expert_returns_mean = jnp.array(expert_returns).mean()


            # FQE
            ReplayBuffer.batch_size = 512*20
            obs, g_obs, act, next_obs, rew, done, mask = ReplayBuffer.sample_value_batch()
            ReplayBuffer.batch_size = args.batch_size

            # Normalize action
            act_norm = (act - action_bias) / action_scale
            done =  np.expand_dims(done, axis=1)
            obs_Q_feature = jax.vmap(td3.qf1.get_features)(obs, g_obs, act_norm)
            next_pi_action_norm = jax.vmap(td3.actor)(next_obs, g_obs)
            next_obs_Q_feature = jax.vmap(td3.qf1.get_features)(next_obs, g_obs, next_pi_action_norm)
            fqe_loss = fqe.linear_regression(z=obs_Q_feature, next_z=next_obs_Q_feature, rewards=rew, done=done)
            
            # eva
            obs_s0 = jnp.array(ReplayBuffer.S0)
            obs_sg = jnp.array(ReplayBuffer.Send)
            pi_action_at_s_norm = jax.vmap(td3.actor)(obs_s0, obs_sg)
            S0_Q_feature = jax.vmap(td3.qf1.get_features)(obs_s0, obs_sg, pi_action_at_s_norm)

            fqe_value = fqe.evaluate(z=S0_Q_feature)

            if args.track:
                    wandb.log({
                        'Metrics/Mean of Metrics 1:E_expert(Q(s_expert,a,s_g) -(r+ r(Q(s,pi(s),s_g))))': EE1,
                        'Metrics/Mean of Metrics 2: E(Q(s_expert, pi(s), s_g)) ': EE2,
                        'Metrics/Mean of Metrics 3: E(Q(s_expert, pi(s), s_g)) - E_expert(Q(s_expert,a,s_g) -(r+ r(Q(s,pi(s),s_g)))) ': EE3,
                        'Metrics/Mean of Metrics 4: E(Q(s_expert, pi(s), s_g) - Q(s_expert,a,s_g)) ': EE4,
                        'Metrics/Expert Discounted Return mean': expert_returns_mean,
                        'Metrics/FQE Loss': fqe_loss,
                        'Metrics/FQE Estimated Value': fqe_value.mean(),
                    #    'The mean of Cosine Similarity to next_state from Q': cos_sim_qpred_mean,
                    #    'The mean of Cosine Similarity to next_state from policy': cos_sim_apred_mean
                    })
            print(f'spend time for calculating metrics: {time.time() - t1} seconds at step {step} ')

            


        # Online Evaluation every 10k steps
        if (step+1) % 30000 ==0:

            # our metrics
            tn = 0
            DR_values = []
            success_count = 0
            min_distance = 0
            Total_Dis = 0

            
            for goal in ReplayBuffer.GOALS:
                tn +=1
                name2 = f"expert {goal}_step{step}"
                goal_feature = ReplayBuffer.GOALS[goal]['goal_feature'][:]
                goal_pos = ReplayBuffer.GOALS[goal]['goal_position'][:]
                initial_pos = ReplayBuffer.GOALS[goal]['initial_position'][:]
                initial_quat = ReplayBuffer.GOALS[goal]['initial_quaternion'][:]
                initial_distance = ReplayBuffer.GOALS[goal]['initial_distance']
                trans, frame_list = eval_env.online_evaluation(td3.actor, goal_feature, goal_pos, initial_pos, initial_quat, name=name2, args=args) 
                states = trans['features']
                next_states = trans['next_features']
                traj_pos = trans['positions']
                goal_states = jnp.broadcast_to(goal_feature, states.shape)
                rel_pos = np.array(traj_pos - goal_pos)
                distance = np.linalg.norm(rel_pos, axis=-1)

                Total_Dis += initial_distance
                min_distance += distance.min()
                
                if args.track:
                    for x, y in enumerate(distance):
                        wandb.log({
                            f"Distance Change for each one/expert_traj{goal}-distance": y
                        })


                rewards, dones = calculate_reward_batch(
                        calculate_cosine_similarity(states, goal_states),
                        calculate_cosine_similarity(next_states, goal_states),
                        done_threshold=args.done_threshold,
                        scale=args.reward_scale, bonus=args.bonus
                )
                
                gammas = jnp.array([args.gamma**(i) for i in range(len(dones))])
                reach_goal = True
                t = dones.max()
                if t==0:
                    index = len(dones)-1
                    reach_goal = False
                else:
                    index = jnp.where(dones==1)[0][0]
                Discount_Return = jnp.sum(rewards[0:index+1]* (gammas[0:index+1]))
                print(f"Discounted Return for expert {goal}: ", Discount_Return, 'length of trajectory: ', index+1, 'reach goal: ', reach_goal)
                if reach_goal:
                    success_count +=1
                    media.show_video(frame_list, fps=10)
                    media.write_video(f"evaluation_{name}_{name2}.mp4", frame_list, fps=10)

                DR_values.append(Discount_Return)

            Score = (Total_Dis - min_distance) / Total_Dis
            Discount_Return = jnp.mean(jnp.array(DR_values))
            print("Average DR OPE Value over all trajectories: ", Discount_Return)
            if args.track:
                wandb.log({
                       'Online evaluation/Online Evaluation Discounted Return': Discount_Return,
                       'Online evaluation/Success rate %': 100 * success_count/ tn,
                       'Online evaluation/Score(full is 1)': Score
                    })
         
        if (step+1) % 30000 == 0:
            td3.save_model(step=step)
    
    wandb.finish()

if __name__ == "__main__":


    args = tyro.cli(Args)



    
    #args = tyro.cli(Args)
   # args.exp_name = 'spare+HER_reward'
   # args.alpha = 0.05
   # args.env_id = '1Hz-2history_Sparse+HER_0.05bc' 
  #  main(args)

    args = tyro.cli(Args)
    args.value_sample_ratios= (0.1, 0.2, 0.2, 0.5)
    args.exp_name = 'spare+HER_reward'
    args.alpha = 0.01
    args.env_id = '2Hz-2history+filter_Sparse+Red+0.99Gamma+HER_0.01bc' 
    main(args)



   # args = tyro.cli(Args)
   # args.exp_name = 'dense_reward'
   # args.alpha = 50
   # args.env_id = 'TD3_Navigation_dense_reward_alpha = 50' 
  #  main(args)