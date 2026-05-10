"""
Agent
=====
"""


import numpy as np
import tensorflow as tf
import os


class Agent:
    def __init__(self,
                 state_shape,
                 n_actions,
                 action_limits,
                 discount,
                 actor_lr,
                 critic_lr,
                 actor_model,
                 critic_model,
                 target_actor_model,
                 target_critic_model,
                 tau=0.005):

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.discount = discount

        self.actor = actor_model
        self.critic = critic_model

        self.target_actor = target_actor_model
        self.target_critic = target_critic_model

        self.update_target_weights(tau=1.0)

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.tau = tau
        self.action_limits = action_limits

    def prep_state(self, state):
        return state

    def get_action(self, state, noise_object=None):

        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        if noise_object is not None:
            noise = noise_object()
        else:
            noise = 0
        
        sampled_actions = tf.squeeze(self.actor(self.prep_state(tf_state)))
        sampled_actions = sampled_actions.numpy()
        sampled_actions += noise
        action = sampled_actions

        return np.squeeze(action)

    def scale_action(self, action, mini, maxi):
        a = action * (maxi - mini) + mini

        a = np.clip(a,
                    mini,
                    maxi)

        return a
        
    @tf.function
    def train_on(self, batch):
        states, actions, next_states, rewards, done = batch
    
        # Convert all to tensors once
        states = tf.cast(states, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.float32)
        next_states = tf.cast(next_states, dtype=tf.float32)
        rewards = tf.cast(rewards, dtype=tf.float32)
        done = tf.cast(done, dtype=tf.float32)
    
        # Preprocess states in batch, once
        states_prep = self.prep_state(states)
        next_states_prep = self.prep_state(next_states)
    
        # Critic Update
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states_prep, training=True)
            target_q = self.target_critic([next_states_prep, target_actions], training=True)
            y = rewards + self.discount * target_q
            y = tf.clip_by_value(y, -100, 100)
            y = tf.stop_gradient(y)  # Important: DDPG target must not backprop
    
            q_values = self.critic([states_prep, actions], training=True)
            critic_loss = tf.reduce_mean(tf.square(y - q_values))
    
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, 1.0)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
    
        # Actor Update
        with tf.GradientTape() as tape:
            new_actions = self.actor(states_prep, training=True)
            critic_value = self.critic([states_prep, new_actions], training=True)
            actor_loss = -tf.reduce_mean(critic_value)
    
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, 1.0)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
    
        return actor_loss, critic_loss

    def update_target_weights(self, tau=None):
        if tau is None:
            tau = self.tau

        for (a, b) in zip(self.target_actor.variables, self.actor.variables):
            a.assign(b * tau + a * (1 - tau))

        for (a, b) in zip(self.target_critic.variables, self.critic.variables):
            a.assign(b * tau + a * (1 - tau))

    def __prepare_batch(self, batch):
        return batch

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.target_actor.save(os.path.join(path, "actor.keras"), )
        self.target_critic.save(os.path.join(path, "critic.keras"))

    def load_model(self, path):
        self.actor = tf.keras.models.load_model(os.path.join(path, "actor.keras"),
                                                compile=True)
        self.critic = tf.keras.models.load_model(os.path.join(path, "critic.keras"),
                                                 compile=True)

        self.update_target_weights(tau=1.0)
