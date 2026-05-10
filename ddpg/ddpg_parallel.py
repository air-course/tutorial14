"""
DDPG Trainer
============
"""


import numpy as np
import tensorflow as tf

from pendulum_plant.pendulum_plant import PendulumPlant
from pendulum_plant.simulation import Simulator
from pendulum_plant.gym_environment import SimplePendulumEnv
from stable_baselines3.common.vec_env import SubprocVecEnv


from ddpg.replay_buffer import ReplayBuffer
from ddpg.models import get_actor, get_critic
from ddpg.agent import Agent
from ddpg.noise import OUActionNoise
from stable_baselines3 import DDPG

def make_env(simulator_args, env_args):
    """
    Create single instance of enviroment.
    """
    def _init():
        pendulum = PendulumPlant(**simulator_args)
        simulator = Simulator(plant=pendulum)
        env = SimplePendulumEnv(simulator=simulator, **env_args) 
        original_reset = env.reset
        def reset_wrapper(*args, **kwargs):
            return original_reset(*args, random_init="everywhere", **kwargs)
        env.reset = reset_wrapper
        return env
    return _init

class ddpg_trainer:
    def __init__(self,
                 batch_size,
                 validate_every=None,
                 validation_reps=None,
                 train_every_steps=np.inf):

        self.batch_size = batch_size

        self.validate = (validate_every is not None)
        self.validate_every = validate_every
        self.validation_reps = validation_reps

        self.train_every_steps = train_every_steps

        self.noise_object = [OUActionNoise(mean=np.zeros(1),
                                          std_deviation=0.2*np.ones(1)) for _ in range(4)] # TODO:dont hardcoode n_envs

    def init_pendulum(self, mass=0.57288, length=0.5, inertia=None,
                      damping=0.15, coulomb_friction=0.0, gravity=9.81,
                      torque_limit=2.0):
        """
        Initialize the pendulum parameters.

        Parameters
        ----------
        mass : float, default=0.57288
            mass of the pendulum [kg]
        length : float, default=0.5
            length of the pendulum [m]
        inertia : float, default=None
            inertia of the pendulum [kg m^2]
            defaults to point mass inertia (mass*length^2)
        damping : float, default=0.15
            damping factor of the pendulum [kg m/s]
        coulomb_friction : float, default=0.0
            coulomb friciton of the pendulum [Nm]
        gravity : float, default=9.81
            gravity (positive direction points down) [m/s^2]
        torque_limit : float, default=2.0
            the torque_limit of the pendulum actuator
        """

        self.pen_mass = mass
        self.pen_length = length
        if inertia is None:
            inertia = mass*length**2
        self.pen_inertia = inertia
        self.pen_damping = damping
        self.pen_cfric = coulomb_friction
        self.pen_gravity = gravity
        self.pen_torque_limit = torque_limit

        self.pendulum = PendulumPlant(mass=self.pen_mass,
                                      length=self.pen_length,
                                      damping=self.pen_damping,
                                      gravity=self.pen_gravity,
                                      coulomb_fric=self.pen_cfric,
                                      inertia=self.pen_inertia,
                                      torque_limit=self.pen_torque_limit)

        self.simulator = Simulator(plant=self.pendulum)

    def init_environment(self,
                         dt=0.01,
                         integrator="runge_kutta",
                         max_steps=1000,
                         reward_type="open_ai_gym",
                         state_representation=2,
                         validation_limit=-150,
                         target=[np.pi, 0.0],
                         state_target_epsilon=[1e-2, 1e-2],
                         scale_action=True,
                         random_init="everywhere",
                         n_envs=4):
        """
        Initialize the training environment.
        This includes the simulation parameters of the pendulum.

        Parameter
        ---------
        dt : float, default=0.01
            time step [s]
        integrator: string
            integration method to be used
            "euler" for euler integrator,
            "runge_kutta" for Runge-Kutta integrator
        max_steps : int, default=1000
            maximum number of timesteps for one training episode
            i.e. One episode lasts at most max_stepd*dt seconds
        reward_type : string, default=soft_binary_with_repellor
            string which defines the reward function
            options are: 'continuous', 'discrete', 'soft_binary',
                         'soft_binary_with_repellor'
        state_representation : int, default=2
            determines how the state space of the pendulum is represented
            state_representation=2 means state = [position, velocity]
            state_representation=3 means state = [cos(position),
                                                  sin(position),
                                                  velocity]
        target : array-like, default=[np.pi, 0.0]
            The target state of the pendulum
        state_target_epsilon : array-like, default=[1e-2, 1e-2]
            In this vicinity the target counts as reached.
        scale_action : bool, default=True
            whether to scale the output of the model with the torque limit
            of the simulator's plant.
            If True the model is expected so return values in the intervall
            [-1, 1] as action.

        """

        self.max_steps = max_steps
        self.n_envs = n_envs

        self.simulator_args = {
            "mass": self.pen_mass,
            "length": self.pen_length,
            "damping": self.pen_damping,
            "gravity": self.pen_gravity,
            "coulomb_fric": self.pen_cfric,
            "inertia": self.pen_inertia,
            "torque_limit": self.pen_torque_limit,
        }

        self.env_args = {
            "max_steps": max_steps,
            "reward_type": reward_type,
            "dt": dt,
            "integrator": integrator,
            "state_representation": state_representation,
            "validation_limit": validation_limit,
            "scale_action": True,
            "random_init": random_init,
        }

        self.single_env = SimplePendulumEnv(self.simulator, **self.env_args)

        self.env = SubprocVecEnv([make_env(self.simulator_args, self.env_args)
                                  for _ in range(n_envs)])

    def init_agent(self,
                   replay_buffer_size=50000,
                   actor=None,
                   critic=None,
                   discount=0.99,
                   actor_lr=0.0005,
                   critic_lr=0.001,
                   tau=0.005):

        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size,
                                          num_states=self.env.observation_space.shape[0],
                                          num_actions=self.env.action_space.shape[0])

        if actor is None:
            actor = get_actor(self.env.observation_space.shape,
                              float(self.env.action_space.low),
                              verbose=True)
            target_actor = get_actor(self.env.observation_space.shape,
                                     float(self.env.action_space.high),
                                     verbose=False)
        else:
            target_actor = tf.keras.models.clone_model(actor)

        if critic is None:
            critic = get_critic(self.env.observation_space.shape,
                                self.env.action_space.shape[0],
                                verbose=True)
            target_critic = get_critic(self.env.observation_space.shape,
                                       self.env.action_space.shape[0],
                                       verbose=False)
        else:
            target_critic = tf.keras.models.clone_model(critic)

        self.agent = Agent(state_shape=self.env.observation_space.shape,
                           n_actions=self.env.action_space.shape[0],
                           action_limits=(self.env.action_space.low,self.env.action_space.high),
                           discount=discount,
                           actor_lr=actor_lr,
                           critic_lr=critic_lr,
                           actor_model=actor,
                           critic_model=critic,
                           target_actor_model=target_actor,
                           target_critic_model=target_critic,
                           tau=tau)

    def train(self, n_episodes, verbose=True):
        rewards = []
        actor_losses = []
        critic_losses = []

        validation_criterion_passed = False
        for episode in range(n_episodes):
            (total_reward,
             actor_mean_loss,
             critic_mean_loss,
             steps,
             final_state,
             success) = self._train_1_episode()

            rewards.append(total_reward)
            actor_losses.append(actor_mean_loss)
            critic_losses.append(critic_mean_loss)

            if verbose:
                print("Episode: {}".format(episode), end="")
                print(", Steps: {}".format(steps), end="")
                print(", Reward: {}".format([round(rew, 2) for rew in total_reward], end=""))
                print(", Actor Loss: {}".format(round(actor_mean_loss, 5)),
                      end="")
                print(", Critic Loss: {}".format(round(critic_mean_loss, 5)),
                      end="")
                print(", Final State: {}".format(final_state), end="")
                print(", Success: {}".format(success))

            if episode > 0:
                if self.validate and episode % self.validate_every == 0:
                    validation_criterion_passed = self._validate()

            if validation_criterion_passed:
                if verbose:
                    print("Validation criterion passed, stopping early.")
                break

        return rewards, actor_losses, critic_losses

    def _train_1_episode(self):
        '''
        Train every episodes_per_epoch's episodes.
        '''
        rewards = []
        actor_loss_list = []
        critic_loss_list = []

        state = self.env.reset()
        for step in range(self.max_steps):
            action = np.array([self.agent.get_action(s, self.noise_object[i]) for i,s in enumerate(state)]) # action is a list of actions for each enviroment
            next_state, reward, done, _ = self.env.step(action)
            for i in range(len(state)):
                self.replay_buffer.append((state[i],
                                        action[i],
                                        next_state[i],
                                        reward[i],
                                        done[i]))
                rewards.append(reward[i])

            if step % self.train_every_steps == 0:
                if self.replay_buffer.size >= 20*self.batch_size:
                    print("Training at step ", str(step),
                          "/", str(self.max_steps),
                          "state env1: ", state[0],
                          "action env1: ", action[0],
                          "                ",
                          end="\r")
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    actor_loss, critic_loss = self.agent.train_on(batch)
                    actor_loss_list.append(actor_loss)
                    critic_loss_list.append(critic_loss)
                    self.agent.update_target_weights()

            if done.any(): # Break if any parallel env is done
                break
            else:
                state = next_state
        final_state = np.array([np.arctan2(next_state[i][1], next_state[i][0]) for i in range(self.n_envs)])
        success = self.single_env.is_goal(state[0]) #Only check if first env is in goal

        return [sum(rewards[i::self.n_envs]) for i in range(self.n_envs)], np.sum(actor_loss_list), \
            np.sum(critic_loss_list), step, \
            final_state, success

    def _validate(self):
        validation_rewards = []
        final_states = []

        for ep in range(self.validation_reps):
            print("Validation Episode ", str(ep),
                  "/", str(self.validation_reps), end="\r")
            episode_rewards = []
            if ep <= (self.validation_reps / 2.0):
                state = self.single_env.reset(random_init="False")
            else:
                state = self.single_env.reset(random_init="start_vicinity")

            for step in range(self.max_steps):
                action = self.agent.get_action(state, None)
                next_state, reward, done, _ = self.single_env.step(action)

                episode_rewards.append(reward)
                state = next_state

                if done:
                    break
            validation_rewards.append(np.sum(episode_rewards))
            final_states.append(state)

        return self.single_env.validation_criterion(validation_rewards, final_states)

    def save(self, path):
        self.agent.save_model(path)

    def load(self, path):
        self.agent.load_model(path)
