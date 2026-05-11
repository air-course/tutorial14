import os
import csv
import numpy as np
from datetime import datetime
import time

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import SubprocVecEnv

from pendulum_plant.pendulum_plant import PendulumPlant
from pendulum_plant.simulation import Simulator
from pendulum_plant.gym_environment import SimplePendulumEnv
from stable_baselines3.common.logger import configure

import multiprocessing as mp

def make_env(simulator_args, env_args):
    """
    Create single instance of enviroment.
    """
    def _init():
        print("Starting env")
        pendulum = PendulumPlant(**simulator_args)
        simulator = Simulator(plant=pendulum)
        env = SimplePendulumEnv(simulator=simulator, **env_args) 
        original_reset = env.reset
        def reset_wrapper(*args, **kwargs):
            return original_reset(*args, random_init="everywhere", **kwargs)
        env.reset = reset_wrapper
        return env
    return _init



class sac_trainer():
    """
    Class to train a policy for pendulum swingup with the
    state actor critic (sac) method in parallel.
    """
    def __init__(self, log_dir="sac_training"):
        """
        Class to train a policy for pendulum swingup with the
        state actor critic (sac) method.

        Parameter
        ---------
        log_dir : string, default="sac_training"
            path to directory where results and log data will be stored
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.best_mean_reward = -np.inf


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
                         reward_type="soft_binary_with_repellor",
                         state_representation=2,
                         validation_limit=-150,
                         target=[np.pi, 0.0],
                         state_target_epsilon=[1e-2, 1e-2],
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
        """
        
        # Store common simulator and environment arguments
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
            "state_target_epsilon": state_target_epsilon,
        }

        self.env_eval_args = {
            "max_steps": max_steps,
            "reward_type": reward_type,
            "dt": dt,
            "integrator": integrator,
            "state_representation": state_representation,
            "validation_limit": validation_limit,
            "scale_action": True,
            "random_init": "False",
            "state_target_epsilon": state_target_epsilon,
        }
        
        #self.env = SubprocVecEnv([make_env(self.simulator_args, self.env_args)
        #                         for _ in range(n_envs-1)] + [make_env(self.simulator_args, self.env_eval_args)])
        self.env = SubprocVecEnv([make_env(self.simulator_args, self.env_args)
                                  for _ in range(n_envs)])
        # Evaluation env is single-process
        eval_simulator = Simulator(plant=PendulumPlant(**self.simulator_args))
        
        self.eval_env = SimplePendulumEnv(simulator=eval_simulator,
                                          **self.env_eval_args)

    def init_agent(self,
                   learning_rate=0.0003,
                   warm_start=False,
                   warm_start_path="",
                   device="auto",
                   
                   verbose=1):
        tensorboard_log = os.path.join(self.log_dir, "tb_logs")

        self.agent = SAC(MlpPolicy,
                         self.env,
                         verbose=verbose,
                         device=device,
                         tensorboard_log=tensorboard_log,
                         learning_rate=learning_rate,
                         train_freq=1,         # still needed but unused in this flow
                         gradient_steps=1,     # overridden manually
                         #ent_coef=4,
                         learning_starts=0)    # allow training from first episode
        
        self.agent._logger = configure(folder=None, format_strings=[])  # disables logging
        self.agent._current_progress_remaining = 1.0  # still required by LR schedule

        if warm_start:
            self.agent.set_parameters(load_path_or_dict=warm_start_path)


    def run_episode(self, env=None, deterministic=False):
        """
        Run a full episode using the current SAC policy.
        Returns transitions and total reward.
        """
        if env is None:
            env = self.env

        obs = env.reset()
        done = np.array([False])
        episode_reward = 0.0
        transitions = []
        while not done.all():
            action, _ = self.agent.predict(obs, deterministic=deterministic)
            next_obs, reward, done, info = env.step(action)
            transitions.append((obs, action, reward, next_obs, done, info))
            obs = next_obs
            episode_reward += reward

        return transitions, episode_reward



        
    
    def store_transitions(self, transitions):
        """
        Add each (s, a, r, s', done) tuple to the replay buffer.
        """
        for obs, action, reward, next_obs, done, info in transitions:
            self.agent.replay_buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                infos=info
            )
    def evaluate_agent(self, n_episodes=5):
        """
        Run evaluation episodes using a separate eval_env.
        Returns mean reward.
        """
        episode_rewards = []

        for _ in range(n_episodes):
            obs = self.eval_env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, reward, done, _ = self.eval_env.step(action)
                total_reward += reward
            episode_rewards.append(total_reward)

        mean_reward = np.mean(episode_rewards)
        return mean_reward

    def train_after_episode(self, gradient_steps=200, batch_size=256):
        """
        Train SAC agent using stored replay buffer samples.
        """
        self.agent.train(batch_size=batch_size,
                         gradient_steps=gradient_steps)
    
    
    def train(self, n_episodes=100, gradient_steps=200, batch_size=256, eval_every=5, eval_episodes=5, start_training=2000,  save_path="best_model", reward_limit=21_000):
        """
        Run one full episode at a time and train after each rollout.
        Suitable for real hardware integration.
        """
        experience_collection_time = 0.
        policy_update_time = 0.
        evaluation_time = 0.
        self.save_path = save_path
        # Initialize csv for reward logging
        train_log_path = os.path.join(save_path, "train_rewards.csv")
        eval_log_path = os.path.join(save_path, "eval_rewards.csv")
        runtime_log_path = os.path.join(os.path.dirname(save_path), "runtime_log.csv")
        
        if not os.path.exists(train_log_path):
            with open(train_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "env_index", "reward", "timestamp"])
        
        if not os.path.exists(eval_log_path):
            with open(eval_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "eval_index", "reward", "timestamp"])

        if not os.path.exists(runtime_log_path):
            with open(runtime_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Experience collecting", "Policy Updates", "Evaluation"])
        
        for ep in range(n_episodes):

            start_time = time.time()
            transitions, ep_reward = self.run_episode()
            self.store_transitions(transitions)
            experience_collection_time+= time.time() - start_time

            start_time = time.time()
            print(f"[Ep {ep}, Rewards: {ep_reward}, Transitions: {len(transitions)}, Initial: {transitions[0][0]} ")
            if self.agent.replay_buffer.size() > start_training:
                self.train_after_episode(gradient_steps, batch_size)
            if self.agent.log_ent_coef != None:
                print(f"Enthropy coefficent: {np.exp(self.agent.log_ent_coef.item())} ")
            avg_reward = np.mean(ep_reward)
            
            # Log training rewards
            with open(train_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                timestamp = datetime.now().strftime("%H:%M:%S")
                for i, r in enumerate(ep_reward):
                    writer.writerow([ep, i, r, timestamp])
            policy_update_time+=time.time() - start_time


            if ep % eval_every == 0 and ep>0:
                start_time = time.time()
                eval_reward = self.evaluate_agent(eval_episodes)
                print(f"Evaluation Reward: {eval_reward}")

                # Log evaluation reward
                with open(eval_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    writer.writerow([ep, 0, eval_reward, timestamp])

                if eval_reward >= self.best_mean_reward:
                    self.best_mean_reward = eval_reward
                    self.agent.save(os.path.join(save_path, "best_model"))
                    print("New best model saved!")
                
                if eval_reward > reward_limit:
                    print("Successful evaluation, ending training early")
                    break
                evaluation_time+=time.time()-start_time


        print("Training done, saving model!")
        print(f"\nExperince collecting time: {experience_collection_time}, \nPolicy Update time: {policy_update_time}")
        with open(runtime_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([experience_collection_time, policy_update_time, evaluation_time])

        