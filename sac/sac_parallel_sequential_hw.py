import os
import time
from datetime import datetime
import numpy as np
import pickle
import threading
import csv

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.logger import configure

from pendulum_plant.pendulum_plant import PendulumPlant
from pendulum_plant.simulation import Simulator
from pendulum_plant.gym_environment_hw import SimplePendulumEnv
from sac.sac_controller import SacController



def make_env(env_id, env_args):
    """
    Factory function for creating environments.
    """
    def _init():
        env = SimplePendulumEnv(
            **env_args
        )
        return env
    return _init



class sac_trainer():
    """
    Class to train a policy for pendulum swingup with the
    state actor critic (sac) method in parallel.
    """
    def __init__(self, log_dir="sac_training", verbose=1):
        """
        Class to train a policy for pendulum swingup with the
        state actor critic (sac) method.

        Parameter
        ---------
        log_dir : string, default="sac_training"
            path to directory where results and log data will be stored     
        verbose : int, default=1
            Verbosity level of model and training.
        """
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
        self.verbose=verbose

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

        self.simulator_args = {
            "mass": self.pen_mass,
            "length": self.pen_length,
            "damping": self.pen_damping,
            "gravity": self.pen_gravity,
            "coulomb_fric": self.pen_cfric,
            "inertia": self.pen_inertia,
            "torque_limit": self.pen_torque_limit,
        }

        self.pendulum = PendulumPlant(mass=self.pen_mass,
                                      length=self.pen_length,
                                      damping=self.pen_damping,
                                      gravity=self.pen_gravity,
                                      coulomb_fric=self.pen_cfric,
                                      inertia=self.pen_inertia,
                                      torque_limit=self.pen_torque_limit)

        self.simulator = Simulator(plant=self.pendulum)

    def init_environment(self,
                         user_token,
                         dt=0.01,
                         max_steps=1000,
                         reward_type="soft_binary_with_repellor",
                         state_representation=2,
                         validation_limit=-150,
                         target=[np.pi, 0.0],
                         state_target_epsilon=[1e-2, 1e-2],
                         random_init="everywhere",
                         random_init_eval="False",
                         torque_limit=0.02,
                         dt_step_scaling=0.8):
        """
        Initialize the environment.

        Parameters
        ---------
        user_token : str
            User token required to run experiments on cloud pendulum.
        dt : float, default=0.01
            time step [s]
        max_steps : int, default=1000
            maximum number of timesteps for one training episode
            i.e. One episode lasts at most max_step*dt seconds
        reward_type : string, default=soft_binary_with_repellor
            string which defines the reward function
            options are: 'continuous', 'discrete', 'soft_binary',
                         'soft_binary_with_repellor', 'combined_reward'
        state_representation : int, default=2
            determines how the state space of the pendulum is represented
            state_representation=2 means state = [position, velocity]
            state_representation=3 means state = [cos(position),
                                                  sin(position),
                                                  velocity]
        validation_limit : float, default=-150
            Lower bound on cumulative reward to consider an episode valid.
        target : array-like, default=[np.pi, 0.0]
            The target state of the pendulum
        state_target_epsilon : array-like, default=[1e-2, 1e-2]
            In this vicinity the target counts as reached.
        random_init : string, default="everywhere"
            Method for choosing initial state.
        random_init_eval : string, default="False"
            Method for choosing initial state during evaluation.
        torque_limit: float, default=0.02
            the torque_limit of the pendulum actuator
        dt_step_scaling : float, default=0.8
            Fraction of the simulation timestep (`dt`) to execute before returning the next state.  
            Allows time for computation or control signal processing.
        """
        
        # Store environment arguments
        self.env_args = {
            "user_token": user_token,
            "max_steps": max_steps,
            "reward_type": reward_type,
            "dt": dt,
            "state_representation": state_representation,
            "validation_limit": validation_limit,
            "scale_action": True,
            "random_init": random_init,
            "target": target,
            "state_target_epsilon": state_target_epsilon,
            "torque_limit": torque_limit,
            "dt_step_scaling": dt_step_scaling,
            "verbose": self.verbose
        }
        self.eval_env_args = {**self.env_args, 
                              "random_init": random_init_eval}
        
        self.env =  SimplePendulumEnv(**self.env_args)
        
        self.eval_env = SimplePendulumEnv(**self.eval_env_args)

    def init_agent(self,
                   learning_rate=0.0003,
                   warm_start=False,
                   warm_start_path=""):
        """
        Initialize the SAC agent.

        Parameters
        ----------
        learning_rate : float, default=0.0003
            The learning rate used by the optimizer.
        
        warm_start : bool, default=False
            If True, loads model parameters from `warm_start_path`.

        warm_start_path : str, default=""
            Path to the pre-trained model to load when warm_start is True.
        """
        tensorboard_log = os.path.join(self.log_dir, "tb_logs")
        self.agent = SAC(MlpPolicy,
                         self.env,
                         verbose=self.verbose,
                         tensorboard_log=tensorboard_log,
                         learning_rate=learning_rate,
                         train_freq=1,         # still needed but unused in this flow
                         gradient_steps=1,     # overridden manually
                         #ent_coef=2,
                         learning_starts=0)    # allow training from first episode
        self.agent._logger = configure(folder=None, format_strings=[])  # disables logging
        self.agent._current_progress_remaining = 1.0  # still required by LR schedule

        if warm_start:
            self.agent.set_parameters(load_path_or_dict=warm_start_path)
    
    
        
    def _run_episode_thread(self, env, deterministic, result_list, index):
        """
        Worker function to run one episode and store result at result_list[index].
        """
        if self.verbose >0:
            print(f"Thread {index}: Env Reset")
        obs = env.reset()
        done = False
        episode_reward = 0.0
        transitions = []
        step_counter = 0
        while not done:
            action, _ = self.agent.predict(obs, deterministic=deterministic)
            next_obs, reward, done, info = env.step(action)
            transitions.append((np.array([obs]), np.array([action]), np.array([reward]), np.array([next_obs]), np.array([done]), np.array([info])))
            obs = next_obs
            episode_reward += reward
            step_counter += 1
        if self.verbose >0:
            print(f"Thread {index}: Episode done, total reward {episode_reward:.2f}")
        result_list[index] = (transitions, float(episode_reward))
    
    def run_parallel_episodes(self, number_of_envs=2, envs=None, deterministic=False):
        """
        Run multiple episodes in parallel across the given list of envs.
        Returns a list of (transitions, episode_reward) tuples, one per env.
        """
        if envs==None:
            env_inits = [make_env(i, {**self.env_args, "env_index": i}) for i in range(number_of_envs)]
            envs = [env_init() for env_init in env_inits]
        num_envs = len(envs)
        results = [None] * num_envs
        threads = []
    
        for i, env in enumerate(envs):
            t = threading.Thread(target=self._run_episode_thread, args=(env, deterministic, results, i))
            threads.append(t)
            t.start()
    
        for t in threads:
            t.join()
        # convert results to correct format
        all_transitions = []
        episode_rewards = []
        for transitions, ep_reward in results:
            all_transitions.extend(transitions)
            episode_rewards.append(ep_reward)
        return all_transitions, episode_rewards




    
    def store_transitions(self, transitions):
        """
        Add each (s, a, r, s', done, info) tuple to the replay buffer.
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
            
    def evaluate_agent(self, n_episodes=5, ep=0):
        """
        Run evaluation episodes using a separate eval_env.
        Returns mean reward.
        """
        episode_rewards = []

        for i in range(n_episodes):
            obs = self.eval_env.reset(record=True)
            done = False
            total_reward = 0.0
            
            while not done:
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, reward, done, _ = self.eval_env.step(action, stop_when_done=False)
                total_reward += reward
            self.eval_env.close(save_video=True, video_path = os.path.join(self.save_path, f"evaluation_videos/evaluation_{ep}_{i}.mp4"))
            episode_rewards.append(total_reward)
            # Save video
            
            

        mean_reward = np.mean(episode_rewards)
        return mean_reward

    def train_after_episode(self, gradient_steps=200):
        """
        Train SAC agent using stored replay buffer samples.
        """
        self.agent.train(batch_size=self.agent.batch_size,
                         gradient_steps=gradient_steps)
    
    
    def train(self, n_episodes=100, gradient_steps=500, eval_every=5, eval_episodes=1, start_training=2000,  save_path="best_model", number_of_envs=2):
        """
        Train the agent by running full episodes on cloud pendulum and updating after each rollout.

        Parameters
        ----------
        n_episodes : int, default=100
            Number of episodes to run for training.

        gradient_steps : int, default=500
            Number of gradient steps to perform after each episode.

        eval_every : int, default=5
            Frequency (in episodes) at which the agent is evaluated.

        eval_episodes : int, default=1
            Number of evaluation episodes to average over during evaluation.

        start_training : int, default=2000
            Minimum number of transitions in the replay buffer before training begins.

        save_path : str, default="best_model"
            Directory path to save the best-performing model and replay buffer.

        number_of_envs : int, default=2
            Number of parallel environments to use for collecting transitions.
        """
        experience_collection_time = 0.
        policy_update_time = 0.
        evaluation_time = 0.
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "evaluation_videos"), exist_ok=True)
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
            transitions, ep_reward = self.run_parallel_episodes(number_of_envs=number_of_envs)
            experience_collection_time+= time.time() - start_time
            
            start_time = time.time()
            self.store_transitions(transitions)
            print(f"[Ep {ep}, Rewards: {ep_reward}, Transitions: {len(transitions)}, Initial: {transitions[0][0]} ")
            
            if self.agent.replay_buffer.size() > start_training:
                self.train_after_episode(gradient_steps)
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
                eval_reward = self.evaluate_agent(eval_episodes, ep)
                print(f"Evaluation Reward: {eval_reward}")

                #Save replay buffer
                with open(os.path.join(save_path,"replay_buffer.pkl"), "wb") as f:
                    pickle.dump(self.agent.replay_buffer, f)

                # Log evaluation reward
                with open(eval_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    writer.writerow([ep, 0, eval_reward, timestamp])
                    
                #Save new best model
                if eval_reward >= self.best_mean_reward:
                    self.best_mean_reward = eval_reward
                    self.agent.save(os.path.join(save_path, "best_model"))
                    print("New best model saved!")
                
                evaluation_time+=time.time()-start_time
            
            

        print("Training done, saving model!")
        print(f"\nExperince collecting time: {experience_collection_time}, \nPolicy Update time: {policy_update_time}")
        with open(runtime_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([experience_collection_time, policy_update_time, evaluation_time])