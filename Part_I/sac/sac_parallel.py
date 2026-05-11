import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import SubprocVecEnv

from pendulum_plant.pendulum_plant import PendulumPlant
from pendulum_plant.simulation import Simulator
from pendulum_plant.gym_environment import SimplePendulumEnv


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
            "random_init": "start_vicinity",
            "state_target_epsilon": state_target_epsilon,
        }
        

        self.env = SubprocVecEnv([make_env(self.simulator_args, self.env_args)
                                  for _ in range(n_envs-1)] + [make_env(self.simulator_args, self.env_eval_args)])

        # Evaluation env is single-process
        eval_simulator = Simulator(plant=PendulumPlant(**self.simulator_args))
        
        self.eval_env = SimplePendulumEnv(simulator=eval_simulator,
                                          **self.env_eval_args)

    def init_agent(self,
                   learning_rate=0.0003,
                   warm_start=False,
                   warm_start_path="",
                   verbose=1):
        tensorboard_log = os.path.join(self.log_dir, "tb_logs")

        self.agent = SAC(MlpPolicy,
                         self.env,
                         verbose=verbose,
                         tensorboard_log=tensorboard_log,
                         learning_rate=learning_rate,
                         #batch_size=64,
                         #gradient_steps=8,
                         #learning_starts=500,
                         #ent_coef='auto',
                         )

        if warm_start:
            self.agent.set_parameters(load_path_or_dict=warm_start_path)

    def train(self,
              training_timesteps=1e6,
              reward_threshold=1000.0,
              eval_frequency=10000,
              n_eval_episodes=20,
              verbose=1):
        
        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=reward_threshold,
            verbose=verbose)

        log_path = os.path.join(self.log_dir, 'best_model')

        eval_callback = EvalCallback(
            self.eval_env,
            callback_on_new_best=callback_on_best,
            best_model_save_path=log_path,
            log_path=log_path,
            eval_freq=eval_frequency,
            verbose=verbose,
            n_eval_episodes=n_eval_episodes)

        self.agent.learn(total_timesteps=int(training_timesteps),
                         callback=eval_callback)
