"""
DDPG Trainer
============
"""


import numpy as np
import tensorflow as tf

from pendulum_plant.pendulum_plantzeroRPC import PendulumPlant
from pendulum_plant.simulation import Simulator
from pendulum_plant.gym_environment import SimplePendulumEnv

from ddpg.replay_buffer import ReplayBuffer
from ddpg.models import get_actor, get_critic
from ddpg.agent import Agent
from ddpg.noise import OUActionNoise
from ddpg.ddpg_controller import ddpg_controller
import csv
import os

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

        self.noise_object = OUActionNoise(mean=np.zeros(1),
                                          std_deviation=0.2*np.ones(1)) # changed from 0.2

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
                                      #coulomb_fric=self.pen_cfric,
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
                         random_init="everywhere"):
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

        self.env = SimplePendulumEnv(simulator=self.simulator,
                                     max_steps=max_steps,
                                     target=target,
                                     state_target_epsilon=state_target_epsilon,
                                     reward_type=reward_type,
                                     dt=dt,
                                     integrator=integrator,
                                     state_representation=state_representation,
                                     validation_limit=validation_limit,
                                     scale_action=scale_action,
                                     random_init=random_init)

    def init_agent(self,
                   replay_buffer_size=50000,
                   actor=None,
                   critic=None,
                   discount=0.99,
                   actor_lr=0.0005,
                   critic_lr=0.001,
                   tau=0.005):

        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size,
                                          num_states=self.env.n_states,
                                          num_actions=self.env.n_actions)

        if actor is None:
            actor = get_actor(self.env.state_shape,
                              float(self.env.action_space.high),
                              verbose=True)
            target_actor = get_actor(self.env.state_shape,
                                     float(self.env.action_space.high),
                                     verbose=False)
        else:
            target_actor = tf.keras.models.clone_model(actor)

        if critic is None:
            critic = get_critic(self.env.state_shape,
                                self.env.n_actions,
                                verbose=True)
            target_critic = get_critic(self.env.state_shape,
                                       self.env.n_actions,
                                       verbose=False)
        else:
            target_critic = tf.keras.models.clone_model(critic)

        self.agent = Agent(state_shape=self.env.state_shape,
                           n_actions=self.env.n_actions,
                           action_limits=self.env.action_limits,
                           discount=discount,
                           actor_lr=actor_lr,
                           critic_lr=critic_lr,
                           actor_model=actor,
                           critic_model=critic,
                           target_actor_model=target_actor,
                           target_critic_model=target_critic,
                           tau=tau)

    def train(self, n_episodes, verbose=True, episode_type="sim"):
        rewards = []
        actor_losses = []
        critic_losses = []
        self.episode_type = episode_type

        validation_criterion_passed = False
        for episode in range(n_episodes):
            
            if episode_type == "hw_parallel":
                result = self._train_1_episode_hw_parallel()
            elif episode_type == "sim":
                result = self._train_1_episode()
            elif episode_type == "hw":
                result = self._train_1_episode_hw()
            else:
                raise ValueError(f"Unknown episode_type: {episode_type}")
            
            (total_reward,
             actor_mean_loss,
             critic_mean_loss,
             steps,
             final_state,
             success) = result


            rewards.append(total_reward)
            actor_losses.append(actor_mean_loss)
            critic_losses.append(critic_mean_loss)

            if verbose:
                print("Episode: {}".format(episode), end="")
                print(", Steps: {}".format(steps), end="")
                print(", Reward: {}".format(round(total_reward, 2)), end="")
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
        Run one episode in simulation, then do the training loop multiple times.
        '''
        self.noise_object.reset()
        rewards = []
        actor_loss_list = []
        critic_loss_list = []

        state = self.env.reset(random_init="everywhere")
        for step in range(self.max_steps):
            action = self.agent.get_action(state,
                                           self.noise_object)
            next_state, reward, done, _ = self.env.step(action)

            self.replay_buffer.append((state,
                                       action,
                                       next_state,
                                       reward,
                                       done))
            rewards.append(reward)


            if done:
                break
            else:
                state = next_state
        final_state = self.env.get_state_from_observation(next_state)
        success = self.env.is_goal(state)

        for step in range(self.max_steps):
            if self.replay_buffer.size >= self.batch_size * 10:
                print(f"Training at step {str(step)}/{self.max_steps}",  end="\r")
                batch = self.replay_buffer.sample_batch(self.batch_size)
                actor_loss, critic_loss = self.agent.train_on(batch)
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)
                self.agent.update_target_weights()
                
        return np.sum(rewards), np.sum(actor_loss_list), \
            np.sum(critic_loss_list), step, \
            final_state, success

    
    def append_experiment_to_csv(self, Treal_es, Xreal_es, Ureal_es, Ureal_es_des, csv_path="real_data_log.csv"):
        header = ["time", "pos", "vel", "action", "action_des"]
        write_header = not os.path.exists(csv_path)
    
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
    
            for t, x, u, u_des in zip(Treal_es, Xreal_es, Ureal_es, Ureal_es_des):
                pos, vel = x  # Assuming 2D state
                writer.writerow([t, pos, vel, u, u_des])

    
    def _train_1_episode_hw(self):
        '''
        Run one episode on real pendulum then do the training loop multiple times.
        '''
        self.noise_object.reset()
        rewards = []
        actor_loss_list = []
        critic_loss_list = []
        class Controller(ddpg_controller):
            def __init__(self, agent, torque_limit, state_representation, noise_object):
                self.agent = agent
                self.torque_limit = torque_limit
                self.state_representation = state_representation
                self.noise_object = noise_object

                if state_representation == 2:
                    # state is [th, th, vel]
                    self.low = np.array([-6*2*np.pi, -20])
                    self.high = np.array([6*2*np.pi, 20])
                elif state_representation == 3:
                    # state is [cos(th), sin(th), vel]
                    self.low = np.array([-1., -1., -20.])
                    self.high = np.array([1., 1., 20.])
                

            def get_control_output(self, meas_pos, meas_vel, meas_tau=0, meas_time=0):
                pos = float(np.squeeze(meas_pos))
                vel = float(np.squeeze(meas_vel))

                state = np.array([pos, vel])
                observation = self.get_observation(state)
                control_output = self.agent.get_action(observation, self.noise_object)
                control_output *= self.torque_limit

                control_output = np.clip(control_output,
                                        -self.torque_limit,
                                        self.torque_limit)
                
                # Limit to prevent overspeed
                if np.abs(state[1]) >= 19:
                    return float(0)
                return float(control_output)
            
        # Create an exploring controller
        controller = Controller(self.agent, self.env.torque_limit, 3, self.noise_object)
        # Run an episode on the pendulum
        #def sample_initial_state(angle_range=(-np.pi, np.pi), vel_range=(-1.0, 1.0)):
        #    theta0 = np.random.uniform(*angle_range)
        #    theta_dot0 = np.random.uniform(*vel_range)
        #    return np.array([theta0, theta_dot0])
        #import numpy as np

        def sample_initial_state(angle_range=(-np.pi, np.pi), vel_range=(-1.0, 1.0)):
            theta0 = np.random.uniform(*angle_range)
            theta_dot0 = np.random.uniform(*vel_range)
            return [theta0, theta_dot0]
        
        initial_state = sample_initial_state()
        Treal_es, Xreal_es, Ureal_es, Ureal_es_des, vod_filepath = self.pendulum.run_on_hardware_hash(10,0.05, controller=controller, starting_pos=initial_state[0])

        # Save experiment data to csv
        self.append_experiment_to_csv(Treal_es, Xreal_es, Ureal_es, Ureal_es_des)
        
        # Save to replay buffer
        for step in range(len(Ureal_es)-1):
            state = Xreal_es[step]
            state = self.env.get_observation(state)
            action = Ureal_es[step]
            next_state = Xreal_es[step+1]
            next_state = self.env.get_observation(next_state)
            reward = self.env.swingup_reward(Xreal_es[step], action)
            #print(f"Step {step} | State: {state} | Action: {action} | Next State: {next_state} | Reward: {reward:.3f}")

            done=False
            self.replay_buffer.append((state,
                                       action,
                                       next_state,
                                       reward,
                                       done))
            rewards.append(reward)

        
        final_state = Xreal_es[-1]
        #success = self.env.is_goal(final_state)
        success = False

        # Do repeated loss calculations and updates on model
        for step in range(self.max_steps):
            if self.replay_buffer.size >= self.batch_size * 10:
                print(f"Training at step {str(step)}/{self.max_steps}",  end="\r")
                batch = self.replay_buffer.sample_batch(self.batch_size)
                actor_loss, critic_loss = self.agent.train_on(batch)
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)
                self.agent.update_target_weights()
                
        return np.sum(rewards), np.sum(actor_loss_list), \
            np.sum(critic_loss_list), step, \
            final_state, success
        
    def _train_1_episode_hw_parallel(self):
        '''
        Run one episode on real pendulum then do the training loop multiple times.
        '''
        self.noise_object.reset()
        rewards = []
        actor_loss_list = []
        critic_loss_list = []
        class Controller(ddpg_controller):
            def __init__(self, agent, torque_limit, state_representation, noise_object):
                self.agent = agent
                self.torque_limit = torque_limit
                self.state_representation = state_representation
                self.noise_object = noise_object

                if state_representation == 2:
                    # state is [th, th, vel]
                    self.low = np.array([-6*2*np.pi, -20])
                    self.high = np.array([6*2*np.pi, 20])
                elif state_representation == 3:
                    # state is [cos(th), sin(th), vel]
                    self.low = np.array([-1., -1., -20.])
                    self.high = np.array([1., 1., 20.])
                

            def get_control_output(self, meas_pos, meas_vel, meas_tau=0, meas_time=0):
                pos = float(np.squeeze(meas_pos))
                vel = float(np.squeeze(meas_vel))

                state = np.array([pos, vel])
                observation = self.get_observation(state)
                control_output = self.agent.get_action(observation, self.noise_object)
                control_output *= self.torque_limit

                control_output = np.clip(control_output,
                                        -self.torque_limit,
                                        self.torque_limit)
                
                # Limit to prevent overspeed
                if np.abs(state[1]) >= 19:
                    return float(0)
                return float(control_output)
            
        # Create an exploring controller
        noise1 = OUActionNoise(mean=np.zeros(1), std_deviation=0.2*np.ones(1))
        noise2 = OUActionNoise(mean=np.zeros(1), std_deviation=0.0*np.ones(1))
        noise3 = OUActionNoise(mean=np.zeros(1), std_deviation=1.0*np.ones(1))
        controller1 = Controller(self.agent, self.env.torque_limit, 3, noise1)
        controller2 = Controller(self.agent, self.env.torque_limit, 3, noise2)
        controller3 = Controller(self.agent, self.env.torque_limit, 3, noise3)

        # Run an episode on the pendulum
        #def sample_initial_state(angle_range=(-np.pi, np.pi), vel_range=(-1.0, 1.0)):
        #    theta0 = np.random.uniform(*angle_range)
        #    theta_dot0 = np.random.uniform(*vel_range)
        #    return np.array([theta0, theta_dot0])
        #import numpy as np

        def sample_initial_state(angle_range=(-np.pi, np.pi), vel_range=(-1.0, 1.0)):
            theta0 = np.random.uniform(*angle_range)
            theta_dot0 = np.random.uniform(*vel_range)
            return [theta0, theta_dot0]
        
        starting_pos = [sample_initial_state()[0] for _ in range(3)]
        Treal_es, Xreal_es, Ureal_es, Ureal_es_des, vod_filepath = self.pendulum.run_on_hardware_hash_parallel(10,0.05, controllers=[controller1, controller2, controller3], starting_pos=starting_pos)

        # Save experiment data to csv
        self.append_experiment_to_csv(Treal_es, Xreal_es, Ureal_es, Ureal_es_des)
        
        # Save to replay buffer
        for step in range(len(Ureal_es)-1):
            state = Xreal_es[step]
            state = self.env.get_observation(state)
            action = Ureal_es[step]
            next_state = Xreal_es[step+1]
            next_state = self.env.get_observation(next_state)
            reward = self.env.swingup_reward(Xreal_es[step], action)
            #print(f"Step {step} | State: {state} | Action: {action} | Next State: {next_state} | Reward: {reward:.3f}")

            done=False
            self.replay_buffer.append((state,
                                       action,
                                       next_state,
                                       reward,
                                       done))
            rewards.append(reward)

        final_state = Xreal_es[-1]
        #success = self.env.is_goal(final_state)
        success = False

        # Do repeated loss calculations and updates on model
        for step in range(self.max_steps*3):
            if self.replay_buffer.size >= self.batch_size * 10:
                print(f"Training at step {str(step)}/{self.max_steps}",  end="\r")
                batch = self.replay_buffer.sample_batch(self.batch_size)
                actor_loss, critic_loss = self.agent.train_on(batch)
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)
                self.agent.update_target_weights()
                
        return np.sum(rewards), np.sum(actor_loss_list), \
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
                state = self.env.reset(random_init="False")
            else:
                state = self.env.reset(random_init="start_vicinity")

            for step in range(self.max_steps):
                action = self.agent.get_action(state, None)
                next_state, reward, done, _ = self.env.step(action)

                episode_rewards.append(reward)
                state = next_state
                if self.max_steps-step < 50:
                    final_states.append(state)
                if done:
                    break
            validation_rewards.append(np.sum(episode_rewards))
            

        return self.env.validation_criterion(validation_rewards, final_states)

    def save(self, path):
        self.agent.save_model(path)

    def load(self, path):
        self.agent.load_model(path)
