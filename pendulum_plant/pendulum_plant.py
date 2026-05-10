"""
Pendulum Plant
==============
"""


import numpy as np
import yaml
import wget
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as mplanimation
#import zerorpc
import wget
import subprocess
import os
import time
import ast
from pathlib import Path




class PendulumPlant:
    def __init__(self, mass=1.0, length=0.5, damping=0.1, gravity=9.81,
                 coulomb_fric=0.0, inertia=None, torque_limit=np.inf):

        """
        The PendulumPlant class contains the kinematics and dynamics
        of the simple pendulum.

        The state of the pendulum in this class is described by
            state = [angle, angular velocity]
            (array like with len(state)=2)
            in units: rad and rad/s
        The zero state of the angle corresponds to the pendulum hanging down.
        The plant expects an actuation input (tau) either as float or
        array like in units Nm.
        (in which case the first entry is used (which should be a float))

        Parameters
        ----------
        mass : float, default=1.0
            pendulum mass, unit: kg
        length : float, default=0.5
            pendulum length, unit: m
        damping : float, default=0.1
            damping factor (proportional to velocity), unit: kg*m/s
        gravity : float, default=9.81
            gravity (positive direction points down), unit: m/s^2
        coulomb_fric : float, default=0.0
            friction term, (independent of magnitude of velocity), unit: Nm
        inertia : float, default=None
            inertia of the pendulum (defaults to point mass inertia)
            unit: kg*m^2
        torque_limit: float, default=np.inf
            maximum torque that the motor can apply, unit: Nm
        """

        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity
        self.coulomb_fric = coulomb_fric
        if inertia is None:
            self.inertia = mass*length*length
        else:
            self.inertia = inertia

        self.torque_limit = torque_limit

        self.dof = 1
        self.n_actuators = 1
        self.base = [0, 0]
        self.n_links = 1
        self.workspace_range = [[-1.2*self.l, 1.2*self.l],
                                [-1.2*self.l, 1.2*self.l]]

    def load_params_from_file(self, filepath):
        """
        Load the pendulum parameters from a yaml file.

        Parameters
        ----------
        filepath : string
            path to yaml file
        """

        with open(filepath, 'r') as yaml_file:
            params = yaml.safe_load(yaml_file)
        self.m = params["mass"]
        self.l = params["length"]
        self.b = params["damping"]
        self.g = params["gravity"]
        self.coulomb_fric = params["coulomb_fric"]
        self.inertia = params["inertia"]
        self.torque_limit = params["torque_limit"]
        self.dof = params["dof"]
        self.n_actuators = params["n_actuators"]
        self.base = params["base"]
        self.n_links = params["n_links"]
        self.workspace_range = [[-1.2*self.l, 1.2*self.l],
                                [-1.2*self.l, 1.2*self.l]]

    def forward_kinematics(self, pos):

        """
        Computes the forward kinematics.

        Parameters
        ----------
        pos : float, angle of the pendulum

        Returns
        -------
        list : A list containing one list (for one end-effector)
              The inner list contains the x and y coordinates
              for the end-effector of the pendulum
        """

        ee_pos_x = float(self.l * np.sin(pos))
        ee_pos_y = float(-self.l * np.cos(pos))
        return [[ee_pos_x, ee_pos_y]]

    def inverse_kinematics(self, ee_pos):

        """
        Comutes inverse kinematics

        Parameters
        ----------
        ee_pos : array like,
            len(state)=2
            contains the x and y position of the end_effector
            floats, units: m

        Returns
        -------
        pos : float
            angle of the pendulum, unit: rad
        """

        pos = np.arctan2(ee_pos[0]/self.l, ee_pos[1]/(-1.0*self.l))
        return pos

    def forward_dynamics(self, state, tau):

        """
        Computes forward dynamics

        Parameters
        ----------
        state : array like
            len(state)=2
            The state of the pendulum [angle, angular velocity]
            floats, units: rad, rad/s
        tau : float
            motor torque, unit: Nm

        Returns
        -------
            - float, angular acceleration, unit: rad/s^2
        """

        torque = np.clip(tau, -np.asarray(self.torque_limit),
                         np.asarray(self.torque_limit))

        accn = (torque - self.m * self.g * self.l * np.sin(state[0]) -
                self.b * state[1] -
                np.sign(state[1]) * self.coulomb_fric) / self.inertia
        return accn

    def inverse_dynamics(self, state, accn):

        """
        Computes inverse dynamics

        Parameters
        ----------
        state : array like
            len(state)=2
            The state of the pendulum [angle, angular velocity]
            floats, units: rad, rad/s
        accn : float
            angular acceleration, unit: rad/s^2

        Returns
        -------
        tau : float
            motor torque, unit: Nm
        """

        tau = accn * self.inertia + \
            self.m * self.g * self.l * np.sin(state[0]) + \
            self.b*state[1] + np.sign(state[1]) * self.coulomb_fric
        return tau

    def rhs(self, t, state, tau):

        """
        Computes the integrand of the equations of motion.

        Parameters
        ----------
        t : float
            time, not used (the dynamics of the pendulum are time independent)
        state : array like
            len(state)=2
            The state of the pendulum [angle, angular velocity]
            floats, units: rad, rad/s
        tau : float or array like
            motor torque, unit: Nm

        Returns
        -------
        res : array like
              the integrand, contains [angular velocity, angular acceleration]
        """

        if isinstance(tau, (list, tuple, np.ndarray)):
            torque = tau[0]
        else:
            torque = tau

        accn = self.forward_dynamics(state, torque)

        res = np.zeros(2*self.dof)
        res[0] = state[1]
        res[1] = accn
        return res

    def potential_energy(self, state):
        Epot = self.m*self.g*self.l*(1-np.cos(state[0]))
        return Epot

    def kinetic_energy(self, state):
        Ekin = 0.5*self.m*(self.l*state[1])**2.0
        return Ekin

    def total_energy(self, state):
        E = self.potential_energy(state) + self.kinetic_energy(state)
        return E

    def activate_hardware(self):
        """
        Activate the pendulum hardware
        """    
        import pyCandle

        # Create CANdle object and set FDCAN baudrate to 1Mbps
        self.candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M,True)

        # Ping FDCAN bus in search of drives
        ids = self.candle.ping()

        # Add all found to the update list
        for id in ids:
            self.candle.addMd80(id)


    
    def run_on_hardware(self, tf, dt, controller=None, user_token = None, preparation_time = 0.0):

        import time
        
        if user_token is None:
            from cloudpendulumlocal.cloud_pendulum_local import Client
            user_token = ""
        else:
            from cloudpendulumclient.client import Client

        self.c = Client()
        
        exp_hash, self.live_url = self.c.start_experiment(
            user_token = user_token,
            experiment_type = "SimplePendulum",
            experiment_time = tf,
            preparation_time = preparation_time,
            record = True
        )
        print("Your experiment hash key is:", exp_hash)
        time.sleep(0.5)
        self.c.set_impedance_controller_params(0.0, 0.0, exp_hash)
    
        tau_scaling = 1.0

        n = int(tf / dt)

        meas_time_vec = np.zeros(n)
        meas_pos = np.zeros(n)
        meas_vel = np.zeros(n)
        meas_tau = np.zeros(n)
        des_tau = np.zeros(n)

        # defining runtime variables
        i = 0
        meas_dt = 0.0
        meas_time = 0.0

        print("Control Loop Started!")

        # Control loop
        while meas_time < tf and i < n:
            start_loop = time.time()
            meas_time += meas_dt
            
            ## Do your stuff here - START
            measured_position = self.c.get_position(exp_hash)
            measured_velocity = self.c.get_velocity(exp_hash)
            measured_torque = self.c.get_torque(exp_hash)
            
            self.x = np.array([measured_position, measured_velocity])
            
            # Control logic
            if controller is not None:
                tau = controller.get_control_output(self.x[0], self.x[1])
                tau_scaled = tau*tau_scaling    # physical torque to motor torque
                self.c.set_torque(tau_scaled, exp_hash)
            else:
                tau = 0                
                       
            # Collect data for plotting
            meas_time_vec[i] = meas_time
            meas_pos[i] = measured_position
            meas_vel[i] = measured_velocity    
            meas_tau[i] = measured_torque/tau_scaling
            des_tau[i] = tau 
                
            ## Do your stuff here - END
            
            i += 1
            exec_time = time.time() - start_loop
            if exec_time > dt:
                print("Control loop is too slow!")
                print("Control frequency:", 1/exec_time, "Hz")
                print("Desired frequency:", 1/dt, "Hz")
                print()
            while time.time() - start_loop < dt:
                pass
            meas_dt = time.time() - start_loop
        print("Control Loop Ended!")
        
        download_url = self.c.stop_experiment(exp_hash)
        print("Experiment Finished!")
        
        filename = wget.download(download_url,".")
        self.vod_filepath = f'{Path(filename).stem}.mp4'
        self.convert_flv_to_mp4(f'{filename}', self.vod_filepath)
        
        self.t_values = meas_time_vec
        self.x_values = np.vstack((meas_pos, meas_vel)).T
        self.tau_values = meas_tau
        self.des_tau_values = des_tau
        
        return self.t_values, self.x_values, self.tau_values, self.des_tau_values, self.vod_filepath

    def convert_flv_to_mp4(self, input_path, output_path):
        """
        Convert an FLV file to MP4 using FFmpeg.
    
        :param input_path: Path to the input FLV file.
        :param output_path: Path to the output MP4 file.
        """
        command = [
            "ffmpeg",
            "-i", input_path,    # Input file
            "-c:v", "copy",      # Copy video stream
            "-c:a", "copy",      # Copy audio stream
            output_path          # Output file
        ]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode == 0:
            print(f"Conversion successful: {output_path}")
        else:
            print(f"Error during conversion: {process.stderr.decode()}")
