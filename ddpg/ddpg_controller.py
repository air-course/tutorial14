"""
DDPG Controller
===============
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from ddpg.abstract_controller import AbstractController




class ddpg_controller(AbstractController):
    """
    DDPG controller class
    """
    def __init__(self, model_path, torque_limit, state_representation=3):
        self.model = load_model(model_path, compile=False)
        self.torque_limit = torque_limit
        self.state_representation = state_representation

        if state_representation == 2:
            # state is [th, th, vel]
            self.low = np.array([-6*2*np.pi, -20])
            self.high = np.array([6*2*np.pi, 20])
        elif state_representation == 3:
            # state is [cos(th), sin(th), vel]
            self.low = np.array([-1., -1., -20.])
            self.high = np.array([1., 1., 20.])
            
    @tf.function
    def fast_predict(self,x):
        return self.model(x)
        
    def get_control_output(self, meas_pos, meas_vel, meas_tau=0, meas_time=0):

        pos = float(np.squeeze(meas_pos))
        vel = float(np.squeeze(meas_vel))

        state = np.array([pos, vel])
        observation = self.get_observation(state)
        #control_output = self.model({'input_layer_1': np.atleast_2d(observation)}, verbose=0, training=False)
        control_output = self.fast_predict(tf.convert_to_tensor(np.atleast_2d(observation), dtype=tf.float32))
        control_output *= self.torque_limit

        control_output = np.clip(control_output,
                                 -self.torque_limit,
                                 self.torque_limit)
        #if np.abs(meas_vel)>=19:,
        #    return None, None, 0.0
        return float(control_output)

    def get_observation(self, state):
        st = np.copy(state)
        st[1] = np.clip(st[1], self.low[-1], self.high[-1])
        if self.state_representation == 2:
            observation = np.array([obs for obs in st], dtype=np.float32)
        elif self.state_representation == 3:
            observation = np.array([np.cos(st[0]),
                                    np.sin(st[0]),
                                    st[1]],
                                   dtype=np.float32)

        return observation
