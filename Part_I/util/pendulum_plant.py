import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as mplanimation
import zerorpc
import wget
import subprocess
import os
import ast
import time
import traceback
from util.misc import download_video
try:
    from cloudpendulumclient.client import Client
    import cloudpendulumclient.data as Data
except:
    print("Warning: Did not find the CloudPendulum Client. Please run on the CloudPendulum to run on the real hardware.")


class PendulumPlant:
    def __init__(self, mass=1.0, length=0.5, damping=0.1, gravity=9.81, inertia=None, torque_limit=np.inf):
        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity
        if inertia is None:
            self.I = mass*length*length
        else:
            self.I = inertia
        self.torque_limit = torque_limit

        self.dof = 1
        self.x = np.zeros(2*self.dof) #position, velocity
        self.t = 0.0 #time

        self.t_values = []
        self.x_values = []
        self.tau_values = []

    def set_state(self, time, x):
        self.x = x
        self.t = time

    def get_state(self):
        return self.t, self.x

    def forward_kinematics(self, pos):
        """
        forward kinematics, origin at fixed point
        """
        ee_pos_x = self.l * np.sin(pos)
        ee_pos_y = -self.l*np.cos(pos)
        return [ee_pos_x, ee_pos_y]

    def forward_dynamics(self, pos, vel, tau):
        """
        return acceleration from current position, velocity and torque.
        use self.m, self.g, self.l, self.b and self.I if needed
        """
        torque = np.clip(tau, -self.torque_limit, self.torque_limit)

        accn = (torque - self.m*self.g*self.l*np.sin(pos) - self.b*vel) / self.I
        
        return accn

    def inverse_dynamics(self, pos, vel, accn):
        """
        return torque acting on the revolute joint (tau) in terms of inputs
        use self.m, self.g, self.l, self.b and self.I if needed
        """
        tau_id = accn*self.I + self.m*self.g*self.l*np.sin(pos) + self.b*vel

        return tau_id

    def rhs(self, t, x, tau):

        """
        Computes the integrand of the equations of motion.
        """
        accn = self.forward_dynamics(x[0], x[1], tau)
        integ = np.array([x[1], accn])
        return integ

    def euler_integrator(self, t, y, dt, tau):
        """
        Implement Forward Euler Integration for a time-step dt and state y
        y = [pos, vel]
        """
        integ = self.rhs(t, y, tau)
        y_new = y + dt*integ
        return y_new

    def runge_integrator(self, t, y, dt, tau):
        """
        Bonus: Implement a fourth order Runge-Kutta Integration scheme
        """
        k1 = self.rhs(t, y, tau)
        k2 = self.rhs(t + 0.5*dt, y + 0.5*dt*k1, tau)
        k3 = self.rhs(t + 0.5*dt, y + 0.5*dt*k2, tau)
        k4 = self.rhs(t + dt, y + dt*k3, tau)
        integ = (k1 + 2*(k2 + k3) + k4) / 6.0

        y_new = y + dt*integ

        return y_new

    def step(self, tau, dt, integrator="euler"):
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        if integrator == "runge_kutta":
            self.x = self.runge_integrator(self.t, self.x, dt, tau)
        elif integrator == "euler":
            self.x = self.euler_integrator(self.t, self.x, dt, tau)
        self.t += dt
        # Store the time series output
        #if len(self.x_values) == 500:
        #    self.x = self.x - [np.pi, 0]
        self.t_values.append(self.t)
        self.x_values.append(self.x.copy())
        self.tau_values.append(tau)

    def simulate(self, t0, y0, tf, dt, controller=None, integrator="euler"):
        self.set_state(t0, y0)
        
        self.t_values = []
        self.x_values = []
        self.tau_values = []
        self.controller = controller
        
        while (self.t <= tf):
            if controller is not None:
                #tau = controller.get_control_output(self.x, self.t)
                tau = controller.get_control_output(self.x)
            else:
                tau = 0
            self.step(tau, dt, integrator=integrator)

        return self.t_values, self.x_values, self.tau_values
    
    def plot_MPC_instant_trajectory(self, x_trj, u_trj, dt, N, i):
        # Plot the trajectory which the system has followed so far
        # as well as the trajectory which has been calculated
        # until the horizon (th)
        ## Executed trajectory up to current point
        plt.plot(self.t_values[:i], np.asarray(self.x_values[:i]).T[0], color = "tab:blue", label="theta")
        plt.plot(self.t_values[:i], np.asarray(self.x_values[:i]).T[1], color = "tab:orange", label="theta dot")
        plt.plot(self.t_values[:i], self.tau_values[:i], color = "tab:green", label="u")
        ## Currently planned trajectory
        self.t_values_th = np.linspace(self.t_values[i], self.t_values[i]+dt*(N+1), N)
        plt.plot(self.t_values_th, np.asarray(x_trj).T[0], color = "tab:blue", linestyle = "dashed", label="theta")
        plt.plot(self.t_values_th, np.asarray(x_trj).T[1], color = "tab:orange", linestyle = "dashed", label="theta dot")
        plt.plot(self.t_values_th[:-1], u_trj, color = "tab:green", linestyle = "dashed", label="u")
        plt.legend(loc="best")
        
        #plt.show()
        
    def animate_MPC_plot2(self, x_trj, u_trj, cost, ite, dt, N, vid = False, vidname = "video"):
        fig, (ax1, ax3) = plt.subplots(2)
        ax2 = ax1.twinx()
        ax4 = ax3.twinx()
        
        ## Executed trajectory up to current point
        x1 = ax1.plot(0, 0, color = "tab:blue", label="theta")[0]
        x2 = ax1.plot(0, 0, color = "tab:orange", label="theta dot")[0]
        u = ax2.plot(0, 0, color = "tab:green", label="u")[0]
        
        ## Currently planned trajectory
        
        x1traj = ax1.plot(0, 0, color = "tab:blue", linestyle = "dashed", label="theta")[0]
        x2traj = ax1.plot(0, 0, color = "tab:orange", linestyle = "dashed", label="theta dot")[0]
        utraj = ax2.plot(0, 0, color = "tab:green", linestyle = "dashed", label="u")[0]

        ax1.set(xlim=[0, 10], ylim=[-7.5, 7.5], xlabel='Time [s]')
        ax1.legend(loc="best")
        ax2.legend(loc="best")

        ax2.set(ylim=[-0.1, 0.1])
        ax2.set_ylabel(ylabel='u [Nm]', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        ## iterations

        costpl = ax3.plot(0, 0, color = "tab:blue", label = "cost")[0]
        itepl = ax4.plot(0, 0, color = "tab:orange", label = "iterations")[0]

        ax3.set(xlim=[0, 10], ylim=[0.001, 100], xlabel='Time [s]', ylabel='Cost')
        ax4.set(ylim = [0, N])
        ax4.set_ylabel(ylabel='Iterations', color='tab:orange')
        ax4.tick_params(axis='y', labelcolor='tab:orange')
        ax3.set_yscale('log')
        
        fig.tight_layout()
        
        def update_MPC_plot(frame):
            frame = frame*10
            
            x1.set_xdata(self.t_values[:frame])
            x1.set_ydata(np.asarray(self.x_values).T[0][:frame])

            x2.set_xdata(self.t_values[:frame])
            x2.set_ydata(np.asarray(self.x_values).T[1][:frame])

            u.set_xdata(self.t_values[:frame])
            u.set_ydata(self.tau_values[:frame])

            t2 = np.linspace(self.t_values[frame], self.t_values[frame]+dt*(N+1), N)
            
            x_trji = np.asarray([x_trj[frame*2], x_trj[frame*2+1]]).T
            u_trji = np.asarray(u_trj)[frame]
            
            x1traj.set_xdata(t2)
            x1traj.set_ydata(np.asarray(x_trji).T[0])

            x2traj.set_xdata(t2)
            x2traj.set_ydata(np.asarray(x_trji).T[1])

            utraj.set_xdata(t2[:-1])
            utraj.set_ydata(u_trji)

            costpl.set_xdata(self.t_values[:frame])
            costpl.set_ydata(cost[:frame])

            itepl.set_xdata(self.t_values[:frame])
            itepl.set_ydata(ite[:frame])

            return (x1, x2, u, x1traj, x2traj, utraj)

        ani = mplanimation.FuncAnimation(fig=fig, func=update_MPC_plot, frames=int(len(self.t_values)/10), interval = 100)
        if vid:
            ani.save(vidname + ".gif")
        else:
            plt.show()    
    '''
    def animate_MPC_plot(self):
        moviewriter = mplanimation.FFMpegWriter(fps = 10)
        fig = plt.figure()
        x_trj , u_trj, dt, N = self.controller.get_animation_data()
        with moviewriter.saving(fig, "vid1.mp4", 100):
            for i in range(0, 1000, 10):
                plt.clf()
                plot_MPC_instant_trajectory(np.asarray([x_trj[i*2], x_trj[i*2+1]]).T, u_trj[i], dt, N, i)
                moviewriter.grab_frame()
        
        #for i in range(self.t_values.len())
    '''         
    def simulate_and_animate(self, t0, y0, tf, dt, controller=None, integrator="euler", save_video=False):
        """
        simulate and animate the pendulum
        """
        self.set_state(t0, y0)

        self.t_values = []
        self.x_values = []
        self.tau_values = []

        #fig = plt.figure(figsize=(6,6))
        #self.animation_ax = plt.axes()
        fig, (self.animation_ax, self.ps_ax) = plt.subplots(1, 2, figsize=(10, 5))
        self.animation_plots = []
        ee_plot, = self.animation_ax.plot([], [], "o", markersize=25.0, color="blue")
        bar_plot, = self.animation_ax.plot([], [], "-", lw=5, color="black")
        #text_plot = self.animation_ax.text(0.1, 0.1, [], xycoords="figure fraction")
        self.animation_plots.append(ee_plot)
        self.animation_plots.append(bar_plot)

        num_steps = int(tf / dt)
        par_dict = {}
        par_dict["dt"] = dt
        par_dict["controller"] = controller
        par_dict["integrator"] = integrator
        frames = num_steps*[par_dict]

        #ps_fig = plt.figure(figsize=(6,6))
        #self.ps_ax = plt.axes()
        #self.ps_plots = []
        ps_plot, = self.ps_ax.plot([], [], "-", lw=1.0, color="blue")
        #self.ps_plots.append(ps_plot)
        self.animation_plots.append(ps_plot)

        animation = FuncAnimation(fig, self._animation_step, frames=frames, init_func=self._animation_init, blit=True, repeat=False, interval=dt*1000)
        animation2 = None
        #if phase_plot:
        #    animation2 = FuncAnimation(fig, self._ps_update, init_func=self._ps_init, blit=True, repeat=False, interval=dt*1000)

        if save_video:
            Writer = mplanimation.writers['ffmpeg']
            writer = Writer(fps=60, bitrate=1800)
            mplanimation.save('pendulum_swingup.mp4', writer=writer)
            #if phase_plot:
            #    Writer2 = mplanimation.writers['ffmpeg']
            #    writer2 = Writer2(fps=60, bitrate=1800)
            #    animation2.save('pendulum_swingup_phase.mp4', writer=writer2)
        #plt.show()

        return self.t_values, self.x_values, self.tau_values, animation#, animation2

    def _animation_init(self):
        """
        init of the animation plot
        """
        self.animation_ax.set_xlim(-1.5*self.l, 1.5*self.l)
        self.animation_ax.set_ylim(-1.5*self.l, 1.5*self.l)
        self.animation_ax.set_xlabel("x position [m]")
        self.animation_ax.set_ylabel("y position [m]")
        for ap in self.animation_plots:
            ap.set_data([], [])

        self._ps_init()
        return self.animation_plots

    def _animation_step(self, par_dict):
        """
        simulation of a single step which also updates the animation plot
        """
        dt = par_dict["dt"]
        controller = par_dict["controller"]
        integrator = par_dict["integrator"]
        if controller is not None:
            tau = controller.get_control_output(self.x)
        else:
            tau = 0
        self.step(tau, dt, integrator=integrator)
        ee_pos = self.forward_kinematics(self.x[0])
        self.animation_plots[0].set_data((ee_pos[0],), (ee_pos[1],))
        self.animation_plots[1].set_data([0, ee_pos[0]], [0, ee_pos[1]])

        self._ps_update(0)

        return self.animation_plots

    def _ps_init(self):
        """
        init of the phase space animation plot
        """
        self.ps_ax.set_xlim(-np.pi, 2*np.pi)
        self.ps_ax.set_ylim(-10, 10)
        self.ps_ax.set_xlabel("degree [rad]")
        self.ps_ax.set_ylabel("velocity [rad/s]")
        for ap in self.animation_plots:
            ap.set_data([], [])
        return self.animation_plots

    def _ps_update(self, i):
        """
        update of the phase space animation plot
        """
        self.animation_plots[-1].set_data(np.asarray(self.x_values).T[0], np.asarray(self.x_values).T[1])
        return self.animation_plots

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

    def CubicTimeScaling(self, Tf, t):
        """Computes s(t) for a cubic time scaling
        Source: Modern Robotics Toolbox (https://github.com/NxRLab/ModernRobotics/blob/master/packages/Python/modern_robotics/core.py#L1455C1-L1469C61)
        :param Tf: Total time of the motion in seconds from rest to rest
        :param t: The current time t satisfying 0 < t < Tf
        :return: The path parameter s(t) corresponding to a third-order
                 polynomial motion that begins and ends at zero velocity
    
        Example Input:
            Tf = 2
            t = 0.6
        Output:
            0.216
        """
        return 3 * (1.0 * t / Tf) ** 2 - 2 * (1.0 * t / Tf) ** 3

    def JointTrajectory(self, thetastart, thetaend, Tf, N):
        """Computes a straight-line trajectory in joint space
        Source: Modern Robotics Toolbox (modified) 
        :param thetastart: The initial joint variables
        :param thetaend: The final joint variables
        :param Tf: Total time of the motion in seconds from rest to rest
        :param N: The number of points N > 1 (Start and stop) in the discrete
                  representation of the trajectory
        :return: A trajectory as an N x n matrix, where each row is an n-vector
                 of joint variables at an instant in time. The first row is
                 thetastart and the Nth row is thetaend . The elapsed time
                 between each row is Tf / (N - 1)
    
        Example Input:
            thetastart = np.array([1, 0, 0, 1, 1, 0.2, 0,1])
            thetaend = np.array([1.2, 0.5, 0.6, 1.1, 2, 2, 0.9, 1])
            Tf = 4
            N = 6
            method = 3
        Output:
            np.array([[     1,     0,      0,      1,     1,    0.2,      0, 1]
                      [1.0208, 0.052, 0.0624, 1.0104, 1.104, 0.3872, 0.0936, 1]
                      [1.0704, 0.176, 0.2112, 1.0352, 1.352, 0.8336, 0.3168, 1]
                      [1.1296, 0.324, 0.3888, 1.0648, 1.648, 1.3664, 0.5832, 1]
                      [1.1792, 0.448, 0.5376, 1.0896, 1.896, 1.8128, 0.8064, 1]
                      [   1.2,   0.5,    0.6,    1.1,     2,      2,    0.9, 1]])
        """
        N = int(N)
        timegap = Tf / (N - 1.0)
        traj = np.zeros((len(thetastart), N))
        for i in range(N):
            s = self.CubicTimeScaling(Tf, timegap * i)
            traj[:, i] = s * np.array(thetaend) + (1 - s) * np.array(thetastart)
        traj = np.array(traj).T
        return traj

    def return_home(self, startconfig = None):
        
        if startconfig is None:            
            startconfig = self.c.get_position(self.cell_id)
            
        endconfig = 0.0
        thetadot_max = 10.0 # max. speed limit (rad/s)
        
        #Tf = 20.0 # either choose a fixed trajectory time in seconds
        Tf = 3.0*np.abs((endconfig - startconfig))/(thetadot_max) # OR calculate Tf such that speed limit is repsected.
        
        thetastart = np.array([startconfig])
        thetaend = np.array([endconfig])

        N = 1000 # Number of points in trajectory
        traj = self.JointTrajectory(thetastart, thetaend, Tf, N)
        
        # defining runtime variables
        i = 0
        meas_dt = 0.0
        meas_time = 0.0
        dt = Tf/N
        
        print("Rezeroing motion started from start configuration = ", startconfig, " rad.")
        while i < N:
            start_loop = time.time()
            meas_time += meas_dt
            
            ## Do your stuff here - START  
            pos = traj[i,:]
            self.c.set_position(pos[0], self.cell_id)
            ## Do your stuff here - END
            
            i += 1
            exec_time = time.time() - start_loop
            #if exec_time > dt:
                #print("Control loop is too slow!")
                #print("Control frequency:", 1/exec_time, "Hz")
                #print("Desired frequency:", 1/dt, "Hz")
                #print()
            while time.time() - start_loop < dt:
                pass
            meas_dt = time.time() - start_loop
        print("Rezeroing motion finished at end configuration = ", self.c.get_position(self.cell_id), " rad.")

    def wait_for_control_loop_end(self, time_to_pass):
            """Delay ending a while loop so that it loops at a desired sampling time dt."""
            if time_to_pass <= 0.0:
                return

            start = time.time()
            
            time.sleep(time_to_pass * 0.7) # sleep
            while time.time() - start < time_to_pass: # busy waiting
                pass

    def run_on_hardware(self, user_token, tf, dt, controller=None, save_video=True):
        client = Client()

        experiment_time = tf
        preparation_time = 1.0
        n = int(experiment_time / dt)

        session_token, url = client.start_experiment(user_token, "SimplePendulum", experiment_time+0.01, preparation_time, record=False)

        # Set Kp, Kd to zero for torque control
        kp = 0.0
        kd = 0.0
        client.set_impedance_controller_params([kp], [kd], session_token)

        meas_time_vec = np.zeros(n)
        meas_x = np.zeros((n, 2))
        meas_u = np.zeros((n, 1))
        des_u = np.zeros((n, 1))

        i = 0
        t_start = time.time()

        try:
            while i < n:
                meas_time_vec[i] = time.time() - t_start
    
                # Measure and record data
                meas_x[i] = np.array( [client.get_position(session_token),  client.get_velocity(session_token)] )
                meas_u[i] = client.get_torque(session_token)

                self.x = meas_x[i]
                tau = 0
                if controller is not None:
                    tau = controller.get_control_output(self.x)
                client.set_torque(tau, session_token)
    
                des_u[i] = tau
    
                i += 1
                sleep_duration = t_start + i * dt - time.time()
                self.wait_for_control_loop_end(sleep_duration)
    
            video_url = client.stop_experiment(session_token)
    
            if save_video:
                video_path = download_video(video_url)
            else:
                video_path = ""
        except:
            print(traceback.format_exc())
            video_path = ""
            video_url = ""
            
        return meas_time_vec, meas_x, meas_u, des_u, video_url, video_path


    def run_on_hardware_phys(self, tf, dt, controller=None):
            
        import pyCandle
        import time

        # Select pendulum from motor list
        self.candle = pyCandle.Candle(pyCandle.CAN_BAUD_8M,True)
        for id in self.candle.ping():
            self.candle.addMd80(id)
        
        # Now we shall loop over all found drives to change control mode and enable them one by one
        for md in self.candle.md80s:
            self.candle.controlMd80SetEncoderZero(md)      #  Reset encoder at current position
            self.candle.controlMd80Mode(md, pyCandle.IMPEDANCE)    # Set mode to impedance control
            self.candle.controlMd80Enable(md, True)     # Enable the drive

        # Begin update loop (it starts in the background)
        self.candle.begin()

        candle_dict = {}
        motornum = 0
        for motor in self.candle.md80s:
            candle_dict[self.candle.md80s[motornum].getId()] = motornum
            motornum += 1

        md80id = 899
        
        md80num = candle_dict[md80id]
        
        # set zero impedance (kp=kd=0) for pure torque control 
        self.candle.md80s[md80num].setImpedanceControllerParams(0, 0)
        
        input("Press bring the pendulum to the starting configuration and press enter to continue...")
    
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
        # Auto update loop is running in the background updating data in candle.md80s vector. Each md80 object can be 
        # Called for data at any time
        while i < n:
            start_loop = time.time()
            meas_time += meas_dt
            
            ## Do your stuff here - START
            
            measured_position = self.candle.md80s[md80num].getPosition()
            measured_velocity = self.candle.md80s[md80num].getVelocity()  
            measured_torque = self.candle.md80s[md80num].getTorque()             
            self.x = np.array([measured_position, measured_velocity])
            
            # Control logic
            if controller is not None:
                tau = controller.get_control_output(self.x)
                tau_scaled = tau*tau_scaling    # physical torque to motor torque
                self.candle.md80s[md80num].setTargetTorque(tau_scaled)
            else:
                tau = 0                
                       
            # Collect data for plotting
            meas_time_vec[i] = meas_time
            meas_pos[i] = measured_position
            meas_vel[i] = measured_velocity    
            meas_tau[i] = self.candle.md80s[md80num].getTorque()/tau_scaling
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

        # Send a few zeros to the motor and then close the update loop
        for i in range(5):
            self.candle.md80s[md80num].setTargetTorque(0.0)
        self.candle.end()
        
        self.t_values = meas_time_vec
        self.x_values = np.vstack((meas_pos, meas_vel)).T
        self.tau_values = meas_tau
        self.des_tau_values = des_tau
        
        return self.t_values, self.x_values, self.tau_values, self.des_tau_values
    
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

def plot_timeseries(T, X, U):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.plot(T, np.asarray(X).T[0], label=r"$\theta$")
    plt.ylabel(r"$\theta$ (rad)")
    plt.xlabel("t (s)")
    plt.grid()
    
    plt.subplot(1, 3, 2)
    plt.plot(T, np.asarray(X).T[0], label=r"$\dot\theta$")
    plt.ylabel(r"$\dot\theta$ (rad/s)")
    plt.grid()
    plt.xlabel("t (s)")

    plt.subplot(1, 3, 3)
    plt.plot(T, U, label="u_main")
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel("t (s)")
    plt.ylabel(r"$\tau$ (Nm)")

    plt.tight_layout()
    plt.show()

