import numpy as np
import matplotlib.pyplot as plt


def visualize_replay_buffer(filepath, plot_range = None, sim=None):
    """
    Visualizes and analyzes the contents of a saved replay buffer.
    
    This function loads a replay buffer from a pickle file, extracts its transitions in order,
    and produces a series of diagnostic plots comparing observed states, predicted next states
    (from a physics simulator), and actual next states. It also visualizes actions, rewards,
    and episode termination flags over a specified range of timesteps.
    
    Parameters
    ----------
    filepath : str
        Path to the pickle file containing the replay buffer.
    plot_range : list[int, int], optional
        Start and end indices of the timesteps to plot. Defaults to the entire replay buffer.
    
    Returns
    -------
    None
        Displays the generated plots and prints summary statistics of prediction errors.
    """

    import pickle
    def load_replay_buffer(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def get_ordered_transitions(rb):
        size = rb.size()
        
        obs = rb.observations
        obs = obs[:size].squeeze()
        next_obs = rb.next_observations
        next_obs = next_obs[:size].squeeze()
        actions = rb.actions
        actions = actions[:size].squeeze()
        rewards = rb.rewards
        rewards = rewards[:size].squeeze()
        dones = rb.dones
        dones = dones[:size].squeeze()
        return obs, next_obs,  actions, rewards, dones
    
    
    # Load and extract
    replay_buffer = load_replay_buffer(filepath)
    obs, next_obs, actions, rewards, dones = get_ordered_transitions(replay_buffer)
    print("Total observations in replay buffer: ", dones.shape)

    if plot_range == None:
        plot_range = [0,dones.shape[0]]
    
    # Transform observations to (angle, velocity)
    angles = np.arctan2(obs[:, 1], obs[:, 0])
    obs = np.column_stack((angles, obs[:, 2]))
    angles_next = np.arctan2(next_obs[:, 1], next_obs[:, 0])
    next_obs = np.column_stack((angles_next, next_obs[:, 2]))
    
    # Prediction using simulator
    filtered_obs = obs[plot_range[0]:plot_range[1]]
    filtered_next_obs = next_obs[plot_range[0]:plot_range[1]]
    torque = actions[plot_range[0]:plot_range[1]]
    dt = 0.02
    t = 0.0

    if sim != None:
        pred_theta = []
        for i in range(len(torque)):
            y = filtered_obs[i]
            tau = torque[i] * sim.plant.torque_limit
            y_next = y + dt * sim.runge_integrator(t, y, dt, tau)
            pred_theta.append(y_next)
        filtered_pred_next_obs = np.array(pred_theta)
    
    # Create Plots
    
    fig, axs = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(f"Replay Buffer Summary (steps {plot_range[0]} to {plot_range[1]})", fontsize=16)
    
    # 1: Raw XY Position
    axs[0, 0].plot(obs[plot_range[0]:plot_range[1], 0], label="θ")
    axs[0, 0].plot(obs[plot_range[0]:plot_range[1], 1], label="ω")
    axs[0, 0].set_title("Observation Angle & Velocity")
    axs[0, 0].set_xlabel("Time step")
    axs[0, 0].set_ylabel("Value")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # 2: Velocity Estimation
    axs[0, 1].plot(obs[plot_range[0]:plot_range[1]-1, 1], label="ω (obs)")
    diff_theta = np.diff(obs[plot_range[0]:plot_range[1], 0]) / (0.02 * 0.7)
    axs[0, 1].plot(diff_theta, label="dθ/dt approx")
    axs[0, 1].set_title("Velocity vs. Estimated dθ/dt")
    axs[0, 1].set_xlabel("Time step")
    axs[0, 1].set_ylabel("Velocity")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # 3: Simulator vs. Actual – θ
    #axs[1, 0].plot(filtered_obs[:, 0], label="θ (obs)")
    axs[1, 0].plot(filtered_next_obs[:, 0], label="θ (next)")
    if sim != None:
        axs[1, 0].plot(filtered_pred_next_obs[:, 0], label="θ (pred)")
    axs[1, 0].set_title("Angle Comparison with Simulator")
    axs[1, 0].set_xlabel("Time step")
    axs[1, 0].set_ylabel("Angle (rad)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # 4: Simulator vs. Actual – ω
    #axs[1, 1].plot(filtered_obs[:, 1], label="ω (obs)")
    axs[1, 1].plot(filtered_next_obs[:, 1], label="ω (next)")
    if sim != None:
        axs[1, 1].plot(filtered_pred_next_obs[:, 1], label="ω (pred)")
    axs[1, 1].set_title("Velocity Comparison with Simulator")
    axs[1, 1].set_xlabel("Time step")
    axs[1, 1].set_ylabel("Velocity (rad/s)")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # 5: Difference in %
    if sim != None:
        avg_step_length_pos = np.mean(np.abs(filtered_pred_next_obs[:, 0] - filtered_obs[:, 0]))
        avg_step_length_vel = np.mean(np.abs(filtered_pred_next_obs[:, 1] - filtered_obs[:, 1]))
        diff_pos = 100 * (filtered_pred_next_obs[:, 0] - filtered_next_obs[:, 0]) / avg_step_length_pos
        diff_vel = 100 * (filtered_pred_next_obs[:, 1] - filtered_next_obs[:, 1]) / avg_step_length_vel
        axs[2, 0].plot(diff_vel, label="ω diff %")
        axs[2, 0].plot(diff_pos, label="θ diff %")
        axs[2, 0].set_title("Prediction Error (% of Avg Step)")
        axs[2, 0].set_xlabel("Time step")
        axs[2, 0].legend()
        axs[2, 0].set_ylim(-150,150)
        axs[2, 0].grid(True)
    
    # 6: Actions
    axs[2, 1].plot(actions[plot_range[0]:plot_range[1]])
    axs[2, 1].set_title("Action Trajectory")
    axs[2, 1].set_xlabel("Time step")
    axs[2, 1].set_ylabel("Torque")
    axs[2, 1].grid(True)
    
    # 7: Rewards
    axs[3, 0].plot(rewards[plot_range[0]:plot_range[1]])
    axs[3, 0].set_title("Reward Trajectory")
    axs[3, 0].set_xlabel("Time step")
    axs[3, 0].set_ylabel("Reward")
    axs[3, 0].set_ylim(-20,20)
    axs[3, 0].grid(True)
    
    # 8: Dones
    axs[3, 1].plot(dones[plot_range[0]:plot_range[1]])
    axs[3, 1].set_title("Done Flags")
    axs[3, 1].set_xlabel("Time step")
    axs[3, 1].set_ylabel("Done")
    axs[3, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    if sim != None:
        print(f"Average step length pos {avg_step_length_pos:.4f}, Average step length vel {avg_step_length_vel:.4f}")
        print(f"Median absolute positional error: {np.median(np.abs(diff_pos)):.2f}%")
        print(f"Median absolute velocity error: {np.median(np.abs(diff_vel)):.2f}%")
    return

def visualize_policy_with_stabilization_view(controller, 
                                             theta_range=(-np.pi, np.pi), 
                                             vel_range=(-8, 8), 
                                             stab_range=0.6,
                                             bins=200,
                                             zoom_bins=100):
    """
    Plots two heatmaps:
    1. Full (theta, velocity) policy
    2. Zoomed view around stabilization point ±π, shown as deviation from upright

    Parameters
    ----------
    controller : controller
        Arbitrary controller object.
    theta_range : tuple
        Range for theta in full plot.
    vel_range : tuple
        Range for velocity in both plots.
    stab_range : float
        Maximum deviation (in radians) from the upright position (π or -π).
    bins : int
        Resolution for full plot.
    zoom_bins : int
        Resolution for zoom plot.
    """
    def compute_actions(theta_vals, vel_vals):
        actions = np.zeros((len(vel_vals), len(theta_vals)))
        for i, th in enumerate(theta_vals):
            for j, vel in enumerate(vel_vals):
                action = controller.get_control_output(meas_pos=th, meas_vel=vel)
                actions[j, i] = action
        return actions

    # === Full policy ===
    thetas = np.linspace(*theta_range, bins)
    vels = np.linspace(*vel_range, bins)
    full_actions = compute_actions(thetas, vels)

    # === Stabilization view around ±π ===
    delta_theta_vals = np.linspace(-stab_range, stab_range, zoom_bins)
    vels_zoom = np.linspace(-4, 4, zoom_bins)

    # Map delta_theta to wrapped angles near ±π
    theta_left = -np.pi + delta_theta_vals  # around -π
    theta_right = np.pi - delta_theta_vals  # around +π

    # Merge both into one batch
    theta_zoom = np.concatenate([theta_left, theta_right])
    delta_zoom = np.concatenate([delta_theta_vals, delta_theta_vals])  # for x-axis
    vels_zoom_full = np.tile(vels_zoom, 2)

    actions_zoom = np.zeros((zoom_bins, 2 * zoom_bins))
    for i, dt in enumerate(delta_theta_vals):
        for j, vel in enumerate(vels_zoom):
            a_left = controller.get_control_output(meas_pos=-np.pi + dt, meas_vel=vel)
            a_right = controller.get_control_output(meas_pos=np.pi + dt, meas_vel=vel)
            actions_zoom[j, i] = a_left
            actions_zoom[j, i + zoom_bins] = a_right

    # === Plotting ===
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Full plot
    im1 = axs[0].imshow(full_actions, extent=[*theta_range, *vel_range],
                        origin='lower', aspect='auto', cmap='coolwarm')
    axs[0].set_title("Full Policy")
    axs[0].set_xlabel("Theta (rad)")
    axs[0].set_ylabel("Angular Velocity")
    fig.colorbar(im1, ax=axs[0], label='Torque')

    # Stabilization plot
    x_extent = [-stab_range, stab_range]
    im2 = axs[1].imshow(actions_zoom, extent=[x_extent[0], x_extent[1], -4, 4],
                        origin='lower', aspect='auto', cmap='coolwarm')
    axs[1].set_title("Stabilization Near ±π")
    axs[1].set_xlabel("ΔTheta from ±π (rad)")
    fig.colorbar(im2, ax=axs[1], label='Torque')

    plt.suptitle("DDPG Policy: Full and Stabilization Region Views")
    plt.tight_layout()
    plt.show()
    return


def create_evaluation_progress_video(folder_path, max_per_row = 6):
    """
    Creates a combined progress video from evaluation video clips.

    This function searches for all video files in the given folder that match
    the pattern `evaluation_<episode_number>_0.mp4`, and compiles them into a grid-style 
    video layout with episode labels.

    The resulting video is saved as `evaluation_progress.mp4` in the same folder.

    Args:
        folder_path (str): Path to the directory containing evaluation videos.
        max_per_row (int, optional): Maximum number of video clips per row 
            in the final output grid. Defaults to 6.

    """

    from moviepy import VideoFileClip, TextClip, CompositeVideoClip, clips_array, ColorClip
    import glob, re, os
    
    # Find all matching videos in the specified folder
    video_files = glob.glob(os.path.join(folder_path, "evaluation_*_0.mp4"))
    
    # Sort numerically by x in evaluation_x_0.mp4
    video_files.sort(key=lambda f: int(re.search(r"evaluation_(\d+)_0\.mp4", os.path.basename(f)).group(1)))
    
    
    video_clips = []
    
    for idx, filename in enumerate(video_files):
        match = re.search(r"evaluation_(\d+)_0\.mp4", os.path.basename(filename))
        if not match:
            continue
        episode_num = match.group(1)
        label = f"Episode {episode_num}"
        clip = VideoFileClip(filename)
    
        # Black background bar
        text_height = 120  # height of black strip
        bar = ColorClip(
            size=(clip.w, text_height),
            color=(0, 0, 0)
        ).with_duration(clip.duration)
        

        # Text on the bar
        txt = TextClip(
            font="Arial.ttf",
            text=label,
            font_size=30,
            color='white'
        ).with_duration(clip.duration).with_position(("center", "center"))
    
        text_bar = CompositeVideoClip([bar, txt])
    
        # Stack text bar above video
        stacked = clips_array([[text_bar], [clip]])
            
        video_clips.append(stacked)
        
        # Add vertical black bar separator between videos (except after last one)
        #vbar = ColorClip(
        #    size=(20, stacked.h),  # 20px wide black strip
        #   color=(0, 0, 0)
        #).with_duration(stacked.duration)
        #video_clips.append(vbar)
    
    # Arrange all videos side by side    
    clip_width, clip_height = video_clips[0].size
    rows = []
    
    for i in range(0, len(video_clips), max_per_row):
        row = video_clips[i:i+max_per_row]
        if len(row) < max_per_row:
            # pad with black clips to match the width of others
            pad = ColorClip(size=(clip_width, clip_height), color=(0, 0, 0)).with_duration(row[0].duration)
            row += [pad] * (max_per_row - len(row))
        rows.append(row)
    
    final = clips_array(rows)
    final = clips_array(rows)
    
    
    # Ensure same duration
    min_duration = min(c.duration for c in video_clips)
    final = final.subclipped(0, min_duration)
    
    final.write_videofile(os.path.join(folder_path,"evaluation_progress.mp4"), fps=24)
    return