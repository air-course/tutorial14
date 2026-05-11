import os
import imageio
import mujoco
import numpy as np


def record_policy_video(
    env,
    agent,
    output_path,
    duration=5.0,
    fps=30,
    height=480,
    width=640,
    noise_std=0.0,
):
    """
    Records one rollout using MuJoCo's offscreen renderer.

    This assumes:
    - env.model is a mujoco.MjModel
    - env.data is a mujoco.MjData
    - agent.select_action(obs) returns action in [-1, 1]
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    renderer = mujoco.Renderer(env.model, height=height, width=width)

    frames = []
    obs = env.reset()
    done = False
    r = 0.

    while env.data.time < duration and not done:
        action = agent.select_action(obs, noise_std=noise_std)
        obs, reward, done, info = env.step(action)
        r += reward

        if len(frames) < env.data.time * fps:
            renderer.update_scene(env.data, camera="track")
            frame = renderer.render()
            frames.append(frame)

    renderer.close()

    imageio.mimsave(output_path, frames, fps=fps)

    return output_path, r
