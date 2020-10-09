import multiprocessing
import sys
import torch
import numpy as np

from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import make_vec_env


def build_env(seed, env_id):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = ncpu
    env_type = 'atari'

    frame_stack_size = 4
    env = VecFrameStack(
        make_vec_env(env_id, env_type, nenv, seed), frame_stack_size)
    # env = TransposeImage(env)

    return env


if __name__ == "__main__":
    env = build_env(seed=1024, env_id="BreakoutNoFrameskip-v4")
    print(env.reset().transpose(0, 3, 1, 2).shape)
    print(env.action_space)
    print(env.observation_space)
    for i in range(100):
        print(env.step(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])))
