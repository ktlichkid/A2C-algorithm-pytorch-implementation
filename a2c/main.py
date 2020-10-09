import sys
import os
import random
sys.path.extend(['/home/kt/Projects/a2c_lstm/a2c/'])
import argparse
import numpy as np
import torch

import a2c.mutli_atari_env
import a2c.modules
import a2c.algo
import a2c.runner


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--env',
        help='environment ID',
        type=str,
        default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=2048)
    parser.add_argument('--num_timesteps', type=float, default=10000),
    parser.add_argument(
        '--num_env',
        help='Number of environment copies being run in parallel. When not '
        'specified, set to number of cpus for Atari, and to 1 for Mujoco',
        default=None,
        type=int)
    parser.add_argument(
        '--reward_scale',
        help='Reward scale factor. Default: 1.0',
        default=1.0,
        type=float)
    parser.add_argument(
        '--save_path',
        help='Path to save trained model to',
        default=None,
        type=str)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--tmax', default=5, type=int, help='placeholder')
    parser.add_argument('--lstm_dim', default=256, type=int, help='placeholder')

    return parser


def deterministic(seed):
    # Set random seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=10)


def main(arg):
    deterministic(arg.seed)
    env = a2c.mutli_atari_env.build_env(seed=arg.seed, env_id=arg.env)
    n_envs = env.num_envs

    model = a2c.algo.PolicyModel(
        batch_size=n_envs * arg.tmax,
        n_step=arg.tmax,
        actor_out_dim=env.action_space.n,
        n_lstm=arg.lstm_dim,
        use_gpu=True)
    runner = a2c.runner.Runner(env=env, model=model, n_step=arg.tmax)

    batch_size = n_envs * arg.tmax

    total_epochs = int(arg.num_timesteps) // batch_size + 1

    for update in range(total_epochs):
        memory_obs, memory_discounted_rewards, memory_actions, memory_lstm_hidden_state, memory_masks = runner.run(
        )
        loss, policy_loss, entropy_loss, value_loss = model.train(
            obs=memory_obs,
            rewards=memory_discounted_rewards,
            actions=memory_actions,
            lstm_hidden_state=memory_lstm_hidden_state,
            masks=memory_masks,
            epoch=update)
        print(
            "Epoch %d, loss %f, policy loss %f, entropy loss %f, value loss %f"
            % (update, loss, policy_loss, entropy_loss, value_loss))


if __name__ == "__main__":
    arg = common_arg_parser().parse_args()
    main(arg)
