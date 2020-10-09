import torch
import numpy as np

import a2c.modules
from a2c.mutli_atari_env import build_env


def batch_to_seq(batch, n_env, t_max):
    batch_first = batch.view(n_env, t_max, -1)
    sequence = batch_first.transpose(0, 1)
    return sequence


class Network(torch.nn.Module):

    def __init__(self,
                 batch_size,
                 n_steps,
                 actor_outdim=1,
                 n_lstm=0,
                 use_gpu=False):
        super(Network, self).__init__()
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_envs = batch_size // n_steps
        self.n_lstm = n_lstm

        if use_gpu:
            self.shared = a2c.modules.ModelNatureConv().cuda()
            next_in_dim = self.shared.feature_dim
            if n_lstm > 0:
                self.lstm = a2c.modules.ModelLSTM(next_in_dim, n_lstm).cuda()
                next_in_dim = n_lstm
            self.actor = a2c.modules.ModelActor(
                input_dim=next_in_dim, output_dim=actor_outdim).cuda()
            self.critic = a2c.modules.ModelCritic(
                input_dim=next_in_dim, output_dim=1).cuda()
        else:
            self.shared = a2c.modules.ModelNatureConv()
            next_in_dim = self.shared.feature_dim
            if n_lstm > 0:
                self.lstm = a2c.modules.ModelLSTM(next_in_dim, n_lstm)
                next_in_dim = n_lstm
            self.actor = a2c.modules.ModelActor(
                input_dim=next_in_dim, output_dim=actor_outdim)
            self.critic = a2c.modules.ModelCritic(
                input_dim=next_in_dim, output_dim=1)

    def forward(self, obs, lstm_states, masks, training):
        if training:
            # In training phase, a batch of training data is feed as a batch of sequence, with the length of t_max
            # LSTM hidden state is passed as parameter
            state_info = self.shared(obs)
            # ==============================================================
            if self.n_lstm >= 1:
                state_info = batch_to_seq(
                    state_info, n_env=self.n_envs, t_max=self.n_steps)
                h, c = lstm_states
                masks_sequence = masks.swapaxes(1, 0)
                lstm_outputs = []

                for i in range(len(masks_sequence)):
                    tensor_masks = torch.tensor(1.0 - masks_sequence[i])
                    tensor_masks = tensor_masks.to(
                        next(self.parameters()).device)
                    h = tensor_masks.view(1, -1, 1).float() * h
                    c = tensor_masks.view(1, -1, 1).float() * c
                    lstm_output, (h, c) = self.lstm(state_info[i].unsqueeze(0),
                                                    (h, c))
                    lstm_outputs.append(lstm_output.squeeze(0))

                lstm_outputs = torch.stack(lstm_outputs).transpose(1, 0)
                lstm_outputs = lstm_outputs.contiguous().view(
                    self.batch_size, -1)
                state_info = lstm_outputs
            # ==============================================================
            action_logits = self.actor(state_info)
            dist = torch.distributions.Categorical(logits=action_logits)
            value = self.critic(state_info)
            return dist, value, action_logits
        else:
            state_info = self.shared(obs)
            # ==============================================================
            if self.n_lstm >= 1:
                # In data generation phase, only length 1 sequence
                # LSTM hidden state is passed from last step as input (a tuple of two tensors h and c)
                state_info = batch_to_seq(
                    state_info, n_env=self.n_envs, t_max=1)
                h, c = lstm_states
                masks = torch.tensor(1.0 - masks)
                masks = masks.to(next(self.parameters()).device)
                h = masks.view(1, -1, 1).float() * h
                c = masks.view(1, -1, 1).float() * c
                lstm_output, lstm_states = self.lstm(state_info, (h, c))
                lstm_output = lstm_output.view(self.n_envs, -1)
                state_info = lstm_output
            # ==============================================================
            action_logits = self.actor(state_info)
            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()
            value = self.critic(state_info)
            return action, value, lstm_states


class PolicyModel(object):
    """
    Replace TF by PyTorch
    Try to match original API
    Now only CNN
    """

    def __init__(self,
                 batch_size=60,
                 n_step=5,
                 learning_rate=7e-5,
                 epsilon=1e-5,
                 entropy_coefficient=0.01,
                 value_coefficient=0.5,
                 actor_out_dim=1,
                 n_lstm=0,
                 use_gpu=False):
        n_envs = batch_size // n_step

        self.network = Network(
            batch_size=batch_size,
            n_steps=n_step,
            actor_outdim=actor_out_dim,
            n_lstm=n_lstm,
            use_gpu=use_gpu)

        self.optimizer = torch.optim.RMSprop(
            params=self.network.parameters(),
            lr=learning_rate,
            eps=epsilon,
            alpha=0.99)

        self.entro_coe = entropy_coefficient
        self.value_coe = value_coefficient

        if n_lstm > 0:
            self.lstm_initial_state = (
                torch.zeros(1, n_envs, n_lstm).to(
                    next(self.network.parameters()).device),
                torch.zeros(1, n_envs, n_lstm).to(
                    next(self.network.parameters()).device))
        else:
            self.lstm_initial_state = None, None

    def loss(self, obs, rewards, actions, lstm_hidden_state, masks):
        dist_train, value_train, _ = self.forward(
            obs, lstm_states=lstm_hidden_state, masks=masks, training=True)
        value_train = value_train.squeeze(1)
        log_prob_train = dist_train.log_prob(actions)
        entropy_train = dist_train.entropy()

        advantages = rewards - value_train
        advantages = advantages.detach()

        policy_loss = -(log_prob_train * advantages).mean()
        value_loss = ((value_train - rewards).pow(2) * 0.5).mean()
        entropy_loss = entropy_train.mean()

        loss = policy_loss - entropy_loss * self.entro_coe + value_loss * self.value_coe

        return loss, policy_loss, entropy_loss, value_loss

    def forward(self, obs, lstm_states, masks, training):
        if training:
            return self.network(obs, lstm_states, masks, training)
        else:
            with torch.no_grad():
                output = self.network(obs, lstm_states, masks, training)
            return output

    def train(self, obs, rewards, actions, lstm_hidden_state, masks, epoch):
        loss, policy_loss, entropy_loss, value_loss = self.loss(
            obs, rewards, actions, lstm_hidden_state, masks)

        # Gradient clipping by 0.5
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        return loss, policy_loss, entropy_loss, value_loss

    def save(self, path_to_model_file):
        torch.save(self.network.state_dict(), path_to_model_file)

    def load(self, path_to_model_file):
        self.network.load_state_dict(torch.load(path_to_model_file))


if __name__ == "__main__":
    # Set seed
    # deteministic(seed=1024)

    # Init environments
    n_envs = 16  # From arg? or N cpu?
    env = build_env(seed=1024, env_id="BreakoutNoFrameskip-v4")
    init_state = env.reset()
    init_state = init_state.transpose(0, 3, 1, 2)

    # Model
    shared = a2c.modules.ModelNatureConv(in_channels=4, out_dim=512)
    actor = a2c.modules.ModelActor(input_dim=512, output_dim=4)
    critic = a2c.modules.ModelCritic(input_dim=512)

    # Init policy & value network

    state_info = shared(init_state)
    action_logits = actor(state_info)
    value = critic(state_info)
    print(value)
