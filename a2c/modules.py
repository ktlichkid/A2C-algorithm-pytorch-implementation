import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class ModelNatureConv(nn.Module):
    """
    A three layer CNN, configuration is from DQN Nature paper
    For state generating module
    (conv1): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
    (conv2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
    (conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (fc4): Linear(in_features=3136, out_features=512, bias=True)
    """

    def __init__(self, in_channels=4, out_dim=512):
        super(ModelNatureConv, self).__init__()
        self.in_channels = in_channels
        self.feature_dim = out_dim

        self.conv1 = layer_init(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            w_scale=np.sqrt(2))
        self.conv2 = layer_init(
            nn.Conv2d(32, 64, kernel_size=4, stride=2), w_scale=np.sqrt(2))
        self.conv3 = layer_init(
            nn.Conv2d(64, 64, kernel_size=3, stride=1), w_scale=np.sqrt(2))
        self.fc4 = layer_init(
            nn.Linear(7 * 7 * 64, self.feature_dim), w_scale=np.sqrt(2))

        # if use_gpu:
        #     self.gpu_acceleration()

    def forward(self, x):
        x = x / 255.
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float().to(next(self.parameters()).device)

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc4(out))
        return out


class ModelLSTM(nn.Module):

    def __init__(self, input_dim, lstm_dim):
        super(ModelLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_dim)
        # Layer init function doesn't work on LSTM, have to do it manually
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.constant_(self.lstm.bias_ih_l0, 0.0)
        nn.init.constant_(self.lstm.bias_hh_l0, 0.0)

    def forward(self, x, hidden_state):
        x, hidden_state = self.lstm(x, hidden_state)
        return x, hidden_state


class ModelActor(nn.Module):
    """
    Fully connected network .
    Compute Action
    """

    def __init__(self, input_dim=512, output_dim=1):
        super(ModelActor, self).__init__()
        self.fc_out = layer_init(nn.Linear(input_dim, output_dim), w_scale=0.01)

    def forward(self, x):
        x = self.fc_out(x)
        return x


class ModelCritic(nn.Module):

    def __init__(self, input_dim=512, output_dim=1):
        super(ModelCritic, self).__init__()
        self.fc_out = layer_init(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        x = self.fc_out(x)
        return x


if __name__ == "__main__":
    from a2c.algo import PolicyModel
    model = PolicyModel(learning_rate=0.01)

    # for p in model.actor.parameters():
    #     print(p)
    # print("=================================================================")

    for epoch in range(100):
        init_state = np.ones((60, 4, 84, 84), dtype=np.int64)
        init_state = torch.Tensor(init_state)

        dist, val, action_logits = model.forward(
            init_state, lstm_states=0, masks=None, training=True)

        loss = torch.abs(val.mean() + action_logits.mean())

        # model.scheduler.step(epoch)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        for param_group in model.optimizer.param_groups:
            print(epoch)
            print(param_group['lr'])
            print(loss)
