import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, n_filter):
        super().__init__()

        self.conv1 = nn.Conv2d(n_filter, n_filter, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_filter)
        self.conv2 = nn.Conv2d(n_filter, n_filter, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_filter)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + identity)
        return x


class OmegaNet(nn.Module):
    def __init__(self, board_size, n_action, n_res_block, res_filter, policy_filter, value_filter, value_hidden):
        super().__init__()

        # input channel : black / white / side
        self.conv = nn.Sequential(
            nn.Conv2d(2, res_filter, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(res_filter),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(*[ResBlock(res_filter) for i in range(n_res_block)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(res_filter, policy_filter, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(policy_filter),
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Linear(policy_filter * board_size ** 2, n_action)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(res_filter, value_filter, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(value_filter),
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Linear(value_filter * board_size ** 2, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1)
        )

    def forward(self, black_board, white_board, side, legal_flags):
        side_board = side[:, None, None]
        player_board = black_board * (1 - side_board) + white_board * side_board
        opponent_board = black_board * side_board + white_board * (1 - side_board)
        x = torch.stack([player_board, opponent_board], dim=1)
        x = self.res_blocks(self.conv(x))

        policy_logit = self.policy_head(x)
        policy_logit = policy_logit + (legal_flags + 1e-45).log()
        policy_logit = F.log_softmax(policy_logit, dim=1)

        value_pred = torch.tanh(self.value_head(x)).squeeze(dim=1)

        return policy_logit, value_pred
