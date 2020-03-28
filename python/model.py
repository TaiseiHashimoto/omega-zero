import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ResBlock(nn.Module):
    def __init__(self, n_filter):
        super().__init__()

        self.conv1 = nn.Conv2d(n_filter, n_filter, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filter)
        self.conv2 = nn.Conv2d(n_filter, n_filter, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filter)
    
    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + identity)
        return x


class OmegaNet(nn.Module):
    def __init__(self, board_size, n_action, n_res_block, res_filter, head_filter, value_hidden):
        super().__init__()

        # input channel : black / white / side
        self.conv = nn.Sequential(
            nn.Conv2d(3, res_filter, kernel_size=3, padding=1),
            nn.BatchNorm2d(res_filter),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.ModuleList()
        for i in range(n_res_block):
            self.res_blocks.append(ResBlock(res_filter))
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(res_filter, head_filter, kernel_size=1, padding=0),
            nn.BatchNorm2d(head_filter),
            Flatten(),
            nn.ReLU(inplace=True),
            nn.Linear(head_filter * board_size ** 2, n_action)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(res_filter, head_filter, kernel_size=1, padding=0),
            nn.BatchNorm2d(head_filter),
            Flatten(),
            nn.ReLU(inplace=True),
            nn.Linear(head_filter * board_size ** 2, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1)
        )
    
    def forward(self, black_board, white_board, side, legal_flags):
        side_board = torch.ones_like(black_board) * side[:, None, None]
        x = torch.stack([black_board, white_board, side_board], dim=1)
        
        x = self.conv(x)
        for block in self.res_blocks:
            x = block(x)
        
        policy_logit = self.policy_head(x)
        policy_logit = policy_logit + (legal_flags + 1e-45).log()
        policy_logit = F.log_softmax(policy_logit, dim=1)

        value_pred = torch.tanh(self.value_head(x)).squeeze(dim=1)

        return policy_logit, value_pred
