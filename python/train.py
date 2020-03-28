import sys
import torch

from model import OmegaNet
from mldata import DataLoader


def train(file_names, device):
    omega_net = OmegaNet(
        board_size=8,
        n_action=64,
        n_res_block=5,
        res_filter=64,
        head_filter=32,
        value_hidden=64
    )
    omega_net.to(device)

    optim = torch.optim.AdamW(omega_net.parameters())
    loader = DataLoader(["../cpp/game0.dat"], batch_size=512, n_iter=1000)

    for black_board_b, white_board_b, side_b, legal_flags_b, result_b, Q_b, posteriors_b in loader:
        policy_logit_b, value_pred_b = omega_net(black_board_b, white_board_b, side_b, legal_flags_b)
        
        policy_loss = -(posteriors_b * policy_logit_b).sum(dim=1).mean(dim=0)

        value_target_b = (result_b + Q_b) / 2  # TODO: how to create target
        value_loss = (value_pred_b - value_target_b).pow(2).mean(dim=0)
        
        loss = policy_loss + value_loss
        optim.zero_grad()
        loss.backward()
        optim.step()


if __name__ == '__main__':
    cuda_id = 0
    if sys.argc == 2:
        cuda_id = int(sys.argv[1])

    if torch.cuda.is_available():
        device = torch.device('cuda')

    print(f"using {device}")
    train(file_names, device)
