import torch
from model import OmegaNet
import pathlib

if __name__ == '__main__':

    omega_net = OmegaNet(
        board_size=8,
        n_action=64,
        n_res_block=5,
        res_filter=64,
        head_filter=32,
        value_hidden=64
    )

    root_path = pathlib.Path(__file__).resolve().parents[1]
    new_model_path = root_path / 'model' / f'model_0.pt'
    new_model_jit_path = root_path / 'model' / f'model_jit_0.pt'

    torch.save(omega_net.state_dict(), new_model_path)

    black_board_dummy = torch.rand(3, 8, 8)
    white_board_dummy = torch.rand(3, 8, 8)
    side_dummy = torch.rand(3)
    legal_flags_dummy = torch.rand(3, 64)

    omega_net_jit = torch.jit.trace(omega_net, (black_board_dummy, white_board_dummy, side_dummy, legal_flags_dummy))
    omega_net_jit.save(str(new_model_jit_path))

    # p, v = omega_net(black_board_dummy, white_board_dummy, side_dummy, legal_flags_dummy)
    # print(v)
