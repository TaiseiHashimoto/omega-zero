import torch
from model import OmegaNet
import pathlib
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_id', type=int)
    args = parser.parse_args()

    root_path = pathlib.Path(__file__).resolve().parents[1]
    exp_path = root_path / "exp" / str(args.exp_id)
    config_path = exp_path / "config.json"
    model_dir_path = exp_path / "model"
    model_path = model_dir_path / 'model_0.pt'
    model_jit_path = model_dir_path / 'model_jit_0.pt'
    # print(f"config_path={config_path}")
    # print(f"model_path={model_path}")
    # print(f"model_jit_path={model_jit_path}")

    with open(config_path, "r") as f:
        values = json.load(f)

    omega_net = OmegaNet(
        board_size=values["board_size"],
        n_action=values["n_action"],
        n_res_block=values["n_res_block"],
        res_filter=values["res_filter"],
        head_filter=values["head_filter"],
        value_hidden=values["value_hidden"]
    )

    torch.save(omega_net.state_dict(), model_path)

    black_board_dummy = torch.rand(1, values["board_size"], values["board_size"])
    white_board_dummy = torch.rand(1, values["board_size"], values["board_size"])
    side_dummy = torch.rand(1)
    legal_flags_dummy = torch.rand(1, values["n_action"])

    omega_net_jit = torch.jit.trace(omega_net, (black_board_dummy, white_board_dummy, side_dummy, legal_flags_dummy))
    omega_net_jit.save(str(model_jit_path))
