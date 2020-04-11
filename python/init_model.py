import torch
from model import OmegaNet
import pathlib
import argparse
import json
import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_id', type=int)
    args = parser.parse_args()

    root_path = pathlib.Path(__file__).resolve().parents[1]
    exp_path = root_path / "exp" / str(args.exp_id)
    config_path = exp_path / "config.json"
    model_path = exp_path / "model"
    new_model_path = model_path / 'model_0.pt'
    new_model_jit_path = model_path / 'model_jit_0.pt'
    best_model_path = model_path / 'model_best.pt'
    best_model_jit_path = model_path / 'model_jit_best.pt'
    # print(f"config_path={config_path}")
    # print(f"new_model_path={new_model_path}")
    # print(f"new_model_jit_path={new_model_jit_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    omega_net = OmegaNet(
        board_size=config["board_size"],
        n_action=config["n_action"],
        n_res_block=config["n_res_block"],
        res_filter=config["res_filter"],
        policy_filter=config["policy_filter"],
        value_filter=config["value_filter"],
        value_hidden=config["value_hidden"]
    )

    torch.save(omega_net.state_dict(), new_model_path)

    black_board_dummy = torch.rand(1, config["board_size"], config["board_size"])
    white_board_dummy = torch.rand(1, config["board_size"], config["board_size"])
    side_dummy = torch.rand(1)
    legal_flags_dummy = torch.rand(1, config["n_action"])

    omega_net_jit = torch.jit.trace(omega_net, (black_board_dummy, white_board_dummy, side_dummy, legal_flags_dummy))
    omega_net_jit.save(str(new_model_jit_path))

    shutil.copy(new_model_path, best_model_path)
    shutil.copy(new_model_jit_path, best_model_jit_path)
