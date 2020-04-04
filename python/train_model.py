import sys
import torch
import numpy as np
import argparse
import pathlib
import json
import time

from model import OmegaNet
from mldata import DataLoader


def get_window_size(generation, max_size=20, ratio=0.6):
    return min(int(np.ceil((generation+1) * ratio)), max_size)


def get_file_names(exp_path, generation, delete_old=True):
    window_size = get_window_size(generation)
    print(f"window_size = {window_size}")

    file_names = []
    for i in range(generation - window_size + 1, generation + 1):
        path = exp_path / pathlib.Path(f"mldata/{i}.dat")
        file_names.append(str(path))

    if delete_old:
        for i in range(generation - window_size + 1):
            path = exp_path / pathlib.Path(f"mldata/{i}.dat")
            if path.exists():
                print(f"unlink {path}")
                path.unlink()

    return file_names


def train(args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_id}")
    print(f"using {device}")

    root_path = pathlib.Path(__file__).resolve().parents[1]
    exp_path = root_path / "exp" / str(args.exp_id)
    config_path = exp_path / "config.json"
    old_model_path = exp_path / "model" / f"model_{args.generation}.pt"
    old_model_jit_path = exp_path / "model" / f"model_jit_{args.generation}.pt"
    new_model_path = exp_path / "model" / f"model_{args.generation+1}.pt"
    new_model_jit_path = exp_path / "model" / f"model_jit_{args.generation+1}.pt"

    if new_model_path.exists():
        print(f"ERROR: model already exists ({new_model_path})")
        exit(-1)

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

    epoch = values["epoch"]
    batch_size = values["batch_size"]

    # load latest model
    omega_net.load_state_dict(torch.load(old_model_path))
    print(f"load {old_model_path}")

    omega_net.to(device)
    optim = torch.optim.AdamW(omega_net.parameters())

    file_names = get_file_names(exp_path, args.generation)
    start = time.time()
    loader = DataLoader(file_names, batch_size)
    elapsed = time.time() - start
    print(f"load time : {elapsed:.2f} sec")

    start = time.time()
    for e in range(epoch):
        for black_board_b, white_board_b, side_b, legal_flags_b, result_b, Q_b, posteriors_b in loader:
            black_board_b = black_board_b.to(device)
            white_board_b = white_board_b.to(device)
            side_b = side_b.to(device)
            legal_flags_b = legal_flags_b.to(device)
            result_b = result_b.to(device)
            Q_b = Q_b.to(device)
            posteriors_b = posteriors_b.to(device)

            policy_logit_b, value_pred_b = omega_net(black_board_b, white_board_b, side_b, legal_flags_b)

            policy_loss = -(posteriors_b * policy_logit_b).sum(dim=1).mean(dim=0)

            value_target_b = result_b * (1 - args.Q_frac) + Q_b * args.Q_frac
            value_loss = (value_pred_b - value_target_b).pow(2).mean(dim=0)

            loss = policy_loss + value_loss
            optim.zero_grad()
            loss.backward()
            optim.step()

        entropy = -(posteriors_b * (posteriors_b + 1e-45).log()).sum(dim=1).mean(dim=0).item()
        uniform = legal_flags_b / (legal_flags_b.sum(dim=1, keepdim=True) + 1e-8)
        entropy_uni = -(uniform * (uniform + 1e-45).log()).sum(dim=1).mean(dim=0).item()
        elapsed = time.time() - start
        print(f"epoch={e+1}  ({elapsed:.2f} sec)  policy_loss={policy_loss:.3f} (entropy={entropy:.3f}, entropy_uni={entropy_uni:.3f}) value_loss={value_loss:.3f}")


    omega_net.cpu()
    torch.save(omega_net.state_dict(), new_model_path)

    # save model as ScriptModule (for c++)
    black_board_s = black_board_b[:1].cpu()
    white_board_s = white_board_b[:1].cpu()
    side_s = side_b[:1].cpu()
    legal_flags_s = legal_flags_b[:1].cpu()
    omega_net_traced = torch.jit.trace(omega_net, (black_board_s, white_board_s, side_s, legal_flags_s))
    omega_net_traced.save(str(new_model_jit_path))

    print(f"model saved! ({new_model_path}, {new_model_jit_path})")

    if args.generation % 5 != 0:  # delete old model
        print(f"unlink {old_model_path}, {old_model_jit_path}")
        old_model_path.unlink()
        old_model_jit_path.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # generation >= 0
    # train (generation+1)-th model using data from (generation)-th model
    parser.add_argument('exp_id', type=int)
    parser.add_argument('generation', type=int)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--Q-frac', type=float, default=0)
    args = parser.parse_args()

    print(args)
    train(args)
