import sys
import torch
import numpy as np
import argparse
import pathlib
import json
import time

from model import OmegaNet
from mldata import DataLoader


def get_window_size(generation, window_size_max, ratio=0.6):
    return min(int(np.ceil((generation+1) * ratio)), window_size_max)


def get_file_paths(exp_path, generation, window_size_max):
    window_size = get_window_size(generation, window_size_max)
    print(f"window_size = {window_size}")

    file_paths = []
    for i in range(generation - window_size + 1, generation + 1):
        path = exp_path / pathlib.Path(f"mldata/{i}.dat")
        file_paths.append(path)

    for i in range(generation - window_size + 1):
        path = exp_path / "mldata" / f"{i}.dat"
        if path.exists():
            print(f"unlink {path.name}")
            path.unlink()
        bu_path = exp_path / "mldata" / f"{i}.pt"
        if bu_path.exists():
            print(f"unlink {bu_path.name}")
            bu_path.unlink()

    return file_paths


def train(args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_id}")
    print(f"using {device}")

    root_path = pathlib.Path(__file__).resolve().parents[1]
    exp_path = root_path / "exp" / str(args.exp_id)
    config_path = exp_path / "config.json"
    best_model_path = exp_path / "model" / f"model_best.pt"
    new_model_path = exp_path / "model" / f"model_{args.generation+1}.pt"
    new_model_jit_path = exp_path / "model" / f"model_jit_{args.generation+1}.pt"
    # optim_path = exp_path / "model" / "optim.pt"

    if new_model_path.exists():
        print(f"ERROR: model already exists ({new_model_path})")
        exit(-1)

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

    # load latest model
    omega_net.load_state_dict(torch.load(best_model_path))
    print(f"load {best_model_path.name}")

    omega_net.to(device)
    optim = torch.optim.AdamW(omega_net.parameters(), weight_decay=config["weight_decay"])
    assert omega_net.training

    # if optim_path.exists():
    #     print(f"load optimizer {optim_path.name}")
    #     optim.load_state_dict(torch.load(optim_path))

    file_paths = get_file_paths(exp_path, args.generation, config["window_size_max"])
    start = time.time()
    loader = DataLoader(file_paths, config["batch_size"], config["augmentation"], config["unique"])
    elapsed = time.time() - start
    print(f"load time : {elapsed:.2f} sec")

    n_batch = len(loader)
    start = time.time()
    for e in range(config["epoch"]):
        policy_loss_avg = 0
        value_loss_avg = 0
        entropy_avg = 0
        # loss_uni_avg = 0
        for black_board_b, white_board_b, side_b, legal_flags_b, result_b, posteriors_b in loader:
            black_board_b = black_board_b.to(device)
            white_board_b = white_board_b.to(device)
            side_b = side_b.to(device)
            legal_flags_b = legal_flags_b.to(device)
            result_b = result_b.to(device)
            posteriors_b = posteriors_b.to(device)

            policy_logit_b, value_pred_b = omega_net(black_board_b, white_board_b, side_b, legal_flags_b)

            policy_loss = -(posteriors_b * policy_logit_b).sum(dim=1).mean(dim=0)

            value_loss = (value_pred_b - result_b).pow(2).mean(dim=0)

            loss = policy_loss + value_loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            with torch.no_grad():
                policy_loss_avg += policy_loss.item() / n_batch
                value_loss_avg += value_loss.item() / n_batch
                entropy_avg += -(posteriors_b * (posteriors_b + 1e-45).log()).sum(dim=1).mean(dim=0).item() / n_batch

        elapsed = time.time() - start
        print(f"epoch={e+1}  ({elapsed:.2f} sec)  policy_loss={policy_loss_avg:.3f} (entropy={entropy_avg:.3f}) value_loss={value_loss_avg:.3f}")


    omega_net.cpu()
    torch.save(omega_net.state_dict(), new_model_path)
    # torch.save(optim.state_dict(), optim_path)

    # save model as ScriptModule (for c++)
    black_board_s = black_board_b[:1].cpu()
    white_board_s = white_board_b[:1].cpu()
    side_s = side_b[:1].cpu()
    legal_flags_s = legal_flags_b[:1].cpu()
    omega_net_traced = torch.jit.trace(omega_net, (black_board_s, white_board_s, side_s, legal_flags_s))
    omega_net_traced.save(str(new_model_jit_path))

    print(f"save model {new_model_path.name}, {new_model_jit_path.name}")


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
