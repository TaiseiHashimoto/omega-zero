import sys
import torch
import numpy as np
import argparse
import pathlib
import json
import datetime

from model import OmegaNet
from mldata import DataLoader


def save_model(model, exp_path, generation, config):
    # save as tmp file and rename afterwards
    # in order to avoid save (python) / load (c++) conflict
    # TODO: create lock file

    model_path = exp_path / "model" / f"{generation}.pt"
    tmp_path = exp_path / "model" / f"tmp_{generation}.pt"
    model_jit_path = exp_path / "model_jit" / f"{generation}.pt"
    tmp_jit_path = exp_path / "model_jit" / f"tmp_{generation}.pt"

    torch.save(model.state_dict(), tmp_path)

    # save model as ScriptModule (for c++)
    black_board_dummy = torch.rand(1, config["board_size"], config["board_size"])
    white_board_dummy = torch.rand(1, config["board_size"], config["board_size"])
    side_dummy = torch.rand(1)
    legal_flags_dummy = torch.rand(1, config["n_action"])
    model_jit = torch.jit.trace(model, (black_board_dummy, white_board_dummy, side_dummy, legal_flags_dummy))
    model_jit.save(str(tmp_jit_path))

    tmp_path.rename(str(model_path))
    tmp_jit_path.rename(str(model_jit_path))

    print(f"PY/TRAIN_MODEL  model saved! ({model_path.name}, {model_jit_path.name})")


def train(args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_id}")
    print(f"PY/TRAIN_MODEL  using {device}")

    root_path = pathlib.Path(__file__).resolve().parents[1]
    exp_path = root_path / "exp" / str(args.exp_id)
    config_path = exp_path / "config.json"
    mldata_dir_path = exp_path / "mldata"

    with open(config_path, "r") as f:
        config = json.load(f)

    omega_net = OmegaNet(
        board_size=config["board_size"],
        n_action=config["n_action"],
        n_res_block=config["n_res_block"],
        res_filter=config["res_filter"],
        head_filter=config["head_filter"],
        value_hidden=config["value_hidden"]
    )

    Q_frac = config["Q_frac"]
    batch_size = config["batch_size"]
    n_update = config["n_update"]
    n_thread = config["n_thread"]
    window_size = config["window_size"] // n_thread
    total_game_each = config["total_game"] // n_thread

    loader = DataLoader(mldata_dir_path, window_size, batch_size, n_update, n_thread)

    # find the latest model
    generation = -1
    for model_path in (exp_path / "model").glob("*.pt"):
        # delete ".pt"
        generation = max(int(model_path.name[:-3]), generation)

    print(f"PY/TRAIN_MODEL  generation initialized to {generation}")
    assert generation >= 0

    model_path = exp_path / "model" / f"{generation}.pt"
    print(f"PY/TRAIN_MODEL  load latest model {model_path.name}")
    omega_net.load_state_dict(torch.load(model_path))
    omega_net.to(device)
    optim = torch.optim.AdamW(omega_net.parameters())

    epoch = 0
    start = datetime.datetime.now()

    while True:
        loader.load_data()

        policy_loss_avg = 0
        value_loss_avg = 0
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

            value_target_b = result_b * (1 - Q_frac) + Q_b * Q_frac
            value_loss = (value_pred_b - value_target_b).pow(2).mean(dim=0)

            loss = policy_loss + value_loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            policy_loss_avg += policy_loss.item() / n_update
            value_loss_avg += value_loss.item() / n_update

        generation += 1
        elapsed = datetime.datetime.now() - start
        print(f"PY/TRAIN_MODEL  generation={generation} game_count={loader.game_count} ({elapsed})")
        entropy = -(posteriors_b * (posteriors_b + 1e-45).log()).sum(dim=1).mean(dim=0).item()
        uniform = legal_flags_b / (legal_flags_b.sum(dim=1, keepdim=True) + 1e-8)
        entropy_uni = -(uniform * (uniform + 1e-45).log()).sum(dim=1).mean(dim=0).item()
        print(f"PY/TRAIN_MODEL  policy_loss={policy_loss_avg:.3f} (entropy={entropy:.3f}, entropy_uni={entropy_uni:.3f}) value_loss={value_loss_avg:.3f}")

        omega_net.cpu()
        save_model(omega_net, exp_path, generation, config)
        omega_net.to(device)

        if (generation-1) % 25 != 0:  # delete old model
            old_model_path = exp_path / "model" / f"{generation-1}.pt"
            old_model_jit_path = exp_path / "model_jit" / f"{generation-1}.pt"
            print(f"PY/TRAIN_MODEL  unlink {old_model_path.name}, {old_model_jit_path.name}")
            old_model_path.unlink()
            old_model_jit_path.unlink()

        if loader.game_count == total_game_each:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_id', type=int)
    parser.add_argument('--device-id', type=int, default=0)
    args = parser.parse_args()

    train(args)
