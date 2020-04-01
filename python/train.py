import sys
import torch
import numpy as np
import argparse
import pathlib

from model import OmegaNet
from mldata import DataLoader


def get_window_size(generation, max_size=20, ratio=0.8):
    return min(int(np.ceil((generation+1) * ratio)), max_size)


def get_file_names(root_path, generation, delete_old=True):
    window_size = get_window_size(generation)
    print(f"window_size = {window_size}")

    file_names = []
    for i in range(generation - window_size + 1, generation + 1):
        path = root_path / pathlib.Path(f"mldata/{i}.dat")
        file_names.append(str(path))

    if delete_old:
        for i in range(generation - window_size + 1):
            path = root_path / pathlib.Path(f"mldata/{i}.dat")
            if path.exists():
                print(f"unlink {path}")
                path.unlink()

    return file_names


def train(args):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_id}")
    print(f"using {device}")

    root_path = pathlib.Path(__file__).resolve().parents[1]
    old_model_path = root_path / 'model' / f'model_{args.generation}.pt'
    new_model_path = root_path / 'model' / f'model_{args.generation+1}.pt'
    new_model_jit_path = root_path / 'model' / f'model_jit_{args.generation+1}.pt'

    if new_model_path.exists():
        print(f"ERROR: model already exists ({new_model_path})")
        exit(-1)

    omega_net = OmegaNet(
        board_size=8,
        n_action=64,
        n_res_block=5,
        res_filter=64,
        head_filter=32,
        value_hidden=64
    )

    # load latest model
    omega_net.load_state_dict(torch.load(old_model_path))
    print(f"load {old_model_path}")

    omega_net.to(device)
    optim = torch.optim.AdamW(omega_net.parameters())

    file_names = get_file_names(root_path, args.generation)
    loader = DataLoader(file_names, args.batch_size, args.n_iter, device)

    total_len = len(loader)
    count = 0
    for black_board_b, white_board_b, side_b, legal_flags_b, result_b, Q_b, posteriors_b in loader:
        policy_logit_b, value_pred_b = omega_net(black_board_b, white_board_b, side_b, legal_flags_b)

        policy_loss = -(posteriors_b * policy_logit_b).sum(dim=1).mean(dim=0)

        value_target_b = (result_b + Q_b) / 2  # TODO: how to create target
        value_loss = (value_pred_b - value_target_b).pow(2).mean(dim=0)

        loss = policy_loss + value_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        count += 1
        if count % 500 == 0:
            entropy = -(posteriors_b * (posteriors_b + 1e-45).log()).sum(dim=1).mean(dim=0).item()
            print(f"{count:6d}/{total_len:6d}  policy_loss={policy_loss:.4f} (entropy={entropy:.3f}) value_loss={value_loss:.4f}")

    omega_net.cpu()

    torch.save(omega_net.state_dict(), new_model_path)

    # save model as ScriptModule (for c++)
    black_board_s = black_board_b.cpu()
    white_board_s = white_board_b.cpu()
    side_s = side_b.cpu()
    legal_flags_s = legal_flags_b.cpu()
    omega_net_traced = torch.jit.trace(omega_net, (black_board_s, white_board_s, side_s, legal_flags_s))
    omega_net_traced.save(str(new_model_jit_path))

    print(f"model saved! ({new_model_path}, {new_model_jit_path})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # generation >= 0
    # train (generation+1)-th model using data from (generation)-th model
    parser.add_argument('generation', type=int)
    parser.add_argument('-d', '--device_id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=512)
    # n_iter approximately corresponds to # epochs
    parser.add_argument('--n-iter', type=int, default=20000)
    args = parser.parse_args()

    print(args)
    train(args)
