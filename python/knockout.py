import argparse
import pathlib
import shutil
import subprocess
import re
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", type=int)
    parser.add_argument("generation", type=int)
    parser.add_argument("--n-games", type=int, default=3)
    parser.add_argument("--n-simulation", type=int, default=50)
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    root_path = pathlib.Path(__file__).resolve().parents[1]
    python_path = root_path / "python"
    build_path = root_path / "cpp" / "build"
    exp_path = root_path / "exp" / str(args.exp_id)
    model_path = exp_path / "model" / f"model_{args.generation}.pt"
    model_jit_path = exp_path / "model" / f"model_jit_{args.generation}.pt"
    best_model_path = exp_path / "model" / f"model_best.pt"
    best_model_jit_path = exp_path / "model" / f"model_jit_best.pt"

    file = open(exp_path / f"ko_record_{args.exp_id}.txt", 'a')
    file.write(f"{args.generation}\n")

    cmd1 = f"{build_path}/play {args.exp_id} --generation {args.generation} --n_simulation {args.n_simulation} --device_id {args.device_id}"
    cmd2 = f"{build_path}/play {args.exp_id} --n_simulation {args.n_simulation} --device_id {args.device_id}"

    update = False
    start = time.time()

    for side1 in ['b', 'w']:
        cmd = f"python {python_path}/mediator.py \"{cmd1}\" \"{cmd2}\" --n-games {args.n_games} --side {side1} --quiet"
        print(cmd)
        result = subprocess.check_output(cmd, shell=True).decode('utf-8')

        if side1 == 'b':
            mr = re.search(r"black win rate = ([0-9\.]+) %", result)
        else:
            mr = re.search(r"white win rate = ([0-9\.]+) %", result)
        assert mr

        win_rate = float(mr.group(1))
        file.write(f"{side1} {win_rate:.4f}\n")
        file.flush()

        if win_rate < args.threshold:
            break
        if side1 == 'w':
            update = True

    elapsed = time.time() - start
    print(f"elapsed : {elapsed:.2f} sec")

    if update:
        file.write("update best model\n")
        print("update best model")
        shutil.copy(model_path, best_model_path)
        shutil.copy(model_jit_path, best_model_jit_path)

    if args.generation % 10 != 0:  # delete old model
        print(f"unlink {model_path.name}, {model_jit_path.name}")
        model_path.unlink()
        model_jit_path.unlink()

    file.close()
