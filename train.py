import argparse
import pathlib
import subprocess
import re
import datetime
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_id', type=int)
    parser.add_argument('--device-id', type=int, default=0)
    args = parser.parse_args()

    exp_path = pathlib.Path(f"exp/{args.exp_id}")
    config_path = exp_path / "config.json"
    model_dir_path = exp_path / "model"
    mldata_dir_path = exp_path / "mldata"

    with open(config_path, "r") as f:
        config = json.load(f)
    end_generation = config["end_generation"]

    if model_dir_path.exists():
        print(f"{model_dir_path} already exists")
    else:
        model_dir_path.mkdir()
        init_model_cmd = f"python python/init_model.py {args.exp_id}"
        print(init_model_cmd)
        subprocess.check_call(init_model_cmd.split())

    start_time = datetime.datetime.now()

    if mldata_dir_path.exists():
        generations = []
        for file_path in mldata_dir_path.glob('*.dat'):
            generations.append(int(file_path.name[:-4]))
        print(f"generations = {generations}")
        assert len(generations) > 0
        start_generation = max(generations)
    else:
        mldata_dir_path.mkdir()
        start_generation = 0


    for generation in range(start_generation, end_generation):
        print(f"generation = {generation}")

        mldata_path = mldata_dir_path / f'{generation}.dat'
        if mldata_path.exists():
            print(f"\n{mldata_path} already exists")
        else:
            self_play_cmd = f"./cpp/build/main {args.exp_id} {generation} --device_id {args.device_id}"
            print("\n" + self_play_cmd)
            subprocess.check_call(self_play_cmd.split())

            print("\nmerge mldata files")
            with open(mldata_dir_path / f"{generation}.dat", "wb") as output_file:
                for input_path in mldata_dir_path.glob(f"{generation}_*.dat"):
                    with open(input_path, 'rb') as input_file:
                        output_file.write(input_file.read())
                    input_path.unlink()

        model_path = model_dir_path / f"model_{generation+1}.pt"
        if model_path.exists():
            print(f"\n{model_path} already exists")
        else:
            Q_frac = config["Q_frac"]
            if config["anneal_Q_frac"]:
                Q_frac *= generation / end_generation
            model_train_cmd = f"python python/train_model.py {args.exp_id} {generation} --device-id {args.device_id} --Q-frac {Q_frac}"
            print("\n" + model_train_cmd)
            subprocess.check_call(model_train_cmd.split())

        delta1 = datetime.datetime.now() - start_time
        remaining_sec = delta1.total_seconds() * (end_generation-generation-1) / (generation-start_generation+1)
        delta2 = datetime.timedelta(seconds=remaining_sec)
        print(f"\nelapsed time: {delta1}  remaining time: {delta2}\n")
