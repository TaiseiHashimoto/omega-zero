import argparse
import pathlib
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_id', type=int)
    parser.add_argument('--device-id', type=int, default=0)
    args = parser.parse_args()

    exp_path = pathlib.Path(f"exp/{args.exp_id}")
    if not exp_path.exists():
        print(f"exp dir {exp_path} not found")
        exit(-1)

    model_dir_path = exp_path / "model"
    mldata_dir_path = exp_path / "mldata"

    if model_dir_path.exists():
        print(f"{model_dir_path} already exists")
    else:
        print(f"mkdir {model_dir_path}")
        model_dir_path.mkdir()
        init_model_cmd = f"python python/init_model.py {args.exp_id}"
        print(init_model_cmd)
        subprocess.check_call(init_model_cmd.split())

    if mldata_dir_path.exists():
        print(f"{mldata_dir_path} already exists")
    else:
        print(f"mkdir {model_dir_path}")
        mldata_dir_path.mkdir()


    print("\nTraining start\n")

    self_play_cmd = f"./cpp/build/main {args.exp_id} --device_id {args.device_id}"
    print(self_play_cmd)
    proc_self_play = subprocess.Popen(self_play_cmd.split())

    model_train_cmd = f"python python/train_model.py {args.exp_id} --device-id {args.device_id}"
    print(model_train_cmd)
    proc_model_train = subprocess.Popen(model_train_cmd.split())

    proc_self_play.wait()
    proc_model_train.wait()

    print("\nDone!")
