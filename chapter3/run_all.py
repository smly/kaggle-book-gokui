import argparse
import datetime
import json
import pathlib
import subprocess
import time

import pandas as pd


def read_val_score(stdout_path):
    with open(stdout_path) as f:
        lines = f.readlines()

    last_line = lines[-1]

    if last_line.startswith("val loss="):
        return {
            "val method": "CV",
            "val score": float(last_line.split("=")[1]),
        }
    else:
        tokens = last_line.split(",")
        tokens = [token.strip() for token in tokens]
        tokens = [token for token in tokens if token.startswith("val loss=")]
        assert len(tokens) == 1
        token = tokens[0]
        return {"val method": "split", "val score": float(token.split("=")[1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="./out")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--configs")
    args = parser.parse_args()

    root_out_dir = pathlib.Path(args.out_dir)
    root_out_dir = root_out_dir / datetime.datetime.now().strftime(
        "runall_%Y-%m-%d_%H-%M-%S"
    )
    print("Out dir:", root_out_dir)

    if args.configs:
        configs = args.configs.split(",")
    else:
        configs = ["5", "6", "7-1", "7-3", "8-1", "8-2", "9-1"]
    print(f"Configs: {configs}")

    results = []
    for config in configs:
        out_dir = root_out_dir / f"config{config}"
        out_dir.mkdir(parents=True)

        #
        # Train
        #
        argv = [
            "python",
            "train.py",
            "--data_dir",
            args.data_dir,
            "--out_dir",
            str(out_dir),
            "--device",
            args.device,
            "--config",
            config,
        ]
        if args.dryrun:
            argv.append("--dryrun")

        stdout_path = out_dir / "stdout.txt"
        with open(stdout_path, "w") as f:
            subprocess.run(argv, stdout=f)

        #
        # Report
        #
        val_report = read_val_score(stdout_path)
        val_report["config"] = config
        results.append(val_report)
        print(val_report)
        with open(out_dir / "report.json", "w") as f:
            json.dump(val_report, f)

        #
        # Submit
        #
        csv_path = out_dir / "out_clip.csv"
        argv = [
            "kaggle",
            "competitions",
            "submit",
            "-f",
            str(csv_path),
            "-m",
            str(csv_path),
            "dogs-vs-cats-redux-kernels-edition",
        ]
        if not args.dryrun:
            subprocess.run(argv)

    print(pd.DataFrame(results))

    time.sleep(60)  # スコアが出るまで少し時間がかかる
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "submissions",
            "dogs-vs-cats-redux-kernels-edition",
        ]
    )


if __name__ == "__main__":
    main()
