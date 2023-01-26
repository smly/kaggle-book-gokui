import json
import os
from pathlib import Path

import click
import pandas as pd

from common.constants import (
    NUM_DRYRUN_SAMPLES,
    NUM_TEST_SAMPLES,
    NUM_TRAIN_SAMPLES,
    OUTPUT_DIR,
)


def get_output_dir(experiment_name, dryrun=False):
    output_dir = (
        OUTPUT_DIR / "dryrun" if dryrun else OUTPUT_DIR
    ) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_model_output_path(output_dir, fold_id, is_text=False):
    return output_dir / f"model_{fold_id}.{'txt' if is_text else 'bin'}"


def get_test_prediction_output_path(output_dir, fold_id):
    return output_dir / f"tst_prediction_{fold_id}.csv"


def get_result_output_path(output_dir: Path, fold_id):
    return output_dir / f"result_{fold_id}.json"


def get_config_output_path(output_dir: Path):
    return output_dir / f"config.json"


def get_num_train_samples(dryrun=False):
    return NUM_TRAIN_SAMPLES if not dryrun else NUM_DRYRUN_SAMPLES


def get_num_test_samples(dryrun=False):
    return NUM_TEST_SAMPLES if not dryrun else NUM_DRYRUN_SAMPLES


def save_results(output_dir, config, fold_id, result, oof_df):
    oof_df.to_csv(
        output_dir / f"oof_prediction_{fold_id}.csv",
        columns=["id", "is_duplicate"],
        index=False,
    )
    get_config_output_path(output_dir).write_text(json.dumps(config, indent=2))
    get_result_output_path(output_dir, fold_id).write_text(
        json.dumps(result, indent=2)
    )


def aggregate_oof_predictions(log_dir):
    oof_df = None
    for file in log_dir.glob("oof_prediction_*.csv"):
        fold_oof_df = pd.read_csv(file)
        if oof_df is None:
            oof_df = fold_oof_df
        else:
            oof_df.loc[
                ~fold_oof_df["is_duplicate"].isnull(), "is_duplicate"
            ] = fold_oof_df[~fold_oof_df["is_duplicate"].isnull()]
    oof_df.to_csv(log_dir / "oof_prediction.csv", index=False)


def aggregate_test_predictions(log_dir):
    files = []
    submission_df = None
    for file in log_dir.glob("tst_prediction_*.csv"):
        files.append(file)
        fold_submission_df = pd.read_csv(file)
        if submission_df is None:
            submission_df = fold_submission_df
        else:
            submission_df["is_duplicate"] += fold_submission_df["is_duplicate"]

    if len(files) > 0:
        submission_df["is_duplicate"] /= len(files)
        submission_df.to_csv(log_dir / "tst_prediction.csv", index=False)
        print(
            f"Aggregated predictions {files} into {log_dir / 'tst_prediction.csv'}"
        )


def summarize_results(log_dir):
    aggregate_oof_predictions(log_dir)

    result_df = pd.DataFrame()
    for file in sorted(log_dir.glob("result_*.json")):
        result_df = result_df.append(json.load(open(file)), ignore_index=True)
    avg = result_df.mean()
    std = result_df.std()
    result_df = result_df.T
    result_df["avg"] = avg
    result_df["std"] = std
    result_df = result_df.T
    result_df.to_csv(log_dir / "summary.csv", float_format="%.5f")
    return result_df


def default_cli(experiment_file, train_1fold, predict_1fold, config):
    experiment_name = os.path.splitext(os.path.basename(experiment_file))[0]

    def train_if_necessary(dryrun, device, num_workers, fold_id):
        output_dir = get_output_dir(experiment_name, dryrun)
        result_path = get_result_output_path(output_dir, fold_id)
        if get_result_output_path(output_dir, fold_id).exists():
            print(
                "Skip training since we already have training results. "
                f"If you want to rerun training, please delete a directory {result_path}"
            )
        else:
            train_1fold(
                output_dir, fold_id, config, num_workers, device, dryrun
            )

    def predict_if_necessary(dryrun, device, num_workers, fold_id):
        output_dir = get_output_dir(experiment_name, dryrun)
        output_file = get_test_prediction_output_path(output_dir, fold_id)
        if output_file.exists():
            print(
                "Skip prediction since we already have prediction results. "
                f"If you want to rerun training, please delete a directory {output_file}"
            )
        else:
            predict_1fold(
                output_dir, fold_id, config, num_workers, device, dryrun
            )

    @click.command()
    @click.option(
        "--dryrun", is_flag=True, help="Trueの時はデバッグ用に作成した小さなデータセットを使用する"
    )
    @click.option(
        "--device",
        default="cuda",
        help="モデルの訓練・推論に用いるデバイス (複数のGPUを利用するケースは考えていない)",
    )
    @click.option(
        "--num_workers", default=4, help="DatasetLoaderで利用するのワーカーの個数"
    )
    @click.option(
        "--fold_id", default=0, help="fold_id番目の分割でtrainとvalidationのデータセットを分ける"
    )
    @click.option(
        "--fold_all",
        is_flag=True,
        help="Trueの時は0からn_splits-1までの全てのfoldで訓練する. この時fold_idのあたいは無視される",
    )
    @click.option(
        "--needs_predict", is_flag=True, help="このフラグが指定されたときは訓練後に予測も行う"
    )
    def train(dryrun, device, num_workers, fold_id, fold_all, needs_predict):
        output_dir = get_output_dir(experiment_name, dryrun)
        folds = range(config["n_splits"]) if fold_all else [fold_id]
        for fold_id in folds:
            train_if_necessary(dryrun, device, num_workers, fold_id)
            if needs_predict:
                predict_if_necessary(dryrun, device, num_workers, fold_id)
        summary_df = summarize_results(output_dir)
        print("Performance summary\n", summary_df)
        if needs_predict:
            aggregate_test_predictions(output_dir)

    return train
