import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common.constants import SPLIT_RANDOM_SEED

NUM_BOOST_ROUND = 5000
STOPPING_ROUNDS = 20
VERBOSE_EVAL = 30


def train_kfold(features, targets, n_splits, model_params, weights=None):
    if weights is None:
        weights = np.ones(len(features))

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=SPLIT_RANDOM_SEED
    )
    boosters = []
    val_losses = []
    oof_preds = np.zeros(len(features))

    for trn_idx, val_idx in skf.split(features, targets):
        trn_data = lgb.Dataset(
            data=features.iloc[trn_idx],
            label=targets.iloc[trn_idx],
            weight=weights[trn_idx],
        )
        val_data = lgb.Dataset(
            data=features.iloc[val_idx],
            label=targets.iloc[val_idx],
            weight=weights[val_idx],
            reference=trn_data,
        )
        eval_result = {}
        callbacks = [
            lgb.early_stopping(STOPPING_ROUNDS),
            lgb.log_evaluation(VERBOSE_EVAL),
            lgb.record_evaluation(eval_result),
        ]
        booster = lgb.train(
            model_params,
            trn_data,
            valid_sets=[trn_data, val_data],
            valid_names=["trn", "val"],
            num_boost_round=NUM_BOOST_ROUND,
            callbacks=callbacks,
        )
        boosters.append(booster)
        val_losses.append(
            eval_result["val"]["binary_logloss"][booster.best_iteration - 1]
        )
        oof_preds[val_idx] = booster.predict(features.iloc[val_idx])

    print(f"Avg. validation loss: {np.mean(val_losses):.4f}")
    return boosters, val_losses, oof_preds


def run_kfold(
    features, trn_targets, n_splits, model_params, save_dir, weights=None
):
    trn_features = features.iloc[: len(trn_targets)]
    tst_features = features.iloc[len(trn_targets) :]
    boosters, losses, oof_preds = train_kfold(
        trn_features, trn_targets, n_splits, model_params, weights
    )
    tst_preds = np.zeros(len(tst_features))
    for i, booster in enumerate(boosters):
        tst_preds += booster.predict(tst_features) / n_splits
        booster.save_model(str(save_dir / f"booster.{i}.txt"))
    tst_preds = pd.DataFrame(
        data={"test_id": np.arange(len(tst_preds)), "is_duplicate": tst_preds}
    )
    tst_preds.to_csv(save_dir / "tst_prediction.csv", index=False)
    oof_preds = pd.DataFrame(
        data={"id": np.arange(len(oof_preds)), "is_duplicate": oof_preds}
    )
    oof_preds.to_csv(save_dir / "oof_prediction.csv", index=False)
    (save_dir / "score.txt").write_text(str(np.mean(losses)))
