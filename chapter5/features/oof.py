import pandas as pd

from common.constants import OUTPUT_DIR


def build_oof_features(exp_name):
    oof_pred = pd.read_csv(OUTPUT_DIR / exp_name / "oof_prediction.csv")
    tst_pred = pd.read_csv(OUTPUT_DIR / exp_name / "tst_prediction.csv")
    features = pd.DataFrame()
    features[f"is_duplicate_{exp_name}"] = (
        oof_pred["is_duplicate"].to_list() + tst_pred["is_duplicate"].to_list()
    )
    return features
