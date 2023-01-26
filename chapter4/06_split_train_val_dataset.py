import pandas as pd

df = pd.read_csv("gldv2_micro/gldv2_micro.csv")
df["landmark_id"] = df["landmark_id"].factorize()[0]
df_val = df.groupby("landmark_id").sample(n=1, random_state=1)
df_trn = df[~df["filename"].isin(df_val["filename"])]
df_trn.to_csv("gldv2_micro/train.csv", index=False)
df_val.to_csv("gldv2_micro/val.csv", index=False)
