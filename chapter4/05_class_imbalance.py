import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv("gldv2_micro/gldv2_micro.csv")
    label_counts = df["landmark_id"].value_counts()

    df["landmark_id"] = df["landmark_id"].factorize()[0]
    df_val = df.groupby("landmark_id").sample(n=1, random_state=1)
    df_trn = df[~df["filename"].isin(df_val["filename"])]
    df_trn.to_csv("gldv2_micro/train.csv", index=False)
    df_val.to_csv("gldv2_micro/val.csv", index=False)

    df = pd.read_csv("gldv2_micro/gldv2_micro.csv")
    plt.hist(label_counts, bins=range(label_counts.max() + 1))
    plt.xlim(xmin=0)
    plt.xlabel("The number of occurences")
    plt.ylabel("The number of classes")
    plt.savefig("out/plot_class_histogram.png")


if __name__ == "__main__":
    main()