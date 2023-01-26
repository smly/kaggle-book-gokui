import numpy as np
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.evaluate import compute_map_and_print

np.random.seed(1)


def main():
    datasets = {
        "roxford5k": configdataset("roxford5k", "./"),
        "rparis6k": configdataset("rparis6k", "./")
    }

    for dataset_name, dataset_config in datasets.items():
        n_images = dataset_config["n"]
        n_queries = dataset_config["nq"]
        scores = np.random.rand(n_images, n_queries)
        ranks = np.argsort(-scores, axis=0)
        compute_map_and_print(dataset_name, ranks, dataset_config["gnd"])


if __name__ == "__main__":
    main()
