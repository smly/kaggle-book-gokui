# 5_4_extract_local_features.py
# 注意：以下の手順に従って delf や依存パッケージをセットアップする必要があります
# https://github.com/tensorflow/models/blob/master/research/delf/delf/python/delg/DELG_INSTRUCTIONS.md
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
from delf import delf_config_pb2, extractor, utils
from google.protobuf import text_format

# /opt/models に tensorflow/models をチェックアウトした場合の例
MODEL_CONFIG = (
    "/opt/models/research/"
    "delf/delf/python/delg/"
    "r101delg_gldv2clean_config.pbtxt"
)


def main():
    config = delf_config_pb2.DelfConfig()
    with tf.io.gfile.GFile(MODEL_CONFIG, 'r') as f:
        text_format.Parse(f.read(), config)

    extractor_fn = extractor.MakeExtractor(config)

    filepath_list = [
        "gldv2_micro/images/457cb65ba4a1ee3d.jpg",
        "gldv2_micro/images/1382256e230d5696.jpg",
    ]
    for filepath in filepath_list:
        resize_factor = 1.0
        im = np.array(utils.RgbLoader(filepath))

        extracted_features = extractor_fn(im, resize_factor)
        locations = extracted_features['local_features']['locations']
        descriptors = extracted_features['local_features']['descriptors']

        Path("gldv2_micro/delg_locations").mkdir(parents=True, exist_ok=True)
        delg_locations_path = Path("gldv2_micro/delg_locations") / (Path(filepath).name + ".npy")
        np.save(delg_locations_path, locations.astype(np.int16))

        Path("gldv2_micro/delg_descriptors").mkdir(parents=True, exist_ok=True)
        delg_descriptors_path = Path("gldv2_micro/delg_descriptors") / (Path(filepath).name + ".npy")
        np.save(delg_descriptors_path, descriptors.astype(np.float32))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    for idx, path in enumerate(filepath_list):
        locations_path = (
            Path("gldv2_micro/delg_locations") / (Path(path).name + ".npy")
        )
        locations = np.load(locations_path)
        im = np.array(utils.RgbLoader(path))

        axes[idx][0].imshow(im)
        axes[idx][1].imshow(im)
        axes[idx][1].plot(locations[:, 1], locations[:, 0], 'rx', markersize=4)

    plt.savefig("out/plot_local_features.png")


if __name__ == '__main__':
    main()
