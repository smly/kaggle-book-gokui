# 注意：以下の手順に従って delf や依存パッケージをセットアップする必要があります
# https://github.com/tensorflow/models/blob/master/research/delf/delf/python/delg/DELG_INSTRUCTIONS.md
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydegensac
import tensorflow as tf
from delf import delf_config_pb2, extractor, utils
from google.protobuf import text_format
from scipy import spatial
from skimage.feature import plot_matches


def main():
    config = delf_config_pb2.DelfConfig()
    with tf.io.gfile.GFile("models/research/delf/delf/python/delg/r101delg_gldv2clean_config.pbtxt", 'r') as f:
        text_format.Parse(f.read(), config)

    extractor_fn = extractor.MakeExtractor(config)

    filepath_list = [
        "gldv2_micro/images/457cb65ba4a1ee3d.jpg",
        "gldv2_micro/images/1382256e230d5696.jpg",
    ]
    results = {}
    for filepath in filepath_list:
        resize_factor = 1.0
        im = np.array(utils.RgbLoader(filepath))

        extracted_features = extractor_fn(im, resize_factor)

        # 大域特徴量
        global_descriptor = extracted_features['global_descriptor']

        # 局所特徴量
        locations = extracted_features['local_features']['locations']
        descriptors = extracted_features['local_features']['descriptors']
        feature_scales = extracted_features['local_features']['scales']
        attention = extracted_features['local_features']['attention']

        results[filepath] = {
            "locations": locations.astype(np.int16),
            "descriptors": descriptors.astype(np.float32),
            "im": im,
        }

    kps1 = results[filepath_list[0]]["locations"]
    descs1 = (results[filepath_list[0]]["descriptors"])
    im1 = results[filepath_list[0]]["im"]

    kps2 = results[filepath_list[1]]["locations"]
    descs2 = (results[filepath_list[1]]["descriptors"])
    im2 = results[filepath_list[1]]["im"]

    _DISTANCE_THRESHOLD = 0.8
    d1_tree = spatial.cKDTree(descs1)
    _, indices = d1_tree.query(descs2, distance_upper_bound=_DISTANCE_THRESHOLD)

    threshold = 4.0
    conf = 0.99
    n_iters = 2000

    pts_2 = np.array([kps2[i,] for i in range(kps2.shape[0])
                    if indices[i] != kps1.shape[0]])
    pts_1 = np.array([kps1[indices[i],] for i in range(kps1.shape[0])
                    if indices[i] != kps1.shape[0]])
    _, mask = pydegensac.findFundamentalMatrix(pts_1, pts_2, threshold, conf, n_iters)

    print ('pydegensac found {} inliers'.format(int(deepcopy(mask).astype(np.float32).sum())))
    inlier_idxs = np.nonzero(mask)[0]
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=300)
    plot_matches(
        ax,
        im1,
        im2,
        pts_1,
        pts_2,
        np.column_stack((inlier_idxs, inlier_idxs)),
        matches_color='r')
    plt.savefig("out/plot_image_matching.png")


if __name__ == '__main__':
    main()
