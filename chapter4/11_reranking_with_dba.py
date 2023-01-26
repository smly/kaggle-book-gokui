# 5_4_extract_local_features.py
# 注意：以下の手順に従って delf や依存パッケージをセットアップする必要があります
# https://github.com/tensorflow/models/blob/master/research/delf/delf/python/delg/DELG_INSTRUCTIONS.md
from pathlib import Path

import faiss
import numpy as np
import pydegensac
import tensorflow as tf
import tqdm
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.evaluate import compute_map_and_print
from delf import delf_config_pb2, extractor, utils
from google.protobuf import text_format
from scipy import spatial

# /opt/models に tensorflow/models をチェックアウトした場合の例
MODEL_CONFIG = (
    "/opt/models/research/"
    "delf/delf/python/delg/"
    "r101delg_gldv2clean_config.pbtxt"
)


def search_with_faiss_cpu(feat_test, feat_index, topk=5):
    feat_index = np.ascontiguousarray(feat_index, dtype=np.float32)
    feat_test = np.ascontiguousarray(feat_test, dtype=np.float32)

    n_dim = feat_index.shape[1]
    cpu_index = faiss.IndexFlatIP(n_dim)
    cpu_index.add(feat_index)

    dists, topk_idx = cpu_index.search(x=feat_test, k=topk)

    cpu_index.reset()
    del cpu_index

    return dists, topk_idx


def l2norm_numpy(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)


def qe_dba(
    feats_test, feats_index, sims, topk_idx, alpha=3.0, qe=True, dba=True
):
    # alpha-query expansion (DBA)
    feats_concat = np.concatenate([feats_test, feats_index], axis=0)
    weights = sims ** alpha

    feats_tmp = np.zeros(feats_concat.shape)
    for i in range(feats_concat.shape[0]):
        feats_tmp[i, :] = weights[i].dot(feats_concat[topk_idx[i]])

    del feats_concat
    feats_concat = l2norm_numpy(feats_tmp).astype(np.float32)

    split_at = [len(feats_test)]
    if qe and dba:
        return np.split(feats_concat, split_at, axis=0)
    elif not qe and dba:
        _, feats_index = np.split(feats_concat, split_at, axis=0)
        return feats_test, feats_index
    elif qe and not dba:
        feats_test, _ = np.split(feats_concat, split_at, axis=0)
        return feats_test, feats_index
    else:
        raise ValueError


def get_query_index_images(cfg):
    index_images = [cfg["im_fname"](cfg, i) for i in range(cfg["n"])]
    query_images = [cfg["qim_fname"](cfg, i) for i in range(cfg["nq"])]

    try:
        bbxs = [tuple(cfg["gnd"][i]["bbx"]) for i in range(cfg["nq"])]
    except KeyError:
        bbxs = None

    return index_images, query_images, bbxs


def get_extractor_function():
    config = delf_config_pb2.DelfConfig()
    with tf.io.gfile.GFile(MODEL_CONFIG, 'r') as f:
        text_format.Parse(f.read(), config)

    return extractor.MakeExtractor(config)


def extract_local_features():
    datasets = {
        "roxford5k": configdataset("roxford5k", "./"),
        "rparis6k": configdataset("rparis6k", "./")
    }

    extractor_fn = get_extractor_function()

    # それぞれのデータセットで大域特徴を抽出して、中間ファイルに保存
    resize_factor = 1.0
    for dataset_name, dataset_config in datasets.items():
        index_images, query_images, bbxs = get_query_index_images(
            dataset_config
        )
        all_images = index_images + query_images
        for image_path in tqdm.tqdm(all_images):
            delg_locations_path = Path(
                f"{dataset_name}/delg_locations"
            ) / (Path(image_path).name + ".npy")
            delg_descriptors_path = Path(
                f"{dataset_name}/delg_descriptors"
            ) / (Path(image_path).name + ".npy")
            delg_locations_path.parent.mkdir(parents=True, exist_ok=True)
            delg_descriptors_path.parent.mkdir(parents=True, exist_ok=True)

            im = np.array(utils.RgbLoader(image_path))
            extracted_features = extractor_fn(im, resize_factor)
            locations = extracted_features['local_features']['locations']
            descriptors = extracted_features['local_features']['descriptors']
            np.save(delg_locations_path, locations.astype(np.int16))
            np.save(delg_descriptors_path, descriptors.astype(np.float32))


def count_inliers(dataset_name, index_path, query_path):
    kps1 = np.load(Path(
        f"{dataset_name}/delg_locations"
    ) / (Path(index_path).name + ".npy"))
    desc1 = np.load(Path(
        f"{dataset_name}/delg_descriptors"
    ) / (Path(index_path).name + ".npy"))

    kps2 = np.load(Path(
        f"{dataset_name}/delg_locations"
    ) / (Path(query_path).name + ".npy"))
    desc2 = np.load(Path(
        f"{dataset_name}/delg_descriptors"
    ) / (Path(query_path).name + ".npy"))

    _DISTANCE_THRESHOLD = 0.8
    d1_tree = spatial.cKDTree(desc1)
    _, indices = d1_tree.query(desc2, distance_upper_bound=_DISTANCE_THRESHOLD)

    threshold = 4.0
    conf = 0.99
    n_iters = 2000

    try:
        pts_2 = np.array([kps2[i,] for i in range(kps2.shape[0])
                        if indices[i] != kps1.shape[0]])
        pts_1 = np.array([kps1[indices[i],] for i in range(kps1.shape[0])
                        if indices[i] != kps1.shape[0]])
        _, mask = pydegensac.findFundamentalMatrix(pts_1, pts_2, threshold, conf, n_iters)
    except Exception:
        return 0

    inlier_count = int(mask.astype(np.float32).sum())
    return inlier_count


def reranking(ranks, dataset_name, index_images, query_images, topk=10):
    for j, query_image_path in tqdm.tqdm(enumerate(query_images)):
        topk_new_scores = np.array([
            count_inliers(dataset_name, index_images[ranks[rank_i, j]], query_image_path)
            for rank_i in range(topk)
        ])
        ranks[:topk, j] = ranks[:topk, j][np.argsort(-topk_new_scores)]

    return ranks


def evaluate_dba_with_reranking():
    datasets = {
        "roxford5k": configdataset("roxford5k", "./"),
        "rparis6k": configdataset("rparis6k", "./")
    }

    # 大域特徴をロードして、内積に基づいて順位付けして評価
    for dataset_name, dataset_config in datasets.items():
        index_images, query_images, bbxs = get_query_index_images(
            dataset_config
        )

        # shape = (n_dims, n_images)
        index_vectors = np.load(f"{dataset_name}_index.npy")
        query_vectors = np.load(f"{dataset_name}_query.npy")

        # shape = (n_images, n_dims)
        index_vectors = index_vectors.T
        query_vectors = query_vectors.T
        index_vectors = l2norm_numpy(index_vectors)
        query_vectors = l2norm_numpy(query_vectors)

        # reranking
        n_qe, alpha = 3, 3.0
        feat_all = np.concatenate([query_vectors, index_vectors], axis=0)
        dists, topk_idx = search_with_faiss_cpu(feat_all, feat_all, topk=n_qe)
        query_vectors, index_vectors = qe_dba(
            query_vectors, index_vectors, dists, topk_idx, alpha=alpha)

        scores = np.dot(query_vectors, index_vectors.T)
        ranks = np.argsort(-scores, axis=1).T
        ranks = reranking(ranks, dataset_name, index_images, query_images, topk=20)
        compute_map_and_print(dataset_name, ranks, dataset_config["gnd"])


def main():
    evaluate_dba_with_reranking()


if __name__ == "__main__":
    main()
