"""
この kaggle notebook は GPU を必要とします。Accelerator に GPU P100 を設定してください

次のデータセットを追加してください：
* https://www.kaggle.com/datasets/confirm/gldv2micropretrained

また以下のコマンドから必要なパッケージをインストールしてください。
```
# 必要なパッケージをインストール
!pip install -q faiss-gpu
```
"""
from pathlib import Path
import sys
sys.path.append("/kaggle/input/gldv2micropretrained/")

import tqdm
import faiss
import torch
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from PIL import Image

from model import AngularModel


def get_model():
    state_dict = torch.load("/kaggle/input/gldv2micropretrained/resnext101_64x4d_size384_aug_scalerotate_colorjit_cutoff_last.pth")
    model = AngularModel(model_name="resnext101_64x4d", n_classes=3103, pretrained=False)
    model.load_state_dict(state_dict["state_dict"])
    return model.cuda()


class GL21InferenceDataset(Dataset):
    def __init__(self, filelist, transform) -> None:
        self.filelist = filelist
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index):
        filepath = Path(self.filelist[index])
        assert filepath.exists()

        im = np.array(Image.open(str(filepath)))
        im = self.transform(image=im)["image"]
        im = torch.from_numpy(im.transpose((2, 0, 1))).float()
        id_name = filepath.stem

        return im, id_name


def get_dataloaders(batch_size=8, input_size=384):
    test_transform = A.Compose([
        A.Resize(width=input_size, height=input_size),
        A.Normalize(),
    ])

    index_filelist = list(Path("/kaggle/input/landmark-retrieval-2021/index").glob("./**/*.jpg"))
    index_dataset = GL21InferenceDataset(index_filelist, test_transform)
    index_dataloader = DataLoader(
        dataset=index_dataset,
        sampler=SequentialSampler(index_dataset),
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=False,
    )

    query_filelist = list(Path("/kaggle/input/landmark-retrieval-2021/test").glob("./**/*.jpg"))
    query_dataset = GL21InferenceDataset(query_filelist, test_transform)
    query_dataloader = DataLoader(
        dataset=query_dataset,
        sampler=SequentialSampler(query_dataset),
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=False,
    )
    
    return query_dataloader, index_dataloader


model = get_model()
query_dataloader, index_dataloader = get_dataloaders(batch_size=32, input_size=384)

# Query
query_features = []
query_ids = []
for i, (batch, id_names) in tqdm.tqdm(enumerate(query_dataloader), total=len(query_dataloader)):
    with torch.no_grad():
        batch = batch.cuda()
        out = model.extract_features(batch)
        query_features.append(out.data.cpu().numpy())
        query_ids += list(id_names)

query_features = np.vstack(query_features)

# Index
gpu_resource = faiss.StandardGpuResources()
cpu_index = faiss.IndexFlatIP(512)
index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)
index_ids = []
for i, (batch, id_names) in tqdm.tqdm(enumerate(index_dataloader), total=len(index_dataloader)):
    with torch.no_grad():
        batch = batch.cuda()
        out = model.extract_features(batch)
        index_ids += list(id_names)
        index.add(out.data.cpu().numpy())

sims, topk_idx = index.search(x=query_features, k=10)
print(sims.shape, topk_idx.shape)

with open("/kaggle/working/sub.csv", "w") as f:
    f.write("id,images\n")
    for query_idx, query_id in enumerate(query_ids):
        index_images = " ".join(np.array(index_ids)[topk_idx[query_idx]].tolist())
        f.write(f"{query_id},{index_images}\n")
