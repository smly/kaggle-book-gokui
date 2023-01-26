import os
import zipfile
from pathlib import Path

from gensim.models import KeyedVectors

from common.utils import simple_timer


def embedding2bin(bin_path, txt_path, zip_path, is_glove=False):
    txt_path = Path(txt_path)
    if os.path.exists(bin_path):
        print("Binary embedding already exists")
        return

    if not txt_path.exists():
        assert os.path.exists(zip_path)
        with zipfile.ZipFile(zip_path) as existing_zip:
            existing_zip.extract(txt_path.name, txt_path.parent)
            print(f"Extracted {txt_path}")
    assert txt_path.exists()

    with simple_timer(f"Load {txt_path}"):
        model = KeyedVectors.load_word2vec_format(txt_path, no_header=is_glove)
    model.save_word2vec_format(
        bin_path, total_vec=len(model.key_to_index), binary=True
    )
    os.remove(txt_path)
