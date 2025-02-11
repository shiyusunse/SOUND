import os, sys

from gensim.models import Word2Vec

import more_itertools

from DeepLineDP_model import *
from sound.src.utils.config_for_deeplinedp import *


def train_word2vec_model(dataset_name, embedding_dim=50):
    w2v_path = get_w2v_path()

    if not os.path.exists(w2v_path):
        os.makedirs(w2v_path)

    releases = all_releases[dataset_name]

    for i, release in enumerate(releases[:-1]):
        save_path = f"{w2v_path}/{release}-{embedding_dim}dim.bin"

        if os.path.exists(save_path):
            print(f"Word2Vec model at {save_path} already exists")
            continue

        train_df = get_df(release)
        train_code_3d, _ = get_code3d_and_label(train_df, True)
        all_texts = list(more_itertools.collapse(train_code_3d[:], levels=1))

        word2vec = Word2Vec(all_texts, size=embedding_dim, min_count=1, sorted_vocab=1)
        word2vec.save(save_path)
        print(f"Saved Word2Vec model at path {save_path}")


if __name__ == "__main__":
    for dataset_name in all_releases.keys():
        train_word2vec_model(dataset_name, 50)
