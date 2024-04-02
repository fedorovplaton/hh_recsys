import faiss
import numpy as np
import gc
import pickle
import os
from tqdm import tqdm
from os.path import join as path_join

path = "../data/raw/hh_recsys_vacancies.pq"
output_path = "dumps/production/i2i/paraphrase_multilingual_MiniLM_L12_v2_384d.parquet"

vacancies_tmp_path = "dumps/production/i2i/tmp_vacancies_embeddings"

embedding_root = "dumps/production/i2i/paraphrase_multilingual_MiniLM_L12_v2_384d"
shards_root    = "dumps/production/i2i/paraphrase_multilingual_MiniLM_L12_v2_384d/shards"
indexes_root   = "dumps/production/i2i/paraphrase_multilingual_MiniLM_L12_v2_384d/indexes"

if not os.path.exists(embedding_root):
    os.makedirs(embedding_root)

if not os.path.exists(shards_root):
    os.makedirs(shards_root)

if not os.path.exists(indexes_root):
    os.makedirs(indexes_root)

embeddings = pickle.load(open(vacancies_tmp_path, 'rb'))
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

total_n = len(embeddings)
max_i2i_in_shard = 25000
shards_n = (total_n // max_i2i_in_shard) + (0 if total_n % max_i2i_in_shard == 0 else 1)

del embeddings
gc.collect()

for i in tqdm(range(shards_n), total=shards_n):
    idx_from = max_i2i_in_shard * i
    idx_to = max_i2i_in_shard * i + max_i2i_in_shard

    print(f"[{idx_from}, {idx_to})")

    embeddings = pickle.load(open(vacancies_tmp_path, 'rb'))[idx_from:idx_to]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.float32(embeddings)

    neighbors = []

    for index_path in list(map(lambda x: path_join(indexes_root, x), os.listdir(indexes_root))):
        index = faiss.read_index(index_path)
        D, I = index.search(embeddings, 10)

        break

        # del index
        # gc.collect()
        # print(D, I)

    # shard_output_path = path_join(shards_root, f"shard_{idx_from}_{idx_to}")

    # df = pd.DataFrame.from_dict({"idx": list(range(idx_from, idx_to)), "neighbour_idx": I})
    # print(df.head(2))

    # df.to_parquet(shard_output_path)

    if i >= 2:
        break