import datasets

_CACHE_TRAIN_DATA_PATH = '/cognitive_comp/common_data/wudao_180g_10k_for_test/hf_cache/'


def load_dataset(**kargs):
    ds = datasets.load_from_disk(_CACHE_TRAIN_DATA_PATH)
    return ds

if __name__ == '__main__':
    from fs_datasets import load_dataset
    from datasets.arrow_dataset import Dataset
    ds = load_dataset('wudao_180g', num_proc=100)['train'][:10000]
    # print(ds)
    new_ds = Dataset.from_dict(ds)
    # print(new_ds)
    ds = datasets.DatasetDict({'train': new_ds})
    ds.save_to_disk(_CACHE_TRAIN_DATA_PATH)