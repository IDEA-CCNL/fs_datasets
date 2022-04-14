import datasets

ds = datasets.load_from_disk('/cognitive_comp/common_data/wudao_10k_for_test/hf_cache/')


_CACHE_TRAIN_DATA_PATH = '/cognitive_comp/common_data/wudao_10k_for_test/hf_cache/'


def load_dataset(**kargs):
    ds = datasets.load_from_disk(_CACHE_TRAIN_DATA_PATH)
    return ds
