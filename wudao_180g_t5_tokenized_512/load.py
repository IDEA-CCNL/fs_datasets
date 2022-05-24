import datasets
import glob
import os
from concurrent.futures import ProcessPoolExecutor

_SPLIT_DATA_PATH = ''
# 缓存文件
_CACHE_TRAIN_DATA_PATH = '/cognitive_comp/common_data/wudao_180g_t5_tokenized_512/'
_CACHE_TRAIN_DATA_PATH_TRAIN = '/cognitive_comp/common_data/wudao_180g_t5_tokenized_512_train/'
_CACHE_TRAIN_DATA_PATH_TEST = '/cognitive_comp/common_data/wudao_180g_t5_tokenized_512_test/'


feats = datasets.Features({"input_ids": datasets.Value('int32')})


def load_old_dataset(num_proc=1, **kargs):
    cache_dict_paths = glob.glob(os.path.join(_CACHE_TRAIN_DATA_PATH, '*'))
    ds = []
    res = []
    p = ProcessPoolExecutor(max_workers=num_proc)
    for path in cache_dict_paths:
        res.append(p.submit(datasets.load_from_disk,
                            path, **kargs))

    p.shutdown(wait=True)
    for future in res:
        ds.append(future.result())
    return datasets.DatasetDict({"train": datasets.concatenate_datasets(ds)})

def load_dataset(num_proc=1, **kargs):
    '''
    加载缓存的数据
    '''
    cache_dict_paths = glob.glob(os.path.join(_CACHE_TRAIN_DATA_PATH_TRAIN, '*'))
    ds = []
    res = []
    p = ProcessPoolExecutor(max_workers=num_proc)
    for path in cache_dict_paths:
        res.append(p.submit(datasets.load_from_disk,
                            path, **kargs))

    p.shutdown(wait=True)
    for future in res:
        ds.append(future.result())
    train_ds = datasets.concatenate_datasets(ds)
    test_ds = datasets.load_from_disk(_CACHE_TRAIN_DATA_PATH_TEST)
    return datasets.DatasetDict({
        "train": train_ds,
        "test": test_ds})


def _generate_cache_arrow(index, ds):
    print('saving dataset shard {}'.format(index))
    ds.save_to_disk(os.path.join(_CACHE_TRAIN_DATA_PATH_TRAIN, 'part_{}'.format(index)))
    return 'saving dataset shard {} done'.format(index)


def generate_arrow_cache(num_proc=1) -> None:
    '''
    读取wudao_180g_t5_tokenized_512数据，并进行train test split
    同时利用seed 42做shuffle 缓存下来
    '''
    ds = load_old_dataset(num_proc=num_proc)
    ds = ds['train'].train_test_split(train_size=0.999, test_size=0.001, seed=42)
    print(ds)
    p = ProcessPoolExecutor(max_workers=num_proc)
    res = []
    train_shard_part = 800
    for i in range(0, train_shard_part):
        res.append(p.submit(_generate_cache_arrow, i,
                            ds['train'].shard(train_shard_part, i)))

    p.shutdown(wait=True)
    for future in res:
        print(future.result(), flush=True)

    ds['test'].save_to_disk(_CACHE_TRAIN_DATA_PATH_TEST)
    print('done')


if __name__ == '__main__':
    ds = load_dataset(num_proc=100)
    print(ds)
    # generate_arrow_cache(num_proc=100)

    
    
    