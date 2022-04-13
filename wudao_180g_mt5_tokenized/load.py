import datasets
import glob
import os
from concurrent.futures import ProcessPoolExecutor

_SPLIT_DATA_PATH = ''
# 缓存文件
_CACHE_TRAIN_DATA_PATH = '/cognitive_comp/common_data/wudao_180g_mt5_tokenized/'


feats = datasets.Features({"input_ids": datasets.Value('int32')})


def load_dataset(num_proc=1, **kargs):
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


def _generate_cache_arrow(index, path):
    print('saving dataset shard {}'.format(index))
    ds = (datasets.load_dataset('json', data_files=path,
                                cache_dir='',
                                features=feats)['train'])
    ds.save_to_disk(os.path.join(_CACHE_TRAIN_DATA_PATH, os.path.basename(path)))
    return 'saving dataset shard {} done'.format(index)


def generate_cache_arrow(num_proc=1) -> None:
    '''
    生成HF支持的缓存文件，加速后续的加载
    '''
    data_dict_paths = glob.glob(_SPLIT_DATA_PATH)
    p = ProcessPoolExecutor(max_workers=num_proc)
    res = []

    for index, path in enumerate(data_dict_paths):
        res.append(p.submit(_generate_cache_arrow, index, path))

    p.shutdown(wait=True)
    for future in res:
        print(future.result(), flush=True)
