import datasets
import glob
import os
from concurrent.futures import ProcessPoolExecutor

# 缓存文件
_CACHE_TRAIN_DATA_PATH_TRAIN = '/cognitive_comp/common_data/wudao_180g_bert_tokenized_512_train/'
_CACHE_TRAIN_DATA_PATH_TEST = '/cognitive_comp/common_data/wudao_180g_bert_tokenized_512_test/'


feats = datasets.Features({"input_ids": datasets.Value('int32')})

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
