import datasets
import glob
import os


# 多进程加载这个目录下的所有json文件
_SPLIT_DATA_PATH = '/cognitive_comp/gaoxinyu/data/WuDaoCorpus280G_split_100k/*'
# 缓存文件
_CACHE_TRAIN_DATA_PATH = '/cognitive_comp/common_data/wudao_280g/hf_cache_split_100k/'


def load_dataset(**kargs):
    cache_dict_paths = glob.glob(os.path.join(_CACHE_TRAIN_DATA_PATH, '*'))
    ds = []
    for path in cache_dict_paths:
        print('loading ', path, flush=True)
        ds.append(datasets.load_from_disk(path, **kargs))
    return datasets.DatasetDict({"train": datasets.concatenate_datasets(*ds)})


def generate_cache_arrow() -> None:
    '''
    生成HF支持的缓存文件，加速后续的加载
    '''
    f = datasets.Features({"content": datasets.Value('string')})
    data_dict_paths = glob.glob(_SPLIT_DATA_PATH)
    for path in data_dict_paths:
        ds = (datasets.load_dataset('json', data_files=path,
                                    cache_dir='/cognitive_comp/gaoxinyu/data/huggingface-cache',
                                    features=f)['train'])
        ds.save_to_disk(os.path.join(_CACHE_TRAIN_DATA_PATH, os.path.basename(path)))
