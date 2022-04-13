import datasets
import os
import glob
from typing import Union
from importlib import import_module


def list_datasets():
    split_path = os.path.split(os.path.abspath(__file__))
    files = glob.glob(os.path.join(split_path[0], '*'))
    ds_list = []
    ignore_dir = ['__pycache__']
    for file in files:
        if (os.path.isdir(file)):
            h = os.path.split(file)
            if h[1] in ignore_dir:
                continue
            ds_list.append(h[1])
    return ds_list


def load_dataset(name, **kargs) -> Union[
        datasets.DatasetDict,
        datasets.Dataset,
        datasets.IterableDatasetDict,
        datasets.IterableDataset]:
    '''
    通过数据集名字加载数据集
    '''
    load_func = getattr(import_module('.' + name,
                                      package='fs_datasets'), 'load_dataset')
    return load_func(**kargs)


if __name__ == '__main__':
    print(load_dataset('afqmc'))
