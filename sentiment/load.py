import datasets

# 原始数据文件
_TRAIN_DATA_PATH = '/cognitive_comp/common_data/sentiment//mergedata/train.json'
_VAL_DATA_PATH = '/cognitive_comp/common_data/sentiment//mergedata/dev.json'


def load_dataset(**kargs):
    ds = datasets.load_dataset('json',
                               data_files={
                                   'train': _TRAIN_DATA_PATH,
                                   'validation': _VAL_DATA_PATH, },
                               **kargs)
    return ds
