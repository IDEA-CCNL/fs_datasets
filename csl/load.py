import datasets

_TRAIN_DATA_PATH = '/cognitive_comp/common_data/summary/csl_title_public/csl_title_train.json'
_VAL_DATA_PATH = '/cognitive_comp/common_data/summary/csl_title_public/csl_title_dev.json'
_TEST_DATA_PATH = '/cognitive_comp/common_data/summary/csl_title_public/csl_title_test.json'

def load_dataset(**kargs):
    '''
    加载数据
    '''
    train = datasets.load_dataset('json',
                                  data_files=_TRAIN_DATA_PATH,
                                  split=datasets.Split.TRAIN, **kargs)
    val = datasets.load_dataset('json',
                                data_files={
                                    datasets.Split.VALIDATION: _VAL_DATA_PATH},
                                split=datasets.Split.VALIDATION, **kargs)
    test = datasets.load_dataset('json',
                                 data_files={
                                     datasets.Split.TEST: _TEST_DATA_PATH},
                                 split=datasets.Split.TEST, **kargs)

    return datasets.DatasetDict(train=train, val=val, test=test)