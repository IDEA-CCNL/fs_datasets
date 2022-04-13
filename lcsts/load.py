import datasets


_TRAIN_DATA_PATH = '/cognitive_comp/common_data/LCSTS/train.jsonl'
_VAL_DATA_PATH = '/cognitive_comp/common_data/LCSTS/valid.jsonl'
_TEST_DATA_PATH = '/cognitive_comp/common_data/LCSTS/test_public.jsonl'


def load_dataset(**kargs):
    # 因为train和val&test的column name不一样，需要要分开读取
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
