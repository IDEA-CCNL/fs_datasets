import datasets


# 支持远程或者本地的地址
_TRAIN_DATA_PATH = '/cognitive_comp/common_data/afqmc/train.json'
_VAL_DATA_PATH = '/cognitive_comp/common_data/afqmc/dev.json'
_TEST_DATA_PATH = '/cognitive_comp/common_data/afqmc/test.json'


def load_dataset(**kargs):
    afqmc_features = datasets.Features({
        'sentence1': datasets.Value('string'),
        'sentence2': datasets.Value('string'),
        'label': datasets.Value('int32')})

    afqmc_raw_ds = datasets.load_dataset('json',
                                         data_files={
                                             'train': _TRAIN_DATA_PATH,
                                             'validation': _VAL_DATA_PATH, },
                                         # 'test': _TEST_DATA_PATH},
                                         features=afqmc_features)
    return afqmc_raw_ds
