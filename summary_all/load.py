import datasets
import os
import tqdm

_ROOT_PATH = '/cognitive_comp/common_data/summary/'
data_path = '/cognitive_comp/common_data/summary/summary_all/'

def combine_data():
    file_list = ["education_data", "new2016zh_data", "nlpcc", "shence_data", "sohu_data", "thucnews_data", "weibo_data"]
    # file_list = ["thucnews_data", "nlpcc"]
    all_train_data = []
    all_valid_data = []
    all_test_data = []
    # all_data = datasets.DatasetDict()
    for data in tqdm(file_list):
        filepath = os.path.join(_ROOT_PATH, data)
        dataset = load_dataset_old(root_path=filepath)
        print(dataset)
        all_train_data.append(dataset["train"])
        all_valid_data.append(dataset["val"])
        all_test_data.append(dataset["test"])
    final_data_train = datasets.concatenate_datasets(all_train_data)
    final_data_val = datasets.concatenate_datasets(all_valid_data)
    final_data_test = datasets.concatenate_datasets(all_test_data)
    final_data_train.save_to_disk(os.path.join(data_path, "train"))
    final_data_val.save_to_disk(os.path.join(data_path, "valid"))
    final_data_test.save_to_disk(os.path.join(data_path, "test"))
    print(final_data_train)


def load_dataset_old(root_path):
    train = datasets.load_dataset('json',
                                  data_files=os.path.join(root_path, "train.jsonl"),
                                  split=datasets.Split.TRAIN, **kargs)
    val = datasets.load_dataset('json',
                                data_files={
                                    datasets.Split.VALIDATION: os.path.join(root_path, "valid.jsonl")},
                                split=datasets.Split.VALIDATION, **kargs)
    test = datasets.load_dataset('json',
                                 data_files={
                                     datasets.Split.TEST: os.path.join(root_path, "test.jsonl")},
                                 split=datasets.Split.TEST, **kargs)

    return datasets.DatasetDict(train=train, val=val, test=test)


def load_dataset():
    _DATA_PATH = data_path
    train_ds = datasets.load_from_disk(os.path.join(_DATA_PATH, "train"))
    test_ds = datasets.load_from_disk(os.path.join(_DATA_PATH, "test"))
    valid_ds = datasets.load_from_disk(os.path.join(_DATA_PATH, "valid"))
    return datasets.DatasetDict({
        "train": train_ds,
        "test": test_ds,
        "val": valid_ds
        })
