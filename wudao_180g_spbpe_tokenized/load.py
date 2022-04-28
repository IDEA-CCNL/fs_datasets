import datasets
import glob
from transformers import MT5Tokenizer
import os
from concurrent.futures import ProcessPoolExecutor


_SENTENCE_PIECE_TOKENIZERS = {
    'model': '/cognitive_comp/common_data/tokenizers/sentence_piece_bpe/bpe_v40000_s42_cov0.9995_max6_corpus1M.model',
}

# 缓存文件
_CACHE_TRAIN_DATA_PATH = '/cognitive_comp/common_data/wudao_180g_spbpe_tokenized/train_split'
_CACHE_TEST_DATA_PATH = '/cognitive_comp/common_data/wudao_180g_spbpe_tokenized/test'


def load_dataset(num_proc=1, **kargs):
    '''
    加载缓存的数据
    '''
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
    train_ds = datasets.concatenate_datasets(ds)
    test_ds = datasets.load_from_disk(_CACHE_TEST_DATA_PATH)
    return datasets.DatasetDict({
        "train": train_ds,
        "test": test_ds})


def _generate_cache_arrow(index, ds):
    print('saving dataset shard {}'.format(index))
    ds.save_to_disk(os.path.join(_CACHE_TRAIN_DATA_PATH, 'part_{}'.format(index)))
    return 'saving dataset shard {} done'.format(index)


def generate_arrow_cache(num_proc=1) -> None:
    '''
    读取wudao_180g原始数据，并进行切句，切句后tokenizer生成datasets
    同时利用seed 42做shuffle 缓存下来
    '''
    import sys
    sys.path.append('../../')
    from fs_datasets import load_dataset
    ds = load_dataset('wudao_180g', num_proc=num_proc)
    ds = ds['train'].train_test_split(train_size=0.995, test_size=0.005, seed=42)
    print(ds)
    from fs_datasets.utils import ChineseSentenceSplitter
    sentence_splitter = ChineseSentenceSplitter()
    tokenizer = MT5Tokenizer.from_pretrained(_SENTENCE_PIECE_TOKENIZERS['model'])

    def _tokenizer(example):
        sentences = sentence_splitter.tokenize(example['text'])
        samples = [tokenizer.tokenize(s) for s in sentences]
        return {
            'tokenized_text': samples,
        }

    tokenized_ds = ds.map(
        _tokenizer,
        num_proc=num_proc,
        remove_columns=ds['train'].column_names)

    p = ProcessPoolExecutor(max_workers=num_proc)
    res = []
    train_shard_part = 500
    for i in range(0, train_shard_part):
        res.append(p.submit(_generate_cache_arrow, i,
                            tokenized_ds['train'].shard(train_shard_part, i)))

    p.shutdown(wait=True)
    for future in res:
        print(future.result(), flush=True)

    tokenized_ds['test'].save_to_disk(_CACHE_TEST_DATA_PATH)

    print('done')


if __name__ == '__main__':
    # generate_arrow_cache(num_proc=100)
    ds = load_dataset(num_proc=100)
    print(ds)
