from transformers.utils import logging
import tqdm
import copy
import torch
import random
import os
import numpy as np
from collections import defaultdict

logger = logging.get_logger(__name__)


class BucketSampler:
    def __init__(self, dataset, world_size, rank_id, tokenizer, max_token_nums=2048, max_batch_size=400, min_batch_size=1, sep=64, epoch=0, save_info_path = '../../libs/data/gma/total_infos.pt'):
        self.dataset = dataset
        self.tokenizer = tokenizer._tokenizer
        self.world_size = world_size
        self.rank_id = rank_id
        self.sep = sep
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.max_token_nums = max_token_nums
        self.seed = 20
        self.epoch = epoch
        self.save_info_path = save_info_path # for train scripts
        self.save_bucket_path = os.path.join(os.path.dirname(self.save_info_path), 'total_buckets.pt')
        self.batch_size_k = 0.7
        self.cal_batch_size()
        self.cal_buckets()

    def batch_size_tune(self, len_init, batch_size_init, length):
        batch_size = (len_init * batch_size_init) // length # liner
        return int((batch_size * self.batch_size_k) // 2 * 2) # reduce 3 / 10 to avoid potential OOM, round down to even

    def cal_batch_size(self):
        """Supposing that batch size growing linearly with token length, used for mamba"""
        # init
        len_init = 1024
        batch_size_init = 28
        self.length_2_batchsize = {len_init: batch_size_init}
        
        # linear transform
        token_len_lst = list(range(self.sep, self.max_token_nums + self.sep, self.sep))
        for length in token_len_lst:
            batch_size = self.batch_size_tune(len_init, batch_size_init, length)
            self.length_2_batchsize[length] = batch_size
        
        # set special k, v to avoid OOM
        # self.length_2_batchsize[64] = 360
        # self.length_2_batchsize[128] = 180



    def count_keys(self):
        def _worker(idxs):
            sub_info = dict()
            for idx in tqdm.tqdm(idxs):
                info = self.dataset[idx]
                all_texts = None
                if 'texts' in info: # including Ucfs
                    all_texts = info['texts']
                elif 'text' in info: # pubtables
                    all_texts = []
                    for text in info['text']:
                        if text in ['<None>', '']:
                            continue
                        all_texts.append(text)
                token_info = self.tokenizer.encode_batch(all_texts, add_special_tokens=False, is_pretokenized=False)
                token_ids_lst = [t.ids for t in token_info]
                token_ids = [token_i for lst in token_ids_lst for token_i in lst]
                sub_info[idx] = len(token_ids)
            return sub_info

        idxs = list(range(len(self.dataset)))
        info_total = _worker(idxs)
        
        infos = [info_total[idx] for idx in range(len(info_total.keys()))]
        torch.save(infos, self.save_info_path)
        return infos

    def cal_buckets(self):
        if os.path.exists(self.save_info_path):
            infos = torch.load(self.save_info_path)
        else:
            infos = self.count_keys()
        if os.path.exists(self.save_bucket_path):
            valid_buckets = torch.load(self.save_bucket_path)
        else:
            buckets = defaultdict(list)
            for idx, info in enumerate(infos):
                bucket_idx = str(info // self.sep)
                buckets[bucket_idx].append(idx)

            valid_buckets = dict()
            for bucket_key, bucket_samples in buckets.items():
                if len(bucket_samples) < self.min_batch_size:
                    continue

                bucket_len = int(bucket_key) * self.sep
                if bucket_len in self.length_2_batchsize:
                    batch_size = self.length_2_batchsize[bucket_len]
                    valid_buckets[bucket_key] = dict(
                        samples=bucket_samples,
                        batch_size=batch_size
                    )
            
            torch.save(valid_buckets, self.save_bucket_path)
            print('Total %d buckets' % (len(valid_buckets)))
        self.buckets = [valid_buckets[bucket_key] for bucket_key in sorted(valid_buckets.keys(), key=lambda item: int(item))]
        total_nums = len(infos)
        valid_nums = sum([len(item['samples']) for item in valid_buckets.values()])
        print('Total %d samples, but ignore %d (%d%%) samples, remain %d (%d%%) samples.' % (total_nums, total_nums - valid_nums, (total_nums - valid_nums) / total_nums * 100, valid_nums, valid_nums / total_nums * 100))
        sorted_valid_buckets = dict(sorted(valid_buckets.items(), key=lambda item: int(item[0])))
        print('The distrubution of token_length:')
        cur_ratio_sum = 0
        for bucket_key, bucket_info in sorted_valid_buckets.items():
            bucken_len = int(bucket_key) * self.sep
            ratio = len(bucket_info['samples']) / valid_nums * 100
            cur_ratio_sum += ratio
            print(f'token length {bucken_len:4}: {ratio:.1f}%, {cur_ratio_sum:.1f}%')

    def __iter__(self):
        random_inst = random.Random(self.seed + self.epoch)
        batches = list()
        for bucket in self.buckets:

            sample = copy.deepcopy(bucket['samples'])
            
            batch_size = bucket['batch_size']
            # random_inst.shuffle(sample) # hu_debug
            idx = 0
            while idx < len(sample):
                batch = sample[idx:idx+batch_size]
                idx += batch_size
                if len(batch) < self.min_batch_size:
                    continue
                batches.append(batch)
        # random_inst.shuffle(batches) # hu_debug
        
        align_nums = (len(batches) // self.world_size) * self.world_size
        batches = batches[: align_nums]
        for batch_idx in range(self.rank_id, len(batches), self.world_size):
            yield batches[batch_idx]

    def __len__(self):
        batch_nums = 0
        for bucket in self.buckets:
            bucket_sample_nums = len(bucket['samples'])
            bucket_bs = bucket['batch_size']
            batch_nums += bucket_sample_nums // bucket_bs
            if bucket_sample_nums % bucket_bs >= self.min_batch_size:
                batch_nums += 1
        
        return batch_nums // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch


class FunsdBucketSampler:
    def __init__(self, dataset, world_size, rank_id, tokenizer, max_token_nums=2048, batch_size=16, sep=32, epoch=0):
        self.dataset = dataset
        self.tokenizer = tokenizer._tokenizer
        self.world_size = world_size
        self.rank_id = rank_id
        self.sep = sep
        self.batch_size = batch_size
        self.max_token_nums = max_token_nums
        self.seed = 20
        self.epoch = epoch
        self.cal_buckets()

    def count_keys(self):
        def _worker(idxs, result_queue):
            sub_info = dict()
            for idx in tqdm.tqdm(idxs):
                info = self.dataset[idx]
                token_info = self.tokenizer.encode_batch(info['tokens'], add_special_tokens=False, is_pretokenized=False)
                token_ids_lst = [t.ids for t in token_info]
                token_ids = [token_i for lst in token_ids_lst for token_i in lst]
                sub_info[idx] = len(token_ids)
            result_queue.put(sub_info)

        import multiprocessing
        num_workers = 4
        manager = multiprocessing.Manager()
        result_queue = manager.Queue()
        idxs = list(range(len(self.dataset)))
        workers = list()
        for worker_idx in range(num_workers):
            worker = multiprocessing.Process(
                target=_worker,
                args=(
                    idxs[worker_idx::num_workers],
                    result_queue
                )
            )
            worker.daemon = True
            worker.start()
            workers.append(worker)

        info_total = dict()
        for _ in range(num_workers):
           sub_info = result_queue.get()
           info_total.update(sub_info)
        
        infos = [info_total[idx] for idx in range(len(info_total.keys()))]
        return infos

    def cal_buckets(self):
        infos = self.count_keys()

        buckets = defaultdict(list)
        for idx, info in enumerate(infos):
            bucket_idx = str(info // self.sep)
            buckets[bucket_idx].append(idx)

        valid_buckets = dict()
        for bucket_key, bucket_samples in buckets.items():
            batch_size = self.batch_size
            valid_buckets[bucket_key] = dict(
                samples=bucket_samples,
                batch_size=batch_size
            )

        print('Total %d buckets' % (len(valid_buckets)))
        self.buckets = [valid_buckets[bucket_key] for bucket_key in sorted(valid_buckets.keys(), key=lambda item: int(item))]
        total_nums = len(infos)
        valid_nums = sum([len(item['samples']) for item in valid_buckets.values()])
        print('Total %d samples, but ignore %d (%d%%) samples, remain %d (%d%%) samples.' % (total_nums, total_nums - valid_nums, (total_nums - valid_nums) / total_nums * 100, valid_nums, valid_nums / total_nums * 100))
        sorted_valid_buckets = dict(sorted(valid_buckets.items(), key=lambda item: int(item[0])))
        print('The distrubution of token_length:')
        cur_ratio_sum = 0
        for bucket_key, bucket_info in sorted_valid_buckets.items():
            bucken_len = int(bucket_key) * self.sep
            ratio = len(bucket_info['samples']) / valid_nums * 100
            cur_ratio_sum += ratio
            print(f'token length {bucken_len:4}: {ratio:.1f}%, {cur_ratio_sum:.1f}%')

    def __iter__(self):
        random_inst = random.Random(self.seed + self.epoch)
        batches = list()
        for bucket in self.buckets:
            sample = copy.deepcopy(bucket['samples'])
            batch_size = bucket['batch_size']
            random_inst.shuffle(sample)
            idx = 0
            while idx < len(sample):
                batch = sample[idx:idx+batch_size]
                idx += batch_size
                batches.append(batch)
        random_inst.shuffle(batches)
        
        align_nums = (len(batches) // self.world_size) * self.world_size
        batches = batches[: align_nums]
        for batch_idx in range(self.rank_id, len(batches), self.world_size):
            yield batches[batch_idx]

    def __len__(self):
        batch_nums = 0
        for bucket in self.buckets:
            bucket_sample_nums = len(bucket['samples'])
            bucket_bs = bucket['batch_size']
            batch_nums += bucket_sample_nums // bucket_bs
            if bucket_sample_nums % bucket_bs > 0:
                batch_nums += 1
        
        return batch_nums // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

if __name__ == "__main__":
    import sys
    import numpy as np
    sys.path.append('./')
    sys.path.append('../')
    sys.path.append('../../')
    sys.path.append('../../../')
    from libs.utils.comm import distributed, get_rank, get_world_size
    import libs.configs.default as cfg
    from libs.data.gma import create_simple_dataset
    # pretrain
    simple_dataset = create_simple_dataset(cfg.train_lrc_paths)
    BucketSampler(simple_dataset, get_world_size(), get_rank(), cfg.layoutlmv3_tokenizer, sep=cfg.bucket_sep, save_info_path='libs/data/gma/total_infos.pt')

