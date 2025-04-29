import torch
import os
from . import transform as T
from torch.utils.data.distributed import DistributedSampler
from .dataset import Dataset, FunsdDataset, DataCollator, LRCRecordLoader, PubtablesLRCRecordLoader, SimpleRecordLoader, SimpleDataset, WebpageEnglishLRCRecordLoader, SimpleFunsdDataset, FunsdDataCollator, UcfsLRCRecordLoader, CordDataset
from libs.utils.comm import distributed, get_rank, get_world_size
from .bucket_sampler import BucketSampler, FunsdBucketSampler

def create_simple_funsd_dataset(dataset, cfg):
    return SimpleFunsdDataset(dataset, 'train')




def create_simple_dataset(lrc_paths):
    """create a dataset used for bucker_sampler, without loading images or transforms."""
    loaders = list() 
    for info in lrc_paths:
        loader = SimpleRecordLoader(info['path'])
        loaders.append(loader)
    dataset = SimpleDataset(loaders)
    return dataset

def create_dataset(lrc_paths, cfg):
    loaders = list()
    ratios = list()
    ratios_dict = dict()    
    for info in lrc_paths:
        loader = LRCRecordLoader(info['path'], cfg.use_image_token)
        loaders.append(loader)
        ratios.append(info['ratio'])
        if info['name'] not in ratios_dict:
            ratios_dict[info['name']] = 0.
        ratios_dict[info['name']] += info['ratio']
    
    if cfg.use_dataset_ratio:
        # print sample ratio on different datasets
        print('The sample ratio among datasets are as follows:')
        for k, v in ratios_dict.items():
            print('%s: %.3f%%' % (k, 100.0 * v / sum(ratios)))
    # normalize and accumulate ratios
    ratios = [ratios[idx] / sum(ratios) for idx in range(len(ratios))]
    ratios_acc = [0]
    for ratio in ratios:
        ratios_acc.append(ratios_acc[-1] + ratio)    

    transforms = T.Compose([
        T.SortBox(cfg.group_line_threshold),
        T.TokenProcessor(cfg.layoutlmv3_tokenizer, cfg.token_use_text_box, cfg.bucket_sep),
        T.BoxProcessor(cfg.group_line_threshold),
        T.ImageProcessor(cfg.target_h, cfg.target_w, cfg.resample_choice),
        T.MLMProcessor(cfg.mlm_prob, cfg.random_token_prob, cfg.leave_unmasked_prob, cfg.layoutlmv3_tokenizer),
    ])

    pubtable_paths = [lrc_info['path'] for lrc_info in cfg.train_lrc_paths if lrc_info['name'] == 'Pubtables']
    dataset = Dataset(loaders, transforms, ratios_acc, pubtable_paths, cfg.use_dataset_ratio)
    return dataset
 