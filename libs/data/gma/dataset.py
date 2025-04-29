import os
import cv2
import torch
import numpy as np
import random
from PIL import Image
# from .list_record_cache import ListRecordLoader

import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from libs.data.gma.list_record_cache import ListRecordLoader

def edit_pubtables_info(info, img_dir):
    new_texts = []
    new_polys = []
    for i, (text, poly) in enumerate(zip(info['text'], info['bbox'])):
        if text == '<None>' or poly == [[-1, -1], [-1, -1]] or text == '':
            continue
        new_texts.append(text)
        new_polys.append(poly)
    info['texts'] = new_texts
    info['text_polys'] = new_polys
    info['image_path'] = os.path.join(img_dir, os.path.basename(info['image_path']))
    del info['bbox']
    del info['text']

def edit_webpage_info(info, img_dir):
    info['image_path'] = os.path.join(img_dir, os.path.basename(info['image_path']))

def edit_ucfs_info(info, img_dir):
    info['image_path'] = os.path.join(img_dir, os.path.basename(info['image_path']))



class InvalidFormat(Exception):
    pass


class SimpleRecordLoader:
    """used for bucker_sampler, without loading images"""
    def __init__(self, lrc_path):
        self.loader = ListRecordLoader(lrc_path)

    def __len__(self):
        return len(self.loader)

    def get_data(self, idx):
        info = self.loader.get_record(idx)
        return info

class LRCRecordLoader:
    def __init__(self, lrc_path, use_image_token):
        self.loader = ListRecordLoader(lrc_path)
        self.use_image_token = use_image_token

    def __len__(self):
        return len(self.loader)

    def get_data(self, idx):
        info = self.loader.get_record(idx)
        image_path = info['image_path']
        image = Image.open(image_path)
        return info, image

class PubtablesLRCRecordLoader:
    def __init__(self, lrc_path, img_dir, use_image_token):
        self.loader = ListRecordLoader(lrc_path)
        self.img_dir = img_dir
        self.use_image_token = use_image_token

    def __len__(self):
        return len(self.loader)

    def get_data(self, idx):
        info = self.loader.get_record(idx)
        image_path = info['image_path']
        image_path = os.path.join(self.img_dir, os.path.basename(image_path))
        image = Image.open(image_path)
        return info, image

class WebpageEnglishLRCRecordLoader:
    def __init__(self, lrc_path, img_dir, use_image_token):
        self.loader = ListRecordLoader(lrc_path)
        self.img_dir = img_dir
        self.use_image_token = use_image_token

    def __len__(self):
        return len(self.loader)

    def get_data(self, idx):
        info = self.loader.get_record(idx)
        image_path = info['image_path']
        image_path = os.path.join(self.img_dir, os.path.basename(image_path))
        image = Image.open(image_path)
        return info, image

class UcfsLRCRecordLoader:
    def __init__(self, lrc_path, img_dir):
        self.loader = ListRecordLoader(lrc_path)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.loader)

    def get_data(self, idx):
        info = self.loader.get_record(idx)
        image_path = info['image_path']
        image_path = os.path.join(self.img_dir, os.path.basename(image_path))
        image = Image.open(image_path)
        return info, image


class FunsdDataset:
    def __init__(self, raw_dataset, transforms, train_or_test):
        self.transforms = transforms
        self.train_or_test = train_or_test
        assert self.train_or_test in ['train', 'test']
        self.raw_dataset = raw_dataset[self.train_or_test]

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        info = self.raw_dataset[idx]
        info, pixel_values, input_ids, bbox, labels = self.transforms(info)
        return dict(
            pixel_values=torch.tensor(pixel_values), input_ids=torch.tensor(input_ids), bbox=torch.tensor(bbox, dtype=torch.int64), labels=torch.tensor(labels, dtype=torch.int64)
        )

class CordDataset:
    def __init__(self, raw_dataset, transforms, train_or_test):
        self.transforms = transforms
        self.train_or_test = train_or_test
        assert self.train_or_test in ['train', 'validation', 'test']
        self.raw_dataset = raw_dataset[self.train_or_test]

    def __len__(self):
        return len(self.raw_dataset)

    def trans_cord_2_funsd_format(self, info):
        info['tokens'] = info['words']
        del info['words']

    def __getitem__(self, idx):
        info = self.raw_dataset[idx]
        self.trans_cord_2_funsd_format(info)
        info, pixel_values, input_ids, bbox, labels = self.transforms(info)
        return dict(
            pixel_values=torch.tensor(pixel_values), input_ids=torch.tensor(input_ids), bbox=torch.tensor(bbox, dtype=torch.int64), labels=torch.tensor(labels, dtype=torch.int64)
        )

class SimpleFunsdDataset:
    def __init__(self, raw_dataset, train_or_test):
        self.train_or_test = train_or_test
        assert self.train_or_test in ['train', 'test']
        self.raw_dataset = raw_dataset[self.train_or_test]

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        info = self.raw_dataset[idx]
        return info     


class Dataset:
    def __init__(self, loaders, transforms, ratios, pubtable_paths, use_dataset_ratio):
        self.loaders = loaders
        self.transforms = transforms
        self.ratios = ratios
        self.data_num = len(self)
        self.pubtable_paths = pubtable_paths
        self.use_dataset_ratio = use_dataset_ratio
        self.dataset_token_len = torch.load('../../libs/data/gma/total_infos.pt')
        self.batch_sampler_bucket_info = torch.load('../../libs/data/gma/total_buckets.pt')
        self.batch_sampler = None

    def _match_loader(self, idx):
        offset = 0
        for loader in self.loaders:
            if len(loader) + offset > idx:
                return loader, idx - offset
            else:
                offset += len(loader)
        raise IndexError()
    
    def find_loader_index(self, prob):
        for i in range(len(self.ratios) - 1):
            if self.ratios[i] <= prob < self.ratios[i + 1]:
                return i
        return 0

    def find_rela_idx(self, ratio, loader_index):
        ratio = ratio - self.ratios[loader_index]
        ratio_range = self.ratios[loader_index + 1] - self.ratios[loader_index]
        ratio = ratio / (ratio_range + 1e-8)
        ratio = min(1, ratio + random.random() * 0.1)
        rela_idx = int(ratio * (len(self.loaders[loader_index]) - 1))
        return rela_idx

    def get_info(self, idx):
        loader, rela_idx = self._match_loader(idx)
        return loader.get_info(rela_idx)

    def __len__(self):
        return sum([len(loader) for loader in self.loaders])

    def __getitem__(self, idx):
        try:
            if self.use_dataset_ratio:
                # re-sample with `ratio`
                ratio = idx / self.data_num
                loader_index = self.find_loader_index(ratio)
                loader = self.loaders[loader_index]
                rela_idx = self.find_rela_idx(ratio, loader_index)
                info, image = loader.get_data(rela_idx)
            else:
                loader, rela_idx = self._match_loader(idx)
                info, image = loader.get_data(rela_idx)
            if isinstance(loader, PubtablesLRCRecordLoader):
                edit_pubtables_info(info, loader.img_dir)
            if isinstance(loader, WebpageEnglishLRCRecordLoader):
                edit_webpage_info(info, loader.img_dir)
            if isinstance(loader, UcfsLRCRecordLoader):
                edit_ucfs_info(info, loader.img_dir)

            info, pixel_values, input_ids, bbox, unmask_input_ids, mlm_mask = self.transforms(info, image)
            return dict(
                pixel_values=torch.tensor(pixel_values), input_ids=torch.tensor(input_ids), bbox=torch.tensor(bbox, dtype=torch.int64), unmask_input_ids=torch.tensor(unmask_input_ids), mlm_mask=torch.tensor(mlm_mask), data_idx=rela_idx
            )
        except Exception as e:
            print('Error occured while load data: %d' % idx + ', ' + str(e))
            cur_length = self.dataset_token_len[idx]
            cur_bucket_idx = cur_length // self.batch_sampler.sep
            temp_index = self.batch_sampler_bucket_info[str(cur_bucket_idx)]['samples'][0]
            return self[temp_index]


class SimpleDataset:
    """used for bucker_sampler, without loading images or transforms"""
    def __init__(self, loaders):
        self.loaders = loaders

    def _match_loader(self, idx):
        offset = 0
        for loader in self.loaders:
            if len(loader) + offset > idx:
                return loader, idx - offset
            else:
                offset += len(loader)
        raise IndexError()

    def __len__(self):
        return sum([len(loader) for loader in self.loaders])

    def __getitem__(self, idx):
        loader, rela_idx = self._match_loader(idx)
        info = loader.get_data(rela_idx)
        return info

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer_pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, batch_data):

        batch_pixel_values = torch.stack([data['pixel_values'] for data in batch_data], dim=0)
        batch_input_ids = torch.stack([data["input_ids"] for data in batch_data], dim=0)
        batch_bbox = torch.stack([data['bbox'] for data in batch_data], dim=0)
        batch_unmask_input_ids = torch.stack([data["unmask_input_ids"] for data in batch_data], dim=0)
        batch_mlm_mask = torch.stack([data["mlm_mask"] for data in batch_data], dim=0)
        attention_mask = torch.ones_like(batch_input_ids)
        batch_data_idx = torch.tensor([data['data_idx'] for data in batch_data])
        return {
            "pixel_values": batch_pixel_values,
            "input_ids": batch_input_ids,
            "bbox": batch_bbox,
            "unmask_input_ids": batch_unmask_input_ids,
            "mlm_mask": batch_mlm_mask,
            "attention_mask":attention_mask,
            "data_idx":batch_data_idx,
        }


class FunsdDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer_pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, batch_data):

        batch_pixel_values = torch.stack([data['pixel_values'] for data in batch_data], dim=0)
        batch_input_ids = torch.stack([data["input_ids"] for data in batch_data], dim=0)
        batch_bbox = torch.stack([data['bbox'] for data in batch_data], dim=0)
        attention_mask = torch.ones_like(batch_input_ids)
        batch_labels = torch.stack([data['labels'] for data in batch_data], dim=0)
        return {
            "pixel_values": batch_pixel_values,
            "input_ids": batch_input_ids,
            "bbox": batch_bbox,
            "attention_mask":attention_mask,
            "labels":batch_labels
        }
        

