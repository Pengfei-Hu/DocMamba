import os
import cv2
import copy
import math
import torch
import random
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None  # Removes the limit completely
import torch




class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, *data):
        for transform in self.transforms:
            data = transform(*data)
        return data

      
class SortBox:
    def __init__(self, group_line_threshold):
        self.group_line_threshold = group_line_threshold

    def __call__(self, info, image):
        return info, image
    
        

class TokenProcessor:
    def __init__(self, layoutlmv3_tokenizer, token_use_text_box, bucket_sep):
        self.layoutlmv3_tokenizer = layoutlmv3_tokenizer
        self.tokenizer = self.layoutlmv3_tokenizer._tokenizer
        self.token_use_text_box = token_use_text_box # token use its corresponding text box instead of its own box
        self.bucket_sep = bucket_sep
    
    def __call__(self, info, image):
        texts = info['texts']
        token_info = self.tokenizer.encode_batch(texts, add_special_tokens=False, is_pretokenized=False)
        token_ids = [t.ids for t in token_info]
        token_texts = [t.tokens for t in token_info]        
        text_polys = info['text_polys']
        # set text bbox for each token
        total_tokens = []
        total_bbox = []
        for text_p, token_id_lst in zip(text_polys, token_ids):
            for token_id in token_id_lst:
                total_tokens.append(token_id)
                if self.token_use_text_box:
                    total_bbox.append(text_p) # use text bbox of each token for layout embedding
        cur_token_max_length = max(len(total_tokens) // self.bucket_sep * self.bucket_sep - 2, 0)
        if len(total_tokens) > cur_token_max_length:
            start_idx = random.randint(0, max(0, len(total_tokens) - cur_token_max_length))
            total_tokens = total_tokens[start_idx: start_idx + cur_token_max_length]
            total_bbox = total_bbox[start_idx: start_idx + cur_token_max_length]
        # add <s> to the start
        total_tokens.insert(0, self.layoutlmv3_tokenizer.cls_token_id)
        total_bbox.insert(0, [0.0] * 8)
        # add </s> to the end
        total_tokens.append(self.layoutlmv3_tokenizer.sep_token_id)
        total_bbox.append([0] * 8)
        token_ids.insert(0, [self.layoutlmv3_tokenizer.cls_token_id])
        token_ids.append([self.layoutlmv3_tokenizer.sep_token_id])
        token_texts.insert(0, [self.layoutlmv3_tokenizer.cls_token])
        token_texts.append([self.layoutlmv3_tokenizer.sep_token])
        info['token_ids'] = token_ids
        info['token_texts'] = token_texts
        if 'token_polys' in info:
            del info['token_polys']
        # check input_ids
        total_tokens = np.clip(total_tokens, 0, self.tokenizer.get_vocab_size() - 1).tolist()
        return info, image, total_tokens, total_bbox


class BoxProcessor:
    def __init__(self, group_line_threshold, normalize_h=1000, normalize_w=1000):
        self.group_line_threshold = group_line_threshold
        self.normalize_h = normalize_h
        self.normalize_w = normalize_w
    
    def __call__(self, info, image, total_tokens, total_bbox):
        # normalize each bbox to [1, 1000]
        if 'origin_image_size' in info and 'resize_image_size' in info:
            h, w = info['origin_image_size']
        else:
            w, h = image.size
        total_bbox = np.array(total_bbox, dtype=np.float32)
        total_bbox[:, 0::2] = total_bbox[:, 0::2] / w * self.normalize_w
        total_bbox[:, 1::2] = total_bbox[:, 1::2] / h * self.normalize_h
        
        total_bbox[:, 0::2] = np.clip(total_bbox[:, 0::2], 0, self.normalize_w - 1)
        total_bbox[:, 1::2] = np.clip(total_bbox[:, 1::2], 0, self.normalize_h - 1)
        return info, image, total_tokens, total_bbox

class ImageProcessor:
    def __init__(self, target_h, target_w, resample_choice):
        self.target_h = target_h
        self.target_w = target_w
        self.resample_choice = resample_choice # PILImageResampling.BILINEAR for resize
        self.image_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.image_std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.scale = 1 / 255

    def __call__(self, info, image, total_tokens, total_bbox):
        # resize
        img_path = image.filename
        image = image.resize((self.target_w, self.target_h), resample=self.resample_choice)
        # check
        if np.array(image).max() > 255 or np.array(image).min() < 0:
            print('Error occured while load image {}: np.array(image).max() > 255 or np.array(image).min() < 0'.format(img_path))
            image = np.ones_like(np.array(image)) * 255
        # rescale
        image = (np.array(image) * self.scale).astype(np.float32)
        # normalize
        image = (image - self.image_mean) / self.image_std
        # transpose
        image = image.transpose((2, 0, 1)) # (h, w, c) -> (c, h, w)
        # image = torch.tensor(image)
        return info, image, total_tokens, total_bbox

class MLMProcessor:
    def __init__(self, mlm_prob, random_token_prob, leave_unmasked_prob, layoutlmv3_tokenizer):
        self.mlm_prob = mlm_prob
        self.random_token_prob = random_token_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.layoutlmv3_tokenizer = layoutlmv3_tokenizer

    def __call__(self, info, image, total_tokens, total_bbox):
        num_tokens = len(total_tokens)
        mask = np.random.rand(num_tokens) < self.mlm_prob
        mask[0], mask[-1] = False, False # remove [cls] and [sep]
        if not mask.any() and len(mask) > 2:
            mask[random.randint(1, len(mask) - 2)] = True
        # create target
        unmask_total_tokens = copy.deepcopy(total_tokens)
        mlm_mask = copy.deepcopy(mask).astype(np.int64)
        # create mask (for mask tokens), rand_mask (for randomly replace tokens)
        rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
        rand_or_unmask = mask & (np.random.rand(num_tokens) < rand_or_unmask_prob)
        unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
        decision = np.random.rand(num_tokens) < unmask_prob
        unmask = rand_or_unmask & decision
        rand_mask = rand_or_unmask & (~decision)
        mask = mask ^ unmask
        # mask tokens
        total_tokens = np.array(total_tokens)
        total_tokens[mask] = self.layoutlmv3_tokenizer.mask_token_id
        # replace tokens
        random_tokens = np.random.randint(low=3, high=self.layoutlmv3_tokenizer.vocab_size, size=rand_mask.sum()) # set low to 3 to avoid [cls], [sep] and [pad]
        total_tokens[rand_mask] = random_tokens
        return info, image, total_tokens, total_bbox, unmask_total_tokens, mlm_mask
