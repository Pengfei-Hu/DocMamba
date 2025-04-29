import os
from libs.utils.counter import Counter
from transformers import AutoTokenizer

# train path
train_lrc_paths = [
    dict(name='Toy', path='./data_process/output/toy.lrc', ratio=1),
]


# bucket for batch_sampler of dataloader
bucket_sep = 64
# token processor
layoutlmv3_tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")
# box processor
token_use_text_box = True
group_line_threshold = 5
# image processor
target_h = 224
target_w = 224
resample_choice = 2 # PILImageResampling.BILINEAR for resize
# mlm
mlm_prob = 0.15
random_token_prob = 0.1
leave_unmasked_prob = 0.1


use_dataset_ratio = False
use_image_token = False


# fine-tune
pretrain_output_path = ''
finetune_bucket_sep = 32
num_labels = 7 # useless, but keep it
funsd_use_truncation = False


# counter for show each item loss
counter = Counter(cache_nums=1000)