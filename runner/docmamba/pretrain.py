#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys

import numpy as np

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import libs.configs.default as cfg
from libs.data.gma import create_dataset, DataCollator, create_simple_dataset, BucketSampler
from libs.data.gma.list_record_cache import ListRecordLoader
from libs.utils.comm import distributed, get_rank, get_world_size
from layoutlmft.models.docmamba import DocMambaConfig, DocMambaModel

import torch
import transformers
from layoutlmft.data.data_args import DataTrainingArguments
from layoutlmft.models.model_args import ModelArguments
from pretrainer import PreTrainer as Trainer
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoProcessor,
    Pix2StructVisionModel,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from collections import OrderedDict
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from safetensors.torch import load_file


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)


def get_num_params(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    logger.info(f"trainable params: {int(trainable_params / 1e6)}M || all params: {int(all_param / 1e6)}M || trainable: {100 * trainable_params / all_param}%")


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize pretrain dataset and valid dataset
    train_lrc_paths = cfg.train_lrc_paths

    # init datasets, collator and batch_sampler
    train_dataset = create_dataset(train_lrc_paths, cfg)
    data_collator = DataCollator(cfg.layoutlmv3_tokenizer)
    simple_dataset = create_simple_dataset(cfg.train_lrc_paths)
    batch_sampler = BucketSampler(simple_dataset, get_world_size(), get_rank(), cfg.layoutlmv3_tokenizer, sep=cfg.bucket_sep)
    train_dataset.batch_sampler = batch_sampler

    config = DocMambaConfig.from_pretrained(os.path.join(model_args.model_name_or_path, 'config.json'))
    model = DocMambaModel(config, cfg).to('cuda')

    get_num_params(model)



    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=data_collator,
        batch_sampler=batch_sampler
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
