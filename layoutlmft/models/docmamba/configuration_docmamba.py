# coding=utf-8
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig
logger = logging.get_logger(__name__)


class DocMambaConfig(PretrainedConfig):
    model_type = "docmamba"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
