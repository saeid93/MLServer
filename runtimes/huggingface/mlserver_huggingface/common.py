import json
import numpy as np

from typing import Callable
from functools import partial
from mlserver.settings import ModelSettings

import torch
# import tensorflow as tf

from optimum.pipelines import pipeline as opt_pipeline
from transformers.pipelines import pipeline as trf_pipeline
from transformers.pipelines.base import Pipeline

from .settings import HuggingFaceSettings


OPTIMUM_ACCELERATOR = "ort"

_PipelineConstructor = Callable[..., Pipeline]


def load_pipeline_from_settings(
    hf_settings: HuggingFaceSettings, settings: ModelSettings
) -> Pipeline:
    # TODO: Support URI for locally downloaded artifacts
    # uri = model_parameters.uri
    pipeline = _get_pipeline_class(hf_settings)

    batch_size = 1
    if settings.max_batch_size:
        batch_size = settings.max_batch_size

    tokenizer = hf_settings.pretrained_tokenizer
    if not tokenizer:
        tokenizer = hf_settings.pretrained_model
    if hf_settings.framework == "tf":
        if hf_settings.inter_op_threads is not None:
            tf.config.threading.set_inter_op_parallelism_threads(
                hf_settings.inter_op_threads
            )
        if hf_settings.intera_op_threads is not None:
            tf.config.threading.set_intra_op_parallelism_threads(
                hf_settings.intera_op_threads
            )
    elif hf_settings.framework == "pt":
        if hf_settings.inter_op_threads is not None:
            torch.set_num_interop_threads(hf_settings.inter_op_threads)
        if hf_settings.intera_op_threads is not None:
            torch.set_num_threads(hf_settings.intera_op_threads)

    hf_pipeline = pipeline(
        hf_settings.task_name,
        model=hf_settings.pretrained_model,
        tokenizer=tokenizer,
        device=hf_settings.device,
        batch_size=batch_size,
        framework=hf_settings.framework,
    )

    # If max_batch_size > 0 we need to ensure tokens are padded
    if settings.max_batch_size:
        model = hf_pipeline.model
        eos_token_id = model.config.eos_token_id
        hf_pipeline.tokenizer.pad_token_id = [str(eos_token_id)]  # type: ignore

    return hf_pipeline


def _get_pipeline_class(hf_settings: HuggingFaceSettings) -> _PipelineConstructor:
    if hf_settings.optimum_model:
        return partial(opt_pipeline, accelerator=OPTIMUM_ACCELERATOR)

    return trf_pipeline


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
