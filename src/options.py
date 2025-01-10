import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple
from transformers import TrainingArguments

@dataclass
class ModelOptions:
    model_name: Optional[str] = field(default=None)
    model_path: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    add_pooling_layer: Optional[bool] = field(default=False)
    # SSL DR
    pooling: Optional[str] = field(default="mean")
    span_pooling: Optional[str] = field(default="mean")
    norm_doc: Optional[bool] = field(default=False)
    norm_query: Optional[bool] = field(default=False)
    norm_spans: Optional[bool] = field(default=False)
    # span
    # objective, mining source
    temperature: Optional[float] = field(default=1.0)
    temperature_span: Optional[float] = field(default=1.0)
    alpha: float = field(default=1.0)
    beta: float = field(default=1.0) 
    gamma: float = field(default=1.0)
    mine_neg_using: Optional[str] = field(default=None)
    n_negative_samples: Optional[int] = field(default=0)
    # Multivec (previous)
    # late_interaction: Optional[bool] = field(default=False)
    fixed_d_encoder: Optional[bool] = field(default=False)

@dataclass
class DataOptions:
    # positive_sampling: Optional[str] = field(default='ind_cropping')
    corpus_jsonl: Optional[str] = field(default=None)
    corpus_spans_jsonl: Optional[str] = field(default=None)
    # negative sampling
    prebuilt_faiss_dir: Optional[str] = field(default=None)
    prebuilt_negative_jsonl: Optional[str] = field(default=None)
    # independent cropping
    chunk_length: Optional[int] = field(default=256)
    ratio_min: Optional[float] = field(default=0.1)
    ratio_max: Optional[float] = field(default=0.5)
    augmentation: Optional[str] = field(default=None)
    prob_augmentation: Optional[float] = field(default=0.0)
    # preprocessing
    preprocessing: Optional[str] = field(default='replicate')
    min_chunk_length: Optional[int] = field(default=32)
    # span contrastive
    select_span_mode: Optional[str] = field(default=None)
    span_mask: Optional[bool] = field(default=None)
    span_online_update: bool = field(default=False)

@dataclass
class TrainOptions(TrainingArguments):
    output_dir: str = field(default='./')
    seed: int = field(default=42)
    data_seed: int = field(default=None)
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    num_train_epochs: int = field(default=3)
    save_strategy: str = field(default='epoch')
    overwrite_output_dir: bool = field(default=True)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    logging_dir: Optional[str] = field(default='./logs')
    resume_from_checkpoint: Optional[str] = field(default=None)
    save_total_limit: Optional[int] = field(default=4)
    learning_rate: Union[float] = field(default=5e-5)
    remove_unused_columns: bool = field(default=False)
    dataloader_num_workers: int = field(default=1)
    dataloader_prefetch_factor: int = field(default=2)
    fp16: bool = field(default=False)
    wandb_project: Optional[str] = field(default=None)
    # original is linear
