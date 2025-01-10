import os
import sys
import json
from typing import Optional, Union
from transformers import HfArgumentParser
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from dataclasses import asdict

from src.trainer import Trainer
from src.options import ModelOptions, DataOptions, TrainOptions

from src.sampling.data import DatasetIndependentCropping
from src.sampling.collators import Collator
from src.sampling.miner import NegativeSpanMiner

def main():

    parser = HfArgumentParser((ModelOptions, DataOptions, TrainOptions))
    model_opt, data_opt, train_opt = parser.parse_args_into_dataclasses()

    if train_opt.wandb_project:
        os.environ["WANDB_PROJECT"] = train_opt.wandb_project

    # [Config] tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_opt.tokenizer_name or model_opt.model_name)
    tokenizer.bos_token = '[CLS]'
    tokenizer.eos_token = '[SEP]'

    # [Data] train/eval datasets, collator, preprocessor
    train_dataset = DatasetIndependentCropping(data_opt, tokenizer)
    eval_dataset = None
    collator = Collator(opt=data_opt)

    ## [Model] negative miner
    if model_opt.n_negative_samples > 0:
        negative_miner = NegativeSpanMiner(data_opt, train_dataset, tokenizer)
    else:
        negative_miner = None

    # [Model] model architecture (encoders and bi-encoder framework)
    from src.modeling import Contriever
    from src.modeling import InBatchInteraction

    encoder = Contriever.from_pretrained(
            model_opt.model_name, 
            add_pooling_layer=model_opt.add_pooling_layer,
            pooling=model_opt.pooling,
    )
    model = InBatchInteraction(
            opt=model_opt, 
            retriever=encoder, 
            fixed_d_encoder=model_opt.fixed_d_encoder,
            tokenizer=tokenizer,
            miner=negative_miner
    )
    
    trainer = Trainer(
            model=model, 
            tokenizer=tokenizer,
            args=train_opt,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
    )

    # [Training] the first training round. 0<T<t0.5
    if train_opt.resume_from_checkpoint is None:
        trainer.train()
    else:
        # [Training] testing another loop
        if 'true' in train_opt.resume_from_checkpoint.lower():
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train(resume_from_checkpoint=last_checkpoint)

    ## [Round1] 
    ### [t<0.0] prepare spans and kmeans
    ### [t=0.5] Setup span indexing and encoding for negative mining
    ### [t<1.0] Span negative mining with junior encoders
    # trainer.train(resume_from_checkpoint=None)

    ## [Round2] 
    ### [t<1.0] Instead of doing spans and kmeans separately, 
    ### do them at the same time but with m mini-batch b; thus. mb batch_size
    ### [t=1.5] Setup span indexing and encoding for negative mining
    ### [t<2.0] Span negative mining with junior encoders
    # trainer.train(resume_from_checkpoint=train_opt.resume_from_checkpoint)

    ## Setup t0.5 for span indexing and encoding
    # trainer.save_model(os.path.join(train_opt.output_dir))

    final_path = train_opt.output_dir
    if  trainer.is_world_process_zero():
        with open(os.path.join(final_path, "model_opt.json"), "w") as write_file:
            json.dump(asdict(model_opt), write_file, indent=4)
        with open(os.path.join(final_path, "data_opt.json"), "w") as write_file:
            json.dump(asdict(data_opt), write_file, indent=4)
        with open(os.path.join(final_path, "train_opt.json"), "w") as write_file:
            json.dump(train_opt.to_dict(), write_file, indent=4)

if __name__ == '__main__':
    main()
