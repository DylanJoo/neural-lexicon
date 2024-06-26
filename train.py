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

from src.sampling.data import load_dataset
from src.sampling.collators import Collator
from src.sampling.index_utils import NegativeSpanMiner
from src.sampling.encoders import BERTEncoder

os.environ['WANDB_PROJECT'] = 'SSLDR-span-learn'

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
    train_dataset = load_dataset(data_opt, tokenizer)
    eval_dataset = None
    ## if using `precomputed`. The select span mode is `no`
    train_dataset.select_span_mode = data_opt.select_span_mode
    collator = Collator(opt=data_opt)

    ## [Data-2] Before training, init encoder for precomputing if needed
    if train_opt.resume_from_checkpoint is not None:
        if 'true' in train_opt.resume_from_checkpoint.lower():
            last_checkpoint = get_last_checkpoint(train_opt.output_dir)
        else:
            last_checkpoint = train_opt.resume_from_checkpoint
        encoder = BERTEncoder(last_checkpoint, device='cuda')

        ## precompute spans
        train_dataset.init_spans(
                encoder=encoder,
                batch_size=128,
                max_doc_length=384,
                ngram_range=(2,3),
                top_k_spans=10,
                return_doc_embeddings=False,
                doc_embeddings_by_spans=False
        )
        del encoder

    # [Model] model architecture (encoders and bi-encoder framework)
    from src.modeling import Contriever
    from src.modeling import InBatchInteraction

    ## [Model] negative miner
    if train_opt.do_negative_sampling:
        negative_miner = NegativeSpanMiner(
                dataset=train_dataset,
                tokenizer=tokenizer,
                index_dir=data_opt.prebuilt_index_dir,
        )
    else:
        negative_miner = None

    encoder = Contriever.from_pretrained(
            model_opt.model_name, 
            add_pooling_layer=model_opt.add_pooling_layer,
            pooling=model_opt.pooling,
    )
    model = InBatchInteraction(
            opt=model_opt, 
            retriever=encoder, 
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
        trainer.train(resume_from_checkpoint=None)
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
    trainer.save_model(os.path.join(train_opt.output_dir))

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
