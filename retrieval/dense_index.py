# This code is modified from pyserini, please refer to original one for references.

# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import sys

from pyserini.encode import JsonlCollectionIterator, FaissRepresentationWriter
from pyserini.encode import AutoDocumentEncoder

from encoders import ContrieverDocumentEncoder, GTEDocumentEncoder

encoder_class_map = {
    "contriever": ContrieverDocumentEncoder,
    "gte": GTEDocumentEncoder
}

def init_encoder(encoder, encoder_class, device, pooling, l2_norm, prefix):
    _encoder_class = encoder_class

    # determine encoder_class
    if encoder_class is not None:
        encoder_class = encoder_class_map[encoder_class]
    else:
        # if any class keyword was matched in the given encoder name,
        # use that encoder class
        for class_keyword in encoder_class_map:
            if class_keyword in encoder.lower():
                encoder_class = encoder_class_map[class_keyword]
                break

        # if none of the class keyword was matched,
        # use the AutoDocumentEncoder
        if encoder_class is None:
            _encoder_class = "auto"
            encoder_class = AutoDocumentEncoder

    # prepare arguments to encoder class
    kwargs = dict(model_name=encoder, device=device)
    if (_encoder_class == "contriever") or ("contriever" in encoder):
        pooling = pooling or 'mean'
        kwargs.update(dict(pooling=pooling, l2_norm=False))
    if (_encoder_class == "gte") or ("gte" in encoder):
        pooling = pooling or 'mean'
        l2_norm = l2_norm or False
        kwargs.update(dict(pooling=pooling, l2_norm=l2_norm))
    return encoder_class(**kwargs)

def parse_args(parser, commands):
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(title='sub-commands')

    input_parser = commands.add_parser('input')
    input_parser.add_argument('--corpus', type=str, required=True)
    input_parser.add_argument('--fields', nargs='+', default=['text'], required=False)
    input_parser.add_argument('--docid-field', default=None, required=False)


    input_parser.add_argument('--delimiter', help='delimiter for the fields', default='\n', required=False)
    input_parser.add_argument('--shard-id', type=int, help='shard-id 0-based', default=0, required=False)
    input_parser.add_argument('--shard-num', type=int, help='number of shards', default=1, required=False)

    output_parser = commands.add_parser('output')
    output_parser.add_argument('--embeddings', type=str, help='directory to store encoded corpus', required=True)
    output_parser.add_argument('--to-faiss', action='store_true', default=False)

    encoder_parser = commands.add_parser('encoder')
    encoder_parser.add_argument('--encoder', type=str, help='encoder name or path', required=True)
    encoder_parser.add_argument('--encoder-class', type=str, required=False, default=None, choices=["auto", "contriever", "gte"])
    encoder_parser.add_argument('--fields', help='fields to encode', nargs='+', default=['text'], required=False)
    encoder_parser.add_argument('--batch-size', type=int, help='batch size', default=64, required=False)
    encoder_parser.add_argument('--max-length', type=int, help='max length', default=256, required=False)
    encoder_parser.add_argument('--dimension', type=int, help='dimension', default=768, required=False)
    encoder_parser.add_argument('--device', type=str, default='cuda:0', required=False)
    encoder_parser.add_argument('--fp16', action='store_true', default=False)
    encoder_parser.add_argument('--add-sep', action='store_true', default=False)
    # encoder_parser.add_argument('--pooling', type=str, default='cls', help='for auto classes, allow the ability to dictate pooling strategy', choices=['cls', 'mean'], required=False)
    encoder_parser.add_argument('--pooling', type=str, default=None, help='for auto classes, allow the ability to dictate pooling strategy')
    encoder_parser.add_argument('--l2-norm', action='store_true', help='whether to normalize embedding', default=False, required=False)
    encoder_parser.add_argument('--prefix', type=str, help='prefix of document input', default=None, required=False)
    encoder_parser.add_argument('--use-openai', help='use OpenAI text-embedding-ada-002 to retreive embeddings', action='store_true', default=False)
    encoder_parser.add_argument('--rate-limit', type=int, help='rate limit of the requests per minute for OpenAI embeddings', default=3500, required=False)

    args = parse_args(parser, commands)
    delimiter = args.input.delimiter.replace("\\n", "\n")  # argparse would add \ prior to the passed '\n\n'

    # Multi-shards
    if args.input.shard_num > 1:
        print(f'[{args.input.shard_id}/{args.input.shard_num}] is processing')
        args.output.embeddings = args.output.embeddings + f"{args.input.shard_num}_{args.input.shard_id}"

    encoder = init_encoder(
        args.encoder.encoder, args.encoder.encoder_class, 
        device=args.encoder.device, pooling=args.encoder.pooling, l2_norm=args.encoder.l2_norm, prefix=args.encoder.prefix
    )
    embedding_writer = FaissRepresentationWriter(args.output.embeddings, dimension=args.encoder.dimension)
    collection_iterator = JsonlCollectionIterator(args.input.corpus, args.input.fields, args.input.docid_field, delimiter)

    with embedding_writer:
        for batch_info in collection_iterator(args.encoder.batch_size, args.input.shard_id, args.input.shard_num):
            kwargs = {
                'texts': batch_info['text'] if 'text' in args.encoder.fields else None,
                'titles': batch_info['title'] if 'title' in args.encoder.fields else None,
                'expands': batch_info['expand'] if 'expand' in args.encoder.fields else None,
                'fp16': args.encoder.fp16,
                'max_length': args.encoder.max_length,
                'add_sep': args.encoder.add_sep,
            }
            embeddings = encoder.encode(**kwargs)
            batch_info['vector'] = embeddings
            embedding_writer.write(batch_info, args.input.fields)
