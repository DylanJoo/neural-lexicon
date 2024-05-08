# neural-lexicon

## Corpus preprocessing
Instead of pooling all the documents into one single long vectors (the original contriever's pipeline, see [this](#) for details).

We prepare raw bert-tokenized data for document-wise independent cropping. 
```
sbatch scripts/run_corpus_tokenization.sh
```
Some of the document are overlength and will show the warnings. This will be solved later in the following data pipeline.  
