# see the BEIR repo for details. 
# https://github.com/beir-cellar/beir

URL=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/name.zip
data_dir=/home/dju/datasets/beir

# exclude msmaroc and cqadupstack (Lotte has covered these)
for dataset in trec-covid nfcorpus nq hotpotqa fiqa arguana webis-touche2020 quora dbpedia-entity scidocs fever climate-fever scifact;do
    # Download queries and corpus from huggingface
    mkdir -p ${data_dir}/${dataset}
    wget ${URL/name/${dataset}} -O ${data_dir}/${dataset}/temp.zip
    unzip ${data_dir}/${dataset}/temp.zip -d ${data_dir}

    # remove zip files and qrels 
    rm -rvf ${data_dir}/${dataset}/qrels
    rm ${data_dir}/${dataset}/temp.zip 

    # move to an standalone folder (for pyserini API)
    mkdir -p ${data_dir}/${dataset}/collection 
    mv ${data_dir}/${dataset}/corpus.jsonl ${data_dir}/${dataset}/collection

    # Download beir qrels (trec format from anserini)
    wget https://github.com/castorini/anserini-tools/raw/master/topics-and-qrels/qrels.beir-v1.0.0-${dataset}.test.txt -O ${data_dir}/${dataset}/qrels.beir-v1.0.0-${dataset}.test.txt
done

