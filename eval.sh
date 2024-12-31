## beir

## baseline 1: contriever-msmarco
for dataset in trec-covid nfcorpus fiqa arguana webis-touche2020 quora scidocs scifact nq hotpotqa dbpedia-entity fever climate-fever;do
    echo -ne "beir | contriever-ft-on-ms | ${dataset} | reproduced | "
    ~/trec_eval-9.0.7/trec_eval -c -m ndcg_cut.10 \
        /home/dju/datasets/beir/${dataset}/qrels.beir-v1.0.0-${dataset}.test.txt \
        runs/contriever-ft-on-ms/run.beir.${dataset}.contriever.ft-on-ms.txt \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
done

for dataset in lifestyle recreation science technology writing;do
    echo -ne "lotte-test-search | contriever-ft-on-ms | ${dataset} | reproduced | "
    ~/trec_eval-9.0.7/trec_eval -c -m ndcg_cut.10 \
        /home/dju/datasets/lotte/${dataset}/test/qrels.lotte-${dataset}-test.search.txt \
        runs/contriever-ft-on-ms/run.lotte.${dataset}.contriever.ft-on-ms.txt \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'

done
