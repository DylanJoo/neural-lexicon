from tqdm import tqdm
import collections
import json

def load_topic(path):
    topic = {}

    with open(path, 'r') as f:
        if path.endswith('tsv') or path.endswith('txt'):
            for line in f:
                qid, qtext = line.split('\t')
                topic[str(qid.strip())] = qtext.strip()
        if path.endswith('jsonl'):
            for line in f:
                data = json.loads(line.strip())
                qid = data['_id'].strip()
                qtext = data['text'].strip()
                topic[str(qid)] = qtext
    return topic

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def load_results(path, topk):
    input_run = collections.defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if int(rank) <= topk:
                input_run[str(qid)].append(str(docid))

    return input_run

def load_corpus(path):
    corpus = {}
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            docid = data['_id']
            title = data.get('title', "").strip()
            text = data.get('text', "").strip()
            corpus[str(docid)] = (title + " " + text).strip()
    return corpus
