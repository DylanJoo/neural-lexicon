import collections
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def argdiff(iterable):
    '''
    iterable: list. This should be the sorted list with decesending order
    return
        list sorted by difference (largest decreasing to smallest decreasing)
    '''
    # the difference should all be negative or zero.
    iterable_with_sentinel = np.append([iterable[0]], iterable)
    return np.diff( iterable_with_sentinel ).argsort()


def cosine_top_k(
        doc_embedding,
        candidates,
        candidate_embeddings,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:

    scores = cosine_similarity(doc_embedding.reshape(1, -1), candidate_embeddings)
    key_spans = [(candidates[i_ngram], round(float(scores[0][i_ngram]), 4)) \
            for i_ngram in scores.argsort()[0][-top_k:]][::-1]     
    return key_spans

# this is from keybert
def mmr_top_k(
        doc_embedding,
        candidates,
        candidate_embeddings,
        top_k: int = 5,
        diversity: float = 0.8,
    ) -> List[Tuple[str, float]]:

    '''Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.


    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.

    Arguments:
        doc_embedding: The document embeddings
        candidate_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_k: The number of keywords/keyhprases to return
        diversity: How diverse the select keywords/keyphrases are.
                   Values between 0 and 1 with 0 being not diverse at all
                   and 1 being most diverse.

    Returns:
         List[Tuple[str, float]]: The selected keywords/keyphrases with their distances

    '''
    doc_embedding = doc_embedding.reshape(1, -1)

    # Extract similarity within words, and between words and the document
    scores = cosine_similarity(candidate_embeddings, doc_embedding)
    cand_scores = cosine_similarity(candidate_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    selected = [np.argmax(scores)]
    remained = [i for i in range(len(candidates)) if i != selected[0]]

    for _ in range(min(top_k - 1, len(candidates) - 1)):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = scores[remained, :]
        target_similarities = np.max(
                cand_scores[remained][:, selected], axis=1
        )

        # Calculate MMR
        mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = remained[np.argmax(mmr)]

        # Update keywords & candidates
        selected.append(mmr_idx)
        remained.remove(mmr_idx)

    # Extract and sort keywords in descending similarity
    key_spans = [(candidates[i_ngram], round(float(scores[i_ngram, 0]), 4)) for i_ngram in selected]
    return key_spans
