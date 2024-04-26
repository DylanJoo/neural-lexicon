import numpy as np

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
