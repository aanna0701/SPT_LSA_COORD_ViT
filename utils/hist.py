def make_hist(seq, hist=None) -> dict:
    if hist is None:
        hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist