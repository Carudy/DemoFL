def fedavg(params):
    ks = list(params[0].keys())
    n = len(params)
    ret = {k: 0 for k in ks}
    for k in ks:
        for d in params:
            ret[k] += d[k]
        ret[k] /= n
    return ret


def fedadd(params):
    ks = list(params[0].keys())
    ret = {k: 0 for k in ks}
    for k in ks:
        for d in params:
            ret[k] += d[k]
    return ret
