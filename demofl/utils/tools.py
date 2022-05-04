import importlib
import random
import sys
import time
import yaml
import threading

from .log import *

BIG_PRIME = 17


def join_threads(threads):
    for td in threads:
        td.start()
    for td in threads:
        td.join()


def thread_method_map(items, method, args=(), kwargs={}):
    tds = [threading.Thread(target=getattr(item, method), args=args, kwargs=kwargs) for item in items]
    for td in tds:
        td.start()
    for td in tds:
        td.join()


def secret_share(x, n):
    """
        Split tensor x into n pieces.
    """
    rand = [random.choice([1, -1]) * random.random() * BIG_PRIME for _ in range(n - 1)]
    rand.append(-sum(rand))
    p = 1 / n
    return [(x + i) * p for i in rand]


def import_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_yaml(target):
    # if isinstance(target, str) or isinstance(target, bytes):
    #     return yaml.load(target, Loader=yaml.FullLoader)
    with open(target, encoding='utf-8') as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def wait_func(func, t=0.2, in_runtime=True):
    """
        A "wait_func" will repeatly run until it returns True.
        Notice the return type of func must be bool.
    """

    def decor(*args, **kwargs):
        res = None
        while res is not True:
            try:
                if in_runtime and args[0].stopped():
                    return
                res = func(*args, **kwargs)
            except Exception as e:
                log('Exception happend in a wait_func.')
                log(e)
                return
            else:
                time.sleep(t)
        return res

    return decor
