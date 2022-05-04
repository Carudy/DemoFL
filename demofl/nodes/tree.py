from queue import Queue

from .raft import *
from ..aggregation import *


class TreeNode:
    def __init__(self, name, max_child, model, parent=None, verbose=False):
        self.max_m = max_child
        self.name = int(name)
        self.parent = parent
        self.children = []
        self.model = model
        self.verbose = verbose
        self.raft_node = RaftClient(name=self.name, owner=self, verbose=verbose)
        self.pos = 'leaf'
        self.comm_cnt = 0
        self.recv = {}
        self.params = self.model.get_parameters().copy()
        self.online = True
        self.n_comm = 0

    @property
    def id(self):
        return int(self.name)

    def link(self, node):
        self.children.append(node)
        node.parent = self

    def learn_epoch(self, epoch):
        for c in self.children:
            c.params = self.params
            c.model.set_parameters(self.params)
            c.learn_epoch(epoch)
        if self.children:
            group = [self] + self.children
            group_learn(group, epoch)


def group_learn(group, epoch):
    tds = []
    workers = []
    group[0].recv = {}
    group[1].recv = {}
    for c in group:
        if c.model.epoch <= epoch:
            workers.append(c)
            td = threading.Thread(target=c.model.train_epoch)
            tds.append(td)
    log(f'Group leader: {group[0].name} Learning epoch {group[0].model.epoch}, {len(tds)} nodes need to learn.')
    for td in tds: td.start()
    for td in tds: td.join()

    for c in group:
        params = {
            k: secret_share(v * c.model.n_sample, 2)
            for k, v in c.model.get_parameters().items()
        }
        group[0].recv[c.id] = {k: v[0] for k, v in params.items()}
        group[1].recv[c.id] = {k: v[1] for k, v in params.items()}

    for i in range(2):
        c = group[i]
        a = [v.copy() for v in c.recv.values()]
        c.params = fedadd(a)

    keys = list(group[0].recv.values())[0].keys()
    for k in keys:
        group[0].params[k] += group[1].params[k]

    group[0].model.tot_sample = group[0].model.n_sample + sum(c.model.n_sample for c in group[1:] if c.online)


def find_tree_p(root):
    q = Queue()
    q.put(root)
    while not q.empty():
        now = q.get()
        if len(now.children) < now.max_m:
            return now
        for c in now.children:
            q.put(c)
    return None


def make_tree(root, others, verbose=False):
    others = [d for d in others if d.online and d.name != root.name]
    while others:
        p = find_tree_p(root)
        if verbose:
            log(f'Try to add child to {p.name}, others: {", ".join([str(e.name) for e in others])}')
        res = raft_elect([c.raft_node for c in others], verbose=verbose)
        c = 0
        while int(others[c].name) != res:
            c += 1
        p.link(others[c])
        if verbose:
            log(f'{others[c].name} is selected, remain: {len(others) - 1} nodes.')
        others = others[:c] + others[c + 1:]
        others = [d for d in others if d.online]
    log('Tree structure constructed.\n')
