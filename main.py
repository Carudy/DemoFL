import time

from demofl import *
from demofl.model import *

DATA_MODEL = {
    'mnist': CNNWrapper,
    'cifar': CifarCNNWrapper,
}


class Simulator:
    def __init__(self, n_client, max_child, epoch, dataset='mnist', test_build=False, verbose=False):
        self.n_client = n_client
        self.max_child = max_child
        self.epoch = epoch
        self.verbose = verbose
        self.dataset = dataset
        self.model_class = DATA_MODEL[self.dataset]
        self.nodes = None
        self.raft = None
        self.root = None
        self.test_build = test_build
        if not test_build:
            self.data_spliter = DataSpliter(dataset)
        self.get_tree()
        # plot data
        self._x = []
        self.acc = []
        self.acc_pure = []

    def check_tree(self):
        if not self.root.online:
            _param = self.root.params.copy()
            _epoch = self.root.model.epoch
            self.get_tree()
            self.root.params = _param
            self.root.model.epoch = _epoch
            self.root.model.set_parameters(_param)
        else:
            pass

    def get_tree(self):
        if self.nodes is None:
            if not self.test_build:
                self.data_spliter.split_even(self.n_client)
            log(f'Generating tree from {self.n_client} clients')
            self.nodes = [TreeNode(name=i,
                                   max_child=self.max_child,
                                   model=self.model_class(name=str(i),
                                                          dataset=
                                                          None if self.test_build else self.data_spliter.get_piece(i)),
                                   verbose=self.verbose)
                          for i in range(self.n_client)]
            self.raft_nodes = [c.raft_node for c in self.nodes]
            for i in range(self.n_client):
                self.nodes[i].raft_node.connect(others=self.raft_nodes)
        else:
            log('Regenerate tree.')
        st = time.time()
        root = raft_elect(self.raft_nodes, verbose=self.verbose)
        self.root = self.nodes[root]
        self.root.pos = 'root'
        log(f'Root selected: {self.root.name}.')
        make_tree(root=self.root, others=self.nodes, verbose=self.verbose)
        self.build_time = time.time() - st

    def train_pure(self):
        model = self.model_class(name='pure', dataset=self.data_spliter.get_piece('all'))
        for e in range(self.epoch):
            log(f'Using {self.root.model.device}, start training epoch {e}')
            model.train_epoch()
            log('Training epoch done.')
            if (e + 1) % 2 == 0:
                acc = self.test_pure(model)
                self.acc_pure.append(acc)

    def test_pure(self, model):
        _model = model.model
        _model.eval()
        correct = 0
        all = 0
        for (x_test, y_test) in self.data_spliter.test_dataset:
            x_test = x_test.to(model.device)
            y_test = y_test.to(model.device)
            outputs = _model(x_test)
            _, pred = torch.max(outputs, 1)
            correct += torch.sum(pred == y_test.data)
            all += len(y_test)
        acc = correct * 100. / all
        log(f'Acc: {acc:.3f}')
        return acc.item()

    def train(self):
        for e in range(self.epoch):
            self.check_tree()
            log(f'Using {self.root.model.device}, start training epoch {e}')
            self.root.learn_epoch(e)
            log('Training epoch done.')
            for v in self.root.params.values():
                v /= self.root.model.tot_sample
            if (e + 1) % 2 == 0:
                self._x.append(e)
                acc = self.test()
                self.acc.append(acc)

    def test(self):
        self.root.model.set_parameters(self.root.params)
        model = self.root.model.model.to(self.root.model.device)
        model.eval()
        correct = 0
        all = 0
        for (x_test, y_test) in self.data_spliter.test_dataset:
            x_test = x_test.to(self.root.model.device)
            y_test = y_test.to(self.root.model.device)
            outputs = model(x_test)
            _, pred = torch.max(outputs, 1)
            correct += torch.sum(pred == y_test.data)
            all += len(y_test)
        acc = correct * 100. / all
        log(f'Acc: {acc:.3f}')
        return acc.item()

    def save(self):
        fp = (BASE_PATH / f'output/acc-{self.dataset}.txt').open('w', encoding='utf-8')
        fp.write(str(self._x) + '\n' + str(self.acc) + '\n' + str(self.acc_pure))


def test_set_up(ns, mc, verbose=False):
    _sim = Simulator(n_client=ns, max_child=mc, epoch=1, dataset='mnist', test_build=True, verbose=verbose)
    res = sum(i.n_comm for i in _sim.nodes) / len(_sim.nodes)
    return res, _sim.build_time


if __name__ == '__main__':
    if ARGS.mode == 'setup':
        res = test_set_up(ARGS.n_client, ARGS.max_child)
        print(f'{ARGS.n_client} {ARGS.max_child} {res}')
    elif ARGS.mode == 'train':
        sim = Simulator(n_client=ARGS.n_client, max_child=ARGS.max_child, epoch=ARGS.epoch, dataset=ARGS.dataset,
                        verbose=ARGS.verbose)
        sim.train_pure()
        sim.train()
        sim.save()
