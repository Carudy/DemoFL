from demofl import *
from demofl.model import *


class Simulator:
    def __init__(self, n_client, epoch, model_class, dataset='mnist'):
        self.n_client = n_client
        self.epoch = epoch
        self.model_class = model_class
        self.nodes = None
        self.raft = None
        self.root = None
        if dataset == 'mnist':
            self.data_spliter = DataSpliter('mnist')
        self.get_tree()

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
            self.data_spliter.split_even(self.n_client)
            log(f'Generating tree from {self.n_client} clients')
            self.nodes = [TreeNode(name=i, model=self.model_class(name=str(i), dataset=self.data_spliter.get_piece(i)))
                          for i in range(self.n_client)]
            self.raft = RaftMaster(members=[_c.raft_node for _c in self.nodes])
            for i in range(self.n_client):
                self.nodes[i].raft_node.connect(
                    others=[c for c in [_c for _i, _c in enumerate(self.raft.members) if _i != i]])
        else:
            log('Regenerate tree.')
        root = self.raft.elect_root()
        self.root = self.nodes[root]
        self.root.pos = 'root'
        log(f'{self.root.name} is the root.')
        make_tree(self.raft, self.root, [c for c in self.nodes if c.name != self.root.name])

    def train(self):
        for e in range(self.epoch):
            self.check_tree()
            log(f'Using {self.root.model.device}, start training epoch {e}')
            self.root.learn_epoch(e)
            log('Training epoch done.')
            for v in self.root.params.values():
                v /= self.root.model.tot_sample
            if (e + 1) % 2 == 0:
                self.test()

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
        log(f'Acc: {correct * 100. / all:.3f}')


if __name__ == '__main__':
    sim = Simulator(n_client=6, epoch=6, model_class=CNNWrapper)
    sim.train()
    sim.test()
