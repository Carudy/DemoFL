import argparse


class MyParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--mode', default='setup')
        self.parser.add_argument('--dataset', default='mnist')
        self.parser.add_argument('--n_client', default=6, type=int)
        self.parser.add_argument('--max_child', default=4, type=int)
        self.parser.add_argument('--verbose', default=False)
        self.args = self.parser.parse_args()

    def __getitem__(self, item):
        return self.args.__getattribute__(item)

    def __getattr__(self, item):
        return self.args.__getattribute__(item)


ARGS = MyParser()
