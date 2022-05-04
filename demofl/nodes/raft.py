from collections import defaultdict
from concurrent import futures
import grpc

from ..utils import *
from ..proto import *


class RaftService(demo_pb2_grpc.DemoFL):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def broadcast(self, request, context):
        msg = request.msg
        _s = f'\t{self.node.name} received {msg} from {request.source}.'
        self.node.owner.n_comm += 1
        if msg == 'ask':
            if self.node.state == 'follower' and self.node.voted < 0:
                if self.verbose:
                    log(_s + ' vote!')
                self.node.voted = request.source
                self.node.followed = request.source
                return demo_pb2.Res(res='ok')
            else:
                if self.verbose:
                    log(_s + f' not vote! {self.node.state} voted: {self.node.voted} votes: {self.node.votes}')
                return demo_pb2.Res(res='no')
        if msg == 'leader':
            if self.node.state != 'leader':
                if self.verbose:
                    log(_s + ' follow!')
                self.node.state = 'follower'
                self.node.voted = request.source
                self.node.followed = request.source
                return demo_pb2.Res(res='ok')
            else:
                if self.verbose:
                    log(_s + f' not follow! {self.node.state}')
                return demo_pb2.Res(res='no')
        return demo_pb2.Res(res='no')


def raft_elect(members, verbose=False):
    if len(members) <= 2:
        if verbose:
            log('Members less than 3, random choose.')
        return random.choice([int(c.name) for c in members])
    ret = []
    while not ret:
        for c in members:
            c.init_elect()
        thread_method_map([c for c in members if c.online], "campaign",
                          kwargs={'others': [int(d.name) for d in members]})
        ret = [c for c in members if c.online and c.state == 'leader']
        if len(ret):
            ret = sorted(ret, key=lambda c: sum(c.delays) / len(c.delays))
            if verbose:
                _s = sorted(c.votes for c in members)[::-1]
                _s = f'Elected. {", ".join(str(i) for i in _s)}, tot: {sum(_s)}'
                log(_s)
        else:
            if verbose:
                _s = sorted(c.votes for c in members)[::-1]
                _s = f'Elect Failed. {", ".join(str(i) for i in _s)}, tot: {sum(_s)}'
                log(_s)
    return int(ret[0].name)


class RaftClient:
    def __init__(self, name, owner, verbose=False):
        self.verbose = verbose
        self.owner = owner
        self.name = name
        self.rpc_service = RaftService(verbose=self.verbose)
        self.rpc_service.node = self
        self.rpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        demo_pb2_grpc.add_DemoFLServicer_to_server(self.rpc_service, self.rpc_server)
        self.port = int(self.name) + 50000
        self.addr = 'localhost'
        self.init_elect()
        self.start_rpc_serve()

    def init_elect(self):
        self.msg = 'ask'
        self.state = 'follower'
        self.votes = 0
        self.voted = -1
        self.followed = -1
        self.delays = []

    @property
    def online(self):
        return self.owner.online

    def start_rpc_serve(self):
        self.rpc_server.add_insecure_port(f'[::]:{self.port}')
        self.rpc_server.start()

    def connect(self, others):
        self.others = others
        self.conn = defaultdict(lambda: None)
        for c in self.others:
            if c.name != self.name:
                channel = grpc.insecure_channel(f'{c.addr}:{c.port}')
                self.conn[int(c.name)] = demo_pb2_grpc.DemoFLStub(channel)

    def broadcast(self, tar, method='vote', half_num=None):
        st = time.time()
        res = tar.broadcast(demo_pb2.Msg(source=self.owner.id, msg=self.msg))
        self.owner.n_comm += 1
        if method != 'vote':
            return
        self.delays.append((time.time() - st) * 1000)
        if res.res == 'ok':
            if self.msg == 'ask':
                self.votes += 1
            if self.state != 'leader' and (self.votes >= half_num):
                self.state = 'leader'
                self.msg = 'leader'
                self.followed = self.owner.id
                return

    def campaign(self, others):
        conn = [self.conn[i] for i in others if self.conn[i] and i != self.owner.id]
        half_num = len(others) >> 1

        sleep_t = random.random()
        time.sleep(sleep_t)
        if self.verbose:
            log(f'{self.name} sleeped for {sleep_t}')
        if self.voted >= 0 or self.followed >= 0:
            if self.verbose:
                log(f'\t{self.name} stopped. voted: {self.voted} followed: {self.followed}')
            return
        self.voted = self.owner.id
        self.votes = 1
        self.state = 'candidate'
        if self.verbose:
            log(f'\t{self.name} voted for self.')

        tds = [threading.Thread(target=self.broadcast, kwargs={'tar': c, 'half_num': half_num}) for c in conn]
        join_threads(tds)

        if self.msg == 'leader':
            tds = [threading.Thread(target=self.broadcast, kwargs={'tar': c, 'method': 'leader'}) for c in conn]
            join_threads(tds)
