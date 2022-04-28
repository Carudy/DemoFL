from collections import defaultdict
from concurrent import futures
import grpc
import threading

from ..utils import *
from ..proto import *


class RaftMaster:
    def __init__(self, members):
        self.members = members
        self.threads = []

    def elect_root(self):
        root = None
        while root is None:
            self.threads = []
            for c in self.members:
                if c.online:
                    td = threading.Thread(target=c.elect_root)
                    self.threads.append(td)
            for td in self.threads: td.start()
            for td in self.threads: td.join()
            leaders = [c for c in self.members if c.online and c.state == 'leader']
            if len(leaders):
                root = sorted(leaders, key=lambda c: sum(c.delays) / len(c.delays))
        return int(root[0].name)

    def elect_node(self, members):
        if len(members) <= 2:
            return random.choice([int(c.name) for c in members])

        nodes = []
        while not len(nodes):
            self.threads = []
            for c in members:
                if c.online:
                    td = threading.Thread(target=c.elect_node, kwargs={
                        'others': [int(d.name) for d in members if d.name != c.name]
                    })
                    self.threads.append(td)
            for td in self.threads: td.start()
            for td in self.threads: td.join()
            nodes = [c for c in members if c.online and c.state == 'leader']
        return int(nodes[0].name)


class DemoService(demo_pb2_grpc.DemoFL):
    def __init__(self, node):
        self.node = node

    def broadcast(self, request, context):
        msg = request.msg
        if msg == 'ask':
            if self.node.state == 'follower' and not self.node.voted:
                self.node.voted = True
                self.node.followed = True
                return demo_pb2.Res(res='ok')
            else:
                return demo_pb2.Res(res='no')
        if msg == 'leader':
            if self.node.state != 'leader':
                self.node.state = 'follower'
                self.node.voted = True
                self.node.followed = True
                return demo_pb2.Res(res='ok')
            else:
                return demo_pb2.Res(res='no')
        return demo_pb2.Res(res='no')


class RaftClient:
    def __init__(self, name, owner):
        self.owner = owner
        self.name = name
        self.rpc_service = DemoService(self)
        self.rpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        demo_pb2_grpc.add_DemoFLServicer_to_server(self.rpc_service, self.rpc_server)
        self.addr = 'localhost'
        self.port = int(self.name) + 50000
        self.start_rpc_serve()

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

    def elect_root(self):
        msg = 'ask'
        self.state = 'follower'
        self.delays = [0]
        self.term = 1
        self.votes = 0
        self.voted = False
        self.followed = False
        self.msgs = []
        t = random.random()
        time.sleep(t)
        if self.voted:
            return
        self.voted = True
        self.votes = 1
        self.state == 'candidate'
        for c in self.conn.values():
            if c is None: continue
            if self.followed:
                break
            st = time.time()
            res = c.broadcast(demo_pb2.Msg(msg=msg))
            self.delays.append((time.time() - st) * 1000)
            if res.res == 'ok':
                self.votes += 1
                if self.votes >= (len(self.others) + 1) >> 1:
                    self.state = 'leader'
                    msg = 'leader'
                    self.followed = True
                    break

        if msg == 'leader':
            for c in self.conn.values():
                if c is None: continue
                c.broadcast(demo_pb2.Msg(msg=msg))

    def elect_node(self, others):
        conn = [self.conn[i] for i in others]
        tm = (len(others) + 1) >> 1

        msg = 'ask'
        self.state = 'follower'
        self.delays = [0]
        self.term = 1
        self.votes = 0
        self.voted = False
        self.followed = False
        self.msgs = []
        t = random.random()
        time.sleep(t)
        if self.voted:
            return
        self.voted = True
        self.votes = 1
        self.state == 'candidate'
        for c in conn:
            if c is None: continue
            if self.followed:
                break
            st = time.time()
            res = c.broadcast(demo_pb2.Msg(msg=msg))
            self.delays.append((time.time() - st) * 1000)
            if res.res == 'ok':
                self.votes += 1
                if self.votes >= tm:
                    self.state = 'leader'
                    msg = 'leader'
                    self.followed = True
                    break

        if msg == 'leader':
            for c in conn:
                if c is None: continue
                c.broadcast(demo_pb2.Msg(msg=msg))
