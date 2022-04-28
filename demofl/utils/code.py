from enum import Enum


class Code(Enum):
    ok = 0
    no_task = 1
    server_error = 2
    busy = 3
    full = 4
    auth_error = 5
    already = 6
    wrong_epoch = 7
    unkown = 8

    def __int__(self):
        return self.value

    def __eq__(self, other):
        return int(self) == other
