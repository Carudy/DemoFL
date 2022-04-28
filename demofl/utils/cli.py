from .log import *


class SimpleCLI:
    def __init__(self, cmd_dict):
        self.cmd_dict = cmd_dict

    def run(self):
        while True:
            cmd = input().strip().lower().split()
            if cmd[0] in self.cmd_dict:
                try:
                    self.cmd_dict[cmd[0]](*cmd[1:])
                except TypeError:
                    log('Illegal parameters')
            else:
                log('Wrong command.')
