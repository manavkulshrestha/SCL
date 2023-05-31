import time
from socket import socket, AF_INET, SOCK_STREAM

class VacuumGripper:
    def __init__(self, hostname, port, timeout=2, manual=False):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.connect((hostname, port))
        self.socket.settimeout(timeout)

        self.activate(manual=False)

    def _get(self, key):
        self.socket.sendall(f'GET {key}\n'.encode('UTF-8'))
        r_key, val = self.socket.recv(1024).decode('UTF-8').split()

        return int(val)

    def _set(self, key, val):
        self.socket.sendall(f'SET {key} {val}\n'.encode('UTF-8'))
        data = self.socket.recv(1024).decode('UTF-8').split()

        return data == b'ack'

    def _set_until_success(self, key, val):
        while True:
            if self._get(key) == val:
                break
            time.sleep(0.01)
            self._set(key, val)

    def activate(self, manual=False):
        self._set('ACT', 1)
        self._set_until_success('GTO', 0)
        self._set('MOD', 1 if manual else 0)

    def was_timeout(self):
        return self._get('FLT') == 6

    def suction(self, on):
        if self.was_timeout():  # there was a timeout
            self._set_until_success('GTO', 0)

        self._set_until_success('MOD', 0)

        self._set('POS', 0 if on else 255)
        self._set('GTO', 1)

        return self._get('POS')
