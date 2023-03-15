from threading import Lock
from typing import Optional
import os.path


class Logger:
    def __init__(self, filepath: str, mode: str, lock: Optional[Lock] = None):
        """
        Implements a simple logger that writes on both stdout and a log file.
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a', stand for "write" and "append" log mode, the latter of which doesn't erase the
            file once opened
        :param lock: pass a shared lock for multiprocess write access
        """
        self.__filepath: str = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.__mode: str = mode
        self.__lock: Optional[Lock] = lock

    @property
    def filepath(self) -> str:
        return self.__filepath

    @filepath.setter
    def filepath(self, filepath: str):
        self.__filepath = filepath

    @property
    def mode(self) -> str:
        return self.__mode

    @mode.setter
    def mode(self, mode: str):
        self.__mode = mode

    @property
    def lock(self) -> Optional[Lock]:
        return self.__lock

    @lock.setter
    def lock(self, lock: Optional[Lock]):
        self.__lock = lock

    def log(self, string: str, print_to_stdout: bool = True):
        if self.lock is not None:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(string + '\n')
        except Exception as e:
            print(e)

        if print_to_stdout:
            print(string)

        if self.lock:
            self.lock.release()
