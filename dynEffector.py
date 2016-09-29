

from abc import ABCMeta, abstractmethod

class dynEffector:

    __metaclass__ = ABCMeta

    _dynSystem = None

    _effectorName = ''


    def __init__(self, dynSystem, name):
        self._dynSystem = dynSystem
        self._effectorName = name
        return

    def getEffectorName(self):
        return self._effectorName

    @abstractmethod
    def computeRHS(cls): pass

