

import numpy as np

def identity(x): return x

class stateObj:

    _name = ''
    _id = 0
    _value = None
    _multiplicity = 1
    _stateLimitFunc = None

    def __init__(self, name, id, multiplicity, state_limit_func = identity):
        self._name = name
        self._id = id
        self._multiplicity = multiplicity
        self._stateLimitFunc = state_limit_func
        if multiplicity == 1:
            self._value = 0.0
        else:
            self._value = np.zeros(multiplicity)

        return

    def getName(self):
        return self._name

    def getID(self):
        return self._id

    def getMultiplicity(self):
        return self._multiplicity

    def getValue(self):
        return self._value

    def setValue(self, value):
        value = self._stateLimitFunc(value)
        if self._multiplicity == 1:
            self._value = value
        else:
            for i in range(0, self._multiplicity):
                self._value[i] = value[i]
        return