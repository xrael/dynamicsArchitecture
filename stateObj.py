

import numpy as np

def identity(x): return x

class stateObj:
    """
    Represents a basic state and its derivative with a name and id, multiplicity (number of columns), and a function to process
    the state that can be use to impose a constraint (like with quaternions) or switching (like with MRPs).
    Only the numerical value of the state and its derivative can change.
    """

    _name = ''
    _id = 0
    _state = None
    _stateDerivative = None
    _multiplicity = 1
    _stateLimitFunc = None

    def __init__(self, name, id, multiplicity, state_limit_func = identity):
        self._name = name
        self._id = id
        self._multiplicity = multiplicity
        self._stateLimitFunc = state_limit_func
        if multiplicity == 1:
            self._state = 0.0
            self._stateDerivative = 0.0
        else:
            self._state = np.zeros(multiplicity)
            self._stateDerivative = np.zeros(multiplicity)

        return

    #-----------------------------------Getters-------------------------------------#
    def getName(self):
        return self._name

    def getID(self):
        return self._id

    def getMultiplicity(self):
        return self._multiplicity

    def getState(self):
        return self._state

    def getStateDerivatives(self):
        return self._stateDerivative

    #-----------------------------------Setters-------------------------------------#
    def setState(self, value):
        value = self._stateLimitFunc(value) # To switch MRPs, for example
        if self._multiplicity == 1:
            self._state = value
        else:
            for i in range(0, self._multiplicity):
                self._state[i] = value[i]
        return

    def setStateDerivative(self, value):
        if self._multiplicity == 1:
            self._stateDerivative = value
        else:
            for i in range(0, self._multiplicity):
                self._stateDerivative[i] = value[i]
        return

    def propagate(self, dt):
        return self._state + dt * self._stateDerivative

    #-----------------------------------Operators-------------------------------------#
    def __add__(self, other):
        """
        Overloading of the '+' operator.
        :param other:
        :return:
        """
        return self._state + other

    def __mul__(self, other):
        """
        Overloading of the '*' operator.
        :param other:
        :return:
        """
        return self._state * other