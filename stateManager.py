
import numpy as np
from stateObj import identity
from stateObj import stateObj

class stateManager:
    """
    It's the only responsible to handle the state.
    A dictionary with stateObj is used to store the state. However, this might change in the future.
    """

    _stateDic = None
    _all_states = None
    _all_state_derivatives = None

    def __init__(self):
        self._stateDic = {}
        self._all_states = []
        self._all_state_derivatives = []
        return

    def registerState(self, name, multiplicity, state_limit_func = identity):
        """
        Call registerState() to create a new state. The name has to be unique.
        :param st:
        :return:
        """
        if name not in self._stateDic:
            self._stateDic[name] = stateObj(name, 0, multiplicity, state_limit_func)
            return True
        else:
            return False

    def unregisterState(self, name):
        """
        It's supposed to be used to unregister a state, but the feature is not implemented as of yet.
        :param name:
        :return:
        """
        if name in self._stateDic:
            del self._stateDic[name]
            return True
        else:
            return False

    def getStates(self, name):
        """
        Retrieves a stateObj holding the state with the given name.
        :param name:
        :return:
        """
        if name in self._stateDic:
            return self._stateDic[name].getState()
        else:
            return None

    def getStateDerivatives(self, name):
        """
        Retrieves a stateObj holding the state with the given name.
        :param name:
        :return:
        """
        if name in self._stateDic:
            return self._stateDic[name].getStateDerivatives()
        else:
            return None

    def setStates(self, name, array):
        """
        Set the state value.
        :param name:
        :param array:
        :return:
        """
        if name in self._stateDic:
            self._stateDic[name].setState(array)
            return True
        else:
            return False

    def setStateDerivatives(self, name, array):
        """
        Set the state derivative value.
        :param name:
        :param array:
        :return:
        """
        if name in self._stateDic:
            self._stateDic[name].setStateDerivative(array)
            return True
        else:
            return False


    def getStateVector(self):
        """
        Creates a 1-dimensional numpy vector with all the states.
        THE ORDER SHOULD NOT MATTER.
        :return:
        """
        state_list = list()
        for key,value in self._stateDic.items():
            state_vec = value.getState()
            if value.getMultiplicity() == 1:
                state_list.append(state_vec)
            else:
                for st in state_vec:
                    state_list.append(st)

        return np.array(state_list)

    def getStateDerivativesVector(self):
        """
        Creates a 1-dimensional numpy vector with all the state derivatives.
        THE ORDER SHOULD NOT MATTER.
        :return:
        """
        state_der_list = list()
        for key,value in self._stateDic.items():
            state_der_vec = value.getStateDerivatives()
            if value.getMultiplicity() == 1:
                state_der_list.append(state_der_vec)
            else:
                for st in state_der_vec:
                    state_der_list.append(st)

        return np.array(state_der_list)

    def setStateVector(self, array):
        """
        From a 1-dimensional numpy array, it sets all the states.
        This is really horrible
        It assumes that the operation of serializing a dictionary is reversible.
        It works, but it's not very elegant.
        :param array:
        :return:
        """
        #
        i = 0

        for key, value in self._stateDic.items():
            mult = value.getMultiplicity()
            if mult == 1:
                value.setState(array[i])
            else:
                value.setState(array[i:i+mult])
            i = i + mult
        return

    def createStateHistory(self, number_of_elements):
        """
        This method simply creates and holds a matrix to hold the state history.
        Perhaps it's not the best place for doing this.
        :param number_of_elements:
        :return:
        """

        self._all_states = {}
        self._all_state_derivatives = {} # CONSIDER MOVING THE DERIVATIVES HISTORY SOMEWHERE ELSE

        for key,value in self._stateDic.items():
            mult = value.getMultiplicity()
            if mult == 1:
                state_history = np.zeros(number_of_elements)
                state_derivative_history = np.zeros(number_of_elements)
            else:
                state_history = np.zeros((number_of_elements, value.getMultiplicity()))
                state_derivative_history = np.zeros((number_of_elements, value.getMultiplicity()))

            self._all_states[key] = state_history
            self._all_state_derivatives[key] = state_derivative_history

        return

    def getStateHistory(self):
        return self._all_states

    def getStateDerivativesHistory(self):
        return self._all_state_derivatives

    def saveState(self, element):
        """
        Once the state history is created, this method saves a state at a given time in a certain position of the state history.
        :param element:
        :return:
        """
        for key, value in self._stateDic.items():
            state_history = self._all_states[key]
            state_derivative_history = self._all_state_derivatives[key]

            if value.getMultiplicity() == 1:
                state_history[element] = value.getState()
                state_derivative_history[element] = value.getStateDerivatives()
            else:
                state_history[element, :] = value.getState()
                state_derivative_history[element,:] = value.getStateDerivatives()

        return