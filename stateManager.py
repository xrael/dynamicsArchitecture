
import numpy as np
from stateObj import stateObj

class stateManager:

    _stateDic = None
    _stateDerDic = None
    _all_states = None

    def __init__(self):
        self._stateDic = {}
        self._stateDerDic = {}
        self._all_states = []
        return

    def registerState(self, st):

        name = st.getName()

        st_dot = stateObj(name, 0, st.getMultiplicity())

        if name not in self._stateDic:
            self._stateDic[name] = st
            self._stateDerDic[name] = st_dot
            return True
        else:
            return False

    def unregisterState(self, name):
        if name in self._stateDic:
            del self._stateDic[name]
            del self._stateDerDic[name]
            return True
        else:
            return False

    def getStates(self, name):
        if name in self._stateDic:
            return self._stateDic[name].getValue()
        else:
            return None

    def getStateDerivatives(self, name):
        if name in self._stateDerDic:
            return self._stateDerDic[name].getValue()
        else:
            return None

    def setStates(self, name, array):
        if name in self._stateDic:
            self._stateDic[name].setValue(array)
            return True
        else:
            return False

    def setStateDerivatives(self, name, array):
        if name in self._stateDerDic:
            self._stateDerDic[name].setValue(array)
            return True
        else:
            return False


    def getStateVector(self):
        state_list = list()
        for key,value in self._stateDic.items():
            state_vec = value.getValue()
            if value.getMultiplicity() == 1:
                state_list.append(state_vec)
            else:
                for st in state_vec:
                    state_list.append(st)

        return np.array(state_list)

    def setStateVector(self, array):
        # This is really horrible
        # It relies on state order. CHANGE IT!
        i = 0

        for key, value in self._stateDic.items():
            mult = value.getMultiplicity()
            if mult == 1:
                value.setValue(array[i])
            else:
                value.setValue(array[i:i+mult])
            i = i + mult



    def getStateDerivativeVector(self):
        state_list = list()
        for key,value in self._stateDerDic.items():
            state_vec = value.getValue()
            if value.getMultiplicity() == 1:
                state_list.append(state_vec)
            else:
                for st in state_vec:
                    state_list.append(st)

        return np.array(state_list)

    def createStateHistory(self, number_of_elements):

        self._all_states = {}

        for key,value in self._stateDic.items():
            mult = value.getMultiplicity()
            if mult == 1:
                state_history = np.zeros(number_of_elements)
            else:
                state_history = np.zeros((number_of_elements, value.getMultiplicity()))

            self._all_states[key] = state_history

        return

    def getStateHistory(self):
        return self._all_states

    def saveState(self, element):
        for key, value in self._stateDic.items():
            state_history = self._all_states[key]

            if value.getMultiplicity() == 1:
                state_history[element] = value.getValue()
            else:
                state_history[element, :] = value.getValue()

        return











