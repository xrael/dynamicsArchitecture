
import numpy as np
import integrator
import dynamicalSystem
import stateEffector



class simulatorManager:

    _dynObj = None
    _integrator = None

    _t0 = 0.0
    _tf = 0.0
    _dt = 0.0
    _timeVector = None

    _computeEnergy = False
    _energy = None

    def __init__(self):

        self._dynObj = None
        self._integrator = None

        self._t0 = 0.0
        self._tf = 0.0
        self._dt = 0.0
        self._timeVector = None

        self._computeEnergy = False
        self._energy = None

        return

    @classmethod
    def getSimulatorManager(cls, dynObj_str, integrator_str, sc_name):
        simManager = simulatorManager()

        if integrator_str == "rk4":
            simManager._integrator = integrator.rk4Integrator()
        else: # Unrecognized integrator
            return None

        if dynObj_str == 'spacecraft_backSub':
            simManager._dynObj = dynamicalSystem.spacecraft.getSpacecraft(simManager._integrator, sc_name)
        else: # Unrecognized dynamical object
            return None

        return simManager

    def getDynamicalObject(self):
        return self._dynObj


    def setSimulationTimes(self, t0, tf, dt):
        self._t0 = t0
        self._tf = tf
        self._dt = dt

        if (tf - t0) <= 0 or dt <=0 or (t0 + dt) > tf:
            return False
        else:
            self._t0 = t0
            self._tf = tf
            self._dt = dt
            num = int((self._tf - self._t0)/self._dt) + 1
            self._tf = (num - 1) * self._dt + self._t0 # includes the last value
            self._timeVector = np.linspace(self._t0, self._tf, num)
            return True

    def setInitialConditions(self, stateName, value):
        stateManager = self._dynObj.getStateManager()
        stateManager.setStates(stateName,value)
        return

    def computeEnergy(self, bool):
        self._computeEnergy = bool
        return

    def getEnergyVector(self):
        return self._energy

    def simulate(self):

        if self._tf - self._t0 <= 0:
            return

        l = self._timeVector.size

        stateManager = self._dynObj.getStateManager()
        stateManager.createStateHistory(l)

        if self._computeEnergy:
            self._energy = np.zeros(l)

        for i in range(0,l):
            t = self._timeVector[i]

            self._dynObj.integrateState(t, self._dt)
            print i
            stateManager.saveState(i)

            if self._computeEnergy:
                self._energy[i] = self._dynObj.computeEnergy()

        return

    def getStateHistory(self):
        return self._dynObj.getStateManager().getStateHistory()

    def getTimeVector(self):
        return self._timeVector






