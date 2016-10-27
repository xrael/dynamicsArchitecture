
import numpy as np
import integrator
import dynamicalSystem
import flightSoftware


class simulatorManager:
    """
    Class that manage the simulation.
    """

    _dynObj = None
    _integrator = None

    _flightSoftware = None

    _t0 = 0.0
    _tf = 0.0
    _dt = 0.0
    _timeVector = None

    _computeEnergy = False
    _energy = None
    _mechanicalPower = None
    _computeAngularMomentum = False
    _angularMomentum = None

    _controlFlag = False

    def __init__(self):

        self._dynObj = None
        self._integrator = None

        self._t0 = 0.0
        self._tf = 0.0
        self._dt = 0.0
        self._timeVector = None

        self._computeEnergy = False
        self._computeAngularMomentum = False
        self._energy = None
        self._mechanicalPower = None
        self._angularMomentum = None

        self._controlFlag = False
        self._flightSoftware = None

        return

    @classmethod
    def getSimulatorManager(cls, dynObj_str, integrator_str, sc_name):
        """
        Use this factory method to get a simulator object.
        :param dynObj_str: [string] Type of dynamical object to be used.
        :param integrator_str:  [string] Type of integrator to be used.
        :param sc_name:
        :param useOrbitalElements: [bool] Set to true if orbital elements are to be computed.
        :param orbElemObj: [orbitalElements] Object to compute orbital elements (if used).
        :return:
        """
        simManager = simulatorManager()

        if integrator_str == 'rk4':
            simManager._integrator = integrator.rk4Integrator()
        elif integrator_str == 'odeint':
            simManager._integrator = integrator.odeIntIntegrator()
        else: # Unrecognized integrator
            return None

        if dynObj_str == 'spacecraft_backSub':
            simManager._dynObj = dynamicalSystem.spacecraft.getSpacecraft(simManager._integrator, sc_name)
        else: # Unrecognized dynamical object
            return None

        return simManager

    def getDynamicalObject(self):
        return self._dynObj

    def setControls(self, estimator, controller, referenceComp):
        self._flightSoftware = flightSoftware.flightSoftware(estimator, controller, referenceComp)
        self._controlFlag = True
        return

    def unsetControls(self):
        self._flightSoftware = None
        self._controlFlag = False
        return

    def setSimulationTimes(self, t0, tf, dt):
        """
        Define the simulation times before calling simulate().
        :param t0:
        :param tf:
        :param dt:
        :return:
        """
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

    def getNmbrOfPoints(self):
        return self._timeVector.size

    def setInitialConditions(self, stateName, value):
        """
        Set the initial conditions ofr every state before calling simulate()
        :param stateName:
        :param value:
        :return:
        """
        stateManager = self._dynObj.getStateManager()
        stateManager.setStates(stateName,value)
        return

    def computeEnergy(self, bool):
        self._computeEnergy = bool
        return

    def getEnergyVector(self):
        return self._energy

    def getMechanicalPowerVector(self):
        return self._mechanicalPower

    def computeAngularMomentum(self, bool):
        self._computeAngularMomentum = bool

    def getAngularMomentumVector(self):
        return self._angularMomentum

    def simulate(self):
        """
        This is were magic occurs. Runs the simulation.
        :return:
        """
        if self._tf - self._t0 <= 0:
            return

        l = self._timeVector.size

        stateManager = self._dynObj.getStateManager()
        stateManager.createStateHistory(l)

        if self._controlFlag:
            self._flightSoftware.createControlForceHistory(l)
            self._flightSoftware.initFlightSoftware(self._t0)

        if self._computeEnergy:
            self._energy = np.zeros(l)
            self._mechanicalPower = np.zeros(l)

        if self._computeAngularMomentum:
            self._angularMomentum = np.zeros((l,3))

        # Start simulation
        self._dynObj.startSimulation()

        for i in range(0,l):
            t = self._timeVector[i]

            if self._computeEnergy:
                self._energy[i] = self._dynObj.computeEnergy()
                self._mechanicalPower[i] = self._dynObj.computeMechanicalPower()

            if self._computeAngularMomentum:
                self._angularMomentum[i] = self._dynObj.computeAngularMomentum()

            if self._controlFlag:
                self._flightSoftware.runTask(t)
                self._flightSoftware.saveControlForces(i)

            # Computes the non-integrable states (if any). e.g. orbital elements.
            self._dynObj.computeNonIntegrableStates()

            stateManager.saveState(i)

            self._dynObj.integrateState(t, self._dt)
            print i
        # end for

        return

    def getStateHistory(self):
        """
        Get the state history to retrieve the evolution of the states on time.
        :return:
        """
        return self._dynObj.getStateManager().getStateHistory()

    def getStateDerivativesHistory(self):
        """
        Get the state history to retrieve the evolution of the states on time.
        :return:
        """
        return self._dynObj.getStateManager().getStateDerivativesHistory()

    def getTimeVector(self):
        return self._timeVector






