
from abc import ABCMeta, abstractmethod

import integrator
from stateManager import stateManager
import stateEffector


class dynamicalSystem:
    """
    Base class of every dynamical system to be simulated.
    """
    __metaclass__ = ABCMeta

    _integrator = None
    _stateManager = None
    # _stateEffectors = None
    # _dynEffectors = None

    def __init__(self):
        self._integrator = None
        self._stateManager = stateManager()
        # self._stateEffectors = list()
        # self._dynEffectors = list()
        return

    def setIntegrator(self, integrator):
        self._integrator = integrator
        self._integrator.setDynamicalSystem(self)
        return

    def getIntegrator(self):
        return self._integrator

    def getStateManager(self):
        return self._stateManager

    # def addStateEffector(self, stateEffector):
    #     self._stateEffectors.append(stateEffector)
    #     stateEffector.setStateManager(self._stateManager)
    #     return
    #
    # def addDynEffector(self, dynEffector):
    #     self._dynEffectors.append(dynEffector)
    #     dynEffector.setStateManager(self._stateManager)
    #     return

    def getState(self, name):
        return self._stateManager.getStates(name)


    @abstractmethod
    def equationsOfMotion(self, t): pass

    @abstractmethod
    def computeEnergy(self): pass

    @abstractmethod
    def computeAngularMomentum(self): pass

    def integrateState(self, t, dt):
        self._integrator.integrate(t, dt)
        return

class spacecraft(dynamicalSystem):

    _hub = None

    def __init__(self):
        super(spacecraft, self).__init__()
        self._hub = None
        return

    @classmethod
    def getSpacecraft(cls, integrator, sc_name):
        sc = spacecraft()
        sc.setIntegrator(integrator)
        sc._hub = stateEffector.spacecraftHub.getSpacecraftHub(sc, sc_name)

        return sc

    def setHubMass(self, mass):
        self._hub.setHubMass(mass)
        return

    def setHubInertia(self, inertia):
        self._hub.setHubInertia(inertia)

    def addVSCMG(self, name, mass, R_CoM, Ig, Iw, G):
        vscmg = stateEffector.vscmg.getVSCMG(self, name, mass, R_CoM, Ig, Iw, G, self._hub)
        self._hub.addStateEffector(vscmg)
        return vscmg



    def equationsOfMotion(self, t):
        self._hub.computeStateDerivatives(t)
        return

    def computeEnergy(self):
        return

    def computeAngularMomentum(self):
        return