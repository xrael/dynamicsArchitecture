
from abc import ABCMeta, abstractmethod

import integrator
from stateManager import stateManager
import stateEffector

import numpy as np

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
    def equationsOfMotion(self, t):
        """
        Method to compute F(x,t) where x_dot = F(x,t)
        :param t:
        :return:
        """
        pass

    @abstractmethod
    def computeEnergy(self):
        """
        This method should compute the mechanical energy of the system.
        :return:
        """
        pass

    @abstractmethod
    def computeAngularMomentum(self):
        """
        This method should compute the angular momentum of the system in the inertial frame.
        :return:
        """
        pass

    def integrateState(self, t, dt):
        self._integrator.integrate(t, dt)
        return

#----------------------------------------------------------------------------------------------------------------------#

class spacecraft(dynamicalSystem):
    """
    This implementation of dynamicalSystem assumes that there's a central hub. Every single state effector is only
    connected to the hub, as its only parent.
    It is also assumed that every stateEffector solves its contributions to the hub's dynamics using backsubstitution.
    So first the hub state derivatives are solved, and then the rest of the states.
    """

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
        return

    def setHubCoMOffset(self, R_BcB_N):
        self._hub.setHubCoMOffset(R_BcB_N)
        return

    def addVSCMG(self, name, mass, r_OiB, Igs, Igt, Igg, Iws, Iwt, BG0):
        """
        Use this method to add VSCMGs to the spacecraft.
        :param name:
        :param mass:
        :param r_OiB:
        :param Igs:
        :param Igt:
        :param Igg:
        :param Iws:
        :param Iwt:
        :param BG0:
        :return:
        """
        w_BN_name = self._hub.getStateAngularVelocityName()
        sigma_BN_name = self._hub.getStateAttitudeName()
        v_BN_N_name = self._hub.getStateVelocityName()
        vscmg = stateEffector.vscmg.getVSCMG(self, name, mass, r_OiB, Igs, Igt, Igg, Iws, Iwt, BG0, w_BN_name, sigma_BN_name,v_BN_N_name)
        self._hub.addStateEffector(vscmg)
        return vscmg

    def addRW(self, name, mass, r_OiB, Iws, Iwt, BW):
        """
        Use this method to add reaction wheels to the spacecraft.
        :param name:
        :param mass:
        :param r_OiB:
        :param Iws:
        :param Iwt:
        :param BW:
        :return:
        """
        w_BN_name = self._hub.getStateAngularVelocityName()
        sigma_BN_name = self._hub.getStateAttitudeName()
        v_BN_N_name = self._hub.getStateVelocityName()
        rw = stateEffector.reactionWheel.getRW(self, name, mass, r_OiB, Iws, Iwt, BW, w_BN_name, sigma_BN_name,v_BN_N_name)
        self._hub.addStateEffector(rw)
        return rw

    def addCMG(self, name, mass, r_OiB, Igs, Igt, Igg, BG0):
        """
        Use this method to add CMGs to the spacecraft.
        :param name:
        :param mass:
        :param r_OiB:
        :param Igs:
        :param Igt:
        :param Igg:
        :param BG0:
        :return:
        """
        w_BN_name = self._hub.getStateAngularVelocityName()
        sigma_BN_name = self._hub.getStateAttitudeName()
        v_BN_N_name = self._hub.getStateVelocityName()
        cmg = stateEffector.cmg.getCMG(self, name, mass, r_OiB, Igs, Igt, Igg, BG0, w_BN_name, sigma_BN_name,v_BN_N_name)
        self._hub.addStateEffector(cmg)
        return cmg

    def equationsOfMotion(self, t):
        """
        Computes X_dot.
        Assumption: backsubtitution.
        :param t:
        :return:
        """
        self._hub.computeStateDerivatives(t)

        stateEffectors = self._hub.getStateEffectors()
        for effector in stateEffectors:
            effector.computeStateDerivatives(t)

        return

    def computeEnergy(self):

        E = self._hub.computeEnergy()
        stateEffectors = self._hub.getStateEffectors()
        for effector in stateEffectors:
            E += effector.computeEnergy()

        return E

    def computeAngularMomentum(self):
        H = self._hub.computeAngularMomentum()
        stateEffectors = self._hub.getStateEffectors()
        for effector in stateEffectors:
            H += effector.computeAngularMomentum()

        return H