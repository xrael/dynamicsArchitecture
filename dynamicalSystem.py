
from abc import ABCMeta, abstractmethod

import integrator
from stateManager import stateManager
import stateEffector
import coordinateTransformations
import attitudeKinematics

import numpy as np


class dynamicalSystem:
    """
    Base class of every dynamical system to be simulated.
    """
    __metaclass__ = ABCMeta

    _integrator = None
    _stateManager = None
    _stateEffectors = None
    # _dynEffectors = None
    _gravity = None

    def __init__(self):
        self._integrator = None
        self._stateManager = stateManager()
        self._stateEffectors = list()
        # self._dynEffectors = list()
        self._gravity = None
        return

    def setIntegrator(self, integrator):
        self._integrator = integrator
        self._integrator.setDynamicalSystem(self)
        return

    def getIntegrator(self):
        return self._integrator

    def getStateManager(self):
        return self._stateManager

    def addStateEffector(self, stateEffector):
        self._stateEffectors.append(stateEffector)
        return
    #
    # def addDynEffector(self, dynEffector):
    #     self._dynEffectors.append(dynEffector)
    #     dynEffector.setStateManager(self._stateManager)
    #     return

    def addGravity(self, gravityObj):
        """
        Adds a gravity object.
        :param gravityObj:
        :return:
        """
        self._gravity = gravityObj
        return

    def getState(self, name):
        return self._stateManager.getStates(name)

    def getStateDerivative(self, name):
        return self._stateManager.getStateDerivatives(name)

    def startSimulation(self):
        """
        If there's anything to start after the initial conditions have been set and before starting the simulation, do it here.
        :return:
        """
        for effector in self._stateEffectors:
            effector.addGravity(self._gravity)  # Gravity is added to all effectors
            effector.startSimulation()
        return

    def computeNonIntegrableStates(self):
        """
        Computes the non-integrable states of every state effector.
        :return:
        """
        for effector in self._stateEffectors:
            effector.computeNonIntegrableStates()
        return

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
    def computeMechanicalPower(self):
        """
        This method should compute the mechanical power injected on the system (positive).
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
        sc.addStateEffector(sc._hub)

        return sc

    def useOrbitalElements(self, useOrbitalElements, orbElObj):
        """
        Add the computation of orbital elements
        :param useOrbitalElements: [bool] True or False if you want to use orbital elements.
        :param orbElemObj: [orbitalElements] Object to compute the orbital elements.
        :return:
        """
        self._hub.useOrbitalElements(useOrbitalElements, orbElObj)
        return

    def setHubMass(self, mass):
        self._hub.setMass(mass)
        return

    def setHubInertia(self, inertia):
        self._hub.setInertia(inertia)
        return

    def setHubCoMOffset(self, R_BcB_N):
        self._hub.setCoMOffset(R_BcB_N)
        return

    def getTotalInertiaB(self):
        """
        This probably shouldn't be here.
        :return:
        """
        I_B = np.zeros((3,3))
        for effector in self._stateEffectors:
            I_B += effector.getInertiaRelativeToReferenceB()

        return I_B

    def addDynEffectorToTheHub(self, dynEff):
        self._hub.addDynEffector(dynEff)
        return

    def addVSCMG(self, name, mass, r_OiB, Igs, Igt, Igg, Iws, Iwt, BG0, nominal_speed_rpm, us_max, ug_max, ug = 0.0, us = 0.0):
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
        :param nominal_speed_rpm:
        :param us_max:
        :param ug: Optional initial gimbal torque.
        :param us: Optional initial wheel torque.
        :return:
        """
        w_BN_name = self._hub.getStateAngularVelocityName()
        sigma_BN_name = self._hub.getStateAttitudeName()
        v_BN_N_name = self._hub.getStateVelocityName()
        vscmg = stateEffector.vscmg.getVSCMG(self, name, mass, r_OiB, Igs, Igt, Igg, Iws, Iwt, BG0, w_BN_name, sigma_BN_name,v_BN_N_name, nominal_speed_rpm, ug, us)
        vscmg.setMaxGimbalTorque(ug_max)
        vscmg.setMaxWheelTorque(us_max)
        self.addStateEffector(vscmg) # Now I'm adding stateEffectors to both: hub and sc. This might change because it's not clear
        self._hub.addStateEffector(vscmg)
        return vscmg

    def addRW(self, name, mass, r_OiB, Iws, Iwt, BW, nominal_speed_rpm, us_max, us = 0.0):
        """
        Use this method to add reaction wheels to the spacecraft.
        :param name:
        :param mass:
        :param r_OiB:
        :param Iws:
        :param Iwt:
        :param BW:
        :param nominal_speed_rpm: Nominal speed of the wheel in rpm.
        :param us_max:
        :param us: Optional initial wheel torque.
        :return:
        """
        w_BN_name = self._hub.getStateAngularVelocityName()
        sigma_BN_name = self._hub.getStateAttitudeName()
        v_BN_N_name = self._hub.getStateVelocityName()
        rw = stateEffector.reactionWheel.getRW(self, name, mass, r_OiB, Iws, Iwt, BW, w_BN_name, sigma_BN_name,v_BN_N_name, nominal_speed_rpm, us)
        rw.setMaxWheelTorque(us_max)
        self.addStateEffector(rw) # Now I'm adding stateEffectors to both: hub and sc. This might change because is not clear
        self._hub.addStateEffector(rw)
        return rw

    def addCMG(self, name, mass, r_OiB, Igs, Igt, Igg, BG0, ug_max, ug = 0.0):
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
        cmg.setMaxGimbalTorque(ug_max)
        self.addStateEffector(cmg) # Now I'm adding stateEffectors to both: hub and sc. This might change because it's not clear
        self._hub.addStateEffector(cmg)
        return cmg

    def equationsOfMotion(self, t):
        """
        Computes X_dot.
        Assumption: backsubtitution.
        :param t:
        :return:
        """
        if self._gravity is not None:
            self._gravity.computeGravity()

        self._hub.computeStateDerivatives(t)

        stateEffectors = self._hub.getStateEffectors()
        for effector in stateEffectors:
            effector.computeStateDerivatives(t)

        return self._stateManager.getStateDerivativesVector()

    def computeEnergy(self):
        """
        Computes total mechanical energy.
        :return:
        """
        E = self._hub.computeEnergy()
        stateEffectors = self._hub.getStateEffectors()
        for effector in stateEffectors:
            E += effector.computeEnergy()

        return E

    def computeMechanicalPower(self):
        """
        Computes rotational mechanical power.
        :return:
        """
        P = self._hub.computeMechanicalPower()
        stateEffectors = self._hub.getStateEffectors()
        for effector in stateEffectors:
            P += effector.computeMechanicalPower()
        return P


    def computeAngularMomentum(self):
        """
        Computes the total angular momentum of the spacecraft relative to reference point B in inertial frame.
        Beware that momentum relative to the center of mass may be conserved while this angular momentum is not.
        :return:
        """
        H = self._hub.computeAngularMomentum()
        stateEffectors = self._hub.getStateEffectors()
        for effector in stateEffectors:
            H += effector.computeAngularMomentum()

        return H

#----------------------------------------------------------------------------------------------------------------------#

class threeBodyProblemSpacecraft(dynamicalSystem):

    _name = ''

    _mu1 = 0.0
    _mu2 = 0.0
    _stateNames = None

    _statePositionName = ''
    _stateVelocityName = ''
    _stateTwoBodyPrPos = ''
    _stateTwoBodyPrVel = ''
    _stateBody1Pos = ''
    _stateBody1Vel = ''
    _stateBody2Pos = ''
    _stateBody2Vel = ''

    _orbFrame = False
    _mainBody = 1
    _useOrbitalElements = False


    def __init__(self, name):
        super(threeBodyProblemSpacecraft, self).__init__()

        self._name = name

        self._stateNames = ('r', 'r_dot', 'R', 'V', 'R1', 'R1_dot', 'R2', 'R2_dot')

        self._statePositionName = self._name + '_' + self._stateNames[0]
        self._stateVelocityName = self._name + '_' + self._stateNames[1]

        self._stateTwoBodyPrPos = self._name + '_' + self._stateNames[2]
        self._stateTwoBodyPrVel = self._name + '_' + self._stateNames[3]

        self._stateBody1Pos = self._name + '_' + self._stateNames[4]
        self._stateBody1Vel = self._name + '_' + self._stateNames[5]
        self._stateBody2Pos = self._name + '_' + self._stateNames[6]
        self._stateBody2Vel = self._name + '_' + self._stateNames[7]

        self._mu1 = 0.0
        self._mu2 = 0.0

        self._orbFrame = False
        self._useOrbitalElements = False
        self._mainBody = 1

        return

    @classmethod
    def getthreeBodyProblemSpacecraft(cls, integrator, name):
        sc = threeBodyProblemSpacecraft(name)
        sc.setIntegrator(integrator)
        sc._orbFrame = False
        sc.registerStates()

        return sc

    def setOrbitalFrame(self, flag):
        self._orbFrame = flag
        if self._orbFrame:
            self.registerJacobiIntegral()
        return

    def setGravitationalParameters(self, mu1, mu2):
        self._mu1 = mu1
        self._mu2 = mu2
        return

    def registerStates(self):
        """
        This method register the states. Don't use it. Supposed to be private.
        :return:
        """
        stateMan = self.getStateManager()

        if not stateMan.registerState(self._statePositionName, 3):
            return False
        elif not stateMan.registerState(self._stateVelocityName, 3):
            stateMan.unregisterState(self._statePositionName)
            return False
        elif not stateMan.registerState(self._stateTwoBodyPrPos, 3):
            stateMan.unregisterState(self._statePositionName)
            stateMan.unregisterState(self._stateVelocityName)
            return False
        elif not stateMan.registerState(self._stateTwoBodyPrVel, 3):
            stateMan.unregisterState(self._statePositionName)
            stateMan.unregisterState(self._stateVelocityName)
            stateMan.unregisterState(self._stateTwoBodyPrPos)
            return False
        elif not stateMan.registerNonIntegrableState(self._stateBody1Pos, 3):
            stateMan.unregisterState(self._statePositionName)
            stateMan.unregisterState(self._stateVelocityName)
            stateMan.unregisterState(self._stateTwoBodyPrPos)
            stateMan.unregisterState(self._stateTwoBodyPrVel)
            return False
        elif not stateMan.registerNonIntegrableState(self._stateBody1Vel, 3):
            stateMan.unregisterState(self._statePositionName)
            stateMan.unregisterState(self._stateVelocityName)
            stateMan.unregisterState(self._stateTwoBodyPrPos)
            stateMan.unregisterState(self._stateTwoBodyPrVel)
            stateMan.unregisterNonIntegrableState(self._stateBody1Pos)
            return False
        elif not stateMan.registerNonIntegrableState(self._stateBody2Pos, 3):
            stateMan.unregisterState(self._statePositionName)
            stateMan.unregisterState(self._stateVelocityName)
            stateMan.unregisterState(self._stateTwoBodyPrPos)
            stateMan.unregisterState(self._stateTwoBodyPrVel)
            stateMan.unregisterNonIntegrableState(self._stateBody1Pos)
            stateMan.unregisterNonIntegrableState(self._stateBody1Vel)
            return False
        elif not stateMan.registerNonIntegrableState(self._stateBody2Vel, 3):
            stateMan.unregisterState(self._statePositionName)
            stateMan.unregisterState(self._stateVelocityName)
            stateMan.unregisterState(self._stateTwoBodyPrPos)
            stateMan.unregisterState(self._stateTwoBodyPrVel)
            stateMan.unregisterNonIntegrableState(self._stateBody1Pos)
            stateMan.unregisterNonIntegrableState(self._stateBody1Vel)
            stateMan.unregisterNonIntegrableState(self._stateBody2Pos)
            return False
        else:
            return True

    def registerJacobiIntegral(self):
        stateMan = self.getStateManager()
        if self._orbFrame:
            if not stateMan.registerNonIntegrableState(self._name + '_' + 'jacobi', 1):
                return False
            elif not stateMan.registerNonIntegrableState(self._name + '_' + 'w', 3):
                stateMan.unregisterNonIntegrableState(self._name + '_'+ 'jacobi')
                return False
            elif not stateMan.registerNonIntegrableState(self._name + '_' + 'r_inertial', 3):
                stateMan.unregisterNonIntegrableState(self._name + '_'+ 'jacobi')
                stateMan.unregisterNonIntegrableState(self._name + '_'+ 'w')
                return False
            elif not stateMan.registerNonIntegrableState(self._name + '_' + 'r_dot_inertial', 3):
                stateMan.unregisterNonIntegrableState(self._name + '_'+ 'jacobi')
                stateMan.unregisterNonIntegrableState(self._name + '_'+ 'w')
                stateMan.unregisterNonIntegrableState(self._name + '_'+ 'r_inertial')
                return False
            return True
        else:
            return True

    def useOrbitalElements(self, useOrbitalElements, orbElObj, bodyNmbr):
        """
        Add the computation of orbital elements
        :param useOrbitalElements: [bool] True or False if you want to use orbital elements.
        :param orbElemObj: [orbitalElements] Object to compute the orbital elements.
        :return:
        """
        self._useOrbitalElements = useOrbitalElements
        self._mainBody = bodyNmbr

        if useOrbitalElements: # If orbital elements are to be computed
            self.registerOrbitalElements()
            self._orbElObj = orbElObj
        return

    def registerOrbitalElements(self):
        """
        If orbital elements are used, they are registered as non-integrable states.
        This might change in the future to allow an arbitrary set of orbital elements.
        THIS HAS TO CHANGE -> BAD CODE!!!!!!!
        :return:
        """
        stateMan = self.getStateManager()

        if not stateMan.registerNonIntegrableState(self._name + '_' + 'semimajor_axis', 1):
            return False
        elif not stateMan.registerNonIntegrableState(self._name + '_' + 'eccentricity', 1):
            stateMan.unregisterNonIntegrableState(self._name + '_'+ 'semimajor_axis')
            return False
        elif not stateMan.registerNonIntegrableState(self._name + '_'+ 'inclination', 1):
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'semimajor_axis')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'eccentricity')
            return False
        elif not stateMan.registerNonIntegrableState(self._name + '_'+ 'long_asc_node', 1):
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'semimajor_axis')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'eccentricity')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'inclination')
            return False
        elif not stateMan.registerNonIntegrableState(self._name + '_' + 'arg_periapsis', 1):
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'semimajor_axis')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'eccentricity')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'inclination')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'long_asc_node')
            return False
        elif not stateMan.registerNonIntegrableState(self._name + '_' + 'true_anomaly', 1):
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'semimajor_axis')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'eccentricity')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'inclination')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'long_asc_node')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'arg_periapsis')
            return False
        elif not stateMan.registerNonIntegrableState(self._name + '_' + 'eccentric_anomaly', 1):
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'semimajor_axis')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'eccentricity')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'inclination')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'long_asc_node')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'arg_periapsis')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'true_anomaly')
            return False
        elif not stateMan.registerNonIntegrableState(self._name + '_' + 'longitude_of_periapse', 1):
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'semimajor_axis')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'eccentricity')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'inclination')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'long_asc_node')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'arg_periapsis')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'true_anomaly')
            stateMan.unregisterNonIntegrableState(self._name + '_' + 'eccentric_anomaly')
            return False
        else:
            return True

    def computeNonIntegrableStates(self):
        """
        Computes the non-integrable states (orbital elements and jacobi integral in this case).
        Overrides the base method.
        :return:
        """
        stateMan = self.getStateManager()

        R = stateMan.getStates(self._stateTwoBodyPrPos) # in the inertial frame
        R_dot = stateMan.getStates(self._stateTwoBodyPrVel)

        nu = self._mu2/(self._mu1 + self._mu2)

        R1 = -nu*R
        R2 = (1-nu)*R
        R1_dot = -nu*R_dot
        R2_dot = (1-nu)*R_dot

        stateMan.setNonIntegrableStates(self._stateBody1Pos, R1)
        stateMan.setNonIntegrableStates(self._stateBody1Vel, R1_dot)
        stateMan.setNonIntegrableStates(self._stateBody2Pos, R2)
        stateMan.setNonIntegrableStates(self._stateBody2Vel, R2_dot)

        if self._useOrbitalElements:

            r = stateMan.getStates(self._statePositionName)
            r_dot = stateMan.getStates(self._stateVelocityName)
            # The orbital elements are computed relative to body 1
            if self._mainBody == 1:
                r_rel = r - R1
                r_rel_dot = r_dot - R1_dot
            else:
                r_rel = r - R2
                r_rel_dot = r_dot - R2_dot
            self._orbElObj.setOrbitalElementsFromPosVel(r_rel, r_rel_dot) # This is a pretty general interface

            # This is not general for every single orbital element set.
            stateMan.setNonIntegrableStates(self._name + '_' + 'semimajor_axis', self._orbElObj.a)
            stateMan.setNonIntegrableStates(self._name + '_' + 'eccentricity', self._orbElObj.e)
            stateMan.setNonIntegrableStates(self._name + '_' + 'inclination', self._orbElObj.i)
            stateMan.setNonIntegrableStates(self._name + '_' + 'long_asc_node', self._orbElObj.raan)
            stateMan.setNonIntegrableStates(self._name + '_' + 'arg_periapsis', self._orbElObj.w)
            stateMan.setNonIntegrableStates(self._name + '_' + 'true_anomaly', self._orbElObj.nu)
            stateMan.setNonIntegrableStates(self._name + '_' + 'longitude_of_periapse', self._orbElObj.w_true)
            stateMan.setNonIntegrableStates(self._name + '_' + 'eccentric_anomaly', self._orbElObj.E)

        if self._orbFrame:
            r = stateMan.getStates(self._statePositionName) # in the rotating frame
            r_dot = stateMan.getStates(self._stateVelocityName) # in the rotating frame
            R = stateMan.getStates(self._stateTwoBodyPrPos) # in the inertial frame
            R_dot = stateMan.getStates(self._stateTwoBodyPrVel)

            R_ddot = -(self._mu1 + self._mu2)/np.linalg.norm(R)**3 * R

            H = np.cross(R, R_dot)
            H_dot = np.cross(R, R_ddot)
            H_ddot = np.array([0,0,0]) # Assume this is zero

            (b1, b1_dot, b1_ddot) = coordinateTransformations.computeUnitVectorDerivatives(R, R_dot, R_ddot)
            (b3, b3_dot, b3_ddot) = coordinateTransformations.computeUnitVectorDerivatives(H, H_dot, H_ddot)

            b2 = np.cross(b3, b1)     # along-track
            b2_dot = np.cross(b3_dot, b1) + np.cross(b3, b1_dot)
            b2_ddot = np.cross(b3_ddot, b1) + 2*np.cross(b3_dot, b1_dot) + np.cross(b3, b1_ddot)

            BN = np.array([b1, b2, b3])
            BN_dot = np.array([b1_dot, b2_dot, b3_dot])
            BN_ddot = np.array([b1_ddot, b2_ddot, b3_ddot])

            w = attitudeKinematics.DCMrate2angVel(BN, BN_dot)
            w_dot = attitudeKinematics.DCMdoubleRate2angVelDot(BN, BN_ddot, w)

            nu = self._mu2/(self._mu1 + self._mu2)

            R1 = -nu*BN.dot(R)
            R2 = (1-nu)*BN.dot(R)

            r_1 = r - R1
            r_2 = r - R2

            U = self._mu1/np.linalg.norm(r_1) + self._mu2/np.linalg.norm(r_2)

            w_x_r = np.cross(w, r)

            J = 0.5 * np.inner(r_dot, r_dot) - 0.5 * np.inner(w_x_r, w_x_r) - U

            r_inertial = BN.T.dot(r)
            r_dot_inertial = BN.T.dot(r_dot + w_x_r)

            stateMan.setNonIntegrableStates(self._name + '_' + 'jacobi', J)
            stateMan.setNonIntegrableStates(self._name + '_' + 'w', w)
            stateMan.setNonIntegrableStates(self._name + '_' + 'r_inertial', r_inertial)
            stateMan.setNonIntegrableStates(self._name + '_' + 'r_dot_inertial', r_dot_inertial)

        return

    #------------------State Name getters------------------#
    def getStatePositionName(self):
        return self._statePositionName

    def getStateVelocityName(self):
        return self._stateVelocityName

    def getStateTwoBodyPositionName(self):
        return self._stateTwoBodyPrPos

    def getStateTwoBodyVelocityVelocityName(self):
        return self._stateTwoBodyPrVel
    #------------------------------------------------------#

    def equationsOfMotion(self, t):
        if self._orbFrame:
            self.equationsOfMotionRotating(t)
        else:
            self.equationsOfMotionInertial(t)
        return self._stateManager.getStateDerivativesVector()


    def equationsOfMotionInertial(self, t):

        stateMan = self.getStateManager()

        r = stateMan.getStates(self._statePositionName)
        v = stateMan.getStates(self._stateVelocityName)
        R = stateMan.getStates(self._stateTwoBodyPrPos)
        R_dot = stateMan.getStates(self._stateTwoBodyPrVel)

        nu = self._mu2/(self._mu1 + self._mu2)

        R1 = -nu*R
        R2 = (1-nu)*R

        r_1 = r - R1
        r_2 = r - R2

        r_ddot = -self._mu1/np.linalg.norm(r_1)**3 * r_1 - self._mu2/np.linalg.norm(r_2)**3 * r_2

        R_ddot = -(self._mu1 + self._mu2)/np.linalg.norm(R)**3 * R

        stateMan.setStateDerivatives(self._statePositionName, v)
        stateMan.setStateDerivatives(self._stateVelocityName, r_ddot)
        stateMan.setStateDerivatives(self._stateTwoBodyPrPos, R_dot)
        stateMan.setStateDerivatives(self._stateTwoBodyPrVel, R_ddot)

        return

    def equationsOfMotionRotating(self, t):

        stateMan = self.getStateManager()

        r = stateMan.getStates(self._statePositionName) # in the rotating frame
        r_dot = stateMan.getStates(self._stateVelocityName) # in the rotating frame
        R = stateMan.getStates(self._stateTwoBodyPrPos) # in the inertial frame
        R_dot = stateMan.getStates(self._stateTwoBodyPrVel)

        R_ddot = -(self._mu1 + self._mu2)/np.linalg.norm(R)**3 * R

        H = np.cross(R, R_dot)
        H_dot = np.cross(R, R_ddot)
        H_ddot = np.array([0,0,0]) # Assume this is zero

        (b1, b1_dot, b1_ddot) = coordinateTransformations.computeUnitVectorDerivatives(R, R_dot, R_ddot)
        (b3, b3_dot, b3_ddot) = coordinateTransformations.computeUnitVectorDerivatives(H, H_dot, H_ddot)

        b2 = np.cross(b3, b1)     # along-track
        b2_dot = np.cross(b3_dot, b1) + np.cross(b3, b1_dot)
        b2_ddot = np.cross(b3_ddot, b1) + 2*np.cross(b3_dot, b1_dot) + np.cross(b3, b1_ddot)

        BN = np.array([b1, b2, b3])
        BN_dot = np.array([b1_dot, b2_dot, b3_dot])
        BN_ddot = np.array([b1_ddot, b2_ddot, b3_ddot])

        w = attitudeKinematics.DCMrate2angVel(BN, BN_dot)
        w_dot = attitudeKinematics.DCMdoubleRate2angVelDot(BN, BN_ddot, w)

        nu = self._mu2/(self._mu1 + self._mu2)

        R1 = -nu*BN.dot(R)
        R2 = (1-nu)*BN.dot(R)

        r_1 = r - R1
        r_2 = r - R2

        dU_dr = -self._mu1/np.linalg.norm(r_1)**3 * r_1 - self._mu2/np.linalg.norm(r_2)**3 * r_2

        r_ddot = -np.cross(w_dot, r) - 2*np.cross(w, r_dot) - np.cross(w, np.cross(w, r)) + dU_dr

        stateMan.setStateDerivatives(self._statePositionName, r_dot)
        stateMan.setStateDerivatives(self._stateVelocityName, r_ddot)
        stateMan.setStateDerivatives(self._stateTwoBodyPrPos, R_dot)
        stateMan.setStateDerivatives(self._stateTwoBodyPrVel, R_ddot)

        return


    def computeEnergy(self):
        return

    def computeMechanicalPower(self):
        return

    def computeAngularMomentum(self):
        return




