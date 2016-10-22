
from abc import ABCMeta, abstractmethod

import attitudeKinematics
import coordinateTransformations
import numpy as np
from stateObj import stateObj



class stateEffector:
    """
    A state effector is an abstract class that is to be used for implementing dynamical effects that have a state.
    Rigid bodies, reaction wheels, slosh, CMGs, etc, have dynamics and state.
    """

    __metaclass__ = ABCMeta

    _dynSystem = None
    _effectorName = ''

    _stateEffectors = None
    _dynEffectors = None
    _stateEffectorParent = None

    _stateNames = ()

    _mass = 0.0            # [kg] mass of the effector
    _I_CoM = None          # [kg m^2] Inertia of the effector relative to its center of mass in the Body frame
    _I_B = None            # [kg m^2] Inertia of the effector relative to reference point B in the Body frame
    _CoM = None            # [m] Center of mass of the effector relative to reference point B in the Body frame
    _CoM_tilde = None      # Tilde matrix
    _mCoM = None           # [kg m] Moment of first order (mass * CoM)

    def __init__(self, dynSystem, name, stateEffectorParent = None):
        self._dynSystem = dynSystem
        self._effectorName = name
        self._stateEffectors = list()
        self._dynEffectors = list()
        self._stateEffectorParent = stateEffectorParent
        self._mass = 1.0
        self._I_CoM = np.eye(3)
        self.setCoMOffset(np.zeros(3))
        self.setInertia(np.eye(3))
        return

    def getEffectorName(self):
        return self._effectorName

    def addStateEffector(self, stateEffector):
        """
        State effectors can have more state effectors; eg, a hub might have wheels and slosh particles.
        :param stateEffector:
        :return:
        """
        self._stateEffectors.append(stateEffector)
        #stateEffector.setStateEffectorParent(self)
        return

    def getStateEffectors(self):
        return self._stateEffectors

    def addDynEffector(self, dynEffector):
        """
        Dynamics effectors, such as gravity or thrusters can be added with this method.
        :param dynEffector:
        :return:
        """
        self._dynEffectors.append(dynEffector)
        return

    def getStateEffectorParent(self):
        """
        Each effector was supposed to have a parent. This is no longer used, but I keep it here just in case.
        :return:
        """
        return self._stateEffectorParent

    def setStateEffectorParent(self, stateEffectorParent):
        self._stateEffectorParent = stateEffectorParent
        return

    def getStateNames(self):
        return self._stateNames

    def setMass(self, mass):
        """
        Sets mass of the effector.
        :param mass:
        :return:
        """
        self._mass = mass
        self._I_B = self._I_CoM + self._mass * self._CoM_tilde.dot(self._CoM_tilde.T)
        self._mCoM = self._mass * self._CoM
        return

    def setInertia(self, inertia):
        """
        Sets the inertia relative to its center of mass in the Body frame.
        :param inertia:
        :return:
        """
        self._I_CoM = inertia
        self._I_B = self._I_CoM + self._mass * self._CoM_tilde.dot(self._CoM_tilde.T)
        return

    def setCoMOffset(self, CoM_offset):
        """
        Sets the center of mass offset relative to the reference point in the Body frame.
        :param CoM_offset: [1-dim numpy array] Position of the center of mass of the effector relative to the reference B.
        :return:
        """
        self._CoM = CoM_offset
        self._mCoM = self._mass * self._CoM
        self._CoM_tilde = attitudeKinematics.getSkewSymmetrixMatrix(CoM_offset)
        self._I_B = self._I_CoM + self._mass * self._CoM_tilde.dot(self._CoM_tilde.T)
        return

    def getMass(self):
        return self._mass

    def getInertiaRelativeToReferenceB(self):
        return self._I_B

    def getInertiaRelativeToCoM(self):
        """
        Returns the inertia relative to its center of mass.
        :return:
        """
        return self._I_CoM

    def getCoM(self):
        return self._CoM

    #----------------------------------Abstract Methods-------------------------------#
    # Real state effectors should implement the following methods.

    @abstractmethod
    def registerStates(self):
        """
        Use this method to register states using the state manager.
        :return: void
        """
        pass

    @abstractmethod
    def startSimulation(self):
        """
        If there's anything to start after the initial conditions have been set and before starting the simulation, do it here.
        :return:
        """
        pass

    @abstractmethod
    def computeContributions(self):
        """
        It is assumed that each effector is only connected to one another, called parent.
        This method is to be used to implement the effect that this effector produces on the equations of motion of its parent.
        :return: [tuple] Contributions to the equations of motion of its parent.
        """
        pass

    @abstractmethod
    def computeStateDerivatives(self, t):
        """
        Computes the equations of motion of the effector.
        :param t:
        :return: void
        """
        pass

    @abstractmethod
    def computeEnergy(self):
        """
        Computes the energy contribution of the effector
        :return: [double] Energy contribution
        """
        pass

    @abstractmethod
    def computeMechanicalPower(self):
        """
        Computes mechanical power given to the system (positive) or taken from it (negative).
        :return:
        """
        pass

    @abstractmethod
    def computeAngularMomentum(self):
        """
        Computes the angular momentum contribution of the effector.
        Beware that the angular momentum is computed about the reference point B. Thus, it might not be conserved
        if B is not the center of mass.
        :return: [double] Angular momentum contribution.
        """
        pass

#---------------------------------------------------------------------------------------------------------------------#

class spacecraftHub(stateEffector):
    """
    Implements a rigid body hub effector.
    """

    #_I_Bc = None            # [kg-m^2] Inertia of the hub relative to the hub's center of mass Bc.
    #_I_B = None             # [kg-m^2] Inertia of the hub relative to the reference point B.
    #_r_BcB_B = None         # [m] Position of the hub's CoM relative to the reference B.
    #_r_BcB_B_tilde = None   # Tilde matrix

    #_mr_BcB_B = None        # [kg m] Momentum of order 1 (_m_hub * _r_BcB_B)

    # RHS, LHS and mass property contributions
    _r_ddot_contr = None
    _w_dot_contr = None
    _m_contr = 0.0
    _m_dot_contr = 0.0
    _I_contr = None
    _I_dot_contr = None
    _com_contr = None
    _com_dot_contr = None
    _A_contr = None
    _B_contr = None
    _C_contr = None
    _D_contr = None

    _statePositionName = ''
    _stateVelocityName = ''
    _stateAttitudeName = ''
    _stateAngularVelocityName = ''

    def __init__(self, dynSystem, name):
        super(spacecraftHub, self).__init__(dynSystem, name)
        #self._m_hub = 100.0
        #self._I_Bc = np.diag([50, 50, 50])
        #self._r_BcB_B = np.zeros(3)
        #self._r_BcB_B_tilde = np.zeros((3,3))
        #self._mr_BcB_B = self._m_hub * self._r_BcB_B
        #self._I_B = self._I_Bc + self._m_hub * self._r_BcB_B_tilde.dot(self._r_BcB_B_tilde.T)
        #self._mr_BcB_B = self._mass * self._CoM

        self._stateNames = ('R_BN', 'R_BN_dot', 'sigma_BN', 'omega_BN')

        self._statePositionName = self._effectorName + '_'+ self._stateNames[0]
        self._stateVelocityName = self._effectorName + '_'+ self._stateNames[1]
        self._stateAttitudeName = self._effectorName + '_' + self._stateNames[2]
        self._stateAngularVelocityName = self._effectorName + '_' + self._stateNames[3]
        self._r_ddot_contr = np.zeros(3)
        self._w_dot_contr = np.zeros(3)
        self._m_contr = 0.0
        self._m_dot_contr = 0.0
        self._I_contr = np.zeros((3,3))
        self._I_dot_contr = np.zeros((3,3))
        self._com_contr = np.zeros(3)
        self._com_dot_contr = np.zeros(3)
        self._A_contr = np.zeros((3,3))
        self._B_contr = np.zeros((3,3))
        self._C_contr = np.zeros((3,3))
        self._D_contr = np.zeros((3,3))

        return

    @classmethod
    def getSpacecraftHub(cls, dynSystem, name):
        """
        Use this factory method to create a hub.
        :param dynSystem: [dynamicalSystem] Dynamical system which the hub is attached to.
        :param name: [string] Name for the hub.
        :return:
        """
        hub = spacecraftHub(dynSystem, name)
        hub.registerStates()

        return hub


    #------------------State Name getters------------------#
    def getStatePositionName(self):
        return self._statePositionName

    def getStateVelocityName(self):
        return self._stateVelocityName

    def getStateAttitudeName(self):
        return self._stateAttitudeName

    def getStateAngularVelocityName(self):
        return self._stateAngularVelocityName
    #------------------------------------------------------#

    #----------------StateEffector interface---------------#
    def registerStates(self):
        """
        This method register the states. Don't use it. Supposed to be private.
        :return:
        """
        stateMan = self._dynSystem.getStateManager()

        if not stateMan.registerState(self._statePositionName, 3):
            return False
        elif not stateMan.registerState(self._stateVelocityName, 3):
            stateMan.unregisterState(self._statePositionName)
            return False
        elif not stateMan.registerState(self._stateAttitudeName, 3, attitudeKinematics.switchMRPrepresentation):
            stateMan.unregisterState(self._statePositionName)
            stateMan.unregisterState(self._stateVelocityName)
            return False
        elif not stateMan.registerState(self._stateAngularVelocityName, 3):
            stateMan.unregisterState(self._statePositionName)
            stateMan.unregisterState(self._stateVelocityName)
            stateMan.unregisterState(self._stateAttitudeName)
            return False
        else:
            return True

    def startSimulation(self):
        return

    def computeStateDerivatives(self, t):

        stateMan = self._dynSystem.getStateManager()

        #r_BN_N = stateMan.getStates(self._statePositionName)
        v_BN_N = stateMan.getStates(self._stateVelocityName)
        sigma_BN = stateMan.getStates(self._stateAttitudeName)
        w_BN = stateMan.getStates(self._stateAngularVelocityName)

        f_r_BN_dot = np.zeros(3)
        f_w_dot = np.zeros(3)
        m = 0.0
        m_prime = 0.0
        I = np.zeros((3,3))
        I_prime = np.zeros((3,3))
        com = np.zeros(3)
        com_prime = np.zeros(3)
        A = np.zeros((3,3))
        B = np.zeros((3,3))
        C = np.zeros((3,3))
        D = np.zeros((3,3))

        for effector in self._stateEffectors:
            (A_contr, B_contr, C_contr, D_contr,
            f_r_BN_dot_contr, f_w_dot_contr,
            m_contr, m_prime_contr, I_contr, I_prime_contr, com_contr, com_prime_contr) = effector.computeContributions()
            #(m_contr, m_prime_contr, I_contr, I_prime_contr, com_contr, com_prime_contr) = effector.computeMassProperties()
            m += m_contr
            m_prime += m_prime_contr
            I += I_contr
            I_prime += I_prime_contr
            com += com_contr
            com_prime += com_prime_contr

            #(f_r_BN_dot_contr, f_w_dot_contr) = effector.computeRHS()
            f_r_BN_dot += f_r_BN_dot_contr
            f_w_dot += f_w_dot_contr

            #(A_contr, B_contr, C_contr, D_contr) = effector.computeLHS()
            A += A_contr
            B += B_contr
            C += C_contr
            D += D_contr

        (A_contr, B_contr, C_contr, D_contr,
         f_r_BN_dot_contr, f_w_dot_contr,
         m_contr, m_prime_contr, I_contr, I_prime_contr, com_contr, com_prime_contr) = self.computeContributions()

        #(m_contr, m_prime_contr, I_contr, I_prime_contr, com_contr, com_prime_contr) = self.computeMassProperties()
        m += m_contr
        m_prime += m_prime_contr
        I += I_contr
        I_prime += I_prime_contr
        com += com_contr
        com_prime += com_prime_contr

        #(f_r_BN_dot_contr, f_w_dot_contr) = self.computeRHS()
        f_r_BN_dot += f_r_BN_dot_contr
        f_w_dot += f_w_dot_contr

        #(A_contr, B_contr, C_contr, D_contr) = self.computeLHS()
        A += A_contr
        B += B_contr
        C += C_contr
        D += D_contr

        BN = attitudeKinematics.mrp2dcm(sigma_BN)

        A_inv = np.linalg.inv(A)

        w_BN_dot = np.linalg.inv(D - C.dot(A_inv).dot(B)).dot(f_w_dot - C.dot(A_inv).dot(f_r_BN_dot))
        r_BN_N_ddot = BN.T.dot(A_inv).dot(f_r_BN_dot - B.dot(w_BN_dot))

        stateMan.setStateDerivatives(self._statePositionName, v_BN_N)
        stateMan.setStateDerivatives(self._stateAttitudeName, attitudeKinematics.angVel2mrpRate(sigma_BN, w_BN))
        stateMan.setStateDerivatives(self._stateVelocityName, r_BN_N_ddot)
        stateMan.setStateDerivatives(self._stateAngularVelocityName, w_BN_dot)

        return

    def computeContributions(self):

        stateMan = self._dynSystem.getStateManager()
        w_BN = stateMan.getStates(self._stateAngularVelocityName)
        w_BN_tilde = attitudeKinematics.getSkewSymmetrixMatrix(w_BN)

        # RHS contributions
        f_r_BN_dot_contr = -w_BN_tilde.dot(w_BN_tilde).dot(self._mCoM)
        f_w_BN_dot_contr = - w_BN_tilde.dot(self._I_B).dot(w_BN)

        # LHS contribution
        A_contr = self._mass * np.eye(3)
        B_contr = -attitudeKinematics.getSkewSymmetrixMatrix(self._mCoM)
        C_contr = attitudeKinematics.getSkewSymmetrixMatrix(self._mCoM)
        D_contr = self._I_B

        # Mass propertiy contributions
        I_contr = self._I_B
        I_prime_contr = np.zeros((3,3))
        m_contr = self._mass
        m_prime_contr = 0.0
        com_contr = self._CoM
        com_prime_contr = np.zeros(3)

        return (A_contr, B_contr, C_contr, D_contr, f_r_BN_dot_contr, f_w_BN_dot_contr, m_contr, m_prime_contr, I_contr, I_prime_contr, com_contr, com_prime_contr)

    def computeEnergy(self):
        stateMan = self._dynSystem.getStateManager()

        v_BN_N = stateMan.getStates(self._stateVelocityName)
        w_BN = stateMan.getStates(self._stateAngularVelocityName)
        sigma_BN = stateMan.getStates(self._stateAttitudeName)
        BN = attitudeKinematics.mrp2dcm(sigma_BN)

        E_contr = 0.5*self._mass * np.inner(v_BN_N, v_BN_N) + 0.5 * np.inner(w_BN, self._I_B.dot(w_BN)) \
                  + self._mass * np.inner(BN.dot(v_BN_N), np.cross(w_BN, self._CoM))
        return E_contr

    def computeAngularMomentum(self):
        stateMan = self._dynSystem.getStateManager()

        w_BN = stateMan.getStates(self._stateAngularVelocityName)
        sigma_BN = stateMan.getStates(self._stateAttitudeName)
        BN = attitudeKinematics.mrp2dcm(sigma_BN)

        H_contr = BN.T.dot(self._I_B.dot(w_BN))
        return H_contr

    def computeMechanicalPower(self):
        return 0.0

#---------------------------------------------------------------------------------------------------------------------#


class vscmg(stateEffector):
    """
    Implements a VSCMG (Variable Speed Control Moment Gyro) state effector.
    """

    _Ig = None                      # [kg m^2] Gimbal inertia relative to its center of mass
    _Iw = None                      # [kg m^2] Wheel inertia relative to its center of mass
    _BG0 = None                     # Rotation matrix from G frame to B frame at initial time
    #_m_vscmg = 0.0                  # [kg] Total VSCMG mass
    #_r_OiB_B = None                 # [m] Position of the CoM of the VSCMG relative to the reference point B
    #_r_OiB_B_tilde = None
    #_mr_OiB_B = None                # [kg m] Momentum of order 1 (_m_vscmg * _r_OiB_B)

    _BG = None                      # Instantaneous DCM from G to B

    _ug = 0.0                       # [Nm] Gimbal torque input
    _us = 0.0                       # [Nm] RW torque input

    _ugMax = 1000.0                 # [Nm] Maximum gimbal torque
    _usMax = 1000.0                 # [Nm] Maximum RW torque

    _nominalSpeedRPM = 0.0          # [rpm] Nominal speed of the wheels

    _w_BN_B_name = ''
    _sigma_BN_name = ''
    _v_BN_N_name = ''

    _stateGimbalAngleName = ''
    _stateGimbalRateName = ''
    _stateRWrateName = ''

    def __init__(self, dynSystem, name, w_BN_B_name, sigma_BN_name, v_BN_N_name):
        super(vscmg, self).__init__(dynSystem, name)
        self._Ig = np.eye(3)
        self._Iw = np.eye(3)
        self._BG0 = np.eye(3)
        #self._m_vscmg = 10.0
        #self._r_OiB_B = np.zeros(3)
        #self._r_OiB_B_tilde = attitudeKinematics.getSkewSymmetrixMatrix(self._r_OiB_B)
        #self._mr_OiB_B = self._mass * self._CoM
        self._BG = self._BG0

        self.setInertia(self._BG.dot(self._Ig + self._Iw).dot(self._BG.T))

        self._ug = 0.0
        self._us = 0.0
        self._ugMax = 1000.0
        self._usMax = 1000.0

        self._nominalSpeedRPM = 1000.0

        self._w_BN_B_name = w_BN_B_name
        self._sigma_BN_name = sigma_BN_name
        self._v_BN_N_name = v_BN_N_name

        self._stateNames = ('gamma', 'gamma_dot', 'Omega')

        self._stateGimbalAngleName = self._effectorName + '_' + self._stateNames[0]
        self._stateGimbalRateName = self._effectorName + '_' + self._stateNames[1]
        self._stateRWrateName = self._effectorName + '_' + self._stateNames[2]
        return

    @classmethod
    def getVSCMG(cls, dynSystem, name, m, r_OiB_B, Igs, Igt, Igg, Iws, Iwt, BG0, w_BN_B_name, sigma_BN_name, v_BN_N_name, nominal_speed_rpm, ug = 0.0, us = 0.0):
        """
        Use this factory method to create a VSCMG.
        :param dynSystem: [dynamicalSystem] System which the VSCMG is attached to.
        :param name:  [string] Unique name to identify the VSCMG.
        :param m: [double] mass.
        :param r_OiB_B: [1-dimensional numpy array] Offset of the VSCMG CoM relative to reference point B.
        :param Igs: [double] Inertia of the gimbal in gs direction.
        :param Igt: [double] Inertia of the gimbal in gt direction.
        :param Igg: [double] Inertia of the gimbal in gg direction.
        :param Iws: [double] Main inertia of the wheel.
        :param Iwt: [double] Transverse inertia of the wheel.
        :param BG0: [2-dimensional numpy array] Rotation matrix BG(t_0)
        :param w_BN_B_name: [string] Name that identifies W_BN_B state.
        :param sigma_BN_name: [string] Name that identifies sigma_BN state.
        :param v_BN_N_name: [string] Name that identifies v_BN_N state.
        :param nominal_speed_rpm: [double] Nominal speed of the wheel in rpm.
        :param ug: Optional initial gimbal torque.
        :param us: Optional initial wheel torque.
        :return: [vscmg]
        """
        vscmgObj = vscmg(dynSystem, name, w_BN_B_name, sigma_BN_name, v_BN_N_name)
        vscmgObj.registerStates()
        vscmgObj._mass = m
        vscmgObj.setCoMOffset(r_OiB_B)
        vscmgObj.setGimbalInertia(Igs, Igt, Igg)
        vscmgObj.setWheelInertia(Iws, Iwt)
        vscmgObj._BG0 = BG0
        vscmgObj._BG = BG0
        vscmgObj.setInertia(BG0.dot(vscmgObj._Ig + vscmgObj._Iw).dot(BG0.T))
        vscmgObj._ug = ug
        vscmgObj._us = us
        vscmgObj._nominalSpeedRPM = nominal_speed_rpm
        return vscmgObj

    def setMaxGimbalTorque(self, ugMax):
        """
        Sets the maximum gimbal torque the VSCMG can apply.
        :param ugMax: [double] torque in Nm.
        :return:
        """
        self._ugMax = ugMax
        return

    def setMaxWheelTorque(self, usMax):
        """
        Sets the maximum wheel torque the VSCMG can apply
        :param usMax: [double] torque in Nm.
        :return:
        """
        self._usMax = usMax
        return

    def setW_BNname(self, w_BN_B_name):
        self._w_BN_B_name = w_BN_B_name
        return

    def setWheelInertia(self, Iws, Iwt):
        self._Iw = np.diag(np.array([Iws, Iwt, Iwt]))
        return

    def setGimbalInertia(self, Igs, Igt, Igg):
        self._Ig = np.diag(np.array([Igs, Igt, Igg]))
        return

    def getGimbalInertia(self):
        return self._Ig

    def getWheelInertia(self):
        return self._Iw

    def getBG0matrix(self):
        return self._BG0

    def getBGmatrix(self):
        stateMan = self._dynSystem.getStateManager()
        gamma = stateMan.getStates(self._stateGimbalAngleName)
        GG0 = coordinateTransformations.ROT3(gamma)
        self._BG = self._BG0.dot(GG0.T)
        return self._BG

    def getNominalRWangularMomentum(self):
        """
        This is a stupid method that should return the nominal angular mometum based on nominal velocity.
        :return:
        """
        nominal_Omega = self._nominalSpeedRPM * 2*np.pi/60
        return self._Iw[0,0] * nominal_Omega


    #-----------------VSCMG interface---------------------#
    # The following methods simulate the software
    # interface of the VSCMG.

    def getOmegaEstimate(self):
        """
        Estimator that gives the value of the RW velocity.
        It's currently simulated as an ideal estimator with infinite frequency
        :return:
        """
        stateMan = self._dynSystem.getStateManager()
        Omega = stateMan.getStates(self._stateRWrateName)
        return Omega

    def getOmegaDotEstimate(self):
        """
        Estimator that gives the value of the RW acceleration.
        It's currently simulated as an ideal estimator with infinite frequency
        :return:
        """
        stateMan = self._dynSystem.getStateManager()
        Omega_dot = stateMan.getStateDerivatives(self._stateRWrateName)
        return Omega_dot

    def getGammaEstimate(self):
        """
        Estimator that gives the value of the Gimbal angle.
        It's currently simulated as an ideal estimator with infinite frequency
        :return:
        """
        stateMan = self._dynSystem.getStateManager()
        gamma = stateMan.getStates(self._stateGimbalAngleName)
        return gamma

    def getGammaDotEstimate(self):
        """
        Estimator that gives the value of the Gimbal rate.
        It's currently simulated as an ideal estimator with infinite frequency
        :return:
        """
        stateMan = self._dynSystem.getStateManager()
        gamma_dot = stateMan.getStates(self._stateGimbalRateName)
        return gamma_dot

    def getGammaDDotEstimate(self):
        """
        Estimator that gives the value of the Gimbal acceleration.
        It's currently simulated as an ideal estimator with infinite frequency
        :return:
        """
        stateMan = self._dynSystem.getStateManager()
        gamma_ddot = stateMan.getStateDerivatives(self._stateGimbalRateName)
        return gamma_ddot

    def setGimbalTorqueCommand(self, ug):
        """
        This is part of the command interface of the VSCMG.
        You can set the gimbal control torque using this method.
        :param ug: [double] Torque in Nm.
        :return:
        """
        if np.abs(ug) <= self._ugMax:
            self._ug = ug
        else:
            self._ug = np.sign(ug) * self._ugMax
        return

    def setWheelTorqueCommand(self, us):
        """
        This is part of the command interface of the VSCMG.
        You can set the wheel control torque using this method.
        :param us: [double] Torque in Nm.
        :return:
        """
        if np.abs(us) <= self._usMax:
            self._us = us
        else:
            self._us = np.sign(us) * self._usMax
        return
    #------------------------------------------------------#

    #----------------StateEffector interface---------------#
    def registerStates(self):

        stateMan = self._dynSystem.getStateManager()

        if not stateMan.registerState(self._stateGimbalAngleName, 1, attitudeKinematics.angleTrimming):
            return False
        elif not stateMan.registerState(self._stateGimbalRateName,1):
            stateMan.unregisterState(self._stateGimbalAngleName)
            return False
        elif not stateMan.registerState(self._stateRWrateName, 1):
            stateMan.unregisterState(self._stateGimbalAngleName)
            stateMan.unregisterState(self._stateGimbalRateName)
            return False
        else:
            return True

    def startSimulation(self):
        BG = self.getBGmatrix() # BG changes after setting gamma_0
        self.setInertia(BG.dot(self._Ig + self._Iw).dot(BG.T))
        return

    def computeStateDerivatives(self, t):

        stateMan = self._dynSystem.getStateManager()

        w_BN = stateMan.getStates(self._w_BN_B_name)
        w_BN_dot = stateMan.getStateDerivatives(self._w_BN_B_name)
        gamma_dot = stateMan.getStates(self._stateGimbalRateName)
        Omega = stateMan.getStates(self._stateRWrateName)

        Iws = self._Iw[0,0]
        Iwt = self._Iw[1,1]
        Js = self._Ig[0,0] + Iws
        Jt = self._Ig[1,1] + Iwt
        Jg = self._Ig[2,2] + Iwt

        BG = self.getBGmatrix()

        gs_hat = BG[:,0]
        gt_hat = BG[:,1]
        gg_hat = BG[:,2]

        ws = np.inner(gs_hat, w_BN)
        wt = np.inner(gt_hat, w_BN)

        Omega_dot = self._us/Iws - np.inner(gs_hat, w_BN_dot) - gamma_dot * wt

        gamma_ddot = self._ug/Jg - np.inner(gg_hat, w_BN_dot) - (Jt - Js)/Jg * ws * wt + Iws/Jg * wt * Omega

        stateMan.setStateDerivatives(self._stateGimbalAngleName, gamma_dot)
        stateMan.setStateDerivatives(self._stateGimbalRateName, gamma_ddot)
        stateMan.setStateDerivatives(self._stateRWrateName, Omega_dot)

        return

    def computeContributions(self):
        stateMan = self._dynSystem.getStateManager()

        w_BN = stateMan.getStates(self._w_BN_B_name)
        gamma_dot = stateMan.getStates(self._stateGimbalRateName)
        Omega = stateMan.getStates(self._stateRWrateName)

        Iws = self._Iw[0,0]
        Iwt = self._Iw[1,1]
        Igs = self._Ig[0,0]
        Js = Igs + Iws
        Jt = self._Ig[1,1] + Iwt
        Jg = self._Ig[2,2] + Iwt

        BG = self.getBGmatrix()

        gs_hat = BG[:,0]
        gt_hat = BG[:,1]
        gg_hat = BG[:,2]
        gs_outer = np.outer(gs_hat, gs_hat)
        gt_outer = np.outer(gt_hat, gt_hat)

        ws = np.inner(gs_hat, w_BN)
        wt = np.inner(gt_hat, w_BN)
        wg = np.inner(gg_hat, w_BN)

        w_BN_tilde = attitudeKinematics.getSkewSymmetrixMatrix(w_BN)

        self.setInertia(self._BG.dot(self._Ig + self._Iw).dot(self._BG.T))

        mr_OiB_B_tilde = self._mass * self._CoM_tilde

        # RHS contributions
        f_r_BN_dot_contr = -w_BN_tilde.dot(w_BN_tilde).dot(self._mCoM)
        f_w_BN_dot_contr = -w_BN_tilde.dot(self._I_B).dot(w_BN) \
                           -gs_hat * ((Js - Jt + Jg) * wt * gamma_dot - Iws * wt * gamma_dot + self._us) \
                           -gt_hat * (((Js - Jt - Jg) * ws + Iws * Omega) * gamma_dot + Iws * wg * Omega) \
                           -gg_hat * (self._ug - (Jt - Js) * ws * wt)

        # Omega_dot = stateMan.getStateDerivatives(self._stateRWrateName)
        # gamma_ddot = stateMan.getStateDerivatives(self._stateGimbalRateName)
        # f_w_BN_dot_contr = -w_BN_tilde.dot(self._I_B).dot(w_BN) \
        #                    -gs_hat * ((Js - Jt + Jg) * wt * gamma_dot + Iws * Omega_dot) \
        #                    -gt_hat * (((Js - Jt - Jg) * ws + Iws * Omega) * gamma_dot + Iws * wg * Omega) \
        #                    -gg_hat * (Jg * gamma_ddot - Iws * Omega * wt)

        # LHS contributions
        A_contr = self._mass * np.eye(3)
        B_contr = -mr_OiB_B_tilde
        C_contr = mr_OiB_B_tilde
        D_contr = -self._mass * self._CoM_tilde.dot(self._CoM_tilde) + Igs * gs_outer + Jt * gt_outer

        # gg_outer = np.outer(gg_hat, gg_hat)
        # D_contr = -self._mass * self._CoM_tilde.dot(self._CoM_tilde) + Js * gs_outer + Jt * gt_outer + Jg * gg_outer


        # Mass properties
        m_contr = self._mass
        m_prime_contr = 0.0
        I_contr = self._I_B #-self._mass * self._CoM_tilde.dot(self._CoM_tilde) + self._BG.dot(self._Ig + self._Iw).dot(self._BG.T)
        I_prime_contr = np.zeros((3,3))
        com_contr = self._CoM
        com_prime_contr = np.zeros(3)

        return (A_contr, B_contr, C_contr, D_contr, f_r_BN_dot_contr, f_w_BN_dot_contr, m_contr, m_prime_contr, I_contr, I_prime_contr, com_contr, com_prime_contr)

    def computeEnergy(self):

        stateMan = self._dynSystem.getStateManager()

        gamma = stateMan.getStates(self._stateGimbalAngleName)
        gamma_dot = stateMan.getStates(self._stateGimbalRateName)
        Omega = stateMan.getStates(self._stateRWrateName)
        w_BN = stateMan.getStates(self._w_BN_B_name)
        sigma_BN = stateMan.getStates(self._sigma_BN_name)
        v_BN_N = stateMan.getStates(self._v_BN_N_name)

        BN = attitudeKinematics.mrp2dcm(sigma_BN)

        # GG0 = coordinateTransformations.ROT3(gamma)
        #
        # BG = self._BG0.dot(GG0.T)

        BG = self.getBGmatrix()

        Igs = self._Ig[0,0]
        Igt = self._Ig[1,1]
        Igg = self._Ig[2,2]

        Iws = self._Iw[0,0]
        Iwt = self._Iw[1,1]

        Jt = Igt + Iwt
        Jg = Igg + Iwt

        gs_hat = BG[:,0]
        gt_hat = BG[:,1]
        gg_hat = BG[:,2]

        ws = np.inner(w_BN, gs_hat)
        wt = np.inner(w_BN, gt_hat)
        wg = np.inner(w_BN, gg_hat)

        I_offset = -self._mass * self._CoM_tilde.dot(self._CoM_tilde)

        E_contr = 0.5 * self._mass * np.inner(v_BN_N, v_BN_N) \
                  + self._mass * np.inner(BN.dot(v_BN_N), np.cross(w_BN, self._CoM)) \
                  + 0.5*np.inner(w_BN, I_offset.dot(w_BN)) \
                  + 0.5 * (Igs * ws**2 + Iws*(ws+Omega)**2 + Jt * wt**2 + Jg*(wg+gamma_dot)**2)

        return E_contr

    def computeAngularMomentum(self):
        stateMan = self._dynSystem.getStateManager()

        gamma = stateMan.getStates(self._stateGimbalAngleName)
        gamma_dot = stateMan.getStates(self._stateGimbalRateName)
        Omega = stateMan.getStates(self._stateRWrateName)
        w_BN = stateMan.getStates(self._w_BN_B_name)
        sigma_BN = stateMan.getStates(self._sigma_BN_name)

        BN = attitudeKinematics.mrp2dcm(sigma_BN)

        # GG0 = coordinateTransformations.ROT3(gamma)
        #
        # BG = self._BG0.dot(GG0.T)

        BG = self.getBGmatrix()

        Igs = self._Ig[0,0]
        Igt = self._Ig[1,1]
        Igg = self._Ig[2,2]

        Iws = self._Iw[0,0]
        Iwt = self._Iw[1,1]

        Js = Igs + Iws
        Jt = Igt + Iwt
        Jg = Igg + Iwt

        gs_hat = BG[:,0]
        gt_hat = BG[:,1]
        gg_hat = BG[:,2]

        ws = np.inner(w_BN, gs_hat)
        wt = np.inner(w_BN, gt_hat)
        wg = np.inner(w_BN, gg_hat)

        I_offset = -self._mass * self._CoM_tilde.dot(self._CoM_tilde)

        H_contr = BN.T.dot(I_offset.dot(w_BN) + (Js * ws + Iws * Omega) * gs_hat + Jt * wt * gt_hat + Jg * (wg + gamma_dot) * gg_hat)
        return H_contr

    def computeMechanicalPower(self):
        stateMan = self._dynSystem.getStateManager()

        gamma_dot = stateMan.getStates(self._stateGimbalRateName)
        Omega = stateMan.getStates(self._stateRWrateName)

        P = self._ug*gamma_dot + Omega * self._us
        return P


#---------------------------------------------------------------------------------------------------------------------#


class reactionWheel(stateEffector):
    """
    Implements a Reaction Wheel (RW) state effector.
    """

    _Iw = None                  # [kg m^2] Inertia of the wheel relative to its center of mass
    _BW = None                  # Rotation matrix from the wheel frame to the body frame
    # _m_rw = 0.0                 # [kg] Mass of the RW
    # _r_OiB_B = None             # [m] RW CoM offset relative to reference point B
    # _r_OiB_B_tilde = None
    #_mr_OiB_B = None            # [kg m] Momentum of order 1 (_m_rw * _r_OiB_B)

    _us = 0.0                   # [Nm] Input torque
    _usMax = 1000.0             # [Nm] Maximum torque

    _nominalSpeedRPM = 0.0

    _w_BN_B_name = ''
    _sigma_BN_name = ''
    _v_BN_N_name = ''

    _stateRWrateName = ''

    def __init__(self, dynSystem, name, w_BN_B_name, sigma_BN_name, v_BN_N_name):
        super(reactionWheel, self).__init__(dynSystem, name)
        self._Iw = np.eye(3)
        self._BW = np.eye(3)
        # self._mass = 10.0
        # self._CoM = np.zeros(3)
        # self._CoM_tilde = attitudeKinematics.getSkewSymmetrixMatrix(self._CoM)
        # self._mr_OiB_B = self._mass * self._CoM
        self.setInertia(self._BW.dot(self._Iw).dot(self._BW.T))

        self._us = 0.0
        self._usMax = 1000.0

        self._nominalSpeedRPM = 1000.0

        self._w_BN_B_name = w_BN_B_name
        self._sigma_BN_name = sigma_BN_name
        self._v_BN_N_name = v_BN_N_name

        self._stateNames = ('Omega',)

        self._stateRWrateName = self._effectorName + '_' + self._stateNames[0]
        return

    @classmethod
    def getRW(cls, dynSystem, name, m, r_OiB_B, Iws, Iwt, BW,  w_BN_B_name, sigma_BN_name, v_BN_N_name, nominal_speed_rpm, us = 0.0):
        """
        Use this factory method to create a RW.
        :param dynSystem: [dynamicalSystem] System which the RW is attached to.
        :param name: [string] Unique name to identify the RW.
        :param m: [double] mass.
        :param r_OiB_B: [1-dimensional numpy array] Offset of the RW CoM relative to reference point B.
        :param Iws: [double] Main inertia of the wheel.
        :param Iwt: [double] Transverse inertia of the wheel.
        :param BW: [2-dimensional numpy array] Rotation matrix from the wheel frame to the bodyframe.
        :param w_BN_B_name: [string] Name that identifies W_BN_B state.
        :param sigma_BN_name: [string] Name that identifies sigma_BN state.
        :param v_BN_N_name: [string] Name that identifies v_BN_N state.
        :param nominal_speed_rpm: [double] Nominal speed of the wheel in rpm.
        :param us: Optional initial wheel torque.
        :return: [rw]
        """
        rwObj = reactionWheel(dynSystem, name, w_BN_B_name, sigma_BN_name, v_BN_N_name)
        rwObj.registerStates()
        rwObj._mass = m
        rwObj.setCoMOffset(r_OiB_B)
        rwObj.setWheelInertia(Iws, Iwt)
        rwObj._BW = BW
        rwObj.setInertia(BW.dot(rwObj._Iw).dot(BW.T))
        rwObj._nominalSpeedRPM = nominal_speed_rpm
        rwObj._us = us
        return rwObj

    def setMaxWheelTorque(self, usMax):
        """
        Sets the maximum wheel torque the VSCMG can apply
        :param usMax: [double] torque in Nm.
        :return:
        """
        self._usMax = usMax
        return

    def setW_BNname(self, w_BN_B_name):
        self._w_BN_B_name = w_BN_B_name
        return

    def setWheelInertia(self, Iws, Iwt):
        self._Iw = np.diag(np.array([Iws, Iwt, Iwt]))
        return

    def getWheelInertia(self):
        return self._Iw

    def getBWmatrix(self):
        return self._BW

    def registerStates(self):
        stateMan = self._dynSystem.getStateManager()

        if not stateMan.registerState(self._stateRWrateName, 1):
            return False
        else:
            return True

    def getNominalRWangularMomentum(self):
        """
        This is a stupid method that should return the nominal angular mometum based on nominal velocity.
        :return:
        """
        nominal_Omega = self._nominalSpeedRPM * 2*np.pi/60
        return self._Iw[0,0] * nominal_Omega

    #-----------------RW interface---------------------#
    # The following methods simulate the software
    # interface of the RW.

    def getOmegaEstimate(self):
        """
        Estimator that gives the value of the RW velocity.
        It's currently simulated as an ideal estimator with infinite frequency
        :return:
        """
        stateMan = self._dynSystem.getStateManager()
        Omega = stateMan.getStates(self._stateRWrateName)
        return Omega

    def getOmegaDotEstimate(self):
        """
        Estimator that gives the value of the RW acceleration.
        It's currently simulated as an ideal estimator with infinite frequency
        :return:
        """
        stateMan = self._dynSystem.getStateManager()
        Omega_dot = stateMan.getStateDerivatives(self._stateRWrateName)
        return Omega_dot

    def setWheelTorqueCommand(self, us):
        """
        This is part of the command interface of the VSCMG.
        You can set the wheel control torque using this method.
        :param us: [double] Torque in Nm.
        :return:
        """
        if np.abs(us) <= self._usMax:
            self._us = us
        else:
            self._us = np.sign(us) * self._usMax
        return
    #------------------------------------------------------#

    #----------------StateEffector interface---------------#
    def startSimulation(self):
        return

    def computeStateDerivatives(self, t):

        stateMan = self._dynSystem.getStateManager()

        w_BN_dot = stateMan.getStateDerivatives(self._w_BN_B_name)

        Iws = self._Iw[0,0]

        gs_hat = self._BW[:,0]

        Omega_dot = self._us/Iws - np.inner(gs_hat, w_BN_dot)

        stateMan.setStateDerivatives(self._stateRWrateName, Omega_dot)

        return

    def computeContributions(self):

        stateMan = self._dynSystem.getStateManager()

        w_BN = stateMan.getStates(self._w_BN_B_name)
        Omega = stateMan.getStates(self._stateRWrateName)

        Iws = self._Iw[0,0]
        Iwt = self._Iw[1,1]

        gs_hat = self._BW[:,0]
        gt_hat = self._BW[:,1]
        gg_hat = self._BW[:,2]
        gt_outer = np.outer(gt_hat, gt_hat)
        gg_outer = np.outer(gg_hat, gg_hat)

        w_BN_tilde = attitudeKinematics.getSkewSymmetrixMatrix(w_BN)

        mr_OiB_B_tilde = self._mass * self._CoM_tilde

        #I_B = -self._mass * self._CoM_tilde.dot(self._CoM_tilde) + self._BW.dot(self._Iw).dot(self._BW.T)

        # RHS contributions
        f_r_BN_dot_contr = -w_BN_tilde.dot(w_BN_tilde).dot(self._mCoM)
        f_w_BN_dot_contr = -w_BN_tilde.dot(self._I_B).dot(w_BN) \
                           - gs_hat * self._us \
                           - (Iws * Omega) * w_BN_tilde.dot(gs_hat)

        # LHS contributions
        A_contr = self._mass * np.eye(3)
        B_contr = -mr_OiB_B_tilde
        C_contr = mr_OiB_B_tilde
        D_contr = -self._mass * self._CoM_tilde.dot(self._CoM_tilde) + Iwt * gt_outer + Iwt * gg_outer

        # Mass property contributions
        m_contr = self._mass
        m_prime_contr = 0.0
        I_contr = -self._mass * self._CoM_tilde.dot(self._CoM_tilde) + self._BW.dot(self._Iw).dot(self._BW.T)
        I_prime_contr = np.zeros((3,3))
        com_contr = self._CoM
        com_prime_contr = np.zeros(3)

        return (A_contr, B_contr, C_contr, D_contr, f_r_BN_dot_contr, f_w_BN_dot_contr, m_contr, m_prime_contr, I_contr, I_prime_contr, com_contr, com_prime_contr)

    def computeEnergy(self):

        stateMan = self._dynSystem.getStateManager()

        Omega = stateMan.getStates(self._stateRWrateName)
        w_BN = stateMan.getStates(self._w_BN_B_name)
        sigma_BN = stateMan.getStates(self._sigma_BN_name)
        v_BN_N = stateMan.getStates(self._v_BN_N_name)

        BN = attitudeKinematics.mrp2dcm(sigma_BN)

        Iws = self._Iw[0,0]
        Iwt = self._Iw[1,1]

        gs_hat = self._BW[:,0]
        gt_hat = self._BW[:,1]
        gg_hat = self._BW[:,2]

        ws = np.inner(w_BN, gs_hat)
        wt = np.inner(w_BN, gt_hat)
        wg = np.inner(w_BN, gg_hat)

        I_offset = -self._mass * self._CoM_tilde.dot(self._CoM_tilde)

        E_contr = 0.5 * self._mass * np.inner(v_BN_N, v_BN_N) \
                  + self._mass * np.inner(BN.dot(v_BN_N), np.cross(w_BN, self._CoM)) \
                  + 0.5*np.inner(w_BN, I_offset.dot(w_BN)) \
                  + 0.5 * (Iws*(ws+Omega)**2 + Iwt * wt**2 + Iwt*wg**2)

        return E_contr

    def computeAngularMomentum(self):
        stateMan = self._dynSystem.getStateManager()

        Omega = stateMan.getStates(self._stateRWrateName)
        w_BN = stateMan.getStates(self._w_BN_B_name)
        sigma_BN = stateMan.getStates(self._sigma_BN_name)

        BN = attitudeKinematics.mrp2dcm(sigma_BN)

        Iws = self._Iw[0,0]
        Iwt = self._Iw[1,1]

        gs_hat = self._BW[:,0]
        gt_hat = self._BW[:,1]
        gg_hat = self._BW[:,2]

        ws = np.inner(w_BN, gs_hat)
        wt = np.inner(w_BN, gt_hat)
        wg = np.inner(w_BN, gg_hat)

        I_offset = -self._mass * self._CoM_tilde.dot(self._CoM_tilde)

        H_contr = BN.T.dot(I_offset.dot(w_BN) + Iws * (ws + Omega) * gs_hat + Iwt * wt * gt_hat + Iwt * wg * gg_hat)
        return H_contr

    def computeMechanicalPower(self):
        stateMan = self._dynSystem.getStateManager()
        Omega = stateMan.getStates(self._stateRWrateName)

        P = Omega * self._us
        return P


#---------------------------------------------------------------------------------------------------------------------#


class cmg(stateEffector):
    """
    Implements a CMG (Control Moment Gyro) state effector.
    """

    _Ig = None                      # [kg m^2] Gimbal inertia relative to its center of mass
    _BG0 = None                     # Rotation matrix from G frame to B frame at initial time
    # _m_cmg = 0.0                    # [kg] Total CMG mass
    # _r_OiB_B = None                 # [m] Position of the CoM of the CMG relative to the reference point B
    # _r_OiB_B_tilde = None
    # _mr_OiB_B = None                # [kg m] Momentum of order 1 (_m_cmg * _r_OiB_B)

    _BG = None                      # DCM from G to B

    _ug = 0.0                       # [Nm] Gimbal torque
    _ugMax = 1000.0                 # [Nm] Maximum gimbal torque

    _w_BN_B_name = ''
    _sigma_BN_name = ''
    _v_BN_N_name = ''

    _stateGimbalAngleName = ''
    _stateGimbalRateName = ''


    def __init__(self, dynSystem, name, w_BN_B_name, sigma_BN_name, v_BN_N_name):
        super(cmg, self).__init__(dynSystem, name)
        self._Ig = np.eye(3)
        self._BG0 = np.eye(3)
        # self._m_cmg = 10.0
        # self._r_OiB_B = np.zeros(3)
        # self._r_OiB_B_tilde = attitudeKinematics.getSkewSymmetrixMatrix(self._r_OiB_B)
        # self._mr_OiB_B = self._m_cmg * self._r_OiB_B
        self._BG = self._BG0
        self.setInertia(self._BG.dot(self._Ig).dot(self._BG.T))

        self._ug = 0.0
        self._ugMax = 1000.0

        self._w_BN_B_name = w_BN_B_name
        self._sigma_BN_name = sigma_BN_name
        self._v_BN_N_name = v_BN_N_name

        self._stateNames = ('gamma', 'gamma_dot')

        self._stateGimbalAngleName = self._effectorName + '_' + self._stateNames[0]
        self._stateGimbalRateName = self._effectorName + '_' + self._stateNames[1]
        return

    @classmethod
    def getCMG(cls, dynSystem, name, m, r_OiB_B, Igs, Igt, Igg, BG0, w_BN_B_name, sigma_BN_name, v_BN_N_name, ug = 0.0):
        """
        Use this factory method to create a VSCMG.
        :param dynSystem: [dynamicalSystem] System which the VSCMG is attached to.
        :param name:  [string] Unique name to identify the VSCMG.
        :param m: [double] mass.
        :param r_OiB_B: [1-dimensional numpy array] Offset of the VSCMG CoM relative to reference point B.
        :param Igs: [double] Inertia of the gimbal in gs direction.
        :param Igt: [double] Inertia of the gimbal in gt direction.
        :param Igg: [double] Inertia of the gimbal in gg direction.
        :param BG0: [2-dimensional numpy array] Rotation matrix BG(t_0)
        :param w_BN_B_name: [string] Name that identifies W_BN_B state.
        :param sigma_BN_name: [string] Name that identifies sigma_BN state.
        :param v_BN_N_name: [string] Name that identifies v_BN_N state.
        :param ug: Optional initial gimbal torque.
        :return: [vscmg]
        """
        cmgObj = vscmg(dynSystem, name, w_BN_B_name, sigma_BN_name, v_BN_N_name)
        cmgObj.registerStates()
        cmgObj._mass = m
        cmgObj.setCoMOffset(r_OiB_B)
        cmgObj.setGimbalInertia(Igs, Igt, Igg)
        cmgObj._BG0 = BG0
        cmgObj._BG = BG0
        cmgObj._ug = ug
        cmgObj.setInertia(BG0.dot(cmgObj._Ig).dot(BG0.T))
        return cmgObj

    def setMaxGimbalTorque(self, ugMax):
        """
        Sets the maximum gimbal torque the VSCMG can apply.
        :param ugMax: [double] torque in Nm.
        :return:
        """
        self._ugMax = ugMax
        return

    def setW_BNname(self, w_BN_B_name):
        self._w_BN_B_name = w_BN_B_name
        return

    def setGimbalInertia(self, Igs, Igt, Igg):
        self._Ig = np.diag(np.array([Igs, Igt, Igg]))
        return

    def getBGmatrix(self):
        stateMan = self._dynSystem.getStateManager()
        gamma = stateMan.getStates(self._stateGimbalAngleName)
        GG0 = coordinateTransformations.ROT3(gamma)
        self._BG = self._BG0.dot(GG0.T)
        return self._BG

    def registerStates(self):

        stateMan = self._dynSystem.getStateManager()

        if not stateMan.registerState(self._stateGimbalAngleName, 1):
            return False
        elif not stateMan.registerState(self._stateGimbalRateName,1):
            stateMan.unregisterState(self._stateGimbalAngleName)
            return False
        else:
            return True

     #-----------------CMG interface---------------------#
    # The following methods simulate the software
    # interface of the CMG.

    def getGammaEstimate(self):
        """
        Estimator that gives the value of the Gimbal angle.
        It's currently simulated as an ideal estimator with infinite frequency
        :return:
        """
        stateMan = self._dynSystem.getStateManager()
        gamma = stateMan.getStates(self._stateGimbalAngleName)
        return gamma

    def getGammaDotEstimate(self):
        """
        Estimator that gives the value of the Gimbal rate.
        It's currently simulated as an ideal estimator with infinite frequency
        :return:
        """
        stateMan = self._dynSystem.getStateManager()
        gamma_dot = stateMan.getStates(self._stateGimbalRateName)
        return gamma_dot

    def getGammaDDotEstimate(self):
        """
        Estimator that gives the value of the Gimbal acceleration.
        It's currently simulated as an ideal estimator with infinite frequency
        :return:
        """
        stateMan = self._dynSystem.getStateManager()
        gamma_ddot = stateMan.getStateDerivatives(self._stateGimbalRateName)
        return gamma_ddot

    def setGimbalTorqueCommand(self, ug):
        """
        This is part of the command interface of the VSCMG.
        You can set the gimbal control torque using this method.
        :param ug: [double] Torque in Nm.
        :return:
        """
        if np.abs(ug) <= self._ugMax:
            self._ug = ug
        else:
            self._ug = np.sign(ug) * self._ugMax
        return
    #------------------------------------------------------#


    #----------------StateEffector interface---------------#
    def startSimulation(self):
        BG = self.getBGmatrix() # BG changes after setting gamma_0
        self.setInertia(BG.dot(self._Ig).dot(BG.T))
        return

    def computeStateDerivatives(self, t):

        stateMan = self._dynSystem.getStateManager()

        w_BN = stateMan.getStates(self._w_BN_B_name)
        w_BN_dot = stateMan.getStateDerivatives(self._w_BN_B_name)
        gamma = stateMan.getStates(self._stateGimbalAngleName)
        gamma_dot = stateMan.getStates(self._stateGimbalRateName)

        Igs = self._Ig[0,0]
        Igt = self._Ig[1,1]
        Igg = self._Ig[2,2]

        GG0 = coordinateTransformations.ROT3(gamma)

        BG = self._BG0.dot(GG0.T)

        gs_hat = BG[:,0]
        gt_hat = BG[:,1]
        gg_hat = BG[:,2]

        ws = np.inner(gs_hat, w_BN)
        wt = np.inner(gt_hat, w_BN)

        gamma_ddot = self._ug/Igg - np.inner(gg_hat, w_BN_dot) - (Igt - Igs)/Igg * ws * wt

        stateMan.setStateDerivatives(self._stateGimbalAngleName, gamma_dot)
        stateMan.setStateDerivatives(self._stateGimbalRateName, gamma_ddot)

        return

    def computeContributions(self):

        stateMan = self._dynSystem.getStateManager()

        w_BN = stateMan.getStates(self._w_BN_B_name)
        gamma = stateMan.getStates(self._stateGimbalAngleName)
        gamma_dot = stateMan.getStates(self._stateGimbalRateName)

        Igs = self._Ig[0,0]
        Igt = self._Ig[1,1]
        Igg = self._Ig[2,2]

        GG0 = coordinateTransformations.ROT3(gamma)

        self._BG = self._BG0.dot(GG0.T)

        gs_hat = self._BG[:,0]
        gt_hat = self._BG[:,1]
        gg_hat = self._BG[:,2]
        gs_outer = np.outer(gs_hat, gs_hat)
        gt_outer = np.outer(gt_hat, gt_hat)

        ws = np.inner(gs_hat, w_BN)
        wt = np.inner(gt_hat, w_BN)

        self.setInertia(self._BG.dot(self._Ig).dot(self._BG.T))

        w_BN_tilde = attitudeKinematics.getSkewSymmetrixMatrix(w_BN)

        mr_OiB_B_tilde = self._mass * self._CoM_tilde

        #I_B = -self._m_cmg * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde) + self._BG.dot(self._Ig).dot(self._BG.T)

        # RHS contributions
        f_r_BN_dot_contr = -w_BN_tilde.dot(w_BN_tilde).dot(self._mCoM)
        f_w_BN_dot_contr = -w_BN_tilde.dot(self._I_B).dot(w_BN) \
                           -gs_hat * ((Igs - Igt + Igg) * wt * gamma_dot) \
                           -gt_hat * ((Igs - Igt - Igg) * ws * gamma_dot) \
                           -gg_hat * (self._ug - (Igt - Igs) * ws * wt)

        # LHS contributions
        A_contr = self._mass * np.eye(3)
        B_contr = -mr_OiB_B_tilde
        C_contr = mr_OiB_B_tilde
        D_contr = -self._mass * self._CoM_tilde.dot(self._CoM_tilde) + Igs * gs_outer + Igt * gt_outer

        # Mass propery contributions
        m_contr = self._mass
        m_prime_contr = 0.0
        I_contr = self._I_B #-self._mass * self._CoM_tilde.dot(self._CoM_tilde) + self._BG.dot(self._Ig).dot(self._BG.T)
        I_prime_contr = np.zeros((3,3))
        com_contr = self._CoM
        com_prime_contr = np.zeros(3)

        return (A_contr, B_contr, C_contr, D_contr, f_r_BN_dot_contr, f_w_BN_dot_contr, m_contr, m_prime_contr, I_contr, I_prime_contr, com_contr, com_prime_contr)


    def computeEnergy(self):

        stateMan = self._dynSystem.getStateManager()

        gamma = stateMan.getStates(self._stateGimbalAngleName)
        gamma_dot = stateMan.getStates(self._stateGimbalRateName)
        w_BN = stateMan.getStates(self._w_BN_B_name)
        sigma_BN = stateMan.getStates(self._sigma_BN_name)
        v_BN_N = stateMan.getStates(self._v_BN_N_name)

        BN = attitudeKinematics.mrp2dcm(sigma_BN)

        GG0 = coordinateTransformations.ROT3(gamma)

        BG = self._BG0.dot(GG0.T)

        Igs = self._Ig[0,0]
        Igt = self._Ig[1,1]
        Igg = self._Ig[2,2]

        gs_hat = BG[:,0]
        gt_hat = BG[:,1]
        gg_hat = BG[:,2]

        ws = np.inner(w_BN, gs_hat)
        wt = np.inner(w_BN, gt_hat)
        wg = np.inner(w_BN, gg_hat)

        I_offset = -self._mass * self._CoM_tilde.dot(self._CoM_tilde)

        E_contr = 0.5 * self._mass * np.inner(v_BN_N, v_BN_N) \
                  + self._mass * np.inner(BN.dot(v_BN_N), np.cross(w_BN, self._CoM)) \
                  + 0.5*np.inner(w_BN, I_offset.dot(w_BN)) \
                  + 0.5 * (Igs * ws**2 + Igt * wt**2 + Igg*(wg+gamma_dot)**2)

        return E_contr

    def computeAngularMomentum(self):
        stateMan = self._dynSystem.getStateManager()

        gamma = stateMan.getStates(self._stateGimbalAngleName)
        gamma_dot = stateMan.getStates(self._stateGimbalRateName)
        w_BN = stateMan.getStates(self._w_BN_B_name)
        sigma_BN = stateMan.getStates(self._sigma_BN_name)

        BN = attitudeKinematics.mrp2dcm(sigma_BN)

        GG0 = coordinateTransformations.ROT3(gamma)

        BG = self._BG0.dot(GG0.T)

        Igs = self._Ig[0,0]
        Igt = self._Ig[1,1]
        Igg = self._Ig[2,2]

        gs_hat = BG[:,0]
        gt_hat = BG[:,1]
        gg_hat = BG[:,2]

        ws = np.inner(w_BN, gs_hat)
        wt = np.inner(w_BN, gt_hat)
        wg = np.inner(w_BN, gg_hat)

        I_offset = -self._mass * self._CoM_tilde.dot(self._CoM_tilde)

        H_contr = BN.T.dot(I_offset.dot(w_BN) + Igs * ws * gs_hat + Igt * wt * gt_hat + Igg * (wg + gamma_dot) * gg_hat)
        return H_contr

    def computeMechanicalPower(self):
        stateMan = self._dynSystem.getStateManager()

        gamma_dot = stateMan.getStates(self._stateGimbalRateName)

        P = self._ug*gamma_dot
        return P