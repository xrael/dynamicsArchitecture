
from abc import ABCMeta, abstractmethod

from dynEffector import dynEffector
import attitudeKinematics
import coordinateTransformations
import numpy as np
from stateObj import stateObj



class stateEffector(dynEffector):
    """
    A state effector is an abstract class that is to be used for implementing dynamical effects that have a state.
    Rigid bodies, reaction wheels, slosh, CMGs, etc, have dynamics and state.
    """

    __metaclass__ = ABCMeta

    _stateEffectors = None
    _dynEffectors = None
    _stateEffectorParent = None

    _stateNames = ()

    def __init__(self, dynSystem, name, stateEffectorParent = None):
        super(stateEffector, self).__init__(dynSystem, name)
        self._stateEffectors = list()
        self._dynEffectors = list()
        self._stateEffectorParent = stateEffectorParent
        return

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

    #----------------------------------Abstract Methods-------------------------------#
    # Real state effectors should implement the following methods.

    @abstractmethod
    def computeRHS(self):
        """
        It is assumed that each effector is only connected to one another, called parent.
        This method is to be used to implement the effect that this effector produces on the Right Hand Side
        of the equations of motion of its parent.
        This method is inherited from dynamicEffector which also produces a RHS effect.
        :return: [tuple] Contributions to the RHS of the equations of motion of its parent.
        """
        pass

    @abstractmethod
    def registerStates(self):
        """
        Use this method to register states using the state manager.
        :return: void
        """
        pass

    @abstractmethod
    def computeLHS(self):
        """
        It is assumed that each effector is only connected to one another, called parent.
        This method is to be used to implement the effect that this effector produces on the Left Hand Side
        of the equations of motion of its parent.
        This is characteristic of a state effector.
        :return:
        """
        pass

    @abstractmethod
    def computeMassProperties(self):
        """
        It is assumed that each effector is only connected to one another, called parent.
        This method is to be used to implement the effect that this effector produces on the mass properties
        (mass, inertia, center of mass) of the whole system.
        This is characteristic of a state effector.
        :return: [tuple] mass, mass rate, inertia, inertia rate, CoM, CoM rate.
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
    def computeAngularMomentum(self):
        """
        Computes the angular momentum contribution of the effector
        :return: [double] Angular momentum contribution.
        """
        pass

#---------------------------------------------------------------------------------------------------------------------#

class spacecraftHub(stateEffector):
    """
Implements a rigid body hub effector.
    """

    _m_hub = 0.0            # [kg] Hub mass
    _I_Bc = None            # [kg-m^2] Inertia of the hub relative to the hub's center of mass Bc.
    _I_B = None             # [kg-m^2] Inertia of the hub relative to the reference point B.
    _r_BcB_B = None         # [m] Position of the hub's CoM relative to the reference B.
    _r_BcB_B_tilde = None   # Tilde matrix

    _mr_BcB_B = None        # [kg m] Momentum of order 1 (_m_hub * _r_BcB_B)

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
        self._m_hub = 100.0
        self._I_Bc = np.diag([50, 50, 50])
        self._r_BcB_B = np.zeros(3)
        self._r_BcB_B_tilde = np.zeros((3,3))
        self._mr_BcB_B = self._m_hub * self._r_BcB_B
        self._I_B = self._I_Bc + self._m_hub * self._r_BcB_B_tilde.dot(self._r_BcB_B_tilde.T)

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

    def setHubInertia(self, inertia):
        """
        Sets the Hub inertia relative to its center of mass.
        :param inertia:
        :return:
        """
        self._I_Bc = inertia
        self._I_B = self._I_Bc + self._m_hub * self._r_BcB_B_tilde.dot(self._r_BcB_B_tilde.T)
        return

    def setHubMass(self, mass):
        self._m_hub = mass
        self._I_B = self._I_Bc + self._m_hub * self._r_BcB_B_tilde.dot(self._r_BcB_B_tilde.T)
        self._mr_BcB_B = self._m_hub * self._r_BcB_B
        return

    def setHubCoMOffset(self, R_BcB_N):
        """
        Sets the hub's center of mass offset relative to the reference point.
        :param R_BcB_N:
        :return:
        """
        self._r_BcB_B = R_BcB_N
        self._r_BcB_B_tilde = attitudeKinematics.getSkewSymmetrixMatrix(R_BcB_N)
        self._I_B = self._I_Bc + self._m_hub * self._r_BcB_B_tilde.dot(self._r_BcB_B_tilde.T)
        self._mr_BcB_B = self._m_hub * self._r_BcB_B
        return

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
            (m_contr, m_prime_contr, I_contr, I_prime_contr, com_contr, com_prime_contr) = effector.computeMassProperties()
            m += m_contr
            m_prime += m_prime_contr
            I += I_contr
            I_prime += I_prime_contr
            com += com_contr
            com_prime += com_prime_contr

            (f_r_BN_dot_contr, f_w_dot_contr) = effector.computeRHS()
            f_r_BN_dot += f_r_BN_dot_contr
            f_w_dot += f_w_dot_contr

            (A_contr, B_contr, C_contr, D_contr) = effector.computeLHS()
            A += A_contr
            B += B_contr
            C += C_contr
            D += D_contr

        (m_contr, m_prime_contr, I_contr, I_prime_contr, com_contr, com_prime_contr) = self.computeMassProperties()
        m += m_contr
        m_prime += m_prime_contr
        I += I_contr
        I_prime += I_prime_contr
        com += com_contr
        com_prime += com_prime_contr

        (f_r_BN_dot_contr, f_w_dot_contr) = self.computeRHS()
        f_r_BN_dot += f_r_BN_dot_contr
        f_w_dot += f_w_dot_contr

        (A_contr, B_contr, C_contr, D_contr) = self.computeLHS()
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

    def computeRHS(self):
        stateMan = self._dynSystem.getStateManager()
        w_BN = stateMan.getStates(self._stateAngularVelocityName)
        w_BN_tilde = attitudeKinematics.getSkewSymmetrixMatrix(w_BN)

        f_r_BN_dot = -w_BN_tilde.dot(w_BN_tilde).dot(self._mr_BcB_B)
        f_w_dot = - w_BN_tilde.dot(self._I_B).dot(w_BN)
        return (f_r_BN_dot, f_w_dot)

    def computeLHS(self):
        A_contr = self._m_hub * np.eye(3)
        B_contr = -attitudeKinematics.getSkewSymmetrixMatrix(self._mr_BcB_B)
        C_contr = attitudeKinematics.getSkewSymmetrixMatrix(self._mr_BcB_B)
        D_contr = self._I_B
        return (A_contr, B_contr, C_contr, D_contr)

    def computeMassProperties(self):
        I_contr = self._I_B
        I_prime_contr = np.zeros((3,3))
        m_contr = self._m_hub
        m_prime_contr = 0.0
        com_contr = self._r_BcB_B
        com_prime_contr = np.zeros(3)
        return (m_contr, m_prime_contr, I_contr, I_prime_contr, com_contr, com_prime_contr)

    def computeEnergy(self):
        stateMan = self._dynSystem.getStateManager()

        v_BN_N = stateMan.getStates(self._stateVelocityName)
        w_BN = stateMan.getStates(self._stateAngularVelocityName)
        sigma_BN = stateMan.getStates(self._stateAttitudeName)
        BN = attitudeKinematics.mrp2dcm(sigma_BN)

        E_contr = 0.5*self._m_hub * np.inner(v_BN_N, v_BN_N) + 0.5 * np.inner(w_BN, self._I_B.dot(w_BN)) \
                  + self._m_hub * np.inner(BN.dot(v_BN_N), np.cross(w_BN, self._r_BcB_B ))
        return E_contr

    def computeAngularMomentum(self):
        stateMan = self._dynSystem.getStateManager()

        w_BN = stateMan.getStates(self._stateAngularVelocityName)
        sigma_BN = stateMan.getStates(self._stateAttitudeName)
        BN = attitudeKinematics.mrp2dcm(sigma_BN)

        H_contr = BN.T.dot(self._I_B.dot(w_BN))
        return H_contr

#---------------------------------------------------------------------------------------------------------------------#


class vscmg(stateEffector):
    """
    Implements a VSCMG (Variable Speed Control Moment Gyro) state effector.
    """

    _Ig = None                      # [kg m^2] Gimbal inertia relative to its center of mass
    _Iw = None                      # [kg m^2] Wheel inertia relative to its center of mass
    _BG0 = None                     # Rotation matrix from G frame to B frame at initial time
    _m_vscmg = 0.0                  # [kg] Total VSCMG mass
    _r_OiB_B = None                 # [m] Position of the CoM of the VSCMG relative to the reference point B
    _r_OiB_B_tilde = None
    _mr_OiB_B = None                # [kg m] Momentum of order 1 (_m_vscmg * _r_OiB_B)

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
        self._m_vscmg = 10.0
        self._r_OiB_B = np.zeros(3)
        self._r_OiB_B_tilde = attitudeKinematics.getSkewSymmetrixMatrix(self._r_OiB_B)
        self._mr_OiB_B = self._m_vscmg * self._r_OiB_B

        self._w_BN_B_name = w_BN_B_name
        self._sigma_BN_name = sigma_BN_name
        self._v_BN_N_name = v_BN_N_name

        self._stateNames = ('gamma', 'gamma_dot', 'Omega')

        self._stateGimbalAngleName = self._effectorName + '_' + self._stateNames[0]
        self._stateGimbalRateName = self._effectorName + '_' + self._stateNames[1]
        self._stateRWrateName = self._effectorName + '_' + self._stateNames[2]
        return

    @classmethod
    def getVSCMG(cls, dynSystem, name, m, r_OiB_B, Igs, Igt, Igg, Iws, Iwt, BG0, w_BN_B_name, sigma_BN_name, v_BN_N_name):
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
        :return: [vscmg]
        """
        vscmgObj = vscmg(dynSystem, name, w_BN_B_name, sigma_BN_name, v_BN_N_name)
        vscmgObj.registerStates()
        vscmgObj._m_vscmg = m
        vscmgObj._r_OiB_B = r_OiB_B
        vscmgObj._r_OiB_B_tilde = attitudeKinematics.getSkewSymmetrixMatrix(r_OiB_B)
        vscmgObj._mr_OiB_B = m * r_OiB_B
        vscmgObj.setGimbalInertia(Igs, Igt, Igg)
        vscmgObj.setWheelInertia(Iws, Iwt)
        vscmgObj._BG0 = BG0
        return vscmgObj

    def setW_BNname(self, w_BN_B_name):
        self._w_BN_B_name = w_BN_B_name
        return

    def setWheelInertia(self, Iws, Iwt):
        self._Iw = np.diag(np.array([Iws, Iwt, Iwt]))
        return

    def setGimbalInertia(self, Igs, Igt, Igg):
        self._Ig = np.diag(np.array([Igs, Igt, Igg]))
        return

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


    #----------------StateEffector interface---------------#

    def computeStateDerivatives(self, t):

        stateMan = self._dynSystem.getStateManager()

        w_BN = stateMan.getStates(self._w_BN_B_name)
        w_BN_dot = stateMan.getStateDerivatives(self._w_BN_B_name)
        gamma = stateMan.getStates(self._stateGimbalAngleName)
        gamma_dot = stateMan.getStates(self._stateGimbalRateName)
        Omega = stateMan.getStates(self._stateRWrateName)

        us = self.computeRWControlTorque()
        ug = self.computeGimbalControlTorque()

        Iws = self._Iw[0,0]
        Iwt = self._Iw[1,1]
        Js = self._Ig[0,0] + Iws
        Jt = self._Ig[1,1] + Iwt
        Jg = self._Ig[2,2] + Iwt

        GG0 = coordinateTransformations.ROT3(gamma)

        BG = self._BG0.dot(GG0.T)

        gs_hat = BG[:,0]
        gt_hat = BG[:,1]
        gg_hat = BG[:,2]

        ws = np.inner(gs_hat, w_BN)
        wt = np.inner(gt_hat, w_BN)

        Omega_dot = us/Iws - np.inner(gs_hat, w_BN_dot) - gamma_dot * wt

        gamma_ddot = ug/Jg - np.inner(gg_hat, w_BN_dot) - (Jt - Js)/Jg * ws * wt + Iws/Jg * wt * Omega

        stateMan.setStateDerivatives(self._stateGimbalAngleName, gamma_dot)
        stateMan.setStateDerivatives(self._stateGimbalRateName, gamma_ddot)
        stateMan.setStateDerivatives(self._stateRWrateName, Omega_dot)

        return

    def computeRHS(self):

        stateMan = self._dynSystem.getStateManager()

        w_BN = stateMan.getStates(self._w_BN_B_name)
        gamma = stateMan.getStates(self._stateGimbalAngleName)
        gamma_dot = stateMan.getStates(self._stateGimbalRateName)
        Omega = stateMan.getStates(self._stateRWrateName)

        Iws = self._Iw[0,0]
        Iwt = self._Iw[1,1]
        Js = self._Ig[0,0] + Iws
        Jt = self._Ig[1,1] + Iwt
        Jg = self._Ig[2,2] + Iwt

        GG0 = coordinateTransformations.ROT3(gamma)

        BG = self._BG0.dot(GG0.T)

        gs_hat = BG[:,0]
        gt_hat = BG[:,1]
        gg_hat = BG[:,2]

        ws = np.inner(gs_hat, w_BN)
        wt = np.inner(gt_hat, w_BN)
        wg = np.inner(gg_hat, w_BN)

        us = self.computeRWControlTorque()
        ug = self.computeGimbalControlTorque()

        w_BN_tilde = attitudeKinematics.getSkewSymmetrixMatrix(w_BN)

        I_B = -self._m_vscmg * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde) + BG.dot(self._Ig + self._Iw).dot(BG.T)

        f_r_BN_dot_contr = -w_BN_tilde.dot(w_BN_tilde).dot(self._mr_OiB_B)
        f_w_BN_dot_contr = -w_BN_tilde.dot(I_B).dot(w_BN) \
                           -gs_hat * ((Js - Jt + Jg) * wt * gamma_dot - Iws * wt * gamma_dot + us) \
                           -gt_hat * (((Js - Jt - Jg) * ws + Iws * Omega) * gamma_dot + Iws * wg * Omega) \
                           -gg_hat * (ug - (Jt - Js) * ws * wt)

        return (f_r_BN_dot_contr, f_w_BN_dot_contr)

    def computeMassProperties(self):

        stateMan = self._dynSystem.getStateManager()

        gamma = stateMan.getStates(self._stateGimbalAngleName)

        GG0 = coordinateTransformations.ROT3(gamma)

        BG = self._BG0.dot(GG0.T)

        m_contr = self._m_vscmg
        m_prime_contr = 0.0
        I_contr = -self._m_vscmg * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde) + BG.dot(self._Ig + self._Iw).dot(BG.T)
        I_prime_contr = np.zeros((3,3))
        com_contr = self._mr_OiB_B
        com_prime_contr = np.zeros(3)

        return (m_contr, m_prime_contr, I_contr, I_prime_contr, com_contr, com_prime_contr)

    def computeLHS(self):

        stateMan = self._dynSystem.getStateManager()

        gamma = stateMan.getStates(self._stateGimbalAngleName)

        Igs = self._Ig[0,0]
        Iwt = self._Iw[1,1]
        Jt = self._Ig[1,1] + Iwt

        GG0 = coordinateTransformations.ROT3(gamma)

        BG = self._BG0.dot(GG0.T)

        gs_hat = BG[:,0]
        gt_hat = BG[:,1]

        gs_outer = np.outer(gs_hat, gs_hat)
        gt_outer = np.outer(gt_hat, gt_hat)

        mr_OiB_B_tilde = attitudeKinematics.getSkewSymmetrixMatrix(self._mr_OiB_B)

        A_contr = self._m_vscmg * np.eye(3)
        B_contr = -mr_OiB_B_tilde
        C_contr = mr_OiB_B_tilde
        D_contr = -self._m_vscmg * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde) + Igs * gs_outer + Jt * gt_outer

        return (A_contr, B_contr, C_contr, D_contr)

    def computeRWControlTorque(self):
        return 0.0

    def computeGimbalControlTorque(self):
        return 0.0

    def computeEnergy(self):

        stateMan = self._dynSystem.getStateManager()

        gamma = stateMan.getStates(self._stateGimbalAngleName)
        gamma_dot = stateMan.getStates(self._stateGimbalRateName)
        Omega = stateMan.getStates(self._stateRWrateName)
        w_BN = stateMan.getStates(self._w_BN_B_name)
        sigma_BN = stateMan.getStates(self._sigma_BN_name)
        v_BN_N = stateMan.getStates(self._v_BN_N_name)

        BN = attitudeKinematics.mrp2dcm(sigma_BN)

        GG0 = coordinateTransformations.ROT3(gamma)

        BG = self._BG0.dot(GG0.T)

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

        I_offset = -self._m_vscmg * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde)

        E_contr = 0.5 * self._m_vscmg * np.inner(v_BN_N, v_BN_N) \
                  + self._m_vscmg * np.inner(BN.dot(v_BN_N), np.cross(w_BN, self._r_OiB_B)) \
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

        GG0 = coordinateTransformations.ROT3(gamma)

        BG = self._BG0.dot(GG0.T)

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

        I_offset = -self._m_vscmg * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde)

        H_contr = BN.T.dot(I_offset.dot(w_BN) + (Js * ws + Iws * Omega) * gs_hat + Jt * wt * gt_hat + Jg * (wg + gamma_dot) * gg_hat)
        return H_contr


#---------------------------------------------------------------------------------------------------------------------#


class reactionWheel(stateEffector):
    """
    Implements a Reaction Wheel (RW) state effector.
    """

    _Iw = None                  # [kg m^2] Inertia of the wheel relative to its center of mass
    _BW = None                  # Rotation matrix from the wheel frame to the body frame
    _m_rw = 0.0                 # [kg] Mass of the RW
    _r_OiB_B = None             # [m] RW CoM offset relative to reference point B
    _r_OiB_B_tilde = None
    _mr_OiB_B = None            # [kg m] Momentum of order 1 (_m_rw * _r_OiB_B)

    _w_BN_B_name = ''
    _sigma_BN_name = ''
    _v_BN_N_name = ''

    _stateRWrateName = ''

    def __init__(self, dynSystem, name, w_BN_B_name, sigma_BN_name, v_BN_N_name):
        super(reactionWheel, self).__init__(dynSystem, name)
        self._Iw = np.eye(3)
        self._BW = np.eye(3)
        self._m_rw = 10.0
        self._r_OiB_B = np.zeros(3)
        self._r_OiB_B_tilde = attitudeKinematics.getSkewSymmetrixMatrix(self._r_OiB_B)
        self._mr_OiB_B = self._m_rw * self._r_OiB_B

        self._w_BN_B_name = w_BN_B_name
        self._sigma_BN_name = sigma_BN_name
        self._v_BN_N_name = v_BN_N_name

        self._stateNames = ('Omega',)

        self._stateRWrateName = self._effectorName + '_' + self._stateNames[0]
        return

    @classmethod
    def getRW(cls, dynSystem, name, m, r_OiB_B, Iws, Iwt, BW,  w_BN_B_name, sigma_BN_name, v_BN_N_name):
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
        :return: [rw]
        """
        rwObj = reactionWheel(dynSystem, name, w_BN_B_name, sigma_BN_name, v_BN_N_name)
        rwObj.registerStates()
        rwObj._m_rw = m
        rwObj._r_OiB_B = r_OiB_B
        rwObj._r_OiB_B_tilde = attitudeKinematics.getSkewSymmetrixMatrix(r_OiB_B)
        rwObj._mr_OiB_B = m * r_OiB_B
        rwObj.setInertia(Iws, Iwt)
        rwObj._BW = BW
        return rwObj

    def setW_BNname(self, w_BN_B_name):
        self._w_BN_B_name = w_BN_B_name
        return

    def setInertia(self, Iws, Iwt):
        self._Iw = np.diag(np.array([Iws, Iwt, Iwt]))
        return

    def registerStates(self):

        stateMan = self._dynSystem.getStateManager()

        if not stateMan.registerState(self._stateRWrateName, 1):
            return False
        else:
            return True

    #----------------StateEffector interface---------------#
    def computeStateDerivatives(self, t):

        stateMan = self._dynSystem.getStateManager()

        w_BN_dot = stateMan.getStateDerivatives(self._w_BN_B_name)
        us = self.computeRWControlTorque()

        Iws = self._Iw[0,0]

        gs_hat = self._BW[:,0]

        Omega_dot = us/Iws - np.inner(gs_hat, w_BN_dot)

        stateMan.setStateDerivatives(self._stateRWrateName, Omega_dot)

        return

    def computeRHS(self):

        stateMan = self._dynSystem.getStateManager()

        w_BN = stateMan.getStates(self._w_BN_B_name)
        Omega = stateMan.getStates(self._stateRWrateName)

        Iws = self._Iw[0,0]
        Iwt = self._Iw[1,1]

        gs_hat = self._BW[:,0]

        us = self.computeRWControlTorque()

        w_BN_tilde = attitudeKinematics.getSkewSymmetrixMatrix(w_BN)

        I_B = -self._m_rw * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde) + self._BW.dot(self._Iw).dot(self._BW.T)

        f_r_BN_dot_contr = -w_BN_tilde.dot(w_BN_tilde).dot(self._mr_OiB_B)
        f_w_BN_dot_contr = -w_BN_tilde.dot(I_B).dot(w_BN) \
                           - gs_hat * us \
                           - (Iws * Omega) * w_BN_tilde.dot(gs_hat)

        return (f_r_BN_dot_contr, f_w_BN_dot_contr)

    def computeMassProperties(self):
        m_contr = self._m_rw
        m_prime_contr = 0.0
        I_contr = -self._m_rw * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde) + self._BW.dot(self._Iw).dot(self._BW.T)
        I_prime_contr = np.zeros((3,3))
        com_contr = self._mr_OiB_B
        com_prime_contr = np.zeros(3)

        return (m_contr, m_prime_contr, I_contr, I_prime_contr, com_contr, com_prime_contr)

    def computeLHS(self):

        Iws = self._Iw[0,0]
        Iwt = self._Iw[1,1]

        gt_hat = self._BW[:,1]
        gg_hat = self._BW[:,2]

        gt_outer = np.outer(gt_hat, gt_hat)
        gg_outer = np.outer(gg_hat, gg_hat)

        mr_OiB_B_tilde = attitudeKinematics.getSkewSymmetrixMatrix(self._mr_OiB_B)

        A_contr = self._m_rw * np.eye(3)
        B_contr = -mr_OiB_B_tilde
        C_contr = mr_OiB_B_tilde
        D_contr = -self._m_rw * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde) + Iwt * gt_outer + Iwt * gg_outer

        return (A_contr, B_contr, C_contr, D_contr)

    def computeRWControlTorque(self):
        return 0.0


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

        I_offset = -self._m_rw * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde)

        E_contr = 0.5 * self._m_rw * np.inner(v_BN_N, v_BN_N) \
                  + self._m_rw * np.inner(BN.dot(v_BN_N), np.cross(w_BN, self._r_OiB_B)) \
                  + 0.5*np.inner(w_BN, I_offset.dot(w_BN)) \
                  + 0.5 * (Iws*(ws+Omega)**2 + Iwt * wt**2 + Iwt*wg**2)

        return E_contr

    def computeAngularMomentum(self):
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

        I_offset = -self._m_rw * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde)

        H_contr = BN.T.dot(I_offset.dot(w_BN) + Iws * (ws + Omega) * gs_hat + Iwt * wt * gt_hat + Iwt * wg * gg_hat)
        return H_contr


#---------------------------------------------------------------------------------------------------------------------#


class cmg(stateEffector):
    """
    Implements a CMG (Control Moment Gyro) state effector.
    """

    _Ig = None                      # [kg m^2] Gimbal inertia relative to its center of mass
    _BG0 = None                     # Rotation matrix from G frame to B frame at initial time
    _m_cmg = 0.0                    # [kg] Total CMG mass
    _r_OiB_B = None                 # [m] Position of the CoM of the CMG relative to the reference point B
    _r_OiB_B_tilde = None
    _mr_OiB_B = None                # [kg m] Momentum of order 1 (_m_cmg * _r_OiB_B)

    _w_BN_B_name = ''
    _sigma_BN_name = ''
    _v_BN_N_name = ''

    _stateGimbalAngleName = ''
    _stateGimbalRateName = ''

    def __init__(self, dynSystem, name, w_BN_B_name, sigma_BN_name, v_BN_N_name):
        super(cmg, self).__init__(dynSystem, name)
        self._Ig = np.eye(3)
        self._BG0 = np.eye(3)
        self._m_cmg = 10.0
        self._r_OiB_B = np.zeros(3)
        self._r_OiB_B_tilde = attitudeKinematics.getSkewSymmetrixMatrix(self._r_OiB_B)
        self._mr_OiB_B = self._m_cmg * self._r_OiB_B

        self._w_BN_B_name = w_BN_B_name
        self._sigma_BN_name = sigma_BN_name
        self._v_BN_N_name = v_BN_N_name

        self._stateNames = ('gamma', 'gamma_dot')

        self._stateGimbalAngleName = self._effectorName + '_' + self._stateNames[0]
        self._stateGimbalRateName = self._effectorName + '_' + self._stateNames[1]
        return

    @classmethod
    def getCMG(cls, dynSystem, name, m, r_OiB_B, Igs, Igt, Igg, BG0, w_BN_B_name, sigma_BN_name, v_BN_N_name):
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
        :return: [vscmg]
        """
        cmgObj = vscmg(dynSystem, name, w_BN_B_name, sigma_BN_name, v_BN_N_name)
        cmgObj.registerStates()
        cmgObj._m_cmg = m
        cmgObj._r_OiB_B = r_OiB_B
        cmgObj._r_OiB_B_tilde = attitudeKinematics.getSkewSymmetrixMatrix(r_OiB_B)
        cmgObj._mr_OiB_B = m * r_OiB_B
        cmgObj.setGimbalInertia(Igs, Igt, Igg)
        cmgObj._BG0 = BG0
        return cmgObj

    def setW_BNname(self, w_BN_B_name):
        self._w_BN_B_name = w_BN_B_name
        return

    def setGimbalInertia(self, Igs, Igt, Igg):
        self._Ig = np.diag(np.array([Igs, Igt, Igg]))
        return

    def registerStates(self):

        stateMan = self._dynSystem.getStateManager()

        if not stateMan.registerState(self._stateGimbalAngleName, 1):
            return False
        elif not stateMan.registerState(self._stateGimbalRateName,1):
            stateMan.unregisterState(self._stateGimbalAngleName)
            return False
        else:
            return True


    #----------------StateEffector interface---------------#

    def computeStateDerivatives(self, t):

        stateMan = self._dynSystem.getStateManager()

        w_BN = stateMan.getStates(self._w_BN_B_name)
        w_BN_dot = stateMan.getStateDerivatives(self._w_BN_B_name)
        gamma = stateMan.getStates(self._stateGimbalAngleName)
        gamma_dot = stateMan.getStates(self._stateGimbalRateName)

        ug = self.computeGimbalControlTorque()

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

        gamma_ddot = ug/Igg - np.inner(gg_hat, w_BN_dot) - (Igt - Igs)/Igg * ws * wt

        stateMan.setStateDerivatives(self._stateGimbalAngleName, gamma_dot)
        stateMan.setStateDerivatives(self._stateGimbalRateName, gamma_ddot)

        return

    def computeRHS(self):

        stateMan = self._dynSystem.getStateManager()

        w_BN = stateMan.getStates(self._w_BN_B_name)
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
        wg = np.inner(gg_hat, w_BN)

        ug = self.computeGimbalControlTorque()

        w_BN_tilde = attitudeKinematics.getSkewSymmetrixMatrix(w_BN)

        I_B = -self._m_cmg * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde) + BG.dot(self._Ig).dot(BG.T)

        f_r_BN_dot_contr = -w_BN_tilde.dot(w_BN_tilde).dot(self._mr_OiB_B)
        f_w_BN_dot_contr = -w_BN_tilde.dot(I_B).dot(w_BN) \
                           -gs_hat * ((Igs - Igt + Igg) * wt * gamma_dot) \
                           -gt_hat * ((Igs - Igt - Igg) * ws * gamma_dot) \
                           -gg_hat * (ug - (Igt - Igs) * ws * wt)

        return (f_r_BN_dot_contr, f_w_BN_dot_contr)

    def computeMassProperties(self):

        stateMan = self._dynSystem.getStateManager()

        gamma = stateMan.getStates(self._stateGimbalAngleName)

        GG0 = coordinateTransformations.ROT3(gamma)

        BG = self._BG0.dot(GG0.T)

        m_contr = self._m_cmg
        m_prime_contr = 0.0
        I_contr = -self._m_cmg * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde) + BG.dot(self._Ig).dot(BG.T)
        I_prime_contr = np.zeros((3,3))
        com_contr = self._mr_OiB_B
        com_prime_contr = np.zeros(3)

        return (m_contr, m_prime_contr, I_contr, I_prime_contr, com_contr, com_prime_contr)

    def computeLHS(self):

        stateMan = self._dynSystem.getStateManager()

        gamma = stateMan.getStates(self._stateGimbalAngleName)

        Igs = self._Ig[0,0]
        Igt = self._Ig[1,1]

        GG0 = coordinateTransformations.ROT3(gamma)

        BG = self._BG0.dot(GG0.T)

        gs_hat = BG[:,0]
        gt_hat = BG[:,1]

        gs_outer = np.outer(gs_hat, gs_hat)
        gt_outer = np.outer(gt_hat, gt_hat)

        mr_OiB_B_tilde = attitudeKinematics.getSkewSymmetrixMatrix(self._mr_OiB_B)

        A_contr = self._m_cmg * np.eye(3)
        B_contr = -mr_OiB_B_tilde
        C_contr = mr_OiB_B_tilde
        D_contr = -self._m_cmg * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde) + Igs * gs_outer + Igt * gt_outer

        return (A_contr, B_contr, C_contr, D_contr)

    def computeRWControlTorque(self):
        return 0.0

    def computeGimbalControlTorque(self):
        return 0.0

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

        I_offset = -self._m_cmg * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde)

        E_contr = 0.5 * self._m_cmg * np.inner(v_BN_N, v_BN_N) \
                  + self._m_cmg * np.inner(BN.dot(v_BN_N), np.cross(w_BN, self._r_OiB_B)) \
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

        I_offset = -self._m_cmg * self._r_OiB_B_tilde.dot(self._r_OiB_B_tilde)

        H_contr = BN.T.dot(I_offset.dot(w_BN) + Igs * ws * gs_hat + Igt * wt * gt_hat + Igg * (wg + gamma_dot) * gg_hat)
        return H_contr