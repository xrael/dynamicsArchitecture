
from abc import ABCMeta, abstractmethod

from dynEffector import dynEffector
import attitudeKinematics
import coordinateTransformations
import numpy as np
from stateObj import stateObj

class stateEffector(dynEffector):

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
        self._stateEffectors.append(stateEffector)
        #stateEffector.setStateEffectorParent(self)
        return

    def getStateEffectors(self):
        return self._stateEffectors

    def addDynEffector(self, dynEffector):
        self._dynEffectors.append(dynEffector)
        return

    def getStateEffectorParent(self):
        return self._stateEffectorParent

    def setStateEffectorParent(self, stateEffectorParent):
        self._stateEffectorParent = stateEffectorParent
        return

    def getStateNames(self):
        return self._stateNames

    @abstractmethod
    def computeRHS(self): pass

    @abstractmethod
    def registerStates(self): pass

    @abstractmethod
    def computeLHS(self): pass

    @abstractmethod
    def computeMassProperties(self): pass

    @abstractmethod
    def computeStateDerivatives(self, t): pass

    @abstractmethod
    def computeEnergy(self): pass


class spacecraftHub(stateEffector):

    _m_hub = 0.0
    _I_Bc = None
    _I_B = None
    _r_BcB_B = None
    _r_BcB_B_tilde = None

    _mr_BcB_B = None

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
        hub = spacecraftHub(dynSystem, name)
        hub.registerStates()

        return hub

    def setHubInertia(self, inertia):
        self._I_Bc = inertia
        self._I_B = self._I_Bc + self._m_hub * self._r_BcB_B_tilde.dot(self._r_BcB_B_tilde.T)
        return

    def setHubMass(self, mass):
        self._m_hub = mass
        self._I_B = self._I_Bc + self._m_hub * self._r_BcB_B_tilde.dot(self._r_BcB_B_tilde.T)
        self._mr_BcB_B = self._m_hub * self._r_BcB_B
        return

    def setHubCoMOffset(self, R_BcB_N):
        self._r_BcB_B = R_BcB_N
        self._r_BcB_B_tilde = attitudeKinematics.getSkewSymmetrixMatrix(R_BcB_N)
        self._I_B = self._I_Bc + self._m_hub * self._r_BcB_B_tilde.dot(self._r_BcB_B_tilde.T)
        self._mr_BcB_B = self._m_hub * self._r_BcB_B
        return

    def registerStates(self):

        stateMan = self._dynSystem.getStateManager()

        state_pos =  stateObj(self._statePositionName, 0, 3)
        state_vel =  stateObj(self._stateVelocityName, 0, 3)
        state_att =  stateObj(self._stateAttitudeName, 0, 3, attitudeKinematics.switchMRPrepresentation)
        state_w =  stateObj(self._stateAngularVelocityName, 0, 3)

        if not stateMan.registerState(state_pos):
            return False
        elif not stateMan.registerState(state_vel):
            stateMan.unregisterState(self._statePositionName)
            return False
        elif not stateMan.registerState(state_att):
            stateMan.unregisterState(self._statePositionName)
            stateMan.unregisterState(self._stateVelocityName)
            return False
        elif not stateMan.registerState(state_w):
            stateMan.unregisterState(self._statePositionName)
            stateMan.unregisterState(self._stateVelocityName)
            stateMan.unregisterState(self._stateAttitudeName)
            return False
        else:
            return True

    def getStatePositionName(self):
        return self._statePositionName

    def getStateVelocityName(self):
        return self._stateVelocityName

    def getStateAttitudeName(self):
        return self._stateAttitudeName

    def getStateAngularVelocityName(self):
        return self._stateAngularVelocityName


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

        sigma_BN = stateMan.getStates(self._stateAttitudeName)

        BN = attitudeKinematics.mrp2dcm(sigma_BN)

        v_BN_N = stateMan.getStates(self._stateVelocityName)
        w_BN = stateMan.getStates(self._stateAngularVelocityName)

        E_contr = 0.5*self._m_hub * np.inner(v_BN_N, v_BN_N) + 0.5 * np.inner(w_BN, self._I_B.dot(w_BN)) \
                  + self._m_hub * np.inner(BN.dot(v_BN_N), np.cross(w_BN, self._r_BcB_B ))

        return E_contr








class vscmg(stateEffector):

    _Ig = None
    _Iw = None
    _BG0 = None
    _m_vscmg = 0.0
    _r_OiB_B = None
    _r_OiB_B_tilde = None
    _mr_OiB_B = None

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

        state_gimb_ang =  stateObj(self._stateGimbalAngleName, 0, 1)
        state_gimb_rate =  stateObj(self._stateGimbalRateName, 0, 1)
        state_rw_rate =  stateObj(self._stateRWrateName, 0, 1)

        if not stateMan.registerState(state_gimb_ang):
            return False
        elif not stateMan.registerState(state_gimb_rate):
            stateMan.unregisterState(self._stateGimbalAngleName)
            return False
        elif not stateMan.registerState(state_rw_rate):
            stateMan.unregisterState(self._stateGimbalAngleName)
            stateMan.unregisterState(self._stateGimbalRateName)
            return False
        else:
            return True

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

