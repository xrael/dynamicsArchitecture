
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
        stateEffector.setStateEffectorParent(self)
        return

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
    def computeRHS(cls): pass

    @abstractmethod
    def registerStates(self): pass

    @abstractmethod
    def computeLHS(self): pass

    @abstractmethod
    def computeMassProperties(self): pass

    @abstractmethod
    def computeStateDerivatives(self, t): pass


class spacecraftHub(stateEffector):

    _mass_hub = 0.0
    _inertia_hub_B = None
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

    _statePosition = ''
    _stateVelocity = ''
    _stateAttitude = ''
    _stateAngularVelocity = ''

    def __init__(self, dynSystem, name):
        super(spacecraftHub, self).__init__(dynSystem, name)
        self._mass_hub = 100.0
        self._inertia_hub_B = np.diag([50, 50, 50])

        self._stateNames = ('R_BN', 'R_BN_dot', 'sigma_BN', 'ohmega_BN')

        self._statePosition = self._effectorName + '_'+ self._stateNames[0]
        self._stateVelocity = self._effectorName + '_'+ self._stateNames[1]
        self._stateAttitude = self._effectorName + '_' + self._stateNames[2]
        self._stateAngularVelocity = self._effectorName + '_' + self._stateNames[3]
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
        hub._mass_hub = 100.0
        hub._inertia_hub_B = np.diag([50, 50, 50])
        return hub

    def setHubInertia(self, inertia):
        self._inertia_hub_B = inertia
        return

    def setHubMass(self, mass):
        self._mass_hub = mass
        return

    def registerStates(self):

        stateMan = self._dynSystem.getStateManager()

        state_pos =  stateObj(self._statePosition, 0, 3)
        state_vel =  stateObj(self._stateVelocity, 0, 3)
        state_att =  stateObj(self._stateAttitude, 0, 3, attitudeKinematics.switchMRPrepresentation)
        state_w =  stateObj(self._stateAngularVelocity, 0, 3)

        if not stateMan.registerState(state_pos):
            return False
        elif not stateMan.registerState(state_vel):
            stateMan.unregisterState(self._statePosition)
            return False
        elif not stateMan.registerState(state_att):
            stateMan.unregisterState(self._statePosition)
            stateMan.unregisterState(self._stateVelocity)
            return False
        elif not stateMan.registerState(state_w):
            stateMan.unregisterState(self._statePosition)
            stateMan.unregisterState(self._stateVelocity)
            stateMan.unregisterState(self._stateAttitude)
            return False
        else:
            return True

    def getStatePositionName(self):
        return self._statePosition

    def getStateVelocityName(self):
        return self._stateVelocity

    def getStateAtttiudeName(self):
        return self._stateAttitude

    def getStateAngularVelocityName(self):
        return self._stateAngularVelocity

    def addAccelerationContribution(self, contr):
        self._r_ddot_contr += contr
        return

    def addAngularAccelerationContribution(self, contr):
        self._w_dot_contr += contr
        return

    def addMassContribution(self, contr):
        self._m_contr += contr
        return

    def addMassRateContribution(self, contr):
        self._m_dot_contr += contr
        return

    def addInertiaContribution(self, contr):
        self._I_contr += contr
        return

    def addInertiaDotContribution(self, contr):
        self._I_dot_contr += contr
        return

    def addCoMContribution(self, contr):
        self._com_contr += contr
        return

    def addCoMdotContribution(self, contr):
        self._com_dot_contr += contr
        return

    def addMassMatrixContribution(self, A_contr, B_contr, C_contr, D_contr):
        self._A_contr += A_contr
        self._B_contr += B_contr
        self._C_contr += C_contr
        self._D_contr += D_contr
        return


    def computeStateDerivatives(self, t):

        stateMan = self._dynSystem.getStateManager()

        r_BN_N = stateMan.getStates(self._statePosition)
        v_BN_N = stateMan.getStates(self._stateVelocity)
        sigma = stateMan.getStates(self._stateAttitude)
        w_BN = stateMan.getStates(self._stateAngularVelocity)

        w_BN_tilde = attitudeKinematics.getSkewSymmetrixMatrix(w_BN)

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


        for effector in self._stateEffectors:

            effector.computeRHS()
            effector.computeMassProperties()
            effector.computeLHS()

        self._I_contr += self._inertia_hub_B
        self._m_contr += self._mass_hub

        self._A_contr += self._mass_hub * np.eye(3)
        self._B_contr += np.zeros((3,3))
        self._C_contr += np.zeros((3,3))
        self._D_contr += self._I_contr

        self._r_ddot_contr += np.zeros(3)
        self._w_dot_contr += - w_BN_tilde.dot(self._I_contr).dot(w_BN)


        A_inv = np.linalg.inv(self._A_contr)

        w_BN_dot = np.linalg.inv(self._D_contr - self._C_contr.dot(A_inv).dot(self._B_contr)).dot(self._w_dot_contr - self._C_contr.dot(A_inv).dot(self._r_ddot_contr))
        r_BN_N_ddot = A_inv.dot(self._r_ddot_contr - self._B_contr.dot(w_BN_dot))

        stateMan.setStateDerivatives(self._statePosition, v_BN_N)
        stateMan.setStateDerivatives(self._stateAttitude, attitudeKinematics.angVel2mrpRate(sigma, w_BN))
        stateMan.setStateDerivatives(self._stateVelocity, r_BN_N_ddot)
        stateMan.setStateDerivatives(self._stateAngularVelocity, w_BN_dot)

        for effector in self._stateEffectors:
            effector.computeStateDerivatives(t)

        return


    def computeRHS(cls):
        return

    def computeLHS(self):
        return

    def computeMassProperties(self):
        return












class vscmg(stateEffector):

    _Ig = None
    _Iw = None
    _BG = None
    _m = 0.0
    _R_CoM = None

    def __init__(self, dynSystem, name, hubParent):
        super(vscmg, self).__init__(dynSystem, name, hubParent)
        self._Ig = np.eye(3)
        self._Iw = np.eye(3)
        self._BG = np.eye(3)
        self._m = 10.0
        self._R_CoM = np.zeros(3)

        self._stateNames = ('gamma', 'gamma_dot', 'Ohmega')

        self._stateGimbalAngle = self._effectorName + '_' + self._stateNames[0]
        self._stateGimbalRate = self._effectorName + '_' + self._stateNames[1]
        self._stateRWrate = self._effectorName + '_' + self._stateNames[2]
        return

    @classmethod
    def getVSCMG(cls, dynSystem, name, m, R_CoM, Ig, Iw, BG, hubParent):
        vscmgObj = vscmg(dynSystem, name, hubParent)
        vscmgObj.registerStates()
        vscmgObj._m = m
        vscmgObj._R_CoM = R_CoM
        vscmgObj._Ig = Ig
        vscmgObj._Iw = Iw
        vscmgObj._BG = BG
        return vscmgObj


    def registerStates(self):

        stateMan = self._dynSystem.getStateManager()

        state_gimb_ang =  stateObj(self._stateGimbalAngle, 0, 1)
        state_gimb_rate =  stateObj(self._stateGimbalRate, 0, 1)
        state_rw_rate =  stateObj(self._stateRWrate, 0, 1)

        if not stateMan.registerState(state_gimb_ang):
            return False
        elif not stateMan.registerState(state_gimb_rate):
            stateMan.unregisterState(self._stateGimbalAngle)
            return False
        elif not stateMan.registerState(state_rw_rate):
            stateMan.unregisterState(self._stateGimbalAngle)
            stateMan.unregisterState(self._stateGimbalRate)
            return False
        else:
            return True

    def computeStateDerivatives(self, t):

        stateMan = self._dynSystem.getStateManager()

        w_BN_name = self._stateEffectorParent.getStateAngularVelocityName()

        w_BN = stateMan.getStates(w_BN_name)
        w_BN_dot = stateMan.getStateDerivatives(w_BN_name)
        gamma = stateMan.getStates(self._stateGimbalAngle)
        gamma_dot = stateMan.getStates(self._stateGimbalRate)
        Omega = stateMan.getStates(self._stateRWrate)

        us = self.computeRWControlTorque()
        ug = self.computeGimbalControlTorque()

        Iws = self._Iw[0]
        Iwt = self._Iw[1]
        Js = self._Ig[0] + Iws
        Jt = self._Ig[1] + Iwt
        Jg = self._Ig[2] + Iwt

        GG0 = coordinateTransformations.ROT3(gamma)

        BG = self._BG.dot(GG0.T)

        gs_hat = BG[:,0]
        gt_hat = BG[:,1]
        gg_hat = BG[:,2]

        ws = np.inner(gs_hat, w_BN)
        wt = np.inner(gt_hat, w_BN)

        Ohmega_dot = us/Iws - np.inner(gs_hat, w_BN_dot) - gamma_dot * wt

        gamma_ddot = ug/Jg - np.inner(gg_hat, w_BN_dot) - (Jt - Js)/Jg * ws * wt + Iws/Jg * wt * Omega

        stateMan.setStateDerivatives(self._stateGimbalAngle, gamma_dot)
        stateMan.setStateDerivatives(self._stateGimbalRate, gamma_ddot)
        stateMan.setStateDerivatives(self._stateRWrate, Ohmega_dot)

        return

    def computeRHS(self):

        stateMan = self._dynSystem.getStateManager()

        w_BN_name = self._stateEffectorParent.getStateAngularVelocityName()

        w_BN = stateMan.getStates(w_BN_name)
        gamma = stateMan.getStates(self._stateGimbalAngle)
        gamma_dot = stateMan.getStates(self._stateGimbalRate)
        Ohmega = stateMan.getStates(self._stateRWrate)

        Iws = self._Iw[0]
        Iwt = self._Iw[1]
        Js = self._Ig[0] + Iws
        Jt = self._Ig[1] + Iwt
        Jg = self._Ig[2] + Iwt

        GG0 = coordinateTransformations.ROT3(gamma)

        BG = self._BG.dot(GG0.T)

        gs_hat = BG[:,0]
        gt_hat = BG[:,1]
        gg_hat = BG[:,2]

        ws = np.inner(gs_hat, w_BN)
        wt = np.inner(gt_hat, w_BN)
        wg = np.inner(gg_hat, w_BN)

        us = self.computeRWControlTorque()
        ug = self.computeGimbalControlTorque()

        #w_BN_tilde = attitudeKinematics.getSkewSymmetrixMatrix(w_BN)

        angularAccelerationContr = -gs_hat * ((Js - Jt + Jg) * wt * gamma_dot - Iws * wt * gamma_dot + us) \
                                   -gt_hat * (((Js - Jt - Jg) * ws + Iws * Ohmega) * gamma_dot + Iws * wg * Ohmega) \
                                   -gg_hat * (ug - (Jt - Js) * ws * wt)

        self._stateEffectorParent.addAccelerationContribution(np.zeros(3))
        self._stateEffectorParent.addAngularAccelerationContribution(angularAccelerationContr)

        return

    def computeMassProperties(self):

        stateMan = self._dynSystem.getStateManager()

        Ig = np.diag(self._Ig)
        Iw = np.diag(np.array([self._Iw[0], self._Iw[1], self._Iw[1]]))

        gamma = stateMan.getStates(self._stateGimbalAngle)

        GG0 = coordinateTransformations.ROT3(gamma)

        BG = self._BG.dot(GG0.T)

        R_CoM_tilde = attitudeKinematics.getSkewSymmetrixMatrix(self._R_CoM)

        self._stateEffectorParent.addMassContribution(self._m)
        self._stateEffectorParent.addInertiaContribution(-self._m * R_CoM_tilde.dot(R_CoM_tilde) + BG.dot(Ig + Iw).dot(BG.T))

        return

    def computeLHS(self):

        stateMan = self._dynSystem.getStateManager()

        gamma = stateMan.getStates(self._stateGimbalAngle)

        zer = np.zeros((3,3))

        Iws = self._Iw[0]
        Iwt = self._Iw[1]
        Jg = self._Ig[2] + Iwt

        GG0 = coordinateTransformations.ROT3(gamma)

        BG = self._BG.dot(GG0.T)

        gs_hat = BG[:,0]
        gg_hat = BG[:,2]

        gs_outer = np.outer(gs_hat, gs_hat)
        gg_outer = np.outer(gg_hat, gg_hat)

        self._stateEffectorParent.addMassMatrixContribution(self._m * np.eye(3), zer, zer, - Iws * gs_outer - Jg * gg_outer)

        return

    def computeRWControlTorque(self):
        return 0.0

    def computeGimbalControlTorque(self):
        return 0.0


