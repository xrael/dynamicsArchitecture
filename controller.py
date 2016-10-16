
from abc import ABCMeta, abstractmethod

import numpy as np
import attitudeKinematics
import stateManager

class controller:
    """
    This is an abstract class for all controllers.
    """

    __metaclass__ = ABCMeta

    _estimator = None
    _referenceComputer = None
    _period = 0.0

    _controlForceManager = None

    def __init__(self, estimator, referenceComp, period, t0):
        """

        :param hubEstimator: [estimator] Object that estimates the state.
        :param referenceComp: [referenceComputer] Object that computes reference trajectories.
        :param period: [double] 1/period is the frequency of the control loop in Hz.
        :param t0:
        :param controlForces: [tuple] \
        :return:
        """
        self._estimator = estimator
        self._referenceComputer = referenceComp
        self._period = period
        self._lastTime = t0 - period
        self._controlForceManager = stateManager.stateManager()
        return

    @abstractmethod
    def start(self):
        """
        Run this method to get the controller started before calling runControl()
        :return:
        """
        pass


    def runControl(self, t):
        """
        This method should run the control algorithm running at a certain frequency.
        :param t:
        :param dt:
        :return:
        """
        if (t-self._lastTime) >= self._period:
            self._lastTime = t
            self.computeControl(t)
        return

    @abstractmethod
    def computeControl(self, t):
        """
        Method in charge of doing the computations.
        :return:
        """
        pass

    def createControlForceHistory(self, number_of_elements):
        """
        This method simply creates and holds a matrix to hold the state history.
        Perhaps it's not the best place for doing this.
        :param number_of_elements:
        :return:
        """
        self._controlForceManager.createStateHistory(number_of_elements)

        return

    def getControlForceHistory(self):
        return self._controlForceManager.getStateHistory()

    def saveControlForces(self, element):
        """
        Once the state history is created, this method saves a state at a given time in a certain position of the state history.
        :param element:
        :return:
        """
        self._controlForceManager.saveState(element)
        return

class vscmgSteeringController(controller):
    """
    Implements the 2-loop controller used in Schaub-Junkins.
    """

    _K = 1.0                    # Proportional gain
    _P = np.eye(3)              # Derivative gain

    _K_gamma_dot = None         # gamma-dot steering-law gain
    _K_Omega = None             # Omega-dot steering-law gain

    _nmbrVSCMG = 0
    _vscmgs = None

    _weights = None

    _Omega_desired = None
    _Omega_dot_desired = None
    _gamma_dot_desired = None
    _gamma_ddot_desired = None

    _periodOuterLoop = 0.0
    _lastTimeOuterLoop = 0.0

    _ug = None
    _us = None

    _ugMax = 0.0
    _usMax = 0.0

    def __init__(self, estimator, vscmgs, referenceComp, period_outer_loop, period_inner_loop, t0, K_sigma, P_omega, K_gamma_dot, K_Omega_dot, weights, mu_weights):
        """

        :param estimator:
        :param vscmgs:
        :param referenceComp:
        :param period_outer_loop:
        :param period_inner_loop:
        :param t0:
        :param K_sigma:
        :param P_omega:
        :param K_gamma_dot:
        :param K_Omega_dot:
        :param weights:
        :param mu_weights:
        :return:
        """
        super(vscmgSteeringController, self).__init__(estimator, referenceComp, period_inner_loop, t0)

        self._vscmgs = vscmgs
        self._nmbrVSCMG = len(vscmgs)

        self._weights = weights
        self._mu = mu_weights

        self._K = K_sigma
        self._P = P_omega
        self._K_gamma_dot = K_gamma_dot
        self._K_Omega = K_Omega_dot

        self._Omega_desired = np.zeros(self._nmbrVSCMG)
        self._Omega_dot_desired = np.zeros(self._nmbrVSCMG)
        self._gamma_dot_desired = np.zeros(self._nmbrVSCMG)
        self._gamma_ddot_desired = np.zeros(self._nmbrVSCMG)

        self._ug = np.zeros(self._nmbrVSCMG)
        self._us = np.zeros(self._nmbrVSCMG)

        self._ugMax = 100.0
        self._usMax = 100.0

        self._lastTimeOuterLoop = t0 - period_outer_loop
        self._periodOuterLoop = period_outer_loop

        self._controlForceManager.registerState('sigma_BR', 3)
        self._controlForceManager.registerState('w_BR_B', 3)
        self._controlForceManager.registerState('sigma_RN', 3)
        self._controlForceManager.registerState('w_RN_B', 3)
        self._controlForceManager.registerState('Lr', 3)
        for vscmg in self._vscmgs:
            vscmg_name = vscmg.getEffectorName()
            self._controlForceManager.registerState(vscmg_name + '_ug', 1)
            self._controlForceManager.registerState(vscmg_name + '_us', 1)
            self._controlForceManager.registerState(vscmg_name + '_Omega_desired', 1)
            self._controlForceManager.registerState(vscmg_name + '_Omega_dot_desired', 1)
            self._controlForceManager.registerState(vscmg_name + '_gamma_dot_desired', 1)
            self._controlForceManager.registerState(vscmg_name + '_gamma_ddot_desired', 1)
            self._controlForceManager.registerState(vscmg_name + '_delta', 1)

        return

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

    def start(self):
        """
        Method to be called before starting the simulation.
        :return:
        """
        i = 0
        for vscmg in self._vscmgs:
            Omega_est = vscmg.getOmegaEstimate()
            gamma_dot_est = vscmg.getGammaDotEstimate()
            self._Omega_desired[i] = Omega_est
            self._gamma_dot_desired[i] = gamma_dot_est
            i += 1

        return

    def computeControl(self, t):
        """

        :param t:
        :return:
        """
        if (t-self._lastTimeOuterLoop) >= self._periodOuterLoop:
            self._lastTimeOuterLoop = t
            Lc = np.zeros(3) # For now, we're not using external torques
            self.computeVelocityBasedSteeringLaw(Lc,t)

        #self.steeringTest(t)

        self.computeAccelerationBasedSteeringLaw(t)

        return

    def computeVelocityBasedSteeringLaw(self, Lc,t):

        sigma_BN_est = self._estimator.get_sigma_BN_estimation()
        w_BN_B_est = self._estimator.get_w_BN_B_estimation()
        BN_est = attitudeKinematics.mrp2dcm(sigma_BN_est)

        I_est = self._estimator.get_I_B_estimation()

        RN = self._referenceComputer.get_RN()
        sigma_RN = self._referenceComputer.get_sigma_RN()
        w_RN_R = self._referenceComputer.get_w_RN_R()
        w_RN_R_dot = self._referenceComputer.get_w_RN_R_dot()

        w_RN_B = BN_est.dot(RN.T.dot(w_RN_R))
        w_RN_B_dot = BN_est.dot(RN.T.dot(w_RN_R_dot))

        sigma_BR = attitudeKinematics.computeErrorMRP(sigma_BN_est, sigma_RN)
        w_BR_B = w_BN_B_est - w_RN_B

        self._controlForceManager.setStates('sigma_BR', sigma_BR)
        self._controlForceManager.setStates('w_BR_B', w_BR_B)

        self._controlForceManager.setStates('sigma_RN', sigma_RN)
        self._controlForceManager.setStates('w_RN_B', w_RN_B)

        Lr = np.zeros(3)
        D0 = np.zeros((3, self._nmbrVSCMG))
        D1 = np.zeros((3, self._nmbrVSCMG))
        D4 = np.zeros((3, self._nmbrVSCMG))
        h_nominal = np.zeros(self._nmbrVSCMG)
        i=0
        for vscmg in self._vscmgs:

            Omega_est = vscmg.getOmegaEstimate()
            gamma_est = vscmg.getGammaEstimate()

            GG_0 = attitudeKinematics.ROT3(gamma_est)
            BG_0 = vscmg.getBG0matrix()
            BG = BG_0.dot(GG_0.T)

            h_nominal[i] = vscmg.getNominalRWangularMomentum()

            Ig = vscmg.getGimbalInertia()
            Iw = vscmg.getWheelInertia()

            Iws = Iw[0,0]
            Iwt = Iw[1,1]

            Js = Ig[0,0] + Iws
            Jt = Ig[1,1] + Iwt

            gs_hat = BG[:,0]
            gt_hat = BG[:,1]
            gg_hat = BG[:,2]

            ws = np.inner(gs_hat, w_BN_B_est)
            wt = np.inner(gt_hat, w_BN_B_est)
            wg = np.inner(gg_hat, w_BN_B_est)

            Lr += -Iws * (wg * gt_hat - wt * gg_hat) * Omega_est

            D0[:,i] = Js * gs_hat
            D1[:,i] = Js * ((Omega_est + ws/2) * gt_hat + wt/2 * gs_hat)
            D4[:,i] = 0.5 * (Js-Jt) * (np.outer(gs_hat, gt_hat).dot(w_RN_B) + np.outer(gt_hat, gs_hat).dot(w_RN_B))

            # D0[:,i] = Js * gs_hat
            # D1[:,i] = Js * (Omega_est + ws) * gt_hat

            i += 1

        Lr += self._K * sigma_BR + self._P.dot(w_BR_B) + Lc - np.cross(w_BN_B_est, I_est.dot(w_BN_B_est)) - I_est.dot(w_RN_B_dot - np.cross(w_BN_B_est, w_RN_B))
        self._controlForceManager.setStates('Lr', Lr)

        D = D1 + D4
        Q = np.zeros((3, 2*self._nmbrVSCMG))

        Q[:, :self._nmbrVSCMG] = D0
        Q[:, self._nmbrVSCMG:] = D

        det_D2 = np.linalg.det(D.dot(D.T))

        delta = det_D2/h_nominal**2

        i = 0
        for vscmg in self._vscmgs:
            vscmg_name = vscmg.getEffectorName()
            self._controlForceManager.setStates(vscmg_name + '_delta', delta[i])
            i+=1

        print 'delta', delta

        W = np.copy(self._weights)
        W[:self._nmbrVSCMG] = self._weights[:self._nmbrVSCMG] * np.exp(-self._mu * delta)

        print 'W', W

        W = np.diag(W)

        nu_dot = W.dot(Q.T).dot(np.linalg.inv(Q.dot(W).dot(Q.T))).dot(Lr)

        Omega_dot_desired = nu_dot[:self._nmbrVSCMG]
        gamma_dot_desired = nu_dot[self._nmbrVSCMG:]

        # Omega_dot_desired = np.zeros(self._nmbrVSCMG) #D0.T.dot(np.linalg.inv(D0.dot(D0.T))).dot(Lr)
        # gamma_dot_desired = np.zeros(self._nmbrVSCMG)

        if gamma_dot_desired.any() >np.deg2rad(30):
            a=1

        # Numerical integration
        self._Omega_desired = self._Omega_desired + Omega_dot_desired * self._periodOuterLoop#self._periodOuterLoop

        # Numerical derivative
        self._gamma_ddot_desired = (gamma_dot_desired - self._gamma_dot_desired)/self._periodOuterLoop

        self._Omega_dot_desired = Omega_dot_desired
        self._gamma_dot_desired = gamma_dot_desired

        # stateMan = self._vscmgs[0]._dynSystem.getStateManager()
        # i = 0
        # for vscmg in self._vscmgs:
        #     vscmg_name = vscmg.getEffectorName()
        #     #stateMan.setStates(vscmg._stateGimbalRateName, gamma_dot_desired[i])
        #     #stateMan.setStates(self._vscmgs[i]._stateRWrateName, self._Omega_desired[i])
        #     stateMan.setStateDerivatives(vscmg._stateGimbalAngleName, gamma_dot_desired[i])
        #     stateMan.setStateDerivatives(vscmg._stateGimbalRateName, self._gamma_ddot_desired[i])
        #     stateMan.setStateDerivatives(vscmg._stateRWrateName, self._Omega_dot_desired[i])
        #     self._controlForceManager.setStates(vscmg_name + '_Omega_desired', self._Omega_desired[i])
        #     self._controlForceManager.setStates(vscmg_name + '_Omega_dot_desired', self._Omega_dot_desired[i])
        #     self._controlForceManager.setStates(vscmg_name + '_gamma_dot_desired', self._gamma_dot_desired[i])
        #     self._controlForceManager.setStates(vscmg_name + '_gamma_ddot_desired', self._gamma_ddot_desired[i])
        #     # self._ug[i] = Jg*(self._gamma_ddot_desired[i] - self._K_gamma_dot[i] * delta_gamma_dot) - (Js - Jt) * ws * wt - Iws * wt * Omega_est
        #     # self._us[i] = Iws*(self._Omega_dot_desired[i] + gamma_dot_est * wt - self._K_Omega[i] * delta_Omega)
        #     #
        #     # self._controlForceManager.setStates(vscmg_name + '_ug', self._ug[i])
        #     # self._controlForceManager.setStates(vscmg_name + '_us', self._us[i])
        #     i += 1

        return

    def computeAccelerationBasedSteeringLaw(self,t):

        w_BN_B_est = self._estimator.get_w_BN_B_estimation()

        i = 0
        for vscmg in self._vscmgs:

            gamma_dot_est = vscmg.getGammaDotEstimate()
            Omega_est = vscmg.getOmegaEstimate()

            Ig = vscmg.getGimbalInertia()
            Iw = vscmg.getWheelInertia()
            BG = vscmg.getBGmatrix()

            Iws = Iw[0,0]
            Iwt = Iw[1,1]

            Js = Ig[0,0] + Iws
            Jt = Ig[1,1] + Iwt
            Jg = Ig[2,2] + Iwt

            gs_hat = BG[:,0]
            gt_hat = BG[:,1]
            gg_hat = BG[:,2]

            ws = np.inner(gs_hat, w_BN_B_est)
            wt = np.inner(gt_hat, w_BN_B_est)

            delta_Omega = Omega_est - self._Omega_desired[i]
            delta_gamma_dot = gamma_dot_est - self._gamma_dot_desired[i]

            # print 'dOmega', delta_Omega
            # print 'd_gamma_dot', delta_gamma_dot

            self._ug[i] = Jg*(self._gamma_ddot_desired[i] - self._K_gamma_dot[i] * delta_gamma_dot) - (Js - Jt) * ws * wt - Iws * wt * Omega_est
            self._us[i] = Iws*(self._Omega_dot_desired[i] + gamma_dot_est * wt - self._K_Omega[i] * delta_Omega)


            # Omega_dot_est = vscmg.getOmegaDotEstimate()
            # gamma_ddot_est = vscmg.getGammaDDotEstimate()
            # self._ug[i] = Jg*(gamma_ddot_est + np.inner(gg_hat, w_BN_B_dot_est)) - (Js - Jt) * ws * wt - Iws * wt * Omega_est
            # self._us[i] = Iws*(Omega_dot_est + np.inner(gs_hat, w_BN_B_dot_est) + gamma_dot_est * wt)




            if np.abs(self._ug[i]) > self._ugMax:
                self._ug[i] = np.sign(self._ug[i]) * self._ugMax

            if np.abs(self._us[i]) > self._usMax:
                self._us[i] = np.sign(self._us[i]) * self._usMax

            vscmg_name = vscmg.getEffectorName()
            self._controlForceManager.setStates(vscmg_name + '_Omega_desired', self._Omega_desired[i])
            self._controlForceManager.setStates(vscmg_name + '_Omega_dot_desired', self._Omega_dot_desired[i])
            self._controlForceManager.setStates(vscmg_name + '_gamma_dot_desired', self._gamma_dot_desired[i])
            self._controlForceManager.setStates(vscmg_name + '_gamma_ddot_desired', self._gamma_ddot_desired[i])
            self._controlForceManager.setStates(vscmg_name + '_ug', self._ug[i])
            self._controlForceManager.setStates(vscmg_name + '_us', self._us[i])

            vscmg.setGimbalTorqueCommand(self._ug[i])
            vscmg.setWheelTorqueCommand(self._us[i])

            i += 1

        return


    def steeringTest(self,t):

        frq = 1.0/300

        omega_amp = np.array([50, 30, 20, 10]) # rpm

        gamma_amp = np.array([2,1,0.7,0.6])   # deg

        self._Omega_desired = (40  + np.sin(2*np.pi* frq * t) * omega_amp) * 2*np.pi/60
        self._Omega_dot_desired = 2*np.pi* frq *np.cos(2*np.pi* frq * t) * omega_amp * 2*np.pi/60

        self._gamma_dot_desired = (1 + np.sin(2*np.pi* frq * t) * gamma_amp) * np.pi/180
        self._gamma_ddot_desired = 2*np.pi* frq * np.cos(2*np.pi* frq * t) * gamma_amp * np.pi/180

        return
