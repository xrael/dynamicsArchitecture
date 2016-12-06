
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

    _controlForceManager = None # Object which is in charge of saving variables of interest

    def __init__(self, estimator, referenceComp, period):
        """

        :param hubEstimator: [estimator] Object that estimates the state.
        :param referenceComp: [referenceComputer] Object that computes reference trajectories.
        :param period: [double] 1/period is the frequency of the control loop in Hz.
        :param controlForces: [tuple] \
        :return:
        """
        self._estimator = estimator
        self._referenceComputer = referenceComp
        self._period = period
        self._lastTime = 0.0
        self._controlForceManager = stateManager.stateManager()
        return

    def start(self, t0):
        """
        Run this method to get the controller started before calling runControl()
        :return:
        """
        self._lastTime = t0 - self._period
        return

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

#---------------------------------------------------------------------------------------------------------------------#

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

    _setNullMotion = False
    _alpha = None
    _A = None
    _ke = 1.0
    _condNmbrDeadbandMin = 1.0
    _condNmbrDeadbandMax = 1.0
    _condNmbrDeadband = 1.0
    _Omega_prefered = None

    _ug = None
    _us = None

    _ugMax = 0.0
    _usMax = 0.0

    def __init__(self, estimator, vscmgs, referenceComp, period_outer_loop, period_inner_loop, K_sigma, P_omega, K_gamma_dot, K_Omega_dot, weights, mu_weights):
        """

        :param estimator:
        :param vscmgs:
        :param referenceComp:
        :param period_outer_loop:
        :param period_inner_loop:
        :param K_sigma:
        :param P_omega:
        :param K_gamma_dot:
        :param K_Omega_dot:
        :param weights:
        :param mu_weights:
        :return:
        """
        super(vscmgSteeringController, self).__init__(estimator, referenceComp, period_inner_loop)

        self._vscmgs = vscmgs
        self._nmbrVSCMG = len(vscmgs)

        self._setNullMotion = False
        self._A = None
        self._alpha = None
        self._ke = 1.0
        self._condNmbrDeadbandMin = 1.0
        self._condNmbrDeadbandMax = 1.0
        self._condNmbrDeadband = 1.0
        self._Omega_prefered = None

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

        self._lastTimeOuterLoop = 0.0
        self._periodOuterLoop = period_outer_loop

        self._controlForceManager.registerState('sigma_BR', 3)
        self._controlForceManager.registerState('w_BR_B', 3)
        self._controlForceManager.registerState('sigma_RN', 3)
        self._controlForceManager.registerState('w_RN_B', 3)
        self._controlForceManager.registerState('Lr', 3)
        self._controlForceManager.registerState('condition_number', 1)
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

    def configureNullMotion(self, alpha, ke, cond_nmbr_deadband_min, cond_nmbr_deadband_max, Omega_preferred, driveRW = True, driveCMG = True):
        """
        Configures the null motion algorithm to avoid CMG singularities.
        Check Schaub-Junkins.
        :param alpha:
        :param ke:
        :param cond_nmbr_deadband_min:
        :param cond_nmbr_deadband_max:
        :param Omega_preferred:
        :param driveRW:
        :param driveCMG:
        :return:
        """
        self._setNullMotion = True
        self._alpha = alpha
        self._ke = ke
        self._condNmbrDeadbandMin = cond_nmbr_deadband_min
        self._condNmbrDeadbandMax = cond_nmbr_deadband_max
        self._Omega_prefered = Omega_preferred
        if driveRW:
            a_RW = 1
        else:
            a_RW = 0

        if driveCMG:
            a_CMG = 1
        else:
            a_CMG = 0

        self._A = np.zeros((2*self._nmbrVSCMG,2*self._nmbrVSCMG))
        self._A[:self._nmbrVSCMG, :self._nmbrVSCMG] = a_RW * np.eye(self._nmbrVSCMG)
        self._A[self._nmbrVSCMG:, self._nmbrVSCMG:] = a_CMG * np.eye(self._nmbrVSCMG)

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

    def start(self, t0):
        """
        Method to be called before starting the simulation.
        :param t0:
        :return:
        """
        super(vscmgSteeringController, self).start(t0)
        self._lastTimeOuterLoop = t0 - self._periodOuterLoop
        self._condNmbrDeadband = self._condNmbrDeadbandMin
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

        (s, U, V, cond_nmbr) = self.computeConditionNumber(D)

        self._controlForceManager.setStates('condition_number', cond_nmbr)

        if self._setNullMotion:
            nu_dot_null = self.computeNullSpace(D, U, V, s, cond_nmbr, Q, self._Omega_prefered, w_RN_B)
            nu_dot = nu_dot + nu_dot_null

        Omega_dot_desired = nu_dot[:self._nmbrVSCMG]
        gamma_dot_desired = nu_dot[self._nmbrVSCMG:]

        # Omega_dot_desired = np.zeros(self._nmbrVSCMG) #D0.T.dot(np.linalg.inv(D0.dot(D0.T))).dot(Lr)
        # gamma_dot_desired = np.zeros(self._nmbrVSCMG)

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
            #gg_hat = BG[:,2]

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

    def computeConditionNumber(self, mat):
        U, s, V = np.linalg.svd(mat)

        V = V.T

        cond_nmbr = np.max(s)/np.min(s)

        return (s,U, V, cond_nmbr)


    def computeNullSpace(self, D, U, V, s, cond_nmbr, Q, Omega_preferred, w_RN_B):
        """
        Computes the null space desired values to avoid singularities continously.
        Check Schaub-Junkins.
        :param D1:
        :param Q:
        :param Omega_preferred:
        :return:
        """
        u1 = U[:,0]
        u3 = U[:,2]

        sigma_1 = np.max(s)
        sigma_3 = np.min(s)

        w_BN_B_est = self._estimator.get_w_BN_B_estimation()

        cond_nmbr_grad = np.zeros(self._nmbrVSCMG)
        delta_Omega = np.zeros(self._nmbrVSCMG)

        i = 0
        for vscmg in self._vscmgs:
            Omega_est = vscmg.getOmegaEstimate()

            Ig = vscmg.getGimbalInertia()
            Iw = vscmg.getWheelInertia()
            BG = vscmg.getBGmatrix()

            Iws = Iw[0,0]
            Iwt = Iw[1,1]

            Js = Ig[0,0] + Iws
            Jt = Ig[1,1] + Iwt

            gs_hat = BG[:,0]
            gt_hat = BG[:,1]

            ws = np.inner(gs_hat, w_BN_B_est)
            wt = np.inner(gt_hat, w_BN_B_est)

            chi = Js * (-(Omega_est + ws) * gs_hat + wt * gt_hat)
            xi = (Js - Jt) * (np.outer(gt_hat, gt_hat) - np.outer(gs_hat, gs_hat)).dot(w_RN_B)

            dsigma1_dgammai = np.inner(u1, chi + xi) * V[i,0]
            dsigma3_dgammai = np.inner(u3, chi + xi) * V[i,2]

            cond_nmbr_grad[i] = 1.0/sigma_3 * dsigma1_dgammai - sigma_1/sigma_3**2 * dsigma3_dgammai

            delta_Omega[i] = Omega_est - Omega_preferred[i]

            i += 1

        if cond_nmbr <= self._condNmbrDeadband:
            # If the condition number is small enough, just turn off this control but still try to steer towards Omega_preferred
            delta_gamma = np.zeros(self._nmbrVSCMG)
            #delta_Omega = np.zeros(self._nmbrVSCMG)
            self._condNmbrDeadband = self._condNmbrDeadbandMax
        else:
            delta_gamma = self._alpha * (cond_nmbr -1) * cond_nmbr_grad
            self._condNmbrDeadband = self._condNmbrDeadbandMin

        delta_eta = np.concatenate([delta_Omega, delta_gamma])
        I = np.eye(2*self._nmbrVSCMG)
        eta_dot_null = -self._ke * (I - Q.T.dot(np.linalg.inv(Q.dot(Q.T))).dot(Q)).dot(self._A).dot(delta_eta)

        return eta_dot_null

    def steeringTest(self,t):

        frq = 1.0/300

        omega_amp = np.array([50, 30, 20, 10]) # rpm

        gamma_amp = np.array([2,1,0.7,0.6])   # deg

        self._Omega_desired = (40  + np.sin(2*np.pi* frq * t) * omega_amp) * 2*np.pi/60
        self._Omega_dot_desired = 2*np.pi* frq *np.cos(2*np.pi* frq * t) * omega_amp * 2*np.pi/60

        self._gamma_dot_desired = (1 + np.sin(2*np.pi* frq * t) * gamma_amp) * np.pi/180
        self._gamma_ddot_desired = 2*np.pi* frq * np.cos(2*np.pi* frq * t) * gamma_amp * np.pi/180

        return

#---------------------------------------------------------------------------------------------------------------------#

class reactionWheelSteeringLawController(controller):

    _P = np.eye(3)              # Derivative gain
    _Ki = np.eye(3)             # Integral gain

    _K1_sigma = 1.0
    _K3_sigma = 1.0

    _w_BR_B_desired = None
    _w_BR_B_dot_desired = None
    _resetDerivative = True

    _z = None

    _Gs = None
    _torqueMappingMatrix = None

    _w_max = 1.0

    _nmbrRW = 0
    _rw = None

    _periodOuterLoop = 0.0
    _lastTimeOuterLoop = 0.0

    _us = None
    _usMax = 0.0

    def __init__(self, estimator, rws, referenceComp, period_outer_loop, period_inner_loop, P_omega, K_integral, K1_sigma, K3_sigma, w_max):
        """

        :param estimator:
        :param rws:
        :param referenceComp:
        :param period_outer_loop:
        :param period_inner_loop:
        :param P_omega:
        :param K_integral:
        :param K1_sigma:
        :param K3_sigma:
        :param w_max:
        :return:
        """
        super(reactionWheelSteeringLawController, self).__init__(estimator, referenceComp, period_inner_loop)

        self._rw = rws
        self._nmbrRW = len(rws)

        self._P = P_omega
        self._Ki = K_integral
        self._K1_sigma = K1_sigma
        self._K3_sigma = K3_sigma

        self._us = np.zeros(self._nmbrRW)

        self._usMax = 100.0

        self._w_max = w_max

        self._lastTimeOuterLoop = 0.0
        self._periodOuterLoop = period_outer_loop

        self._w_BR_B_desired = np.zeros(3)
        self._w_BR_B_dot_desired = np.zeros(3)
        self._resetDerivative = True

        self._z = np.zeros(3)

        self._controlForceManager.registerState('sigma_BR', 3)
        self._controlForceManager.registerState('w_BR_B', 3)
        self._controlForceManager.registerState('sigma_RN', 3)
        self._controlForceManager.registerState('w_RN_B', 3)
        self._controlForceManager.registerState('Lr', 3)
        self._controlForceManager.registerState('w_BR_B_desired', 3)
        self._controlForceManager.registerState('w_BR_B_dot_desired', 3)

        for rw in self._rw:
            rw_name = rw.getEffectorName()
            self._controlForceManager.registerState(rw_name + '_us', 1)

        return

    def setMaxWheelTorque(self, usMax):
        """
        Sets the maximum wheel torque the VSCMG can apply
        :param usMax: [double] torque in Nm.
        :return:
        """
        self._usMax = usMax
        return

    def start(self, t0):
        """
        Method to be called before starting the simulation.
        :param t0:
        :return:
        """
        super(reactionWheelSteeringLawController, self).start(t0)
        self._lastTimeOuterLoop = t0 - self._periodOuterLoop

        self._Gs = np.zeros((3, self._nmbrRW))
        i = 0
        for rw in self._rw:
            BW = rw.getBWmatrix()
            gs_hat = BW[:,0]
            self._Gs[:,i] = gs_hat
            i +=1

        self._torqueMappingMatrix = self._Gs.T.dot(np.linalg.inv(self._Gs.dot(self._Gs.T)))

        self._w_BR_B_desired = np.zeros(3)
        self._w_BR_B_dot_desired = np.zeros(3)
        self._resetDerivative = True

        self._z = np.zeros(3)

        return

    def computeControl(self, t):
        """

        :param t:
        :return:
        """
        if (t-self._lastTimeOuterLoop) >= self._periodOuterLoop:
            self._lastTimeOuterLoop = t

            self.computeSteeringLaw(t)

        Lc = np.zeros(3) # For now, we're not using external torques
        self.computeServoControlLaw(Lc, t)

        return

    def computeServoControlLaw(self, Lc, t):
        """
        Servo control law that keeps w_BB_desired -> 0, sigma_BB_desired -> 0. This law must run faster than the servo.
        :param Lc:
        :param t:
        :return:
        """

        sigma_BN_est = self._estimator.get_sigma_BN_estimation()
        w_BN_B_est = self._estimator.get_w_BN_B_estimation()
        BN_est = attitudeKinematics.mrp2dcm(sigma_BN_est)

        I_est = self._estimator.get_I_B_estimation()
        I_RW = np.copy(I_est)
        hs = np.zeros(self._nmbrRW)
        i = 0
        for rw in self._rw:
            Omega_est = rw.getOmegaEstimate()
            BW = rw.getBWmatrix()
            Iw = rw.getWheelInertia()
            Iws = Iw[0,0]
            gs_hat = BW[:,0]
            I_RW -= Iws * np.outer(gs_hat, gs_hat)
            hs[i] = Iws * (np.inner(gs_hat, w_BN_B_est) + Omega_est) # Angular momentum of the wheels
            i +=1

        RN = self._referenceComputer.get_RN()
        sigma_RN = self._referenceComputer.get_sigma_RN()
        w_RN_R = self._referenceComputer.get_w_RN_R()
        w_RN_R_dot = self._referenceComputer.get_w_RN_R_dot()

        w_RN_B = BN_est.dot(RN.T.dot(w_RN_R))
        w_RN_B_dot = BN_est.dot(RN.T.dot(w_RN_R_dot))

        sigma_BR = attitudeKinematics.computeErrorMRP(sigma_BN_est, sigma_RN)
        w_BR_B = w_BN_B_est - w_RN_B

        # _w_BR_B_desired is set by the steering law
        w_BN_desired = self._w_BR_B_desired + w_RN_B

        w_BB_desired = w_BN_B_est - w_BN_desired

        # Integral term
        self._z += w_BB_desired * self._period

        Lr = self._P.dot(w_BB_desired) + self._Ki.dot(self._z) \
             - attitudeKinematics.getSkewSymmetrixMatrix(w_BN_desired).dot(I_RW.dot(w_BN_B_est) + self._Gs.dot(hs)) \
             - I_RW.dot(self._w_BR_B_dot_desired + w_RN_B_dot - np.cross(w_BN_B_est, w_RN_B)) + Lc

        self._us = self._torqueMappingMatrix.dot(Lr)

        i = 0
        for rw in self._rw:
            rw_name = rw.getEffectorName()

            if np.abs(self._us[i]) > self._usMax:
                self._us[i] = np.sign(self._us[i]) * self._usMax
            rw.setWheelTorqueCommand(self._us[i])
            self._controlForceManager.setStates(rw_name + '_us', self._us[i])
            i += 1

        self._controlForceManager.setStates('sigma_BR', sigma_BR)
        self._controlForceManager.setStates('w_BR_B', w_BR_B)

        self._controlForceManager.setStates('sigma_RN', sigma_RN)
        self._controlForceManager.setStates('w_RN_B', w_RN_B)

        self._controlForceManager.setStates('Lr', Lr)

        return

    def computeSteeringLaw(self, t):
        """
        Steering law that computes w_BR_desired in the B frame.
        :param f:
        :param t:
        :return:
        """

        (f, offset) = self.computeFfunction()

        w_BR_B_desired_last = np.copy(self._w_BR_B_desired)
        self._w_BR_B_desired = -2*self._w_max/np.pi * np.arctan((self._K1_sigma * f + self._K3_sigma * f**3) * np.pi/(2*self._w_max)) + offset

        # Compute the derivative numerically
        if self._resetDerivative:
            self._w_BR_B_dot_desired = np.zeros(3)
            self._resetDerivative = False
        else:
            self._w_BR_B_dot_desired = (self._w_BR_B_desired - w_BR_B_desired_last)/self._periodOuterLoop

        self._controlForceManager.setStates('w_BR_B_desired', self._w_BR_B_desired)
        self._controlForceManager.setStates('w_BR_B_dot_desired', self._w_BR_B_dot_desired)

        return


    def computeFfunction(self):

        sigma_BN_est = self._estimator.get_sigma_BN_estimation()
        sigma_RN = self._referenceComputer.get_sigma_RN()
        sigma_BR = attitudeKinematics.computeErrorMRP(sigma_BN_est, sigma_RN)
        offset = np.array([0,0,0])

        return (sigma_BR, offset)

#---------------------------------------------------------------------------------------------------------------------#

class constrainedSteeringLawController(reactionWheelSteeringLawController):
    """

    """

    _constraints = None

    def __init__(self, estimator, rws, referenceComp, period_outer_loop, period_inner_loop, P_omega, K_integral, K1_sigma, K3_sigma, w_max):
        """

        :param estimator:
        :param rws:
        :param referenceComp:
        :param period_outer_loop:
        :param period_inner_loop:
        :param P_omega:
        :param K_integral:
        :param K1_sigma:
        :param K3_sigma:
        :param w_max:
        :return:
        """
        super(constrainedSteeringLawController, self).__init__(estimator, rws, referenceComp, period_outer_loop, period_inner_loop, P_omega, K_integral, K1_sigma, K3_sigma, w_max)

        self._constraints = list()

        self._controlForceManager.registerState('V_lyapunov', 1)
        self._controlForceManager.registerState('V_dot_lyapunov', 1)
        self._controlForceManager.registerState('det_Q', 1)
        self._controlForceManager.registerState('Vect', 3)
        self._controlForceManager.registerState('Uvect', 3)
        self._controlForceManager.registerState('offset_norm', 1)

        return


    def addConstraint(self, x_N, y_B, angle, name, angle_thres_min, angle_thres_max):
        cons = constraint(x_N, y_B, angle, name, angle_thres_min, angle_thres_max)
        self._constraints.append(cons)
        self._controlForceManager.registerState(cons.getName() + '_y_N', 3)
        self._controlForceManager.registerState(cons.getName() + '_theta', 1)
        self._controlForceManager.registerState(cons.getName() + '_constraint', 1)
        return

    def computeFfunction(self):

        sigma_BN_est = self._estimator.get_sigma_BN_estimation()
        BN_est = attitudeKinematics.mrp2dcm(sigma_BN_est)
        w_BN_est = self._estimator.get_w_BN_B_estimation()

        sigma_RN_est = self._referenceComputer.get_sigma_RN()
        RN_est = self._referenceComputer.get_RN()
        w_RN_R_est = self._referenceComputer.get_w_RN_R()

        sigma_BR_est = attitudeKinematics.computeErrorMRP(sigma_BN_est, sigma_RN_est)

        w_RN_B_est = BN_est.dot(RN_est.T.dot(w_RN_R_est))

        w_BR_est = w_BN_est - w_RN_B_est

        #B_BR = attitudeKinematics.getBmatrix(sigma_BR_est) # Matrix [B(sigma_BR)]
        #sigma_BN_2 = np.inner(sigma_BN_est, sigma_BN_est)
        sigma_BR_2 = np.inner(sigma_BR_est, sigma_BR_est)

        #dsigma_BN_dsigma_BR = attitudeKinematics.computeMRPsensitivity(sigma_BN_est, sigma_RN_est)

        #gamma = constraint.getConstraintMaximumValue() * np.e

        v_vec = np.zeros(3)
        u_vec = np.zeros(3)
        check_vec = np.zeros(3)
        V_lyapunov = 0.0
        V_dot_lyapunov = 0.0
        #Q = np.zeros((3,3))
        nmbr_constr = len(self._constraints)
        for constr in self._constraints:
            (change, useConstraint) = constr.computeConstraint(sigma_BN_est)
            c = constr.getConstraint()
            #grad_c = constr.getConstraintGradient()
            c_dot_coeff = constr.getConstraintTimeDerivativeCoefficient()
            yN = constr.getBoresight()
            angle = constr.getAngle()
            alpha = constr.getAlpha()

            self._controlForceManager.setStates(constr.getName() + '_y_N', yN)
            self._controlForceManager.setStates(constr.getName() + '_theta', angle)
            self._controlForceManager.setStates(constr.getName() + '_constraint', c)

            if change:
                # if there's a change in the law, don't approximate the angular velocity derivative numerically to avoid spikes
                self._resetDerivative = True

            if useConstraint:
                #Vect += -1.0/nmbr_constr * (np.log(-c/gamma) * sigma_BR_est + 0.5 * np.log(1 + sigma_BR_2) * B_BR.T.dot(dsigma_BN_dsigma_BR.T).dot(grad_c/c))
                v_vec += -1.0/nmbr_constr * (np.log(-c/alpha) * sigma_BR_est + 2 * np.log(1 + sigma_BR_2) * c_dot_coeff/c)
                check_vec += c_dot_coeff/c
                u_vec += -2.0/nmbr_constr * np.log(1 + sigma_BR_2) * c_dot_coeff/c
            else:
                v_vec += 1.0/nmbr_constr * sigma_BR_est
                u_vec += np.array([0,0,0])

            if c >= 0:
                print constr.getName() + ': ' + str(c)

            #Vect += np.log(-c/8) * sigma_BN_est + 0.5 * np.log(1 + sigma_BN_2) * B.T.dot(grad_c/c)

            # if sigma_BN_2 > 0.001:
            #     Q += np.log(-c/gamma) * np.eye(3) + 0.5 * np.log(1+sigma_BN_2)/sigma_BN_2 * B.T.dot(np.outer(grad_c/c, sigma_BN_est))
            # else:
            #     Q += np.log(-c/gamma) * np.eye(3) + 0.5 * 1.0 * B.T.dot(np.outer(grad_c/c, sigma_BN_est))
            #
            V_lyapunov += -2.0/nmbr_constr * np.log(1 + sigma_BR_2) * np.log(-c/alpha)

            V_dot_lyapunov += -1.0/nmbr_constr * np.log(-c/alpha) * np.inner(sigma_BR_est, w_BR_est) \
                              - 2.0/nmbr_constr * np.log(1 + sigma_BR_2) * c_dot_coeff.dot(w_BN_est)/c
        # end for

        if np.linalg.norm(v_vec) < 0.01 and sigma_BR_2 > 0.1**2:
            v_vec = 0.05*np.array([sigma_BR_est[2], 0, -sigma_BR_est[0]])

        v2 = np.inner(v_vec, v_vec)
        if  v2 > 0.001:
            #offset = -2*self._w_max/np.pi * np.arctan(np.outer(Vect, Uvect).dot(w_RN_B_est)/Vect_2 * np.pi/(2*self._w_max))
            offset = -np.outer(v_vec, u_vec).dot(w_RN_B_est)/v2
        else:
            offset = np.array([0,0,0])



        #     F = constr.getF()
        #     g = constr.getg()
        #     d = constr.getd()
        #
        #     if sigma_BN_2 < 0.01:
        #         M_1 = 2*F * sigma_BN_2 + (1.0 - sigma_BN_2) * np.outer(sigma_BN_est, g) - 2*np.outer(g,sigma_BN_est) * sigma_BN_2 + 4*d*np.outer(sigma_BN_est, sigma_BN_est) * sigma_BN_2
        #         matr_1 = np.log(-c/8.0)*np.eye(3) + 0.5 * M_1.dot(B)/c
        #
        #     else:
        #         M_1 = 2*F + (1.0/sigma_BN_2 - 1) * np.outer(sigma_BN_est, g) - 2*np.outer(g,sigma_BN_est) + 4*d*np.outer(sigma_BN_est, sigma_BN_est)
        #         matr_1 = np.log(-c/8.0)*np.eye(3) + 0.5*np.log(1+sigma_BN_2)/c * M_1.dot(B)
        #
        #     Q += matr_1
        #
        #
        # Vect = Q.T.dot(sigma_BN_est)

        #Vect = -Vect # There's a (-) sign in the steering law

        print v_vec


        # det_Q = np.linalg.det(Q)
        #
        # if np.abs(det_Q) < 1:
        #     a=1


        # self._controlForceManager.setStates('det_Q', det_Q)
        self._controlForceManager.setStates('V_lyapunov', V_lyapunov)
        self._controlForceManager.setStates('V_dot_lyapunov', V_dot_lyapunov)

        self._controlForceManager.setStates('Vect', v_vec)
        self._controlForceManager.setStates('Uvect', u_vec)
        self._controlForceManager.setStates('offset_norm', np.linalg.norm(offset))

        #print 'offset', offset

        return (v_vec, offset)

#----------------------------------------------------------------------------------------------------------------------#

class constraint:

    _xN = None
    _yB = None
    _yN = None
    _angle_min = 0.0
    _angle = 0.0

    _angle_thres_min = 0.0
    _angle_thres_max = 0.0

    _c = 0.0
    #_grad_c = 0.0
    _c_dot_coeff = 0.0

    _useConstraint = False

    _alpha = np.e

    # _F = None
    # _g = None
    # _d = None

    _name = ''

    def __init__(self, x_N, y_B, angle_min, name, angle_thres_min, angle_thres_max):
        """

        :param x_N: [1-dimensional numpy array] Vector pointing towards the constraint in inertial frame.
        :param y_B: [1-dimensional numpy array] Vector pointing in the boresight direction of the instrument to protect in body frame.
        :param angle_min: [double] angle of protection in rad.
        :param angle_thres_min: [double] The control has to be turned on when the angle is less than this threshold. Depends on the dynamics.
        :param angle_thres_max: [double] The control has to be turned off when the angle is larger than this threshold. Depends on the dynamics.
        :return:
        """
        self._xN = x_N
        self._yB = y_B
        self._angle_min = angle_min

        # self._F = 4*(np.outer(x_N, y_B) + np.outer(y_B,x_N)) - (6*np.inner(x_N,y_B) + 2*np.cos(angle_min))*np.eye(3)
        # self._g = 4*np.cross(y_B,x_N)
        # self._d = np.inner(x_N,y_B) - np.cos(angle_min)

        self._name = name

        self._yN = np.zeros(3)
        self._angle = 0.0
        self._c = 0.0
        # self._grad_c = 0.0
        self._c_dot_coeff = 0.0

        self._angle_thres_min = angle_thres_min
        self._angle_thres_max = angle_thres_max

        self._useConstraint = False
        self._alpha = 2*np.e

        return

    @classmethod
    def getConstraintMaximumValue(cls):
        return 2.0

    def getxN(self):
        return self._xN

    def getyB(self):
        return self._yB

    def getBoresight(self):
        return self._yN

    def getAngle(self):
        return self._angle

    def getMinimumAngle(self):
        return self._angle_min

    def getAngleThresholdMin(self):
        return self._angle_thres_min

    def getAngleThresholdMax(self):
        return self._angle_thres_max

    def getAlpha(self):
        return self._alpha

    def getConstraint(self):
        return self._c

    # def getConstraintGradient(self):
    #     return self._grad_c

    def getConstraintTimeDerivativeCoefficient(self):
        return self._c_dot_coeff

    # def getF(self):
    #     return self._F
    #
    # def getg(self):
    #     return self._g
    #
    # def getd(self):
    #     return self._d

    def getName(self):
        return self._name

    def computeConstraint(self, sigma_BN):
        """
        Computes the constraint surface c(sigma), its gradient, the angle and the boresight vector.
        The constraint is just c(sigma) < 0
        :param sigma_BN:
        :return:
        """
        #sigma_BN_2 = np.inner(sigma_BN, sigma_BN)
        BN = attitudeKinematics.mrp2dcm(sigma_BN)

        # F = self._F
        # g = self._g
        # d = self._d

        # self._c = np.inner(sigma_BN, F.dot(sigma_BN)) \
        #     + np.inner(g, sigma_BN * (1 - sigma_BN_2)) + d * (1 + sigma_BN_2**2)

        # self._grad_c = 2*F.dot(sigma_BN) + g * (1 - sigma_BN_2) - 2 * np.inner(g, sigma_BN) * sigma_BN + 4 * d * sigma_BN_2 * sigma_BN

        self._c = np.inner(BN.dot(self._xN), self._yB) - np.cos(self._angle_min)

        self._c_dot_coeff = np.cross(self._yB, BN.dot(self._xN))

        self._yN = BN.T.dot(self._yB)
        self._angle = np.arccos(np.inner(self._yN, self._xN))

        change = False
        # self._useConstraint = True
        if self._angle < self._angle_thres_min:
            # Use the constraint control
            if self._useConstraint == False:
                change = True
                self._alpha = np.abs(np.cos(self._angle_thres_max) - np.cos(self._angle_min)) * np.e
            self._useConstraint = True
        elif self._angle > self._angle_thres_max:
            # Do not use the constraint control
            if self._useConstraint == True:
                change = True
                #self._alpha = np.abs(np.cos(self._angle_thres_min) - np.cos(self._angle_min)) * np.e
            self._useConstraint = False
        # else: use whatever you're using so far

        return (change, self._useConstraint)
