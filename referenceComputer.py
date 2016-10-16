
from abc import ABCMeta, abstractmethod
import numpy as np
import attitudeKinematics
import coordinateTransformations
import stateManager

class referenceComputer:
    """
    Computes a reference trajectory relative to inertial frame.
    """

    __metaclass__ = ABCMeta

    _RN = None              # DCM of the reference frame relative to inertial
    _sigma_RN = None        # MRP of the reference frame relative to inertial
    _w_RN_R = None          # [rad/sec] Angular velocity of the reference frame relative to inertial expressed in the reference frame
    _w_RN_R_dot = None      # [rad/sec^2] Angular acceleration of the reference frame relative to inertial expressed in the reference frame

    _period = 0.0           # [sec] The estimator process should run every _period seconds
    _lastTime = 0.0

    _referenceManager = None

    def __init__(self, period, t0):
        self._period = period
        self._lastTime = t0 - period
        self._referenceManager = stateManager.stateManager()

        self._referenceManager.registerState('sigma_RN', 3)
        self._referenceManager.registerState('w_RN_R', 3)
        self._referenceManager.registerState('w_RN_R_dot', 3)
        return

    @abstractmethod
    def start(self):
        """
        Run this method to get the reference computer started before calling runReferenceComputing()
        :return:
        """
        pass

    def runReferenceComputing(self, t):
        """
        This method should trigger the reference motion computing process at a certain frequency.
        :param t:
        :return:
        """
        if (t-self._lastTime) >= self._period:
            self._lastTime = t
            self.computeReference(t)

            self._referenceManager.setStates('sigma_RN', self._sigma_RN)
            self._referenceManager.setStates('w_RN_R', self._w_RN_R)
            self._referenceManager.setStates('w_RN_R_dot', self._w_RN_R_dot)

        return

    @abstractmethod
    def computeReference(self, t):
        """
        This is the method that is to do the work computing sigma_RN, RN, w_RN_R, w_RN_R_dot
        :return:
        """
        pass

    def get_sigma_RN(self):
        return self._sigma_RN

    def get_RN(self):
        return self._RN

    def get_w_RN_R(self):
        return self._w_RN_R

    def get_w_RN_R_dot(self):
        return self._w_RN_R_dot

    def createReferenceHistory(self, number_of_elements):
        """
        This method simply creates and holds a matrix to hold the reference history.
        Perhaps it's not the best place for doing this.
        :param number_of_elements:
        :return:
        """
        self._referenceManager.createStateHistory(number_of_elements)

        return

    def getReferenceHistory(self):
        return self._referenceManager.getStateHistory()

    def saveReference(self, element):
        """
        Once the state history is created, this method saves a state at a given time in a certain position of the state history.
        :param element:
        :return:
        """
        self._referenceManager.saveState(element)
        return


class regulationReference(referenceComputer):
    """
    Implements a simple regulation problem:
    """

    def __init__(self, period, t0):
        super(regulationReference, self).__init__(period, t0)

        return

    def setPoint(self, sigma_RN):
        self._sigma_RN = sigma_RN
        self._RN = attitudeKinematics.mrp2dcm(sigma_RN)
        self._w_RN_R = np.zeros(3)
        self._w_RN_R_dot = np.zeros(3)
        return

    def start(self):
        return

    def computeReference(self,t):
        # Nothing to compute!
        return

class nadirPointingReference(referenceComputer):
    """
    For circular orbits
    """

    _gamma = 7.2925e-05          # [rad/sec] Rotation rate

    _raan = 0.0
    _inc = 0.0
    _a = 0.0
    _meanAnomaly = 0.0
    _mu = 0.0
    _r_HN_H = np.zeros(3)

    def __init__(self, period, t0, raan, inc, a, mu):
        super(nadirPointingReference, self).__init__(period, t0)
        self._raan = raan
        self._inc = inc
        self._a = a
        self._mu = mu
        self._meanAnomaly = np.sqrt(mu/a**3)
        self._r_HN_H = np.array([a, 0.0, 0.0])                    # Satellite position in Hill Frame
        return

    def start(self):
        return

    def computeReference(self,t):

        EN = coordinateTransformations.ROT3(self._gamma*t) # ECEF to Inertial attitude
        EN_dot = coordinateTransformations.ROT3_DOT(self._gamma*t, self._gamma)
        EN_ddot = coordinateTransformations.ROT3_DDOT(self._gamma*t, self._gamma, 0.0)
        (HN, HN_dot, HN_ddot) = coordinateTransformations.HillFrameRotationMatrix(self._raan, self._inc, self._meanAnomaly*t, self._meanAnomaly)

        (self._RN, RN_dot, RN_ddot) = self.computeReferenceFrameFromTargetPoint(HN, HN_dot, HN_ddot, EN, EN_dot, EN_ddot, self._r_HN_H, np.zeros(3), 'ECEF')

        self._w_RN_R = attitudeKinematics.DCMrate2angVel(self._RN, RN_dot)
        self._w_RN_R_dot = attitudeKinematics.DCMdoubleRate2angVelDot(self._RN, RN_ddot, self._w_RN_R) # Derivative relative to R or N (both are equal)
        beta_RN = attitudeKinematics.dcm2quat(self._RN)
        self._sigma_RN = attitudeKinematics.quat2mrp(beta_RN)

        return

    def computeReferenceFrameFromTargetPoint(self, HN, HN_dot, HN_ddot, EN, EN_dot, EN_ddot, r_HN_H, r_TN, target_frame):
        """
        Computes the reference frame relative to inertial.
        :param HN: [2-dimensional numpy array] DCM from inertial to Hill-Frame.
        :param HN_dot: [2-dimensional numpy array] Derivative of the DCM from inertial to Hill-Frame.
        :param HN_ddot:
        :param EN: [2-dimensional numpy array] DCM from ECEF to Inertial Frame.
        :param EN_dot: [2-dimensional numpy array] Derivative of the DCM from ECEF to Inertial Frame.
        :param EN_ddot:
        :param r_HN_H: [1-dimensional numpy array] Position of the satellite wrt the center of the Earth in the Hill frame.
        :param r_TN_: [1-dimensional numpy array] Position of the target wrt the center of the Earth in the frame specified by frame.
        :param target_frame: [string] Frame in which r_TN is given.
        :return:
        """

        if target_frame == 'ECEF': # r_TN is constant in ECEF frame
            r_HT_N = HN.T.dot(r_HN_H) - EN.T.dot(r_TN)    # Position of the satellite wrt the target in the inertial frame
            r_HT_N_dot = HN_dot.T.dot(r_HN_H) - EN_dot.T.dot(r_TN)
            r_HT_N_ddot = HN_ddot.T.dot(r_HN_H) - EN_ddot.T.dot(r_TN)
        else: # frame == ECI. r_TN is constant in inertial frame
            r_HT_N = HN.T.dot(r_HN_H) - r_TN
            r_HT_N_dot = HN_dot.T.dot(r_HN_H)
            r_HT_N_ddot = HN_ddot.T.dot(r_HN_H)

        i_h_N = HN.T.dot(np.array([0.0,0.0,1.0]))       # Normal vector to the orbit in inertial frame

        (r1_vec, r1_vec_dot, r1_vec_ddot) = self.computeUnitVectorDerivatives(-r_HT_N, -r_HT_N_dot, -r_HT_N_ddot)
        (r2_vec, r2_vec_dot, r2_vec_ddot) = self.computeUnitVectorDerivatives(np.cross(r1_vec, i_h_N), np.cross(r1_vec_dot, i_h_N), np.cross(r1_vec_ddot, i_h_N))

        r3_vec = np.cross(r1_vec, r2_vec)
        r3_vec_dot = np.cross(r1_vec_dot, r2_vec) + np.cross(r1_vec, r2_vec_dot)
        r3_vec_ddot = np.cross(r1_vec_ddot, r2_vec) + 2*np.cross(r1_vec_dot, r2_vec_dot) + np.cross(r1_vec, r2_vec_ddot)

        RN = np.array([r1_vec, r2_vec, r3_vec])         # Reference wrt inertial attitude
        RN_dot = np.array([r1_vec_dot, r2_vec_dot, r3_vec_dot])
        RN_ddot = np.array([r1_vec_ddot, r2_vec_ddot, r3_vec_ddot])
        return (RN, RN_dot, RN_ddot)

    def computeUnitVectorDerivatives(self, a, a_dot, a_ddot):
        """

        :param a:
        :param a_dot:
        :param a_ddot:
        :return:
        """
        a_inner = np.inner(a,a)
        a_norm = np.sqrt(a_inner)
        a_outer = np.outer(a,a)

        a_dot_inner = np.inner(a_dot, a_dot)
        a_dot_outer = np.outer(a_dot, a_dot)

        r = a/a_norm
        r_dot = a_dot/a_norm - a_outer.dot(a_dot)/a_norm**3
        r_ddot = a_ddot/a_norm - (2*a_dot_outer.dot(a) + a_dot_inner * a + a_outer.dot(a_ddot))/a_norm**3 + (a_outer.dot(a_dot_outer).dot(a))/a_norm**5

        return (r, r_dot, r_ddot)


