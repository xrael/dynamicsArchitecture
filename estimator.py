
from abc import ABCMeta, abstractmethod


class estimator:
    """
    estimator is a base class for all estimators.
    It has mainly two methods:
    - runEstimator(): triggers the estimation process. It only triggers at a given frequency.
    - estimate(): Abstract methods to be implemented with the estimation algorithm.
    """

    __metaclass__ = ABCMeta

    _period = 0.0       # [sec] The estimator process should run every _period seconds
    _lastTime = 0.0

    def __init__(self, period):
        """

        :param period: [sec] 1/period is the estimation frequency.
        :return:
        """
        self._period = period
        self._lastTime = 0.0
        return

    def start(self, t0):
        """
        Run this method to get the estimator started before calling runControl().
        :param t0:
        :return:
        """
        self._lastTime = t0 - self._period
        return

    def runEstimation(self, t):
        """
        This method should trigger the estimation process at a certain frequency.
        :param t:
        :return:
        """
        if (t-self._lastTime) >= self._period:
            self._lastTime = t
            self.estimate(t)
        return

    @abstractmethod
    def estimate(self, t):
        """
        This method performs the estimation
        :param t:
        :return:
        """
        pass


class idealEstimator(estimator):
    """
    This is an ideal estimator. It has a frequency, but it assumes perfect estimation.
    Used only with a spacial type of dynamicalSystem: spacecraft.
    """

    _spacecraft = None

    _sigma_BN_est = None
    _w_BN_B_est = None
    _w_BN_B_dot_est = None
    _I_est = None

    _w_BN_B_name = ''
    _sigma_BN_name = ''

    def __init__(self, spacecraft, period, sigma_BN_name, w_BN_B_name):
        super(idealEstimator, self).__init__(period)
        self._spacecraft = spacecraft
        self._w_BN_B_name = w_BN_B_name
        self._sigma_BN_name = sigma_BN_name
        return

    def start(self, t0):
        super(idealEstimator, self).start(t0)
        return

    def estimate(self, t):
        """
        Should trigger the estimation process given a frequency.
        Here, we're only implementing an ideal estimator that captures the states.
        :return:
        """
        self._sigma_BN_est = self._spacecraft.getState(self._sigma_BN_name)
        self._w_BN_B_est = self._spacecraft.getState(self._w_BN_B_name)
        self._w_BN_B_dot_est = self._spacecraft.getStateDerivative(self._w_BN_B_name)
        self._I_est = self._spacecraft.getTotalInertiaB()
        return

    def get_sigma_BN_estimation(self):
        return self._sigma_BN_est

    def get_w_BN_B_estimation(self):
        return self._w_BN_B_est

    def get_w_BN_B_dot_estimation(self):
        return self._w_BN_B_dot_est

    def get_I_B_estimation(self):
        return self._I_est