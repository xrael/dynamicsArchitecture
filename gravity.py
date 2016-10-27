
from abc import ABCMeta, abstractmethod
import numpy as np

class gravity:
    """
    Base class for all gravity models.
    """

    __metaclass__ = ABCMeta

    _dynSystem = None

    _g = None

    _R_BN_N_name = ''

    def __init__(self, dynSystem, R_BN_name):
        self._dynSystem = dynSystem
        self._R_BN_N_name = R_BN_name
        return

    @abstractmethod
    def computeGravity(self):
        pass

    def getGravityField(self):
        return self._g

#----------------------------------------------------------------------------------------------------------------------#

class gravityPointMass(gravity):
    """
    Simple point-mass gravity model.
    """

    _mu = 0.0

    def __init__(self, dynSystem, R_BN_name, mu):
        super(gravityPointMass, self).__init__(dynSystem, R_BN_name)

        self._mu = mu

        return

    def computeGravity(self):
        stateMan = self._dynSystem.getStateManager()

        # assumes position described in inertial frame
        R_BN_N = stateMan.getStates(self._R_BN_N_name)

        r = np.linalg.norm(R_BN_N)

        self._g = -self._mu/r**3 * R_BN_N

        return self._g


