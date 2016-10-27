

from abc import ABCMeta, abstractmethod
import numpy as np

class dynEffector:

    __metaclass__ = ABCMeta

    _dynSystem = None

    _effectorName = ''


    def __init__(self, dynSystem, name):
        self._dynSystem = dynSystem
        self._effectorName = name
        return

    def getEffectorName(self):
        return self._effectorName

    @abstractmethod
    def computeRHS(self, mass, I_B, CoM, BN):
        """

        :param mass:
        :param I_B:
        :param CoM:
        :param BN:
        :return:
        """
        pass

class constantForceDynEffector(dynEffector):
    """
    Dynamic effector representing a contant force (in inertial frame) applied to the center of mass.
    """
    _force = None

    def __init__(self, dynSystem, name, force):
        super(constantForceDynEffector, self).__init__(dynSystem, name)
        self._force = force
        return

    def computeRHS(self, mass, I_B, CoM, BN):
        f_r_BN_dot_contr =  BN.dot(self._force)
        f_w_dot_contr = np.cross(CoM, f_r_BN_dot_contr) # Torque due to the fact that the reference might not be the CoM.
        return (f_r_BN_dot_contr, f_w_dot_contr)

