
from abc import ABCMeta, abstractmethod
import dynamicalSystem
import stateManager


class integrator:
    """
    Base class of every integrator
    """
    __metaclass__ = ABCMeta

    _dynSystem = None

    def __init__(self):
        self._dynSystem = None
        return

    def setDynamicalSystem(self, dynSystem):
        self._dynSystem = dynSystem
        return

    @abstractmethod
    def integrate(self, t, dt): pass


class rk4Integrator(integrator):

    def __init__(self):
        super(rk4Integrator, self).__init__()
        return

    def integrate(self, t, dt):

        stateMan = self._dynSystem.getStateManager()

        x = stateMan.getStateVector()
        t_half = t + 0.5 * dt

        # k1
        self._dynSystem.equationsOfMotion(t)
        k1 = stateMan.getStateDerivativesVector()
        x_k1 = x + 0.5 * k1 * dt
        stateMan.setStateVector(x_k1)

        # k2
        self._dynSystem.equationsOfMotion(t_half)
        k2 = stateMan.getStateDerivativesVector()
        x_k2 = x + 0.5 * k2 * dt
        stateMan.setStateVector(x_k2)

        # k3
        self._dynSystem.equationsOfMotion(t_half)
        k3 = stateMan.getStateDerivativesVector()
        x_k3 = x + k3 * dt
        stateMan.setStateVector(x_k3)

        # k4
        self._dynSystem.equationsOfMotion(t + dt)
        k4 = stateMan.getStateDerivativesVector()

        x_next = x + 1.0/6.0 * dt * (k1 + 2*k2 + 2*k3 + k4)

        stateMan.setStateVector(x_next)

        return