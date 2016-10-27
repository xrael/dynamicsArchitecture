
from abc import ABCMeta, abstractmethod
from scipy.integrate import odeint
import numpy as np
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

#----------------------------------------------------------------------------------------------------------------------#

class rk4Integrator(integrator):
    """
    Fixed-step rk4 integrator.
    """

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

#----------------------------------------------------------------------------------------------------------------------#

class odeIntIntegrator(integrator):
    """
    Encapsulates odeint.
    THIS IS A VERY INEFFICIENT METHOD since it'se setting odeint at every time step.
    """

    _func = None
    _atol = 0
    _rtol = 0

    def __init__(self, atol = 1e-12, rtol = 1e-12):
        """

        :param atol: [double] Optional absolute tolerance.
        :param rtol: [double] Optional relative tolerance.
        :return:
        """
        super(odeIntIntegrator, self).__init__()

        self._func = None
        self._atol = atol
        self._rtol = rtol

        return

    def setDynamicalSystem(self, dynSystem):
        super(odeIntIntegrator, self).setDynamicalSystem(dynSystem)

        self._func = lambda state, t :   self.func(state, t)
        return

    def func(self, state, t):
        """
        Receives state and current time and returns the derivative of the state.
        :param state:
        :param t:
        :return:
        """
        self._dynSystem.getStateManager().setStateVector(state)
        return self._dynSystem.equationsOfMotion(t)


    def integrate(self, t, dt):
        stateMan = self._dynSystem.getStateManager()

        x = stateMan.getStateVector()

        time_vec = np.array([t, t+dt])

        x_next = odeint(self._func, x, time_vec, rtol=self._rtol, atol=self._atol)

        stateMan.setStateVector(x_next[-1,:])
        return