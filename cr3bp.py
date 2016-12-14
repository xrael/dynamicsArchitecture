
import numpy as np
from scipy.integrate import odeint
from attitudeKinematics import getSkewSymmetrixMatrix

def computePotential(mu1, mu2, R, x, y, z):
    """
    Computes the potential function for the Restricted Circular 3-Body Problem.
    :param mu1:
    :param mu2:
    :param R:
    :param x:
    :param y:
    :param z:
    :return:
    """
    nu = mu2/(mu1 + mu2)
    U = mu1/np.sqrt((x+nu*R)**2 + y**2 + z**2) +  mu2/np.sqrt((x-(1-nu)*R)**2 + y**2 + z**2)
    return U

def computePotentialGradient(mu1, mu2, R, x, y, z):
    nu = mu2/(mu1 + mu2)

    den1 = np.sqrt((x+nu*R)**2 + y**2 + z**2)
    den2 = np.sqrt((x-(1-nu)*R)**2 + y**2 + z**2)

    dUx = -mu1*(x+nu*R)/den1**3 - mu2*(x-(1-nu)*R)/den2**3
    dUy = -mu1*y/den1**3 - mu2*y/den2**3
    dUz = -mu1*z/den1**3 - mu2*z/den2**3
    return np.array([dUx, dUy, dUz])

def computePotentialHessian(mu1, mu2, R, x, y, z):

    nu = mu2/(mu1 + mu2)

    den1 = np.sqrt((x+nu*R)**2 + y**2 + z**2)
    den2 = np.sqrt((x-(1-nu)*R)**2 + y**2 + z**2)

    d2Udx2 = -mu1/den1**3 + 3*mu1*(x+nu*R)**2/den1**5 - mu2/den2**3 + 3*mu2*(x-(1-nu)*R)**2/den2**5
    d2Udy2 = -mu1/den1**3 + 3*mu1*y**2/den1**5 - mu2/den2**3 + 3*mu2*y**2/den2**5
    d2Udz2 = -mu1/den1**3 + 3*mu1*z**2/den1**5 - mu2/den2**3 + 3*mu2*z**2/den2**5

    d2Udxdy = 3*mu1*(x+nu*R)*y/den1**5 + 3*mu2*(x-(1-nu)*R)*y/den2**5
    d2Udxdz = 3*mu1*(x+nu*R)*z/den1**5 + 3*mu2*(x-(1-nu)*R)*z/den2**5
    d2Udydz = 3*mu1*y*z/den1**5 + 3*mu2*y*z/den2**5

    d2Udq2 = np.array([
        [d2Udx2, d2Udxdy, d2Udxdz],
        [d2Udxdy, d2Udy2, d2Udydz],
        [d2Udxdz, d2Udydz, d2Udz2]
    ])
    return d2Udq2


def computeHamiltonian(mu1, mu2, R, x, y, z, p1, p2, p3):

    omega = np.sqrt((mu1+mu2)/R**3)

    H = 0.5*(p1**2+p2**2+p3**2) - omega*(x*p2-y*p1) - computePotential(mu1, mu2, R, x, y, z)
    return H

def computeHamiltonianGradient(mu1, mu2, R, x, y, z, p1, p2, p3):

    omega = np.sqrt((mu1+mu2)/R)

    dU = computePotentialGradient(mu1, mu2, R, x, y, z)

    dUdx = dU[0]
    dUdy = dU[1]
    dUdz = dU[2]

    dHdx = -omega*p2 - dUdx
    dHdy = omega*p1 - dUdy
    dHdz = -dUdz
    dHdp1 = p1 + omega*y
    dHdp2 = p2 - omega*x
    dHdp3 = p3

    return np.array([dHdx, dHdy, dHdz, dHdp1, dHdp2, dHdp3])

def computeHamiltonianHessian(mu1, mu2, R, x, y, z, p1, p2, p3):

    omega = np.sqrt((mu1+mu2)/R)

    d2Udx2 = computePotentialHessian(mu1, mu2, R, x, y, z)

    d2HdX = np.eye(6)

    d2HdX[0:3, 0:3] = -d2Udx2
    d2HdX[3:, 0:3] = -getSkewSymmetrixMatrix([0,0,omega])
    d2HdX[0:3, 3:] = getSkewSymmetrixMatrix([0,0,omega])

    return d2HdX

def computeHamiltonianFfunction(mu1, mu2, R, x, y, z, p1, p2, p3):
    I_3 = np.eye(3)
    J = np.zeros([6,6])
    J[0:3,3:] =  I_3
    J[3:,0:3] = -I_3

    Hx = computeHamiltonianGradient(mu1, mu2, R, x, y, z, p1, p2, p3)
    return J.dot(Hx)

def computeDynamicMatrix(mu1, mu2, R, x, y, z, p1, p2, p3):

    I_3 = np.eye(3)
    J = np.zeros([6,6])
    J[0:3,3:] =  I_3
    J[3:,0:3] = -I_3

    Hxx = computeHamiltonianHessian(mu1, mu2, R, x, y, z, p1, p2, p3)
    A = J.dot(Hxx)
    return A

def computeEquilibriumPointManifolds(mu1, mu2, R):

    nu = mu2/(mu1+mu2)

    omega = np.sqrt((mu1+mu2)/R)

    (LP1, LP2, LP3, LP4, LP5) = computeLagrangePoints(nu, R)

    A_L1 = computeDynamicMatrix(mu1, mu2, R, LP1[0], LP1[1], 0, -omega*LP1[1], omega*LP1[0], 0)
    sigma_L1, u_L1 = np.linalg.eig(A_L1)

    A_L2 = computeDynamicMatrix(mu1, mu2, R, LP2[0], LP2[1], 0, -omega*LP2[1], omega*LP2[0], 0)
    sigma_L2, u_L2 = np.linalg.eig(A_L2)

    A_L3 = computeDynamicMatrix(mu1, mu2, R, LP3[0], LP3[1], 0, -omega*LP3[1], omega*LP3[0], 0)
    sigma_L3, u_L3 = np.linalg.eig(A_L3)

    A_L4 = computeDynamicMatrix(mu1, mu2, R, LP4[0], LP4[1], 0, -omega*LP4[1], omega*LP4[0], 0)
    sigma_L4, u_L4 = np.linalg.eig(A_L4)

    A_L5 = computeDynamicMatrix(mu1, mu2, R, LP5[0], LP5[1], 0, -omega*LP5[1], omega*LP5[0], 0)
    sigma_L5, u_L5 = np.linalg.eig(A_L5)

    return (sigma_L1, u_L1, sigma_L2, u_L2, sigma_L3, u_L3, sigma_L4, u_L4, sigma_L5, u_L5)


def computeZeroVelocityJacobiIntegral(mu1, mu2, R, x, y, z):

    omega = np.sqrt((mu1+mu2)/R)

    p1 = -omega*y
    p2 = omega*x
    p3 = 0.0

    J = computeHamiltonian(mu1, mu2, R, x, y, z, p1, p2, p3)
    #J = -0.5 * (x**2 + y**2) - (1-nu)/np.sqrt((x+ nu)**2 + y**2) - nu/np.sqrt((x-1+ nu)**2 + y**2)
    return J


def computeZeroVelocityPlanarCurves(nu, x_min, x_max, y_min, y_max, step):

    x = np.arange(x_min, x_max, step)
    y = np.arange(y_min, y_max, step)
    [X,Y] = np.meshgrid(x,y)
    J = computeZeroVelocityJacobiIntegral((1-nu), nu, 1, X, Y, 0)
    return (X,Y,J)

def computeLagrangePoints(nu, R):

    psi = 1-nu

    a1 = 2*(2*nu -1)
    a2 = psi**2 - 4*nu*psi + nu**2
    a3 = 2*nu*psi*(1-2*nu)
    a4 = nu**2*psi**2

    # for L1, L2, L3: solving quintic polynomial function of lagrange point positions.
    # Lagrange 1
    #coefL1 = [1, 2*(mu-l), l**2 - 4*l*mu + mu**2, 2*mu*l*(l-mu) + mu - l, mu**2 * l**2 + 2*(l**2 + mu**2), mu**3 - l**3]
    cL1 = [1, a1, a2, a3 - psi + nu, a4 + 2*(psi**2+nu**2), nu**3-psi**3]
    L1roots = np.roots(cL1)
    L1 = 0
    for root in L1roots:
        if np.isreal(root) and (root > -nu) and (nu < psi):
            L1 = np.real(root)
            break
    LP1 = np.array([L1, 0]) * R

    # Lagrange 2
    #coefL2 = [1, 2*(mu-l), l**2 - 4*l*mu+mu**2, 2*mu*l*(l-mu) - (mu+l), mu**2 * l**2 + 2*(l**2 - mu**2), -(mu**3 + l**3)]
    cL2 = [1, a1, a2, a3-psi-nu, a4 + 2*(psi**2-nu**2), -(psi**3+nu**3)]
    L2roots = np.roots(cL2)
    L2 = 0
    for root in L2roots:
        if np.isreal(root) and (root > psi):
            L2 = np.real(root)
    LP2 = np.array([L2, 0]) * R

    # Lagrange 3
    #coefL3 = [1, 2*(mu-l), l**2 - 4*mu*l+mu**2, 2*mu*l*(l-mu) + (l+mu), mu**2 * l**2 + 2*(mu**2 - l**2), l**3 + mu**3]
    cL3 = [1, a1, a2, a3 + (psi+nu), a4 + 2*(nu**2-psi**2), nu**3 + psi**3]
    L3roots = np.roots(cL3)
    L3 = 0
    for root in L3roots:
        if np.isreal(root) and root < -nu:
            L3 = np.real(root)
    LP3 = np.array([L3, 0]) * R

    # Lagrange 4
    LP4 = np.array([0.5 - nu, np.sqrt(3)/2]) * R

    # Lagrange 5
    LP5 = np.array([0.5 - nu, -np.sqrt(3)/2]) * R

    return LP1, LP2, LP3, LP4, LP5

def Func(X, t, mu1, mu2, R):
    x = X[0]
    y = X[1]
    z = X[2]
    p1 = X[3]
    p2 = X[4]
    p3 = X[5]
    # I_3 = np.eye(3)
    # J = np.zeros([6,6])
    # J[0:3,3:] =  I_3
    # J[3:,0:3] = -I_3
    # Hx = computeHamiltonianGradient(mu1,mu2,R,x,y,z,p1,p2,p3)
    X_dot = computeHamiltonianFfunction(mu1,mu2,R,x,y,z,p1,p2,p3)

    A = computeDynamicMatrix(mu1,mu2,R,x,y,z,p1,p2,p3)

    for i in range(0, 6):  # loop for each STM column vector (phi)
        phi = X[(6 + 6*i):(6 + 6 + 6*i):1]
        dphi = A.dot(phi)
        X_dot = np.concatenate([X_dot, dphi])

    return X_dot

def integrateCR3BP(t0, tf, dt, X_0, mu1, mu2, R):#, rtol=1e-12, atol=1e-12):
    num = int((tf - t0)/dt) + 1
    tf = (num - 1) * dt + t0 # includes the last value
    time_vec = np.linspace(t0, tf, num)
    l = time_vec.size

    params = (mu1, mu2, R)

    vec = np.array([1,0,0,0,0,0])
    for i in range(0,6):
        X_0 = np.concatenate([X_0, np.roll(vec,i)])

    X = odeint(Func, X_0, time_vec, args=params)#, rtol=rtol, atol=atol)

    stms = np.zeros([l, 6, 6])
    for i in range(0, l) :
        stms[i,:,:] = X[i,6:].reshape((6,6)).T

    states = X[:,0:6]

    return (time_vec, states, stms)

def singleShootingMethod(X_0, T, X_0_tangent, T_tangent, ds, mu1, mu2, R):
    """

    :param X_0: State on a previously computed periodic orbit.
    :param T: Period of that periodic orbit
    :param X_0_tangent:
    :param T_tangent:
    :param ds:
    :param mu1:
    :param mu2:
    :param R:
    :return:
    """

    # Derivatie of the previous periodic orbit
    f_0 = computeHamiltonianFfunction(mu1, mu2, R, X_0[0],X_0[1],X_0[2],X_0[3],X_0[4],X_0[5])

    # Prediction
    X_0_prediction = X_0 + X_0_tangent * ds
    T_prediction = T + T_tangent * ds

    T_next = T_prediction
    X_0_next = X_0_prediction
    vec_next = np.concatenate([X_0_next, [T_next]])

    # Newton-Raphson iteration
    i = 0
    dvec = np.ones(7)
    while i < 10 and np.linalg.norm(dvec) > 1e-10:
        i += 1

        (time_vec, states, stms) = integrateCR3BP(0, T_next, T_next/2, X_0_next, mu1, mu2, R) #, rtol=1e-12, atol=1e-12)

        phi_T = states[-1]
        stm_T = stms[-1]

        f_T = computeHamiltonianFfunction(mu1, mu2, R, phi_T[0], phi_T[1],phi_T[2],phi_T[3],phi_T[4],phi_T[5])

        F = np.concatenate([phi_T - X_0_next,
                           [np.inner(X_0_next - X_0, f_0)],
                           [np.inner(X_0_next - X_0, X_0_tangent) + (T_next - T)*T_tangent - ds]])

        jacobian_F = np.zeros([8,7])
        jacobian_F[0:6,0:6] = stm_T - np.eye(6)
        jacobian_F[0:6,6] = f_T
        jacobian_F[6,0:6] = f_0
        jacobian_F[7,0:6] = X_0_tangent
        jacobian_F[7,6] = T_tangent

        inverse = np.linalg.inv(jacobian_F.T.dot(jacobian_F)).dot(jacobian_F.T)
        dvec = -inverse.dot(F)
        vec_next = vec_next + dvec
        X_0_next = vec_next[0:6]
        T_next = vec_next[6]

    return (X_0_next, T_next)

def computePeriodicOrbits(eq_point, eigenVal, eigenVect, eps, ds, mu1, mu2, R, nmbrOrbits):

    T_array = np.zeros(nmbrOrbits)
    X_array = np.zeros([nmbrOrbits, 6])

    uR = np.real(eigenVect)
    uI = np.imag(eigenVect)
    omega = np.abs(np.imag(eigenVal))

    # Initial periodic orbit
    T = 2*np.pi/omega
    X_0 = eq_point + eps * uR

    X_0_tangent = uR
    T_tangent = 0.0

    T_array[0] = T
    X_array[0] = X_0

    for i in range(1, nmbrOrbits):
        print "Orbit", i

        (X_0_next, T_next) = singleShootingMethod(X_0, T, X_0_tangent, T_tangent, ds, mu1, mu2, R)

        T_array[i] = T_next
        X_array[i] = X_0_next

        tangent_family = np.concatenate([X_array[i] - X_array[i-1], [T_array[i] - T_array[i-1]]])
        tangent_family = tangent_family/np.linalg.norm(tangent_family)

        T = T_next
        X_0 = X_0_next
        X_0_tangent = tangent_family[0:6]
        T_tangent = tangent_family[6]

    return (X_array, T_array)












