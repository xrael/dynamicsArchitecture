######################################################
# attitudeDetermination
#
# Manuel F. Diaz Ramos
#
# Functions to do non-dynamical attitude determination.
######################################################

import numpy as np
import attitudeKinematics as atKin

f_quest = lambda s, K, I: np.linalg.det(K - s*I)

def triad(v1_b, v2_b, v1_n, v2_n):
    """
    Estimates attitude using the TRIAD method.
    :param v1_b: [1-dimension numpy array] First sensor measurement (the most accurate).
    :param v2_b: [1-dimension numpy array] Second sensor measurement.
    :param v1_n: [1-dimension numpy array] Inertial description of v1
    :param v2_n: [1-dimension numpy array] Inertial description of v2
    :return: [2-dimension numpy array] DCM with the attitude BN (B with respect to N).
    """
    # Normalize
    v1_b_unit = v1_b/np.linalg.norm(v1_b)
    v2_b_unit = v2_b/np.linalg.norm(v2_b)
    v1_n_unit = v1_n/np.linalg.norm(v1_n)
    v2_n_unit = v2_n/np.linalg.norm(v2_n)

    # TRIAD frame in body frame
    t1_b = v1_b_unit
    t2_b = np.cross(v1_b_unit, v2_b_unit)
    t2_b = t2_b/np.linalg.norm(t2_b)
    t3_b = np.cross(t1_b, t2_b)
    BT = np.array([t1_b, t2_b, t3_b]).T

    # TRIAD frame in inertial frame
    t1_n = v1_n_unit
    t2_n = np.cross(v1_n_unit, v2_n_unit)
    t2_n = t2_n/np.linalg.norm(t2_n)
    t3_n = np.cross(t1_n, t2_n)
    NT = np.array([t1_n, t2_n, t3_n]).T

    BN = np.dot(BT, NT.T)
    return BN

def davenport(vk_b, vk_n, weights):
    """
    Computes an attitude estimation using the Davenport's q method.
    :param vk_b: [2-dimension numpy array] N vector observations in body frame. vk_b[i] is the i-th observation.
    :param vk_n: [2-dimension numpy array] N vector observations in inertial frame. vk_n[i] is the i-th observation.
    :param weights: [1-dimension numpy array] N-array of weights.
    :return: [2-dimension numpy array] the estimated DCM
    """
    B = np.zeros((3,3))
    K = np.zeros((4,4))
    obs_nmbr = weights.size

    for i in range(0, obs_nmbr):
        vk_b_unit = vk_b[i]/np.linalg.norm(vk_b[i])
        vk_n_unit = vk_n[i]/np.linalg.norm(vk_n[i])
        B += weights[i] * np.outer(vk_b_unit, vk_n_unit)

    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([B[1,2]-B[2,1], B[2,0]-B[0,2], B[0,1]-B[1,0]])

    K[0,0] = sigma
    K[1:,0] = Z
    K[0,1:] = Z
    K[1:,1:] = S - sigma*np.eye(3)

    w, v = np.linalg.eig(K)

    max_index = np.argmax(w)

    dcm = atKin.quat2dcm(v[:,max_index])
    return dcm

def quest(vk_b, vk_n, weights):
    """

    :param vk_b:
    :param vk_n:
    :param weights:
    :return:
    """
    h = 1e-6
    I = np.eye(3)
    I_4 = np.eye(4)
    B = np.zeros((3,3))
    K = np.zeros((4,4))
    obs_nmbr = weights.size

    for i in range(0, obs_nmbr):
        vk_b_unit = vk_b[i]/np.linalg.norm(vk_b[i])
        vk_n_unit = vk_n[i]/np.linalg.norm(vk_n[i])
        B += weights[i] * np.outer(vk_b_unit, vk_n_unit)

    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([B[1,2]-B[2,1], B[2,0]-B[0,2], B[0,1]-B[1,0]])

    K[0,0] = sigma
    K[1:,0] = Z
    K[0,1:] = Z
    K[1:,1:] = S - sigma * I

    print "B: " + str(B)
    print "K: " + str(K)

    # Newton-Raphson
    lambda_k_1 = np.sum(weights)
    while True:
        f = f_quest(lambda_k_1,  K, I_4)
        f_dot = (f_quest(lambda_k_1+h, K, I_4) - f)/h
        lambda_k = lambda_k_1 - f/f_dot

        if abs(lambda_k-lambda_k_1) < 1e-8:
            break
        lambda_k_1 = lambda_k

    crp = np.linalg.inv((sigma + lambda_k)*I - S).dot(Z) # CRP
    dcm = atKin.crp2dcm(crp)

    print "crp QUEST: " + str(crp)

    return dcm


def olae(vk_b, vk_n, weights):
    """

    :param vk_b:
    :param vk_n:
    :param weights:
    :return:
    """
    obs_nmbr = weights.size
    n = 3*obs_nmbr
    d = np.zeros(n)
    S = np.zeros((n,3))
    W = np.zeros((n, n))
    I = np.eye(3)

    for i in range(0, obs_nmbr):
        l = 3*i
        vk_b_unit = vk_b[i]/np.linalg.norm(vk_b[i])
        vk_n_unit = vk_n[i]/np.linalg.norm(vk_n[i])
        d[l:l+3] = vk_b_unit - vk_n_unit
        S[l:l+3,:] = atKin.getSkewSymmetrixMatrix(vk_b_unit + vk_n_unit)
        W[l:l+3,l:l+3] = weights[i]*I

    crp = np.linalg.inv(S.T.dot(W).dot(S)).dot(S.T).dot(W).dot(d)
    print "crp OLAE: " + str(crp)

    dcm = atKin.crp2dcm(crp)

    return dcm

