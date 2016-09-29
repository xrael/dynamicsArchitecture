#######################################################################################################################
# ASEN 5010: Final Project
#
# Manuel Diaz Ramos
#
#######################################################################################################################


import numpy as np
import coordinateTransformations as coord
import attitudeKinematics
import dynamics
import matplotlib.pyplot as plt

def computeUnitVectorDerivatives(a, a_dot, a_ddot):
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

# def computeReferenceFrame(HN, HN_dot, EN, EN_dot, r_HN_H, r_TN_E):
#     """
#     Computes the reference frame relative to inertial.
#     :param HN: [2-dimensional numpy array] DCM from inertial to Hill-Frame.
#     :param HN_dot: [2-dimensional numpy array] Derivative of the DCM from inertial to Hill-Frame.
#     :param EN: [2-dimensional numpy array] DCM from ECEF to Inertial Frame.
#     :param EN_dot: [2-dimensional numpy array] Derivative of the DCM from ECEF to Inertial Frame.
#     :param r_HN_H: [1-dimensional numpy array] Position of the satellite wrt the center of the Earth in the Hill frame.
#     :param r_TN_E: [1-dimensional numpy array] Position of the target wrt the center of the Earth in ECEF frame
#     :return:
#     """
#     r_HT_N = HN.T.dot(r_HN_H) - EN.T.dot(r_TN_E)    # Position of the satellite wrt the target in the inertial frame
#     r_HT_N_dot = HN_dot.T.dot(r_HN_H) - EN_dot.T.dot(r_TN_E)
#
#     r_HT_N_norm = np.linalg.norm(r_HT_N)
#
#     i_h_N = HN.T.dot(np.array([0.0,0.0,1.0]))       # Normal vector to the orbit in inertial frame
#
#     r1_vec = -r_HT_N/r_HT_N_norm
#     r2_vec = np.cross(r1_vec, i_h_N)/np.linalg.norm(np.cross(r1_vec, i_h_N))
#     r3_vec = np.cross(r1_vec, r2_vec)
#
#     r1_vec_dot = -r_HT_N_dot/r_HT_N_norm + r_HT_N * (np.inner(r_HT_N, r_HT_N_dot))/r_HT_N_norm**3
#     aux = np.cross(r1_vec_dot, i_h_N)/np.linalg.norm(np.cross(r1_vec, i_h_N))
#     r2_vec_dot = aux - r2_vec * np.inner(aux, r2_vec)
#     r3_vec_dot = np.cross(r1_vec_dot, r2_vec) + np.cross(r1_vec, r2_vec_dot)
#
#     RN = np.array([r1_vec, r2_vec, r3_vec])         # Reference wrt inertial attitude
#     RN_dot = np.array([r1_vec_dot, r2_vec_dot, r3_vec_dot])
#     return (RN, RN_dot)

def computeReferenceFrameFromTargetPoint(HN, HN_dot, HN_ddot, EN, EN_dot, EN_ddot, r_HN_H, r_TN, target_frame):
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

    (r1_vec, r1_vec_dot, r1_vec_ddot) = computeUnitVectorDerivatives(-r_HT_N, -r_HT_N_dot, -r_HT_N_ddot)
    (r2_vec, r2_vec_dot, r2_vec_ddot) = computeUnitVectorDerivatives(np.cross(r1_vec, i_h_N), np.cross(r1_vec_dot, i_h_N), np.cross(r1_vec_ddot, i_h_N))

    r3_vec = np.cross(r1_vec, r2_vec)
    r3_vec_dot = np.cross(r1_vec_dot, r2_vec) + np.cross(r1_vec, r2_vec_dot)
    r3_vec_ddot = np.cross(r1_vec_ddot, r2_vec) + 2*np.cross(r1_vec_dot, r2_vec_dot) + np.cross(r1_vec, r2_vec_ddot)

    RN = np.array([r1_vec, r2_vec, r3_vec])         # Reference wrt inertial attitude
    RN_dot = np.array([r1_vec_dot, r2_vec_dot, r3_vec_dot])
    RN_ddot = np.array([r1_vec_ddot, r2_vec_ddot, r3_vec_ddot])
    return (RN, RN_dot, RN_ddot)

def getObservedAttitude(attitude):
    return attitude # Perfect measurement

def getObservedAngVelocity(w):
    return w # Perfect measurement


def getControlTorque(t, quat_BN, w_BN, K, P, quat_BR, w_BR_B, w_RN, w_RN_dot, I):# mrp_RN, w_RN_B):
    """
    Get the control torque.
    :param t:
    :param mrp_BN:
    :param w_BN:
    :param K:
    :param P:
    :param mrp_BR:
    :param w_BR_B:
    :return:
    """
    mrp_BR = attitudeKinematics.quat2mrp(quat_BR)
    u = - K * mrp_BR - P * w_BR_B + np.cross(w_BN, I.dot(w_BN)) + I.dot(w_RN_dot - np.cross(w_BN, w_RN))

    return u

#######################################################################################################################


def runSimulator():

    ##### Earth parameters
    R_T = 6378.0                # [km] Earth radius
    gamma = 7.2925e-05          # [rad/sec] Rotation rate
    mu = 398600.0               # [km^2/s^2] Gravitational parameters

    ##### Orbital parameters
    raan = np.deg2rad(20)       # [rad] Right Ascension of Ascending Node
    inc = np.deg2rad(56)        # [rad] Inclination
    h = 400.0                   # [km] Altitude
    a = R_T + h                 # [km] Semimajor axis
    n = np.sqrt(mu/a**3)        # [rad/sec] Mean Motion

    ##### Parameters
    I = np.diag([10, 5, 2])     # [kg m^2] Principal axis inertia matrix
    I_inv = np.linalg.inv(I)

    ##### Initial conditions
    quat_BN_0 = np.array([0,1,0,0])
    w_BN_0_B = np.array([0.1, 0.4, -0.2])               # Initial angular velocity Body/Inertial written in Body frame

    theta_0 = 0.0                                       # [rad] Initial Argument of Latitude

    #### Target coordinates
    lat = np.deg2rad(48.855)                            # [rad] Latitude
    long = np.deg2rad(2.35)                             # [rad] Longitude
    r_TN_E_nadir = np.array([0,0,0])

    #### Satellite position
    r_HN_H = np.array([a, 0.0, 0.0])                    # Satellite position in Hill Frame

    #### Control parameters
    P = 1
    K = 0.2

    # Target position relative to inertial in ECEF frame
    r_TN_E = r_TN_E_nadir

    t0 = 0.0
    tf = 1000.0
    dt = 0.1

    num = int((tf - t0)/dt) + 1
    tf = (num - 1) * dt + t0 # includes the last value
    time = np.linspace(t0, tf, num)
    l = time.size

    # Body attitude
    quat_BN = np.zeros((l, 4))
    mrp_BN = np.zeros((l, 3))
    BN = np.zeros((l,3,3))
    quat_BN[0,:] = quat_BN_0
    mrp_BN[0,:] = attitudeKinematics.quat2mrp(quat_BN_0)
    BN[0,:,:] = attitudeKinematics.quat2dcm(quat_BN_0)
    w_BN_B = np.zeros((l,3)) # In B frame
    w_BN_B[0,:] = w_BN_0_B

    # Error attitude
    quat_BR = np.zeros((l,4))
    mrp_BR = np.zeros((l, 3))
    w_BR_B = np.zeros((l,3)) # In B frame
    error_angle = np.zeros((l,3))

    # Reference attitude
    quat_RN = np.zeros((l,4))
    mrp_RN = np.zeros((l, 3))
    RN = np.zeros((l,3,3))
    w_RN_R = np.zeros((l,3)) # In R frame


    # Control torque
    u = np.zeros((l, 3))

    for i in range(0, l-1):
        t_i = time[i]


        # Attitude representations
        EN = coord.ROT3(gamma*t_i)
        EN_dot = coord.ROT3_DOT(gamma*t_i, gamma)
        EN_ddot = coord.ROT3_DDOT(gamma*t_i, gamma, 0.0)
        (HN, HN_dot, HN_ddot) = coord.HillFrameRotationMatrix(raan, inc, n*t_i, n)
        (RN[i,:,:], RN_dot, RN_ddot) = computeReferenceFrameFromTargetPoint(HN, HN_dot, HN_ddot, EN, EN_dot, EN_ddot, r_HN_H, r_TN_E, "ECEF")

        #print mrp_BN[i]

        quat_RN[i] = attitudeKinematics.dcm2quat(RN[i])
        mrp_RN[i] = attitudeKinematics.quat2mrp(quat_RN[i])
        w_RN_R[i] = attitudeKinematics.DCMrate2angVel(RN[i], RN_dot)
        w_RN_dot_R = attitudeKinematics.DCMdoubleRate2angVelDot(RN[i], RN_ddot, w_RN_R[i]) # Derivative relative to R or N (both are equal)

        quat_BN_obs = getObservedAttitude(quat_BN[i])

        w_BN_obs = getObservedAngVelocity(w_BN_B[i])

        quat_BR[i] = attitudeKinematics.computeErrorQuat(quat_BN_obs, quat_RN[i])

        #mrp_BR[i] = attitudeKinematics.computeErrorMRP(mrp_BN_obs, mrp_RN[i])
        error_angle[i] = np.arccos(np.dot(RN[i,0,:],BN[i,0,:]))
        #BR = BN.dot(RN[i].T)
        #mrp_BR[i] = attitudeKinematics.quat2mrp(attitudeKinematics.dcm2quat(BR))

        #print "mrp_BR", mrp_BR[i]

        w_BR_B[i] = w_BN_obs - BN[i].dot(RN[i].T).dot(w_RN_R[i])

        # The torque is computed using the current attitude and angular velocity errors
        # It is assumed that this errors are only known at every time step.
        # Therefore, errors cannot be computed in intermediate states.
        torqueParams = (K, P, quat_BR[i], w_BR_B[i], w_RN_R[i], w_RN_dot_R, I) # mrp_RN[i], BN.dot(RN[i].T).dot(w_RN_R[i]))

        (quat_BN[i+1], w_BN_B[i+1], u[i]) = dynamics.rotationDynamicsStep(quat_BN[i], w_BN_B[i], I, I_inv, time[i], dt, attitudeKinematics.angularVelocity2QuaternionRate, getControlTorque, attitudeKinematics.normalizeQuaternion, torqueParams)
        BN[i+1,:,:] = attitudeKinematics.quat2dcm(quat_BN[i+1])
        mrp_BN[i+1,:] = attitudeKinematics.quat2mrp(quat_BN[i+1])

    plt.figure()
    plt.hold(True)
    plt.plot(time, w_BN_B[:,0], label='$w_1$')
    plt.plot(time, w_BN_B[:,1], label='$w_2$')
    plt.plot(time, w_BN_B[:,2], label='$w_3$')
    plt.xlabel("Time [sec]")
    plt.ylabel("$\omega_{B/N}$ $[\\frac{rad}{s}]$", size=18)
    plt.ylim([-0.0012, 0.0002])
    plt.legend()
    plt.savefig('../report/include/problem11_w_BN.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure()
    plt.hold(True)
    plt.plot(time, mrp_BN[:,0], label='$\sigma_1$')
    plt.plot(time, mrp_BN[:,1], label='$\sigma_2$')
    plt.plot(time, mrp_BN[:,2], label='$\sigma_3$')
    plt.xlabel("Time [sec]")
    plt.ylabel("$\sigma_{B/N}$", size=18)
    plt.legend()
    plt.savefig('../report/include/problem11_mrp_BN.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure()
    plt.hold(True)
    plt.plot(time, quat_BN[:,0], label='$\\beta_0$',color='r')
    plt.plot(time, quat_BN[:,1], label='$\\beta_1$',color='g')
    plt.plot(time, quat_BN[:,2], label='$\\beta_2$',color='b')
    plt.plot(time, quat_BN[:,3], label='$\\beta_3$',color='c')
    plt.xlabel("Time [sec]")
    plt.ylabel("$\\beta_{B/N}$", size=18)
    plt.legend()
    plt.savefig('../report/include/problem11_quat_BN.png', bbox_inches='tight', dpi=300)
    plt.close()


    plt.figure()
    plt.hold(True)
    plt.plot(time, w_BR_B[:,0], label='$w_1$')
    plt.plot(time, w_BR_B[:,1], label='$w_2$')
    plt.plot(time, w_BR_B[:,2], label='$w_3$')
    plt.xlabel("Time [sec]")
    plt.ylabel("$\omega_{B/R}$ $[\\frac{rad}{s}]$", size=18)
    plt.legend()
    plt.savefig('../report/include/problem11_w_BR.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure()
    plt.hold(True)
    plt.plot(time, mrp_BR[:,0], label='$\sigma_1$')
    plt.plot(time, mrp_BR[:,1], label='$\sigma_2$')
    plt.plot(time, mrp_BR[:,2], label='$\sigma_3$')
    plt.xlabel("Time [sec]")
    plt.ylabel("$\sigma_{B/R}$", size=18)
    plt.legend()
    plt.savefig('../report/include/problem11_mrp_BR.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure()
    plt.hold(True)
    plt.plot(time, quat_BR[:,0], label='$\\beta_0$',color='r')
    plt.plot(time, quat_BR[:,1], label='$\\beta_1$',color='g')
    plt.plot(time, quat_BR[:,2], label='$\\beta_2$',color='b')
    plt.plot(time, quat_BR[:,3], label='$\\beta_3$',color='c')
    plt.xlabel("Time [sec]")
    plt.ylabel("$\\beta_{B/R}$", size=18)
    plt.legend()
    plt.savefig('../report/include/problem11_quat_BR.png', bbox_inches='tight', dpi=300)
    plt.close()


    plt.figure()
    plt.hold(True)
    plt.plot(time, w_RN_R[:,0], label='$w_1$')
    plt.plot(time, w_RN_R[:,1], label='$w_2$')
    plt.plot(time, w_RN_R[:,2], label='$w_3$')
    plt.xlabel("Time [sec]")
    plt.ylabel("$\omega_{R/N}$ $[\\frac{rad}{s}]$", size=18)
    plt.legend()
    plt.savefig('../report/include/problem11_w_RN.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure()
    plt.hold(True)
    plt.plot(time, mrp_RN[:,0], label='$\sigma_1$')
    plt.plot(time, mrp_RN[:,1], label='$\sigma_2$')
    plt.plot(time, mrp_RN[:,2], label='$\sigma_3$')
    plt.xlabel("Time [sec]")
    plt.ylabel("$\sigma_{R/N}$", size=18)
    plt.legend()
    plt.savefig('../report/include/problem11_mrp_RN.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure()
    plt.hold(True)
    plt.plot(time, quat_RN[:,0], label='$\\beta_0$',color='r')
    plt.plot(time, quat_RN[:,1], label='$\\beta_1$',color='g')
    plt.plot(time, quat_RN[:,2], label='$\\beta_2$',color='b')
    plt.plot(time, quat_RN[:,3], label='$\\beta_3$',color='c')
    plt.xlabel("Time [sec]")
    plt.ylabel("$\\beta_{R/N}$", size=18)
    plt.legend()
    plt.savefig('../report/include/problem11_quat_RN.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure()
    plt.hold(True)
    plt.plot(time, np.rad2deg(error_angle))
    plt.xlabel("Time [sec]")
    plt.ylabel("Error angle $[^\circ]$", size=18)
    plt.ylim([-1,1])
    plt.legend()
    plt.savefig('../report/include/problem11_error_angle.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure()
    plt.hold(True)
    plt.plot(time, u[:,0], label='$u_1$')
    plt.plot(time, u[:,1], label='$u_2$')
    plt.plot(time, u[:,2], label='$u_3$')
    plt.xlabel("Time [sec]")
    plt.ylabel("$u$ $[N m]$", size=18)
    plt.legend()
    plt.savefig('../report/include/problem11_control_torque.png', bbox_inches='tight', dpi=300)
    plt.close()