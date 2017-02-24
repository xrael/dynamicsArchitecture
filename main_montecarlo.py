

# 1) Take a random initial attitude BN.
# 2) take any boresight vector in the outer security cone y_N.
# 3) y_B = BN * y_N
# 4) W_BN = BN * [(y_N x n)/norm(y_N x n)] * w_max. n is the axis of the cone
# Using this algorithm, we'll have our boresight vector in the limit of the outer cone going straight into the inner cone.

import numpy as np
import simulatorManager
import estimator
import controller
import referenceComputer
import coordinateTransformations as coord
import attitudeKinematics
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import stateEffector


##### Parameters
#Hub: cylinder with radius = 0.5 m, height = 1 m
m_hub = 30.0                        # [kg] Hub mass
r_hub = 0.5                         # [m] radius
h_hub = 1.0                         # [m] height
Iz = m_hub * r_hub**2/2
Ix = Iy = m_hub/12 * (3*r_hub**2 + h_hub**2)
I_hub = np.diag([Ix, Iy, Iz])       # [kg m^2] Principal axis inertia matrix

w_max = 2.0*np.pi/180.0             # [rad/sec] Maximum angular velocity

# Reaction wheels
m_rw = 2.0
Iws = 0.03
Iwt = 0.001

# Max torques
us_max = 0.015                  # [Nm]

# Nominal speed
nominal_speed_wheel = 500.0    # [rpm]

# CoM offset
R_O1B_rw = np.array([0.1,0,0])        # [m] Position of the center of mass of the VSCMG relative to the reference B
R_O2B_rw = np.array([0,0.1,0])        # [m] Position of the center of mass of the VSCMG relative to the reference B
R_O3B_rw = np.array([-0.1,0,0])       # [m] Position of the center of mass of the VSCMG relative to the reference B
R_O4B_rw = np.array([0,-0.1,0])       # [m] Position of the center of mass of the VSCMG relative to the reference B

R_O1B_rw_skew = attitudeKinematics.getSkewSymmetrixMatrix(R_O1B_rw)
R_O2B_rw_skew = attitudeKinematics.getSkewSymmetrixMatrix(R_O2B_rw)
R_O3B_rw_skew = attitudeKinematics.getSkewSymmetrixMatrix(R_O3B_rw)
R_O4B_rw_skew = attitudeKinematics.getSkewSymmetrixMatrix(R_O4B_rw)

r_BcB_B = -m_rw * (R_O1B_rw + R_O2B_rw + R_O3B_rw + R_O4B_rw)/m_hub     # [m] Position of the center of mass of the hub relative to the reference B

r_BcB_B_skew = attitudeKinematics.getSkewSymmetrixMatrix(r_BcB_B)

# Inertia of the system including the wheels as point masses
I_s = I_hub + m_hub * r_BcB_B_skew.dot(r_BcB_B_skew.T) +  m_rw * (R_O1B_rw_skew.dot(R_O1B_rw_skew.T) + R_O2B_rw_skew.dot(R_O2B_rw_skew.T) + R_O3B_rw_skew.dot(R_O3B_rw_skew.T) + R_O4B_rw_skew.dot(R_O4B_rw_skew.T))

# Position of the center of mass relative to the reference
r_CB_B =  m_hub * r_BcB_B + m_rw* (R_O1B_rw + R_O2B_rw + R_O3B_rw + R_O4B_rw)

# RW configuration: pyramid
nmbr_rw = 4
pyramid_ang = np.deg2rad(55)        # Internal angle of the pyramid
cos_p_ang = np.cos(pyramid_ang)
sin_p_ang = np.sin(pyramid_ang)


gs1 = np.array([sin_p_ang, 0, cos_p_ang])
gt1 = np.array([0,1,0])
gg1 = np.array([-cos_p_ang, 0, sin_p_ang])
BW1 = np.array([gs1, gt1, gg1]).T     # Attitude of body frame relative to G (gs, gt, gg vectors as columns)

gs2 = np.array([0, sin_p_ang, cos_p_ang])
gt2 = np.array([-1,0,0])
gg2 = np.array([0, -cos_p_ang, sin_p_ang])
BW2 = np.array([gs2, gt2, gg2]).T     # Attitude of body frame relative to G (gs, gt, gg vectors as columns)

gs3 = np.array([-sin_p_ang, 0, cos_p_ang])
gt3 = np.array([0,-1,0])
gg3 = np.array([cos_p_ang, 0, sin_p_ang])
BW3 = np.array([gs3, gt3, gg3]).T     # Attitude of body frame relative to G (gs, gt, gg vectors as columns)

gs4 = np.array([0, -sin_p_ang, cos_p_ang])
gt4 = np.array([1,0,0])
gg4 = np.array([0, cos_p_ang, sin_p_ang])
BW4 = np.array([gs4, gt4, gg4]).T     # Attitude of body frame relative to G (gs, gt, gg vectors as columns)

det_BG1_0 = np.linalg.det(BW1)
det_BG2_0 = np.linalg.det(BW2)
det_BG3_0 = np.linalg.det(BW3)
det_BG4_0 = np.linalg.det(BW4)

Gs = np.array([gs1, gs2, gs3, gs4]).T

L_RW_min = stateEffector.reactionWheel.computeMinMaxTorqueCapability(Gs, us_max)


##### Constraints
x_1 = np.array([0, -np.sin(np.deg2rad(20)), -np.cos(np.deg2rad(20))])       # Constraint unit vector in inertial frame
x_1 = x_1/np.linalg.norm(x_1)
#y_1 = np.array([0, 1, 0])       # Instrument boresight in body frame
cone_angle_1 = np.deg2rad(10) # cone angle around the constraint vector

x_2 = np.array([0, -1, 0])       # Constraint unit vector in inertial frame
x_2 = x_2/np.linalg.norm(x_2)
#y_2 = np.array([0, 1, 0])       # Instrument boresight in body frame
cone_angle_2 = np.deg2rad(30) # cone angle around the constraint vector

x_1_orthogonal = np.array([-x_1[1], x_1[0] - x_1[2], x_1[1]])
x_1_orthogonal = x_1_orthogonal/np.linalg.norm(x_1_orthogonal)
x_1_orthogonal_2 = np.cross(x_1, x_1_orthogonal)

x_2_orthogonal = np.array([-x_2[1], x_2[0] - x_2[2], x_2[1]])
x_2_orthogonal = x_2_orthogonal/np.linalg.norm(x_2_orthogonal)
x_2_orthogonal_2 = np.cross(x_2, x_2_orthogonal)


angle_thres_1 = 0.5*np.max([Ix, Iy, Iz])/L_RW_min * w_max**2 + cone_angle_1
print 'Angle threshold 1', np.rad2deg(angle_thres_1)
angle_thres_1_max = angle_thres_1# * 1.1# + np.deg2rad(10)

angle_thres_2 = 0.5*np.max([Ix, Iy, Iz])/L_RW_min * w_max**2 + cone_angle_2
print 'Angle threshold 2', np.rad2deg(angle_thres_2)
angle_thres_2_max = angle_thres_2# + np.deg2rad(10)


######## Simulation parameters
t0 = 0.0
tf = 200.0
dt = 0.1


######## Control
# Gains
P_w = np.eye(3) * 10.0 #1.0

K1_sigma = 0.1 #0.2 #0.1
K3_sigma = 0.1 #0.2
K_i = 0.01 * np.eye(3)

# Control loop steps
outer_loop_step = 0.1        # 10 Hz
innerLoop_step = 0.1        # 100 Hz

##### Reference computer
reference_computer_step = dt

##### Estimator
estimator_step = dt

simulator = simulatorManager.simulatorManager.getSimulatorManager('spacecraft_backSub', 'rk4', 'hub')
simulator.setSimulationTimes(t0, tf, dt)

spacecraft = simulator.getDynamicalObject()
spacecraft.setHubMass(m_hub)
spacecraft.setHubInertia(I_hub)
spacecraft.setHubCoMOffset(r_BcB_B)

rw1 = spacecraft.addRW('RW_1', m_rw, R_O1B_rw, Iws, Iwt, BW1, nominal_speed_wheel, us_max)
rw2 = spacecraft.addRW('RW_2', m_rw, R_O2B_rw, Iws, Iwt, BW2, nominal_speed_wheel, us_max)
rw3 = spacecraft.addRW('RW_3', m_rw, R_O3B_rw, Iws, Iwt, BW3, nominal_speed_wheel, us_max)
rw4 = spacecraft.addRW('RW_4', m_rw, R_O4B_rw, Iws, Iwt, BW4, nominal_speed_wheel, us_max)

estimator = estimator.idealEstimator(spacecraft, estimator_step,'hub_sigma_BN', 'hub_omega_BN')
referenceComp = referenceComputer.regulationReference(reference_computer_step)

#### Simulations
N_sim = 50

time = np.zeros([N_sim, np.ceil(tf/dt)+1])
mrp_BN = np.zeros([N_sim, np.ceil(tf/dt)+1,3])
w_BN = np.zeros([N_sim, np.ceil(tf/dt)+1,3])
angle_1 = np.zeros([N_sim, np.ceil(tf/dt)+1])
dec = np.zeros([N_sim, np.ceil(tf/dt + 1)])
right_as = np.zeros([N_sim, np.ceil(tf/dt)+1])

rw_us1 = np.zeros([N_sim, np.ceil(tf/dt)+1])
rw_us2 = np.zeros([N_sim, np.ceil(tf/dt)+1])
rw_us3 = np.zeros([N_sim, np.ceil(tf/dt)+1])
rw_us4 = np.zeros([N_sim, np.ceil(tf/dt)+1])

for k in range(0, N_sim):
    print "Simulation: ", k
    ###### Initial attitude
    mrp_BN_0 = np.random.rand(3)
    BN_0 = attitudeKinematics.mrp2dcm(mrp_BN_0)
    ang = 2*np.pi*np.random.rand()
    y_N = np.cos(angle_thres_1_max)*x_1 + np.sin(angle_thres_1_max)*(np.cos(ang)*x_1_orthogonal + np.sin(ang)*x_1_orthogonal_2)
    print np.linalg.norm(y_N)
    y_B = BN_0.dot(y_N)
    w_BN_B_0 = BN_0.dot(np.cross(y_N, x_1)/np.linalg.norm(np.cross(y_N, x_1))) * w_max
    print np.linalg.norm((w_BN_B_0))

    ###### Final attitude
    r_1_N = y_B
    r_2_N = np.array([-y_B[1], y_B[0] - y_B[2], y_B[1]])
    r_2_N = r_2_N/np.linalg.norm(r_2_N)
    r_3_N = np.cross(r_1_N, r_2_N)/np.linalg.norm(np.cross(r_1_N, r_2_N))
    RN = np.array([r_1_N, r_2_N, r_3_N]).T
    mrp_RN = attitudeKinematics.quat2mrp(attitudeKinematics.dcm2quat(RN))


    referenceComp.setPoint(mrp_RN) #np.tan(np.deg2rad(30))/4 * np.array([0,0,1]))
    control = controller.constrainedSteeringLawController(estimator, (rw1, rw2, rw3, rw4), referenceComp, outer_loop_step, innerLoop_step, P_w, K_i, K1_sigma, K3_sigma, w_max, 50)
    control.addExclusionConstraint(x_1, y_B, cone_angle_1, 'Constraint_1', -1, -1)# angle_thres_1, angle_thres_1_max)
    control.addExclusionConstraint(x_2, y_B, cone_angle_2, 'Constraint_2', -1, -1)# angle_thres_2, angle_thres_2_max)
    control.setMaxWheelTorque(us_max)
    simulator.setControls(estimator, control, referenceComp)

    simulator.setInitialConditions('hub_R_BN', np.array([0.0, 0.0, 0.0]))
    simulator.setInitialConditions('hub_R_BN_dot', np.array([0.0, 0.0, 0.0]))
    simulator.setInitialConditions('hub_sigma_BN', mrp_BN_0)
    simulator.setInitialConditions('hub_omega_BN', w_BN_B_0)
    simulator.setInitialConditions('RW_1_Omega', nominal_speed_wheel * 2*np.pi/(60))
    simulator.setInitialConditions('RW_2_Omega', -nominal_speed_wheel * 2*np.pi/(60))
    simulator.setInitialConditions('RW_3_Omega', nominal_speed_wheel * 2*np.pi/(60))
    simulator.setInitialConditions('RW_4_Omega', -nominal_speed_wheel * 2*np.pi/(60))

    simulator.computeEnergy(True)
    simulator.computeAngularMomentum(True)

    simulator.simulate()

    time[k,:] = simulator.getTimeVector()
    state_history = simulator.getStateHistory()
    state_derivatives_history = simulator.getStateDerivativesHistory()

    control_force_history = control.getControlForceHistory()
    reference_history = referenceComp.getReferenceHistory()

    R_BN = state_history['hub_R_BN']
    v_BN = state_history['hub_R_BN_dot']
    mrp_BN[k,:,:] = state_history['hub_sigma_BN']
    w_BN[k,:,:] = state_history['hub_omega_BN']
    rw_Omega_1 = state_history['RW_1_Omega']
    rw_Omega_2 = state_history['RW_2_Omega']
    rw_Omega_3 = state_history['RW_3_Omega']
    rw_Omega_4 = state_history['RW_4_Omega']

    rw_Omega_dot_1 = state_derivatives_history['RW_1_Omega']
    rw_Omega_dot_2 = state_derivatives_history['RW_2_Omega']
    rw_Omega_dot_3 = state_derivatives_history['RW_3_Omega']
    rw_Omega_dot_4 = state_derivatives_history['RW_4_Omega']

    rw_us1[k,:] = control_force_history['RW_1_us']
    rw_us2[k,:] = control_force_history['RW_2_us']
    rw_us3[k,:] = control_force_history['RW_3_us']
    rw_us4[k,:] = control_force_history['RW_4_us']

    sigma_BR = control_force_history['sigma_BR']
    w_BR_B = control_force_history['w_BR_B']
    sigma_RN = control_force_history['sigma_RN']
    w_RN_B = control_force_history['w_RN_B']
    Lr = control_force_history['Lr']

    w_BR_B_desired = control_force_history['w_BR_B_desired']

    y_N_1 = control_force_history['Constraint_1_y_N']
    y_N_2 = control_force_history['Constraint_2_y_N']
    angle_1[k] = control_force_history['Constraint_1_theta']
    angle_2 = control_force_history['Constraint_2_theta']

    l = np.size(y_N_1,0)
    # dec = np.zeros(l-1)
    # right_as = np.zeros(l-1)
    # cone_1_right_as = np.zeros(l-1)
    # cone_1_dec = np.zeros(l-1)
    # cone_2_right_as = np.zeros(l-1)
    # cone_2_dec = np.zeros(l-1)
    #
    # verts_1 = list()
    # codes_1 = list()
    # verts_2 = list()
    # codes_2 = list()

    for i in range(0, l-1):
        y_N_norm = np.linalg.norm(y_N_1[i])

        dec[k,i] = np.arcsin(y_N_1[i,2]/y_N_norm)
        right_as[k,i] = np.arctan2(y_N_1[i,1], y_N_1[i,0])

        # cone_vec_1 = np.cos(cone_angle_1)*x_1 + np.sin(cone_angle_1) *(x_1_orthogonal * np.cos(2*np.pi/(l-1) * i) + x_1_orthogonal_2 * np.sin(2*np.pi/(l-1) * i))
        # cone_vec_1 = cone_vec_1/np.linalg.norm(cone_vec_1)
        # cone_1_right_as[i] = np.arctan2(cone_vec_1[1], cone_vec_1[0])
        # cone_1_dec[i] = np.arcsin(cone_vec_1[2])
        #
        # cone_vec_2 = np.cos(cone_angle_2)*x_2 + np.sin(cone_angle_2) *(x_2_orthogonal * np.cos(2*np.pi/(l-1) * i) + x_2_orthogonal_2 * np.sin(2*np.pi/(l-1) * i))
        # cone_vec_2 = cone_vec_2/np.linalg.norm(cone_vec_2)
        # cone_2_right_as[i] = np.arctan2(cone_vec_2[1], cone_vec_2[0])
        # cone_2_dec[i] = np.arcsin(cone_vec_2[2])
        #
        # if i==0:
        #     codes_1.append(Path.MOVETO)
        #     codes_2.append(Path.MOVETO)
        # elif i < l-2:
        #     codes_1.append(Path.CURVE4)
        #     codes_2.append(Path.CURVE4)
        # else:
        #     codes_1.append(Path.CLOSEPOLY)
        #     codes_2.append(Path.CLOSEPOLY)
        #
        # verts_1.append((np.rad2deg(cone_1_right_as[i]), np.rad2deg(cone_1_dec[i])))
        # verts_2.append((np.rad2deg(cone_2_right_as[i]), np.rad2deg(cone_2_dec[i])))

    # dec_proh_1 = np.arcsin(x_1[2]/np.linalg.norm(x_1))
    # right_as_proh_1 = np.arctan2(x_1[1], x_1[0])
    #
    # dec_proh_2 = np.arcsin(x_2[2]/np.linalg.norm(x_2))
    # right_as_proh_2 = np.arctan2(x_2[1], x_2[0])

l = np.int(np.ceil(tf/dt) + 1)
# dec = np.zeros(l-1)
# right_as = np.zeros(l-1)
cone_1_right_as = np.zeros(l-1)
cone_1_dec = np.zeros(l-1)
cone_2_right_as = np.zeros(l-1)
cone_2_dec = np.zeros(l-1)

cone_1_right_as_ext = np.zeros(l-1)
cone_1_dec_ext = np.zeros(l-1)

verts_1 = list()
codes_1 = list()
verts_2 = list()
codes_2 = list()

verts_1_ext = list()
codes_1_ext = list()

for i in range(0, l-1):
    cone_vec_1 = np.cos(cone_angle_1)*x_1 + np.sin(cone_angle_1) *(x_1_orthogonal * np.cos(2*np.pi/(l-1) * i) + x_1_orthogonal_2 * np.sin(2*np.pi/(l-1) * i))
    cone_vec_1 = cone_vec_1/np.linalg.norm(cone_vec_1)
    cone_1_right_as[i] = np.arctan2(cone_vec_1[1], cone_vec_1[0])
    cone_1_dec[i] = np.arcsin(cone_vec_1[2])

    cone_vec_1_ext = np.cos(angle_thres_1_max)*x_1 + np.sin(angle_thres_1_max) *(x_1_orthogonal * np.cos(2*np.pi/(l-1) * i) + x_1_orthogonal_2 * np.sin(2*np.pi/(l-1) * i))
    cone_vec_1_ext = cone_vec_1_ext/np.linalg.norm(cone_vec_1_ext)
    cone_1_right_as_ext[i] = np.arctan2(cone_vec_1_ext[1], cone_vec_1_ext[0])
    cone_1_dec_ext[i] = np.arcsin(cone_vec_1_ext[2])

    cone_vec_2 = np.cos(cone_angle_2)*x_2 + np.sin(cone_angle_2) *(x_2_orthogonal * np.cos(2*np.pi/(l-1) * i) + x_2_orthogonal_2 * np.sin(2*np.pi/(l-1) * i))
    cone_vec_2 = cone_vec_2/np.linalg.norm(cone_vec_2)
    cone_2_right_as[i] = np.arctan2(cone_vec_2[1], cone_vec_2[0])
    cone_2_dec[i] = np.arcsin(cone_vec_2[2])

    if i==0:
        codes_1.append(Path.MOVETO)
        codes_2.append(Path.MOVETO)
        codes_1_ext.append(Path.MOVETO)
    elif i < l-2:
        codes_1.append(Path.CURVE4)
        codes_2.append(Path.CURVE4)
        codes_1_ext.append(Path.CURVE4)
    else:
        codes_1.append(Path.CLOSEPOLY)
        codes_2.append(Path.CLOSEPOLY)
        codes_1_ext.append(Path.CLOSEPOLY)

    verts_1.append((np.rad2deg(cone_1_right_as[i]), np.rad2deg(cone_1_dec[i])))
    verts_2.append((np.rad2deg(cone_2_right_as[i]), np.rad2deg(cone_2_dec[i])))
    verts_1_ext.append((np.rad2deg(cone_1_right_as_ext[i]), np.rad2deg(cone_1_dec_ext[i])))

plt.figure()
plt.hold(True)
for k in range(0, N_sim):
    plt.plot(time[k,::10], np.rad2deg(mrp_BN[k,::10,0]))
    plt.plot(time[k,::10], np.rad2deg(mrp_BN[k,::10,1]))
    plt.plot(time[k,::10], np.rad2deg(mrp_BN[k,::10,2]))
plt.xlabel("$t$ $[sec]$", size=27)
plt.ylabel("$\sigma_{B/N}$", size=27)
# plt.legend(prop={'size':22})
plt.savefig('figures_montecarlo/sigma_BN.pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
for k in range(0, N_sim):
    plt.plot(time[k,::10], np.rad2deg(w_BN[k,::10,0]))
    plt.plot(time[k,::10], np.rad2deg(w_BN[k,::10,1]))
    plt.plot(time[k,::10], np.rad2deg(w_BN[k,::10,2]))
plt.axhline(np.rad2deg(w_max), label='$\omega_{max}$', color='k', linestyle='--')
plt.axhline(-np.rad2deg(w_max), color='k', linestyle='--')
plt.ylim([-np.rad2deg(w_max)*1.2, np.rad2deg(w_max)*1.2])
plt.xlabel("$t$ $[sec]$", size=27)
plt.ylabel("$\omega_{B/N}$ $[^\circ/sec]$", size=27)
plt.legend(prop={'size':22})
plt.savefig('figures_montecarlo/w_BN.pdf', bbox_inches='tight', dpi=300)
plt.close()


path1 = Path(verts_1, codes_1)
path2 = Path(verts_2, codes_2)
path1_ext = Path(verts_1_ext, codes_1_ext)
fig = plt.figure()
ax = fig.add_subplot(111)
patch1 = patches.PathPatch(path1, facecolor='#CD4141', lw=2, edgecolor="none")
patch2 = patches.PathPatch(path2, facecolor='#CD4141', lw=2, edgecolor="none")
patch1_ext = patches.PathPatch(path1_ext, facecolor='None', lw=2, edgecolor="black",label='$\\theta_0$')
ax.add_patch(patch1)
ax.add_patch(patch2)
ax.add_patch(patch1_ext)
# plt.hold(True)
for k in range(0, N_sim):
    plt.plot(right_as[k,::10]*180/np.pi, dec[k,::10]*180/np.pi, '.',markersize=2)
# plt.plot([right_as[0]*180/np.pi], [dec[0]*180/np.pi], marker='x', markersize=10, color="r")
plt.plot([right_as[0,-1]*180/np.pi], [dec[0,-1]*180/np.pi], marker='x', markersize=10, color="g",label='end')
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xlabel("Right ascension [$^\circ$]",size=22)
plt.ylabel("Declination [$^\circ$]",size=22)
plt.legend(loc='upper right',prop={'size':16})
plt.savefig('figures_montecarlo/map.pdf', bbox_inches='tight', dpi=300)
plt.close()




plt.figure()
plt.hold(True)
for k in range(0, N_sim):
    plt.plot(time[k,::10], np.rad2deg(angle_1[k,::10]))
plt.axhline(np.rad2deg(cone_angle_1), label='$\\theta_{min}$', linestyle='-.', color='k')
plt.axhline(np.rad2deg(angle_thres_1), label='$\\theta_{0}$', linestyle='--', color='c')
# ax_theta1.axhline(np.rad2deg(angle_thres_1_max), label='$\\theta_{thres_{max1}}}$', linestyle='--', color='m')
plt.ylim([0,90])
plt.xlim([0,120])
plt.legend(loc='upper left',prop={'size':22})
plt.xlabel("$t$ $[sec]$", size=27)
plt.ylabel("$\\theta_1$ $[^\circ/sec]$", size=27)
plt.savefig('figures_montecarlo/theta1.pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
for k in range(0, N_sim):
    plt.plot(time[k,::10], rw_us1[k,::10] * 1000)
    plt.plot(time[k,::10], rw_us2[k,::10] * 1000)
    plt.plot(time[k,::10], rw_us3[k,::10] * 1000)
    plt.plot(time[k,::10], rw_us4[k,::10] * 1000)
plt.xlabel("$t$ $[sec]$", size=27)
plt.ylabel("$u_{s}$ $[mN m]$", size=27)
plt.ylim([-us_max*1000*1.1, us_max*1000 * 1.1])
plt.axhline(-us_max*1000, linestyle='-.', color='k')
plt.axhline(us_max*1000, label='$u_{s-max}$', linestyle='-.', color='k')
plt.legend(prop={'size':22})
plt.savefig('figures_montecarlo/rw_us.pdf', bbox_inches='tight', dpi=300)
plt.close()

# plt.figure()
# plt.hold(True)
# plt.plot(time, np.rad2deg(angle_2), label='$\\theta_2$')
# plt.axhline(np.rad2deg(cone_angle_2), label='$\\theta_{min2}$', linestyle='-.', color='k')
# plt.axhline(np.rad2deg(angle_thres_2), label='$\\theta_{thres_{min2}}$', linestyle='--', color='c')
# plt.axhline(np.rad2deg(angle_thres_2_max), label='$\\theta_{thres_{max2}}}$', linestyle='--', color='m')
# plt.ylim([0,180])
# plt.legend(prop={'size':22})
# plt.xlabel("$t$ $[sec]$", size=27)
# plt.ylabel("$\\theta_2$ $[^\circ]$", size=27)
# plt.savefig('figures_montecarlo/theta2.pdf', bbox_inches='tight', dpi=300)
# plt.close()

