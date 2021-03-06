
import numpy as np
import simulatorManager
import estimator
import controller
import referenceComputer
import coordinateTransformations as coord
import attitudeKinematics
import matplotlib
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

L_RW_min = 0.5*stateEffector.reactionWheel.computeMinMaxTorqueCapability(Gs, us_max)


# Orbital parameters
R_T = 6378.0e3              # [m] Earth radius
gamma = 7.2925e-05          # [rad/sec] Rotation rate
mu = 398600.0e9             # [m^3/s^2] Gravitational parameters

raan = np.deg2rad(0)        # [rad] Right Ascension of Ascending Node
inc = np.deg2rad(-90)        # [rad] Inclination
h = 400.0e3                 # [m] Altitude
a = R_T + h                 # [m] Semimajor axis
initialArgLatitude = np.deg2rad(180)

##### Constraints
x_1 = np.array([0, -np.sin(np.deg2rad(20)), -np.cos(np.deg2rad(20))])       # Constraint unit vector in inertial frame
x_1 = x_1/np.linalg.norm(x_1)
y_1 = np.array([0, 1, 0])       # Instrument boresight in body frame
cone_angle_1 = np.deg2rad(10) # cone angle around the constraint vector

x_2 = np.array([0, -1, 0])       # Constraint unit vector in inertial frame
x_2 = x_2/np.linalg.norm(x_2)
y_2 = np.array([0, 1, 0])       # Instrument boresight in body frame
cone_angle_2 = np.deg2rad(30) # cone angle around the constraint vector

x_3 = np.array([1, 1, 0])       # Constraint unit vector in inertial frame
x_3 = x_3/np.linalg.norm(x_3)
y_3 = np.array([0, 1, 0])       # Instrument boresight in body frame
cone_angle_3 = np.deg2rad(20) # cone angle around the constraint vector

x_4 = np.array([-1, 1, 0])       # Constraint unit vector in inertial frame
x_4 = x_4/np.linalg.norm(x_4)
y_4 = np.array([0, 1, 0])       # Instrument boresight in body frame
cone_angle_4 = np.deg2rad(20) # cone angle around the constraint vector

x_inc = np.array([1,0,0])
x_inc = x_inc/np.linalg.norm(x_inc)
y_inc = np.array([1, 0, 0])       # Instrument boresight in body frame
cone_angle_inc = np.deg2rad(70) # cone angle around the constraint vector


angle_thres_1 = 0.5*np.max([Ix, Iy, Iz])/L_RW_min * w_max**2 + cone_angle_1
print 'Angle threshold 1', np.rad2deg(angle_thres_1)
angle_thres_1_max = angle_thres_1 + np.deg2rad(5)

angle_thres_2 = 0.5*np.max([Ix, Iy, Iz])/L_RW_min * w_max**2 + cone_angle_2
print 'Angle threshold 2', np.rad2deg(angle_thres_2)
angle_thres_2_max = angle_thres_2 + np.deg2rad(5)

angle_thres_3 = 0.5*np.max([Ix, Iy, Iz])/L_RW_min * w_max**2 + cone_angle_3
print 'Angle threshold 3', np.rad2deg(angle_thres_3)
angle_thres_3_max = angle_thres_3 + np.deg2rad(5)

angle_thres_4 = 0.5*np.max([Ix, Iy, Iz])/L_RW_min * w_max**2 + cone_angle_4
print 'Angle threshold 4', np.rad2deg(angle_thres_4)
angle_thres_4_max = angle_thres_4 + np.deg2rad(5)

######## Simulation parameters
t0 = 0.0
tf = 300.0
dt = 0.01


######## Control
# Gains
P_w = np.eye(3) * 10.0 #1.0

K1_sigma = 0.1 #0.1
K3_sigma = 0.1
K_i = 0.01 * np.eye(3)

# Control loop steps
outer_loop_step = 0.01        # 10 Hz
innerLoop_step = 0.01        # 100 Hz

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
referenceComp = referenceComputer.nadirPointingReference(reference_computer_step, raan, inc, a, mu, initialArgLatitude)
# referenceComp = referenceComputer.regulationReference(reference_computer_step)
# referenceComp.setPoint(np.array([0,0,0])) #np.tan(np.deg2rad(30))/4 * np.array([0,0,1]))
#control = controller.reactionWheelSteeringLawController(estimator, (rw1, rw2, rw3, rw4), referenceComp, outer_loop_step, innerLoop_step, P_w, K_i, K1_sigma, K3_sigma, w_max)
control = controller.constrainedSteeringLawController(estimator, (rw1, rw2, rw3, rw4), referenceComp, outer_loop_step, innerLoop_step, P_w, K_i, K1_sigma, K3_sigma, w_max, 50)
control.addExclusionConstraint(x_1, y_1, cone_angle_1, 'Constraint_1', angle_thres_1, angle_thres_1_max)
control.addExclusionConstraint(x_2, y_2, cone_angle_2, 'Constraint_2', angle_thres_2, angle_thres_2_max)
control.addExclusionConstraint(x_3, y_3, cone_angle_3, 'Constraint_3', angle_thres_3, angle_thres_3_max)
control.addExclusionConstraint(x_4, y_4, cone_angle_4, 'Constraint_4', angle_thres_4, angle_thres_4_max)
control.addInclusionConstraint(x_inc, y_inc, cone_angle_inc, 'Constraint_inc')
control.setMaxWheelTorque(us_max)
simulator.setControls(estimator, control, referenceComp)


# Initial Conditions
BN_0 = coord.ROT1(-np.deg2rad(135))
quat_BN_0 = attitudeKinematics.dcm2quat(BN_0)
mrp_BN_0 = attitudeKinematics.quat2mrp(quat_BN_0)
w_BN_B_0 = np.array([0,0,w_max])
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

time = simulator.getTimeVector()
state_history = simulator.getStateHistory()
state_derivatives_history = simulator.getStateDerivativesHistory()

control_force_history = control.getControlForceHistory()
reference_history = referenceComp.getReferenceHistory()

R_BN = state_history['hub_R_BN']
v_BN = state_history['hub_R_BN_dot']
sigma_BN = state_history['hub_sigma_BN']
w_BN = state_history['hub_omega_BN']
rw_Omega_1 = state_history['RW_1_Omega']
rw_Omega_2 = state_history['RW_2_Omega']
rw_Omega_3 = state_history['RW_3_Omega']
rw_Omega_4 = state_history['RW_4_Omega']

rw_Omega_dot_1 = state_derivatives_history['RW_1_Omega']
rw_Omega_dot_2 = state_derivatives_history['RW_2_Omega']
rw_Omega_dot_3 = state_derivatives_history['RW_3_Omega']
rw_Omega_dot_4 = state_derivatives_history['RW_4_Omega']


rw_us1 = control_force_history['RW_1_us']
rw_us2 = control_force_history['RW_2_us']
rw_us3 = control_force_history['RW_3_us']
rw_us4 = control_force_history['RW_4_us']

sigma_BR = control_force_history['sigma_BR']
w_BR_B = control_force_history['w_BR_B']
sigma_RN = control_force_history['sigma_RN']
w_RN_B = control_force_history['w_RN_B']
Lr = control_force_history['Lr']

w_BR_B_desired = control_force_history['w_BR_B_desired']

V_lyapunov = control_force_history['V_lyapunov']
V_dot_lyapunov = control_force_history['V_dot_lyapunov']
det_Q = control_force_history['det_Q']

V_T = control_force_history['Vect']
Uvect = control_force_history['Uvect']
offset_norm = control_force_history['offset_norm']

y_N_1 = control_force_history['Constraint_1_y_N']
y_N_2 = control_force_history['Constraint_2_y_N']
y_N_3 = control_force_history['Constraint_3_y_N']
y_N_4 = control_force_history['Constraint_4_y_N']
y_N_inc = control_force_history['Constraint_inc_y_N']

angle_1 = control_force_history['Constraint_1_theta']
angle_2 = control_force_history['Constraint_2_theta']
angle_3 = control_force_history['Constraint_3_theta']
angle_4 = control_force_history['Constraint_4_theta']
angle_inc = control_force_history['Constraint_inc_theta']

l = np.size(y_N_1,0)
dec = np.zeros(l-1)
right_as = np.zeros(l-1)
dec_inc = np.zeros(l-1)
right_as_inc = np.zeros(l-1)
cone_1_right_as = np.zeros(l-1)
cone_1_dec = np.zeros(l-1)
cone_2_right_as = np.zeros(l-1)
cone_2_dec = np.zeros(l-1)
cone_3_right_as = np.zeros(l-1)
cone_3_dec = np.zeros(l-1)
cone_4_right_as = np.zeros(l-1)
cone_4_dec = np.zeros(l-1)
cone_inc_right_as = np.zeros(l-1)
cone_inc_dec = np.zeros(l-1)
verts_1 = list()
codes_1 = list()
verts_2 = list()
codes_2 = list()
verts_3 = list()
codes_3 = list()
verts_4 = list()
codes_4 = list()
verts_inc = list()
codes_inc = list()

x_1_orthogonal = np.array([-x_1[1], x_1[0] - x_1[2], x_1[1]])
x_1_orthogonal = x_1_orthogonal/np.linalg.norm(x_1_orthogonal)
x_1_orthogonal_2 = np.cross(x_1, x_1_orthogonal)

x_2_orthogonal = np.array([-x_2[1], x_2[0] - x_2[2], x_2[1]])
x_2_orthogonal = x_2_orthogonal/np.linalg.norm(x_2_orthogonal)
x_2_orthogonal_2 = np.cross(x_2, x_2_orthogonal)

x_3_orthogonal = np.array([-x_3[1], x_3[0] - x_3[2], x_3[1]])
x_3_orthogonal = x_3_orthogonal/np.linalg.norm(x_3_orthogonal)
x_3_orthogonal_2 = np.cross(x_3, x_3_orthogonal)

x_4_orthogonal = np.array([-x_4[1], x_4[0] - x_4[2], x_4[1]])
x_4_orthogonal = x_4_orthogonal/np.linalg.norm(x_4_orthogonal)
x_4_orthogonal_2 = np.cross(x_4, x_4_orthogonal)

x_inc_orthogonal = np.array([-x_inc[1], x_inc[0] - x_inc[2], x_inc[1]])
x_inc_orthogonal = x_inc_orthogonal/np.linalg.norm(x_inc_orthogonal)
x_inc_orthogonal_2 = np.cross(x_inc, x_inc_orthogonal)



for i in range(0, l-1):
    y_N_norm = np.linalg.norm(y_N_1[i])
    y_N_inc_norm = np.linalg.norm(y_N_inc[i])

    dec[i] = np.arcsin(y_N_1[i,2]/y_N_norm)
    right_as[i] = np.arctan2(y_N_1[i,1], y_N_1[i,0])

    dec_inc[i] = np.arcsin(y_N_inc[i,2]/y_N_inc_norm)
    right_as_inc[i] = np.arctan2(y_N_inc[i,1], y_N_inc[i,0])

    cone_vec_1 = np.cos(cone_angle_1) *x_1 + np.sin(cone_angle_1) *(x_1_orthogonal * np.cos(2*np.pi/(l-1) * i) + x_1_orthogonal_2 * np.sin(2*np.pi/(l-1) * i))
    cone_vec_1 = cone_vec_1/np.linalg.norm(cone_vec_1)
    cone_1_right_as[i] = np.arctan2(cone_vec_1[1], cone_vec_1[0])
    cone_1_dec[i] = np.arcsin(cone_vec_1[2])

    cone_vec_2 = np.cos(cone_angle_2) *x_2 + np.sin(cone_angle_2) *(x_2_orthogonal * np.cos(2*np.pi/(l-1) * i) + x_2_orthogonal_2 * np.sin(2*np.pi/(l-1) * i))
    cone_vec_2 = cone_vec_2/np.linalg.norm(cone_vec_2)
    cone_2_right_as[i] = np.arctan2(cone_vec_2[1], cone_vec_2[0])
    cone_2_dec[i] = np.arcsin(cone_vec_2[2])

    cone_vec_3 = np.cos(cone_angle_3) *x_3 + np.sin(cone_angle_3) *(x_3_orthogonal * np.cos(2*np.pi/(l-1) * i) + x_3_orthogonal_2 * np.sin(2*np.pi/(l-1) * i))
    cone_vec_3 = cone_vec_3/np.linalg.norm(cone_vec_3)
    cone_3_right_as[i] = np.arctan2(cone_vec_3[1], cone_vec_3[0])
    cone_3_dec[i] = np.arcsin(cone_vec_3[2])

    cone_vec_4 = np.cos(cone_angle_4) *x_4 + np.sin(cone_angle_4) *(x_4_orthogonal * np.cos(2*np.pi/(l-1) * i) + x_4_orthogonal_2 * np.sin(2*np.pi/(l-1) * i))
    cone_vec_4 = cone_vec_4/np.linalg.norm(cone_vec_4)
    cone_4_right_as[i] = np.arctan2(cone_vec_4[1], cone_vec_4[0])
    cone_4_dec[i] = np.arcsin(cone_vec_4[2])

    cone_vec_inc = x_inc + np.sin(cone_angle_inc) *(x_inc_orthogonal * np.cos(2*np.pi/(l-1) * i) + x_inc_orthogonal_2 * np.sin(2*np.pi/(l-1) * i))
    cone_vec_inc = cone_vec_inc/np.linalg.norm(cone_vec_inc)
    cone_inc_right_as[i] = np.arctan2(cone_vec_inc[1], cone_vec_inc[0])
    cone_inc_dec[i] = np.arcsin(cone_vec_inc[2])

    if i==0:
        codes_1.append(Path.MOVETO)
        codes_2.append(Path.MOVETO)
        codes_3.append(Path.MOVETO)
        codes_4.append(Path.MOVETO)
        codes_inc.append(Path.MOVETO)
    elif i < l-2:
        codes_1.append(Path.CURVE4)
        codes_2.append(Path.CURVE4)
        codes_3.append(Path.CURVE4)
        codes_4.append(Path.CURVE4)
        codes_inc.append(Path.CURVE4)
    else:
        codes_1.append(Path.CLOSEPOLY)
        codes_2.append(Path.CLOSEPOLY)
        codes_3.append(Path.CLOSEPOLY)
        codes_4.append(Path.CLOSEPOLY)
        codes_inc.append(Path.CLOSEPOLY)

    verts_1.append((np.rad2deg(cone_1_right_as[i]), np.rad2deg(cone_1_dec[i])))
    verts_2.append((np.rad2deg(cone_2_right_as[i]), np.rad2deg(cone_2_dec[i])))
    verts_3.append((np.rad2deg(cone_3_right_as[i]), np.rad2deg(cone_3_dec[i])))
    verts_4.append((np.rad2deg(cone_4_right_as[i]), np.rad2deg(cone_4_dec[i])))
    verts_inc.append((np.rad2deg(cone_inc_right_as[i]), np.rad2deg(cone_inc_dec[i])))

dec_proh_1 = np.arcsin(x_1[2]/np.linalg.norm(x_1))
right_as_proh_1 = np.arctan2(x_1[1], x_1[0])

dec_proh_2 = np.arcsin(x_2[2]/np.linalg.norm(x_2))
right_as_proh_2 = np.arctan2(x_2[1], x_2[0])

dec_proh_3 = np.arcsin(x_3[2]/np.linalg.norm(x_3))
right_as_proh_3 = np.arctan2(x_3[1], x_3[0])

dec_proh_4 = np.arcsin(x_4[2]/np.linalg.norm(x_4))
right_as_proh_4 = np.arctan2(x_4[1], x_4[0])

dec_proh_inc = np.arcsin(x_inc[2]/np.linalg.norm(x_inc))
right_as_proh_inc = np.arctan2(x_inc[1], x_inc[0])





T = simulator.getEnergyVector()
H = simulator.getAngularMomentumVector()
P = simulator.getMechanicalPowerVector()


######## Plots
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
# matplotlib.rcParams.update({'font.size': 14})

plt.figure()
plt.hold(True)
plt.plot(time[::10], sigma_BN[::10,0], label=r'$\sigma_{BN_1}$', color='r')
plt.plot(time[::10], sigma_BN[::10,1], label=r'$\sigma_{BN_2}$', color='g')
plt.plot(time[::10], sigma_BN[::10,2], label=r'$\sigma_{BN_3}$', color='b')
plt.plot(time[::10], sigma_RN[::10,0], '--', label=r'$\sigma_{RN_1}$', color='r')
plt.plot(time[::10], sigma_RN[::10,1], '--', label=r'$\sigma_{RN_2}$', color='g')
plt.plot(time[::10], sigma_RN[::10,2], '--', label=r'$\sigma_{RN_3}$', color='b')
plt.xlabel(r'$t$ [sec]', size=27)
plt.ylabel(r'$\sigma_{B/N}$', size=27)
plt.legend(prop={'size':22})
plt.savefig('figures_tracking/sigma_BN.pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time[::10], np.rad2deg(w_BN[::10,0]), label=r'$\omega_{BN_1}$', color='r')
plt.plot(time[::10], np.rad2deg(w_BN[::10,1]), label=r'$\omega_{BN_1}$', color='g')
plt.plot(time[::10], np.rad2deg(w_BN[::10,2]), label=r'$\omega_{BN_1}$', color='b')
plt.plot(time[::10], np.rad2deg(w_RN_B[::10,0]), '--', label=r'$\omega_{RN_1}$', color='r')
plt.plot(time[::10], np.rad2deg(w_RN_B[::10,1]), '--', label=r'$\omega_{RN_2}$', color='g')
plt.plot(time[::10], np.rad2deg(w_RN_B[::10,2]), '--', label=r'$\omega_{RN_3}$', color='b')
plt.axhline(np.rad2deg(w_max), label=r'$\omega_{max}$', color='k', linestyle='--')
plt.axhline(-np.rad2deg(w_max), color='k', linestyle='--')
plt.ylim([-np.rad2deg(w_max)*1.2, np.rad2deg(w_max)*1.2])
plt.xlabel(r'$t$ [sec]', size=27)
plt.ylabel(r'$\omega_{B/N}$ [$^\circ$/sec]', size=27)
plt.legend(prop={'size':22})
plt.savefig('figures_tracking/w_BN.pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time[::10], rw_Omega_1[::10] * 60.0/(2*np.pi), label=r'$\Omega_{RW-1}$')
plt.plot(time[::10], rw_Omega_2[::10] * 60.0/(2*np.pi), label=r'$\Omega_{RW-2}$')
plt.plot(time[::10], rw_Omega_3[::10] * 60.0/(2*np.pi), label=r'$\Omega_{RW-3}$')
plt.plot(time[::10], rw_Omega_4[::10] * 60.0/(2*np.pi), label=r'$\Omega_{RW-4}$')
plt.xlabel(r'$t$ [sec]', size=27)
plt.ylabel(r'$\Omega$ [rpm]', size=27)
plt.legend(prop={'size':22})
plt.savefig('figures_tracking/rw_Omega.pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time[::10], rw_Omega_dot_1[::10] * 60.0/(2*np.pi), label=r'$\dot\Omega_{RW-1}$')
plt.plot(time[::10], rw_Omega_dot_2[::10] * 60.0/(2*np.pi), label=r'$\dot\Omega_{RW-2}$')
plt.plot(time[::10], rw_Omega_dot_3[::10] * 60.0/(2*np.pi), label=r'$\dot\Omega_{RW-3}$')
plt.plot(time[::10], rw_Omega_dot_4[::10] * 60.0/(2*np.pi), label=r'$\dot\Omega_{RW-4}$')
plt.xlabel(r'$t$ [sec]', size=27)
plt.ylabel(r'$\dot\Omega$ [rpm/sec]', size=27)
plt.legend(prop={'size':22})
plt.savefig('figures_tracking/rw_Omega_dot.pdf', bbox_inches='tight', dpi=300)
plt.close()


plt.figure()
plt.hold(True)
plt.plot(time[::10], rw_us1[::10] * 1000, label=r'$u_{s_1}$')
plt.plot(time[::10], rw_us2[::10] * 1000, label=r'$u_{s_2}$')
plt.plot(time[::10], rw_us3[::10] * 1000, label=r'$u_{s_3}$')
plt.plot(time[::10], rw_us4[::10] * 1000, label=r'$u_{s_4}$')
plt.xlabel(r'$t$ [sec]', size=27)
plt.ylabel(r'"$u_{s}$ [mN-m]', size=27)
plt.legend(prop={'size':22})
plt.savefig('figures_tracking/rw_us.pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time[::10], sigma_BR[::10,0], label=r'$\sigma_1$', color='r')
plt.plot(time[::10], sigma_BR[::10,1], label=r'$\sigma_2$', color='g')
plt.plot(time[::10], sigma_BR[::10,2], label=r'$\sigma_3$', color='b')
#plt.ylim([-0.01, 0.01])
plt.xlabel(r'$t$ [sec]', size=27)
plt.ylabel(r'$\sigma_{B/R}$', size=27)
plt.legend(prop={'size':22})
plt.savefig('figures_tracking/sigma_BR.pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time[::10], np.rad2deg(w_BR_B[::10,0]), label=r'$\omega_1$', color='r')
plt.plot(time[::10], np.rad2deg(w_BR_B[::10,1]), label=r'$\omega_2$', color='g')
plt.plot(time[::10], np.rad2deg(w_BR_B[::10,2]), label=r'$\omega_3$', color='b')
plt.plot(time[::10], np.rad2deg(w_BR_B_desired[::10,0]), '--', label=r'$\omega_{1d}$', color='r')
plt.plot(time[::10], np.rad2deg(w_BR_B_desired[::10,1]), '--', label=r'$\omega_{2d}$', color='g')
plt.plot(time[::10], np.rad2deg(w_BR_B_desired[::10,2]), '--', label=r'$\omega_{3d}$', color='b')
plt.xlabel(r'$t$ [sec]', size=27)
plt.ylabel(r'$\omega_{B/R}$ [$^\circ$/sec]', size=27)
plt.legend(prop={'size':22})
plt.savefig('figures_tracking/w_BR.pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time[::10], Lr[::10,0], label=r'$L_{r1}$')
plt.plot(time[::10], Lr[::10,1], label=r'$L_{r1}$')
plt.plot(time[::10], Lr[::10,2], label=r'$L_{r1}$')
plt.xlabel(r'$t$ [sec]', size=27)
plt.ylabel(r'$L_r$', size=27)
plt.legend(prop={'size':22})
plt.savefig('figures_tracking/Lr.pdf', bbox_inches='tight', dpi=300)
plt.close()

# plt.figure()
# plt.hold(True)
# plt.plot(time, V_lyapunov)
# plt.xlabel("Time [sec]")
# plt.ylabel("$V_{lyap}$", size=18)
# plt.savefig('figures_tracking/V_lyapunov.pdf', bbox_inches='tight', dpi=300)
# plt.close()
#
# plt.figure()
# plt.hold(True)
# plt.plot(time, V_dot_lyapunov)
# plt.xlabel("Time [sec]")
# plt.ylabel("$\dot V_{lyap}$", size=18)
# plt.savefig('figures_tracking/V_dot_lyapunov.pdf', bbox_inches='tight', dpi=300)
# plt.close()


plt.figure()
plt.hold(True)
fig = plt.figure()
plt.plot(time[::10], np.sqrt(V_T[::10,0]**2 + V_T[::10,1]**2 + V_T[::10,2]**2))
plt.xlabel(r'$t$ [sec]', size=27)
plt.ylabel(r'$||v_T||$', size=27)
plt.legend()
plt.savefig('figures_tracking/V_T_norm.pdf', bbox_inches='tight', dpi=300)
plt.close()

#
# plt.figure()
# plt.hold(True)
# plt.plot(time, Uvect[:,0],label="u_1")
# plt.plot(time, Uvect[:,1],label="u_2")
# plt.plot(time, Uvect[:,2],label="u_3")
# plt.xlabel("Time [sec]")
# plt.ylabel("U_T", size=18)
# plt.legend()
# plt.savefig('figures_tracking/Uvect.pdf', bbox_inches='tight', dpi=300)
# plt.close()

plt.figure()
plt.hold(True)
plt.plot(time, offset_norm)
plt.xlabel("Time [sec]")
plt.ylabel("offset norm", size=18)
plt.savefig('figures_tracking/offset_norm.pdf', bbox_inches='tight', dpi=300)
plt.close()

path1 = Path(verts_1, codes_1)
path2 = Path(verts_2, codes_2)
path3 = Path(verts_3, codes_3)
path4 = Path(verts_4, codes_4)
fig = plt.figure()
ax = fig.add_subplot(111)
patch1 = patches.PathPatch(path1, facecolor='#CD4141', lw=2, edgecolor="none")
patch2 = patches.PathPatch(path2, facecolor='#CD4141', lw=2, edgecolor="none")
patch3 = patches.PathPatch(path3, facecolor='#CD4141', lw=2, edgecolor="none")
patch4 = patches.PathPatch(path4, facecolor='#CD4141', lw=2, edgecolor="none")
ax.add_patch(patch1)
ax.add_patch(patch2)
ax.add_patch(patch3)
ax.add_patch(patch4)
plt.hold(True)
plt.plot(right_as[::10]*180/np.pi, dec[::10]*180/np.pi,'.',markersize=2,label='Camera 1 vector tip')
plt.plot([right_as[0]*180/np.pi], [dec[0]*180/np.pi], marker='x', markersize=10, color="r",label='start')
plt.plot([right_as[-1]*180/np.pi], [dec[-1]*180/np.pi], marker='x', markersize=10, color="g",label='end')
ax.text(-97, -76, r'$1$',size=35)
ax.text(-97, -5, r'$2$',size=35)
ax.text(38, -5, r'$3$',size=35)
ax.text(128, -5, r'$4$',size=35)
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xlabel(r'Right ascension [$^\circ$]',size=27)
plt.ylabel(r'Declination [$^\circ$]',size=27)
plt.legend(loc='lower right',prop={'size':19})
plt.savefig('figures_tracking/map.pdf', bbox_inches='tight', dpi=300)
plt.close()

pathinc = Path(verts_inc, codes_inc)
fig = plt.figure()
ax = fig.add_subplot(111)
patchinc = patches.PathPatch(pathinc, facecolor='#08EE64', lw=2, edgecolor="none")
ax.add_patch(patchinc)
plt.hold(True)
plt.plot(right_as_inc[::10]*180/np.pi, dec_inc[::10]*180/np.pi,'.',markersize=2,label='Antenna boresight vector tip')
plt.plot([right_as_inc[0]*180/np.pi], [dec_inc[0]*180/np.pi], marker='x', markersize=10, color="r",label='start')
plt.plot([right_as_inc[-1]*180/np.pi], [dec_inc[-1]*180/np.pi], marker='x', markersize=10, color="g",label='end')
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xlabel(r'Right ascension [$^\circ$]',size=27)
plt.ylabel(r'Declination [$^\circ$]',size=27)
plt.legend(loc='lower right',prop={'size':19})
plt.savefig('figures_tracking/map_inc.pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time[::10], np.rad2deg(angle_1[::10]), label=r'$\theta_1$')
plt.axhline(np.rad2deg(cone_angle_1), label=r'$\theta_{min1}$', linestyle='-.', color='k')
plt.axhline(np.rad2deg(angle_thres_1), label=r'$\theta_{01}$', linestyle='--', color='c')
plt.axhline(np.rad2deg(angle_thres_1_max), label=r'$\theta_{01} + \psi$', linestyle='--', color='m')
plt.ylim([0,180])
plt.legend(prop={'size':22})
plt.xlabel(r'$t$ [sec]', size=27)
plt.ylabel(r'$\theta_1$ [$^\circ$]', size=27)
plt.savefig('figures_tracking/theta1.pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time[::10], np.rad2deg(angle_2[::10]), label=r'$\theta_2$')
plt.axhline(np.rad2deg(cone_angle_2), label=r'$\theta_{min2}$', linestyle='-.', color='k')
plt.axhline(np.rad2deg(angle_thres_2), label=r'$\theta_{02}$', linestyle='--', color='c')
plt.axhline(np.rad2deg(angle_thres_2_max), label=r'$\theta_{02} + \psi$', linestyle='--', color='m')
plt.ylim([0,180])
plt.legend(prop={'size':22})
plt.xlabel(r'$t$ [sec]', size=27)
plt.ylabel(r'$\theta_2$ [$^\circ$]', size=27)
plt.savefig('figures_tracking/theta2.pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time[::10], np.rad2deg(angle_3[::10]), label=r'$\theta_3$')
plt.axhline(np.rad2deg(cone_angle_3), label=r'$\theta_{min3}$', linestyle='-.', color='k')
plt.axhline(np.rad2deg(angle_thres_3), label=r'$\theta_{03}$', linestyle='--', color='c')
plt.axhline(np.rad2deg(angle_thres_3_max), label=r'$\theta_{03} + \psi$', linestyle='--', color='m')
plt.ylim([0,180])
plt.legend(prop={'size':22})
plt.xlabel(r'$t$ [sec]', size=27)
plt.ylabel(r'$\theta_3$ [$^\circ$]', size=27)
plt.savefig('figures_tracking/theta3.pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time[::10], np.rad2deg(angle_4[::10]), label=r'$\theta_4$')
plt.axhline(np.rad2deg(cone_angle_4), label=r'$\theta_{min4}$', linestyle='-.', color='k')
plt.axhline(np.rad2deg(angle_thres_4), label=r'$\theta_{04}$', linestyle='--', color='c')
plt.axhline(np.rad2deg(angle_thres_4_max), label=r'$\theta_{04} + \psi$', linestyle='--', color='m')
plt.ylim([0,180])
plt.legend(prop={'size':22})
plt.xlabel(r'$t$ [sec]', size=27)
plt.ylabel(r'$\theta_4$ [$^\circ$]', size=27)
plt.savefig('figures_tracking/theta4.pdf', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time[::10], np.rad2deg(angle_inc[::10]), label=r'$\theta_{5}$')
plt.axhline(np.rad2deg(cone_angle_inc), label=r'$\theta_{min-5}$', linestyle='-.', color='k')
plt.ylim([0,180])
plt.legend(prop={'size':22})
plt.xlabel(r'$t$ [sec]', size=27)
plt.ylabel(r'$\theta_{5}$ [$^\circ$]', size=27)
plt.savefig('figures_tracking/thetainc.pdf', bbox_inches='tight', dpi=300)
plt.close()


# plt.figure()
# plt.hold(True)
# plt.plot(time, T)
# plt.xlabel("Time [sec]")
# plt.ylabel("$T$ $[J]$", size=18)
# plt.savefig('figures_tracking/energy.pdf', bbox_inches='tight', dpi=300)
# plt.close()
#
# plt.figure()
# plt.hold(True)
# plt.plot(time, P)
# plt.xlabel("Time [sec]")
# plt.ylabel("$P$ $[W]$", size=18)
# plt.savefig('figures_tracking/power.pdf', bbox_inches='tight', dpi=300)
# plt.close()