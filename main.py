
import numpy as np
import simulatorManager
import estimator
import controller
import referenceComputer
import attitudeKinematics
import matplotlib.pyplot as plt


##### Parameters
#Hub: cylinder with radius = 0.5 m, height = 1 m
m_hub = 30.0                        # [kg] Hub mass
r_hub = 0.5                         # [m] radius
h_hub = 1.0                         # [m] height
Iz = m_hub * r_hub**2/2
Ix = Iy = m_hub/12 * (3*r_hub**2 + h_hub**2)
I_hub = np.diag([Ix, Iy, Iz])       # [kg m^2] Principal axis inertia matrix

# VSCMG
m_vscmg = 2.0
Iws = 0.03
Iwt = 0.001
Igs = 0.0005
Igt = 0.0005
Igg = 0.0005

# Max torques
us_max = 0.015                  # [Nm]
ug_max = 1.0#0.5                    # [Nm]

# Nominal speed
nominal_speed_wheel = 200.0    # [rpm]

# CoM offset
R_O1B_vscmg = np.array([0.1,0,0])        # [m] Position of the center of mass of the VSCMG relative to the reference B
R_O2B_vscmg = np.array([0,0.1,0])        # [m] Position of the center of mass of the VSCMG relative to the reference B
R_O3B_vscmg = np.array([-0.1,0,0])       # [m] Position of the center of mass of the VSCMG relative to the reference B
R_O4B_vscmg = np.array([0,-0.1,0])       # [m] Position of the center of mass of the VSCMG relative to the reference B

r_BcB_B = -m_vscmg * (R_O1B_vscmg + R_O2B_vscmg + R_O3B_vscmg + R_O4B_vscmg)/m_hub     # [m] Position of the center of mass of the hub relative to the reference B

# Position of the center of mass relative to the reference
r_CB_B =  m_hub * r_BcB_B + m_vscmg* (R_O1B_vscmg + R_O2B_vscmg + R_O3B_vscmg + R_O4B_vscmg)

# VSCMG configuration: pyramid
nmbr_vscmgs = 4
pyramid_ang = np.deg2rad(55)        # Internal angle of the pyramid
cos_p_ang = np.cos(pyramid_ang)
sin_p_ang = np.sin(pyramid_ang)

gs1 = np.array([sin_p_ang, 0, cos_p_ang])
gt1 = np.array([0,1,0])
gg1 = np.array([-cos_p_ang, 0, sin_p_ang])
BG1_0 = np.array([gs1, gt1, gg1]).T     # Initial attitude of body frame relative to G (gs, gt, gg vectors as columns)

gs2 = np.array([0, sin_p_ang, cos_p_ang])
gt2 = np.array([-1,0,0])
gg2 = np.array([0, -cos_p_ang, sin_p_ang])
BG2_0 = np.array([gs2, gt2, gg2]).T     # Initial attitude of body frame relative to G (gs, gt, gg vectors as columns)

gs3 = np.array([-sin_p_ang, 0, cos_p_ang])
gt3 = np.array([0,-1,0])
gg3 = np.array([cos_p_ang, 0, sin_p_ang])
BG3_0 = np.array([gs3, gt3, gg3]).T     # Initial attitude of body frame relative to G (gs, gt, gg vectors as columns)

gs4 = np.array([0, -sin_p_ang, cos_p_ang])
gt4 = np.array([1,0,0])
gg4 = np.array([0, cos_p_ang, sin_p_ang])
BG4_0 = np.array([gs4, gt4, gg4]).T     # Initial attitude of body frame relative to G (gs, gt, gg vectors as columns)

det_BG1_0 = np.linalg.det(BG1_0)
det_BG2_0 = np.linalg.det(BG2_0)
det_BG3_0 = np.linalg.det(BG3_0)
det_BG4_0 = np.linalg.det(BG4_0)


# Orbital parameters
R_T = 6378.0                # [km] Earth radius
gamma = 7.2925e-05          # [rad/sec] Rotation rate
mu = 398600.0               # [km^2/s^2] Gravitational parameters

raan = np.deg2rad(20)       # [rad] Right Ascension of Ascending Node
inc = np.deg2rad(56)        # [rad] Inclination
h = 400.0                   # [km] Altitude
a = R_T + h                 # [km] Semimajor axis

######## Simulation parameters
t0 = 0.0
tf = 140.0
dt = 0.01

######## Control
USE_CONTROL = True
CONTROL_TWO_LOOPS = True
USE_CONSTANT_TORQUES = False

if not USE_CONTROL and not USE_CONSTANT_TORQUES:
    plot_super = ''
elif USE_CONSTANT_TORQUES:
    plot_super = '_torques'
elif USE_CONTROL and not CONTROL_TWO_LOOPS:
    plot_super = '_control_1_loop'
else:
    plot_super = '_control_2_loops'

# Gains
K_sigma = 1.0
P_w = np.eye(3) * 5.0
K_gamma_dot = np.ones(nmbr_vscmgs) * 10.0
K_Omega_dot = np.ones(nmbr_vscmgs) * 1.0
weights = np.ones(2 * nmbr_vscmgs)
weights[:nmbr_vscmgs] = weights[:nmbr_vscmgs] * 20 #20
mu_weights = np.ones(nmbr_vscmgs) * 1.0 #1e-1 #10

# Control loop steps
outer_loop_step = 0.1        # 10 Hz
innerLoop_step = 0.01        # 100 Hz

# set Point
mrp_RN = np.array([0.0, 0.0, 0.0])

##### Reference computer
reference_computer_step = dt

##### Estimator
estimator_step = dt

# Constant torques
if USE_CONSTANT_TORQUES:
    ug = 0.00001       # [Nm]
    us = 0.001         # [Nm]
else:
    ug = 0.0          # [Nm]
    us = 0.0          # [Nm]

simulator = simulatorManager.simulatorManager.getSimulatorManager('spacecraft_backSub', 'rk4', 'hub')
simulator.setSimulationTimes(t0, tf, dt)

spacecraft = simulator.getDynamicalObject()
spacecraft.setHubMass(m_hub)
spacecraft.setHubInertia(I_hub)
spacecraft.setHubCoMOffset(r_BcB_B)

vscmg1 = spacecraft.addVSCMG('VSCMG_1', m_vscmg, R_O1B_vscmg, Igs, Igt, Igg, Iws, Iwt, BG1_0, nominal_speed_wheel, ug, us)
vscmg2 = spacecraft.addVSCMG('VSCMG_2', m_vscmg, R_O2B_vscmg, Igs, Igt, Igg, Iws, Iwt, BG2_0, nominal_speed_wheel, ug, us)
vscmg3 = spacecraft.addVSCMG('VSCMG_3', m_vscmg, R_O3B_vscmg, Igs, Igt, Igg, Iws, Iwt, BG3_0, nominal_speed_wheel, ug, us)
vscmg4 = spacecraft.addVSCMG('VSCMG_4', m_vscmg, R_O4B_vscmg, Igs, Igt, Igg, Iws, Iwt, BG4_0, nominal_speed_wheel, ug, us)


if USE_CONTROL:
    estimator = estimator.idealEstimator(spacecraft, estimator_step, t0,'hub_sigma_BN', 'hub_omega_BN')
    referenceComp = referenceComputer.nadirPointingReference(reference_computer_step, t0, raan, inc, a, mu)
    control = controller.vscmgSteeringController(estimator, (vscmg1, vscmg2, vscmg3, vscmg4), referenceComp, outer_loop_step, innerLoop_step, t0, K_sigma, P_w, K_gamma_dot, K_Omega_dot, weights, mu_weights)
    control.setMaxGimbalTorque(ug_max)
    control.setMaxWheelTorque(us_max)
    simulator.setControls(estimator, control, referenceComp)

# Initial Conditions
simulator.setInitialConditions('hub_R_BN', np.array([0.0, 0.0, 0.0]))
simulator.setInitialConditions('hub_R_BN_dot', np.array([0.0, 0.0, 0.0]))
simulator.setInitialConditions('hub_sigma_BN', np.array([0.2, 0.3, -0.1]))
simulator.setInitialConditions('hub_omega_BN', np.array([0.02,-0.01,0.02]))
simulator.setInitialConditions('VSCMG_1_gamma', np.deg2rad(90.0))
simulator.setInitialConditions('VSCMG_1_gamma_dot', 0.0)
simulator.setInitialConditions('VSCMG_1_Omega', nominal_speed_wheel * 2*np.pi/(60))
simulator.setInitialConditions('VSCMG_2_gamma', np.deg2rad(90.0))
simulator.setInitialConditions('VSCMG_2_gamma_dot', 0.0)
simulator.setInitialConditions('VSCMG_2_Omega', nominal_speed_wheel * 2*np.pi/(60))
simulator.setInitialConditions('VSCMG_3_gamma', np.deg2rad(0.0))
simulator.setInitialConditions('VSCMG_3_gamma_dot', 0.0)
simulator.setInitialConditions('VSCMG_3_Omega', -nominal_speed_wheel * 2*np.pi/(60))
simulator.setInitialConditions('VSCMG_4_gamma', np.deg2rad(0.0))
simulator.setInitialConditions('VSCMG_4_gamma_dot', 0.0)
simulator.setInitialConditions('VSCMG_4_Omega', -nominal_speed_wheel * 2*np.pi/(60))

simulator.computeEnergy(True)
simulator.computeAngularMomentum(True)

simulator.simulate()

time = simulator.getTimeVector()
state_history = simulator.getStateHistory()
state_derivatives_history = simulator.getStateDerivativesHistory()

if USE_CONTROL:
    control_force_history = control.getControlForceHistory()
    reference_history = referenceComp.getReferenceHistory()

R_BN = state_history['hub_R_BN']
v_BN = state_history['hub_R_BN_dot']
sigma_BN = state_history['hub_sigma_BN']
w_BN = state_history['hub_omega_BN']
vscmg_gamma_1 = state_history['VSCMG_1_gamma']
vscmg_gamma_dot_1 = state_history['VSCMG_1_gamma_dot']
vscmg_Omega_1 = state_history['VSCMG_1_Omega']
vscmg_gamma_2 = state_history['VSCMG_2_gamma']
vscmg_gamma_dot_2 = state_history['VSCMG_2_gamma_dot']
vscmg_Omega_2 = state_history['VSCMG_2_Omega']
vscmg_gamma_3 = state_history['VSCMG_3_gamma']
vscmg_gamma_dot_3 = state_history['VSCMG_3_gamma_dot']
vscmg_Omega_3 = state_history['VSCMG_3_Omega']
vscmg_gamma_4 = state_history['VSCMG_4_gamma']
vscmg_gamma_dot_4 = state_history['VSCMG_4_gamma_dot']
vscmg_Omega_4 = state_history['VSCMG_4_Omega']

vscmg_Omega_dot_1 = state_derivatives_history['VSCMG_1_Omega']
vscmg_Omega_dot_2 = state_derivatives_history['VSCMG_2_Omega']
vscmg_Omega_dot_3 = state_derivatives_history['VSCMG_3_Omega']
vscmg_Omega_dot_4 = state_derivatives_history['VSCMG_4_Omega']

vscmg_gamma_ddot_1 = state_derivatives_history['VSCMG_1_gamma_dot']
vscmg_gamma_ddot_2 = state_derivatives_history['VSCMG_2_gamma_dot']
vscmg_gamma_ddot_3 = state_derivatives_history['VSCMG_3_gamma_dot']
vscmg_gamma_ddot_4 = state_derivatives_history['VSCMG_4_gamma_dot']

if USE_CONTROL:
    vscmg_ug1 = control_force_history['VSCMG_1_ug']
    vscmg_us1 = control_force_history['VSCMG_1_us']
    vscmg_ug2 = control_force_history['VSCMG_2_ug']
    vscmg_us2 = control_force_history['VSCMG_2_us']
    vscmg_ug3 = control_force_history['VSCMG_3_ug']
    vscmg_us3 = control_force_history['VSCMG_3_us']
    vscmg_ug4 = control_force_history['VSCMG_4_ug']
    vscmg_us4 = control_force_history['VSCMG_4_us']

    vscmg_Omega_desired1 = control_force_history['VSCMG_1_Omega_desired']
    vscmg_Omega_dot_desired1 = control_force_history['VSCMG_1_Omega_dot_desired']
    vscmg_gamma_dot_desired1 = control_force_history['VSCMG_1_gamma_dot_desired']
    vscmg_gamma_ddot_desired1 = control_force_history['VSCMG_1_gamma_ddot_desired']
    vscmg_Omega_desired2 = control_force_history['VSCMG_2_Omega_desired']
    vscmg_Omega_dot_desired2 = control_force_history['VSCMG_2_Omega_dot_desired']
    vscmg_gamma_dot_desired2 = control_force_history['VSCMG_2_gamma_dot_desired']
    vscmg_gamma_ddot_desired2 = control_force_history['VSCMG_2_gamma_ddot_desired']
    vscmg_Omega_desired3 = control_force_history['VSCMG_3_Omega_desired']
    vscmg_Omega_dot_desired3 = control_force_history['VSCMG_3_Omega_dot_desired']
    vscmg_gamma_dot_desired3 = control_force_history['VSCMG_3_gamma_dot_desired']
    vscmg_gamma_ddot_desired3 = control_force_history['VSCMG_3_gamma_ddot_desired']
    vscmg_Omega_desired4 = control_force_history['VSCMG_4_Omega_desired']
    vscmg_Omega_dot_desired4 = control_force_history['VSCMG_4_Omega_dot_desired']
    vscmg_gamma_dot_desired4 = control_force_history['VSCMG_4_gamma_dot_desired']
    vscmg_gamma_ddot_desired4 = control_force_history['VSCMG_4_gamma_ddot_desired']

    vscmg_delta_1 = control_force_history['VSCMG_1_delta']
    vscmg_delta_2 = control_force_history['VSCMG_2_delta']
    vscmg_delta_3 = control_force_history['VSCMG_3_delta']
    vscmg_delta_4 = control_force_history['VSCMG_4_delta']

    sigma_BR = control_force_history['sigma_BR']
    w_BR_B = control_force_history['w_BR_B']
    sigma_RN = control_force_history['sigma_RN']
    w_RN_B = control_force_history['w_RN_B']
    Lr = control_force_history['Lr']


gt1 = np.zeros(((tf-t0)/dt + 1,3))
gt2 = np.zeros(((tf-t0)/dt + 1,3))
gt3 = np.zeros(((tf-t0)/dt + 1,3))
gt4 = np.zeros(((tf-t0)/dt + 1,3))
for i in range(0,int((tf-t0)/dt+1)):
    GG1_0 = attitudeKinematics.ROT3(vscmg_gamma_1[i])
    GG2_0 = attitudeKinematics.ROT3(vscmg_gamma_2[i])
    GG3_0 = attitudeKinematics.ROT3(vscmg_gamma_3[i])
    GG4_0 = attitudeKinematics.ROT3(vscmg_gamma_4[i])

    BG1 = BG1_0.dot(GG1_0.T)
    BG2 = BG2_0.dot(GG2_0.T)
    BG3 = BG3_0.dot(GG3_0.T)
    BG4 = BG4_0.dot(GG4_0.T)

    gt1[i,:] = BG1[:,1]
    gt2[i,:] = BG2[:,1]
    gt3[i,:] = BG3[:,1]
    gt4[i,:] = BG4[:,1]

T = simulator.getEnergyVector()
H = simulator.getAngularMomentumVector()
P = simulator.getMechanicalPowerVector()

P_numerical = np.zeros(T.size)
for i in range(1, T.size):
    P_numerical[i] = (T[i] - T[i-1])/dt

P_numerical[0] = P_numerical[1]

plt.figure()
plt.hold(True)
plt.plot(time, R_BN[:,0], label='$x_{BN}$')
plt.plot(time, R_BN[:,1], label='$y_{BN}$')
plt.plot(time, R_BN[:,2], label='$z_{BN}$')
plt.xlabel("Time [sec]")
plt.ylabel("$R_{BN}$ $[m]$", size=18)
plt.legend()
plt.savefig('../report/include/R_BN' + plot_super +'.png', bbox_inches='tight', dpi=300)
plt.close()
#
# plt.figure()
# plt.hold(True)
# plt.plot(time, v_BN[:,0], label='$\dot x_{BN}$')
# plt.plot(time, v_BN[:,1], label='$\dot y_{BN}$')
# plt.plot(time, v_BN[:,2], label='$\dot z_{BN}$')
# plt.xlabel("Time [sec]")
# plt.ylabel("$v_{BN}$ $[m/s]$", size=18)
# plt.legend()
# plt.savefig('../report/include/v_BN' + plot_super +'.png', bbox_inches='tight', dpi=300)
# plt.close()

plt.figure()
plt.hold(True)
plt.plot(time, sigma_BN[:,0], label='$\sigma_1$', color='r')
plt.plot(time, sigma_BN[:,1], label='$\sigma_2$', color='g')
plt.plot(time, sigma_BN[:,2], label='$\sigma_3$', color='b')
if USE_CONTROL and CONTROL_TWO_LOOPS:
    plt.plot(time, sigma_RN[:,0], '--', label='$\sigma_{RN_1}$', color='r')
    plt.plot(time, sigma_RN[:,1], '--', label='$\sigma_{RN_2}$', color='g')
    plt.plot(time, sigma_RN[:,2], '--', label='$\sigma_{RN_3}$', color='b')
plt.xlabel("Time [sec]")
plt.ylabel("$\sigma$", size=18)
plt.legend()
plt.savefig('../report/include/sigma_BN' + plot_super +'.png', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time, w_BN[:,0], label='$\omega_1$', color='r')
plt.plot(time, w_BN[:,1], label='$\omega_2$', color='g')
plt.plot(time, w_BN[:,2], label='$\omega_3$', color='b')
if USE_CONTROL and CONTROL_TWO_LOOPS:
    plt.plot(time, w_RN_B[:,0], '--', label='$\omega_{RN_1}$', color='r')
    plt.plot(time, w_RN_B[:,1], '--', label='$\omega_{RN_2}$', color='g')
    plt.plot(time, w_RN_B[:,2], '--', label='$\omega_{RN_3}$', color='b')
plt.ylim([-0.01,0.01])
plt.xlabel("Time [sec]")
plt.ylabel("$\omega$ $[rad/sec]$", size=18)
plt.legend()
plt.savefig('../report/include/w_BN' + plot_super +'.png', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time, H[:,0], label='$H_1$')
plt.plot(time, H[:,1], label='$H_2$')
plt.plot(time, H[:,2], label='$H_3$')
plt.xlabel("Time [sec]")
plt.ylabel("$H$ $[kg m^2/s]$", size=18)
plt.legend()
plt.savefig('../report/include/angular_momentum' + plot_super +'.png', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time, vscmg_Omega_1 * 60.0/(2*np.pi), label='$\Omega_{VSCMG-1}$')
plt.plot(time, vscmg_Omega_2 * 60.0/(2*np.pi), label='$\Omega_{VSCMG-2}$')
plt.plot(time, vscmg_Omega_3 * 60.0/(2*np.pi), label='$\Omega_{VSCMG-3}$')
plt.plot(time, vscmg_Omega_4 * 60.0/(2*np.pi), label='$\Omega_{VSCMG-4}$')
if USE_CONTROL:
    plt.plot(time, vscmg_Omega_desired1 * 60.0/(2*np.pi), '--', label='$\Omega_{VSCMG-1-d}$')
    plt.plot(time, vscmg_Omega_desired2 * 60.0/(2*np.pi), '--', label='$\Omega_{VSCMG-2-d}$')
    plt.plot(time, vscmg_Omega_desired3 * 60.0/(2*np.pi), '--', label='$\Omega_{VSCMG-3-d}$')
    plt.plot(time, vscmg_Omega_desired4 * 60.0/(2*np.pi), '--', label='$\Omega_{VSCMG-4-d}$')
plt.xlabel("Time [sec]")
plt.ylabel("$\Omega$ $[rpm]$", size=18)
plt.legend()
plt.savefig('../report/include/vscmg_omega' + plot_super +'.png', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time, vscmg_Omega_dot_1 * 60.0/(2*np.pi), label='$\dot\Omega_{VSCMG-1}$')
plt.plot(time, vscmg_Omega_dot_2 * 60.0/(2*np.pi), label='$\dot\Omega_{VSCMG-2}$')
plt.plot(time, vscmg_Omega_dot_3 * 60.0/(2*np.pi), label='$\dot\Omega_{VSCMG-3}$')
plt.plot(time, vscmg_Omega_dot_4 * 60.0/(2*np.pi), label='$\dot\Omega_{VSCMG-4}$')
if USE_CONTROL:
    plt.plot(time, vscmg_Omega_dot_desired1 * 60.0/(2*np.pi), '--', label='$\dot\Omega_{VSCMG-1-d}$')
    plt.plot(time, vscmg_Omega_dot_desired2 * 60.0/(2*np.pi), '--', label='$\dot\Omega_{VSCMG-2-d}$')
    plt.plot(time, vscmg_Omega_dot_desired3 * 60.0/(2*np.pi), '--', label='$\dot\Omega_{VSCMG-3-d}$')
    plt.plot(time, vscmg_Omega_dot_desired4 * 60.0/(2*np.pi), '--', label='$\dot\Omega_{VSCMG-4-d}$')
plt.xlabel("Time [sec]")
plt.ylabel("$\Omega$ $[rpm/sec]$", size=18)
plt.legend()
plt.savefig('../report/include/vscmg_omega_dot' + plot_super +'.png', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time, vscmg_gamma_1 * 180/np.pi, label='$\gamma_{VSCMG-1}$')
plt.plot(time, vscmg_gamma_2 * 180/np.pi, label='$\gamma_{VSCMG-2}$')
plt.plot(time, vscmg_gamma_3 * 180/np.pi, label='$\gamma_{VSCMG-3}$')
plt.plot(time, vscmg_gamma_4 * 180/np.pi, label='$\gamma_{VSCMG-4}$')
plt.xlabel("Time [sec]")
plt.ylabel("$\gamma_{VSCMG}$ $[^\circ]$", size=18)
plt.legend()
plt.savefig('../report/include/vscmg_gamma' + plot_super +'.png', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time, vscmg_gamma_dot_1 * 180/np.pi, label='$\dot\gamma_{VSCMG-1}$')
plt.plot(time, vscmg_gamma_dot_2 * 180/np.pi, label='$\dot\gamma_{VSCMG-2}$')
plt.plot(time, vscmg_gamma_dot_3 * 180/np.pi, label='$\dot\gamma_{VSCMG-3}$')
plt.plot(time, vscmg_gamma_dot_4 * 180/np.pi, label='$\dot\gamma_{VSCMG-4}$')
if USE_CONTROL:
    plt.plot(time, vscmg_gamma_dot_desired1 * 180/np.pi, '--', label='$\dot\gamma_{VSCMG-1-d}$')
    plt.plot(time, vscmg_gamma_dot_desired2 * 180/np.pi, '--', label='$\dot\gamma_{VSCMG-2-d}$')
    plt.plot(time, vscmg_gamma_dot_desired3 * 180/np.pi, '--', label='$\dot\gamma_{VSCMG-3-d}$')
    plt.plot(time, vscmg_gamma_dot_desired4 * 180/np.pi, '--', label='$\dot\gamma_{VSCMG-4-d}$')
plt.xlabel("Time [sec]")
plt.ylabel("$\dot\gamma_{VSCMG}$ $[^\circ/s]$", size=18)
plt.legend()
plt.savefig('../report/include/vscmg_gamma_dot' + plot_super +'.png', bbox_inches='tight', dpi=300)
plt.close()

if USE_CONTROL:
    plt.figure()
    plt.hold(True)
    plt.plot(time, vscmg_ug1 * 1000, label='$u_{g_{VSCMG-1}}$')
    plt.plot(time, vscmg_ug2 * 1000, label='$u_{g_{VSCMG-2}}$')
    plt.plot(time, vscmg_ug3 * 1000, label='$u_{g_{VSCMG-3}}$')
    plt.plot(time, vscmg_ug4 * 1000, label='$u_{g_{VSCMG-4}}$')
    plt.xlabel("Time [sec]")
    plt.ylabel("$u_{g_{VSCMG}}$ $[mN m]$", size=18)
    plt.legend()
    plt.savefig('../report/include/vscmg_ug' + plot_super +'.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure()
    plt.hold(True)
    plt.plot(time, vscmg_us1 * 1000, label='$u_{s_{VSCMG-1}}$')
    plt.plot(time, vscmg_us2 * 1000, label='$u_{s_{VSCMG-2}}$')
    plt.plot(time, vscmg_us3 * 1000, label='$u_{s_{VSCMG-3}}$')
    plt.plot(time, vscmg_us4 * 1000, label='$u_{s_{VSCMG-4}}$')
    plt.xlabel("Time [sec]")
    plt.ylabel("$u_{s_{VSCMG}}$ $[mN m]$", size=18)
    plt.legend()
    plt.savefig('../report/include/vscmg_us' + plot_super +'.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure()
    plt.hold(True)
    plt.plot(time, vscmg_delta_1, label='$\delta_1$')
    plt.plot(time, vscmg_delta_2, label='$\delta_2$')
    plt.plot(time, vscmg_delta_3, label='$\delta_3$')
    plt.plot(time, vscmg_delta_4, label='$\delta_4$')
    plt.ylim([-1,5])
    plt.xlabel("Time [sec]")
    plt.ylabel("$\delta$", size=18)
    plt.legend()
    plt.savefig('../report/include/vscmg_delta' + plot_super +'.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure()
    plt.hold(True)
    plt.plot(time, sigma_BR[:,0], label='$\sigma_{BR-1}$')
    plt.plot(time, sigma_BR[:,1], label='$\sigma_{BR-2}$')
    plt.plot(time, sigma_BR[:,2], label='$\sigma_{BR-3}$')
    plt.xlabel("Time [sec]")
    plt.ylabel("$\sigma_{BR}$", size=18)
    plt.legend()
    plt.savefig('../report/include/sigma_BR' + plot_super +'.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure()
    plt.hold(True)
    plt.plot(time, w_BR_B[:,0], label='$\omega_{BR-1}$')
    plt.plot(time, w_BR_B[:,1], label='$\omega_{BR-2}$')
    plt.plot(time, w_BR_B[:,2], label='$\omega_{BR-3}$')
    plt.xlabel("Time [sec]")
    plt.ylabel("$\omega_{BR}$ $[rad/sec]$", size=18)
    plt.legend()
    plt.savefig('../report/include/w_BR' + plot_super +'.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure()
    plt.hold(True)
    plt.plot(time, Lr[:,0], label='$L_{r1}$')
    plt.plot(time, Lr[:,1], label='$L_{r1}$')
    plt.plot(time, Lr[:,2], label='$L_{r1}$')
    plt.xlabel("Time [sec]")
    plt.ylabel("$L_r$", size=18)
    plt.legend()
    plt.savefig('../report/include/Lr' + plot_super +'.png', bbox_inches='tight', dpi=300)
    plt.close()


if not USE_CONTROL and not USE_CONSTANT_TORQUES:
    plt.figure()
    plt.hold(True)
    plt.plot(time, T-T[0])
    plt.xlabel("Time [sec]")
    plt.ylabel("$\Delta T$ $[J]$", size=18)
    plt.savefig('../report/include/energy' + plot_super +'.png', bbox_inches='tight', dpi=300)
    plt.close()
else:
    plt.figure()
    plt.hold(True)
    plt.plot(time, T)
    plt.xlabel("Time [sec]")
    plt.ylabel("$T$ $[J]$", size=18)
    plt.savefig('../report/include/energy' + plot_super +'.png', bbox_inches='tight', dpi=300)
    plt.close()

plt.figure()
plt.hold(True)
plt.plot(time, P, label='Analitically computed power')
plt.plot(time, P_numerical, label='Numerically computed power')
plt.xlabel("Time [sec]")
plt.ylabel("$P$ $[W]$", size=18)
plt.legend()
plt.savefig('../report/include/power' + plot_super +'.png', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time, gt1[:,2], label='$g_{t-1-z}$')
plt.plot(time, gt2[:,2], label='$g_{t-2-z}$')
plt.plot(time, gt3[:,2], label='$g_{t-3-z}$')
plt.plot(time, gt4[:,2], label='$g_{t-4-z}$')
plt.xlabel("Time [sec]")
plt.ylabel("$g_{t-z}", size=18)
plt.legend()
plt.savefig('../report/include/gt_z' + plot_super +'.png', bbox_inches='tight', dpi=300)
plt.close()



# plt.figure()
# plt.hold(True)
# plt.plot(time, cmg_gamma_1 * 180/np.pi)
# plt.xlabel("Time [sec]")
# plt.ylabel("$\gamma_{CMG-1}$ $[\text{rpm}]$", size=18)
# plt.savefig('../report/include/cmg_gamma_1.png', bbox_inches='tight', dpi=300)
# plt.close()
#
# plt.figure()
# plt.hold(True)
# plt.plot(time, cmg_gamma_dot_1 * 180/np.pi)
# plt.xlabel("Time [sec]")
# plt.ylabel("$\dot\gamma_{CMG-1}$ $[^\circ/seg]$", size=18)
# plt.savefig('../report/include/cmg_gamma_dot_1.png', bbox_inches='tight', dpi=300)
# plt.close()
#
# plt.figure()
# plt.hold(True)
# plt.plot(time, rw_Omega_1 * 60.0/(2*np.pi))
# plt.xlabel("Time [sec]")
# plt.ylabel("$\Omega_{CMG-1}$ $[\text{rpm}]$", size=18)
# plt.savefig('../report/include/rw_omega_1.png', bbox_inches='tight', dpi=300)
# plt.close()