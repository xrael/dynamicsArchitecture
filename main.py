
import numpy as np
import simulatorManager
import coordinateTransformations
import attitudeKinematics
import matplotlib.pyplot as plt


##### Parameters
#Hub
I_hub = np.diag([10, 5, 2])     # [kg m^2] Principal axis inertia matrix
m_hub = 30.0                    # [kg] Hub mass
r_BcB_B = np.array([0.0,0,0])     # [m] Position of the center of mass of the hub relative to the reference B


# VSCMG
Iws = 1.0
Iwt = 1.0

Igs = 1.0
Igt = 1.0
Igg = 1.0

m_vscmg = 10.0
R_OiB_vscmg = np.array([0.0,0,0]) # [m] Position of the center of mass of the VSCMG relative to the reference B
BG_0 = np.eye(3)                # Initial attitude of body frame relative to G (gs, gt, gg vectors as columns)

simulator = simulatorManager.simulatorManager.getSimulatorManager('spacecraft_backSub', 'rk4', 'hub')
simulator.setSimulationTimes(0.0, 100.0, 0.1)

spacecraft = simulator.getDynamicalObject()

spacecraft.setHubMass(m_hub)
spacecraft.setHubInertia(I_hub)
spacecraft.setHubCoMOffset(r_BcB_B)

spacecraft.addVSCMG('VSCMG_1', m_vscmg, R_OiB_vscmg, Igs, Igt, Igg, Iws, Iwt, BG_0)

simulator.setInitialConditions('hub_R_BN', np.array([0.0, 0.0, 0.0]))
simulator.setInitialConditions('hub_R_BN_dot', np.array([0.0, 0.0, 0.0]))
simulator.setInitialConditions('hub_sigma_BN', np.array([0.0, 0.0, 0.0]))
simulator.setInitialConditions('hub_omega_BN', np.array([0.2,0.0,0.1]))

simulator.setInitialConditions('VSCMG_1_gamma', 0.0)
simulator.setInitialConditions('VSCMG_1_gamma_dot', 0.0)
simulator.setInitialConditions('VSCMG_1_Omega', 0.0 * 2*np.pi/(60))

simulator.computeEnergy(True)

simulator.simulate()

time = simulator.getTimeVector()
state_history = simulator.getStateHistory()

R_BN = state_history['hub_R_BN']
v_BN = state_history['hub_R_BN_dot']
sigma_BN = state_history['hub_sigma_BN']
w_BN = state_history['hub_omega_BN']
vscmg_gamma_1 = state_history['VSCMG_1_gamma']
vscmg_gamma_dot_1 = state_history['VSCMG_1_gamma_dot']
vscmg_Ohmega_1 = state_history['VSCMG_1_Omega']

T = simulator.getEnergyVector()

plt.figure()
plt.hold(True)
plt.plot(time, R_BN[:,0], label='$x_{BN}$')
plt.plot(time, R_BN[:,1], label='$y_{BN}$')
plt.plot(time, R_BN[:,2], label='$z_{BN}$')
plt.xlabel("Time [sec]")
plt.ylabel("$R_{BN}", size=18)
plt.legend()
plt.savefig('../report/include/R_BN.png', bbox_inches='tight', dpi=300)
plt.close()


plt.figure()
plt.hold(True)
plt.plot(time, v_BN[:,0], label='$\dot x_{BN}$')
plt.plot(time, v_BN[:,1], label='$\dot y_{BN}$')
plt.plot(time, v_BN[:,2], label='$\dot z_{BN}$')
plt.xlabel("Time [sec]")
plt.ylabel("$v_{BN}", size=18)
plt.legend()
plt.savefig('../report/include/v_BN.png', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time, sigma_BN[:,0], label='$\sigma_1$')
plt.plot(time, sigma_BN[:,1], label='$\sigma_2$')
plt.plot(time, sigma_BN[:,2], label='$\sigma_3$')
plt.xlabel("Time [sec]")
plt.ylabel("$\sigma$", size=18)
plt.legend()
plt.savefig('../report/include/sigma_BN.png', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time, w_BN[:,0], label='$\omega_1$')
plt.plot(time, w_BN[:,1], label='$\omega_2$')
plt.plot(time, w_BN[:,2], label='$\omega_3$')
plt.xlabel("Time [sec]")
plt.ylabel("$\omega$ $[rad/sec]$", size=18)
plt.legend()
plt.savefig('../report/include/w_BN.png', bbox_inches='tight', dpi=300)
plt.close()

plt.figure()
plt.hold(True)
plt.plot(time, T-T[0])
plt.xlabel("Time [sec]")
plt.ylabel("$T$ $[J]$", size=18)
plt.savefig('../report/include/energy.png', bbox_inches='tight', dpi=300)
plt.close()