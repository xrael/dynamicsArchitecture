
import numpy as np
import simulatorManager
import coordinateTransformations
import attitudeKinematics
import matplotlib.pyplot as plt


##### Parameters
I_hub = np.diag([10, 5, 2])     # [kg m^2] Principal axis inertia matrix

m_hub = 30.0

Iw = np.array([1, 0.01])

Ig = np.array([0.001, 0.001, 0.001])

m_vscmg = 2.0

R_CoM_vscmg = np.array([0,0,0])

BG_gamma0 = np.eye(3)


simulator = simulatorManager.simulatorManager.getSimulatorManager('spacecraft_backSub', 'rk4', 'hub')
simulator.setSimulationTimes(0.0, 100.0, 0.01)

spacecraft = simulator.getDynamicalObject()

spacecraft.setHubMass(m_hub)
spacecraft.setHubInertia(I_hub)

spacecraft.addVSCMG('VSCMG_1', m_vscmg, R_CoM_vscmg, Ig, Iw, BG_gamma0)

simulator.setInitialConditions('hub_R_BN', np.array([0.0, 0.0, 0.0]))
simulator.setInitialConditions('hub_R_BN_dot', np.array([0.0, 0.0, 0.0]))
simulator.setInitialConditions('hub_sigma_BN', np.array([0.0, 0.0, 0.0]))
simulator.setInitialConditions('hub_ohmega_BN', np.array([0.1,0.2,0]))

simulator.setInitialConditions('VSCMG_1_gamma', 0.0)
simulator.setInitialConditions('VSCMG_1_gamma_dot', 0.0)
simulator.setInitialConditions('VSCMG_1_Ohmega', 0.0 * 2*np.pi/(60))


simulator.simulate()

time = simulator.getTimeVector()
state_history = simulator.getStateHistory()

sigma_BN = state_history['hub_sigma_BN']
w_BN = state_history['hub_ohmega_BN']
vscmg_gamma_1 = state_history['VSCMG_1_gamma']
vscmg_gamma_dot_1 = state_history['VSCMG_1_gamma_dot']
vscmg_Ohmega_1 = state_history['VSCMG_1_Ohmega']

T = np.zeros(len(time))
for i in range(0,len(time)):

    GG0 = coordinateTransformations.ROT3(vscmg_gamma_1[i])

    BG = BG_gamma0.dot(GG0.T)

    gs_hat = BG[:,0]
    gt_hat = BG[:,1]
    gg_hat = BG[:,2]


    ws = np.inner(gs_hat, w_BN[i])
    wt = np.inner(gt_hat, w_BN[i])
    wg = np.inner(gg_hat, w_BN[i])
    Iws = Iw[0]
    Iwt = Iw[1]
    Igs = Ig[0]
    Js = Igs + Iws
    Jt = Ig[1] + Iwt
    Jg = Ig[2] + Iwt

    R_CoM_vscmg_tilde = attitudeKinematics.getSkewSymmetrixMatrix(R_CoM_vscmg)
    T[i] = 0.5 * w_BN[i].dot(I_hub - m_vscmg * R_CoM_vscmg_tilde.dot(R_CoM_vscmg_tilde)).dot(w_BN[i]) \
           + 0.5 *Igs * ws**2 + 0.5*Iws*(ws+vscmg_Ohmega_1[i])**2 + 0.5*Jt*wt**2 + 0.5*Jg*(wg+vscmg_gamma_dot_1[i])**2

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
plt.plot(time, T)
plt.xlabel("Time [sec]")
plt.ylabel("$T$ $[J]$", size=18)
plt.savefig('../report/include/energy.png', bbox_inches='tight', dpi=300)
plt.close()