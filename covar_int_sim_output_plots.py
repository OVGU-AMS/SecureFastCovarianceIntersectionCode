import pickle as pkl
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import other_helpers.plotting_helper as ph
import covar_int_computation as fci

# Matplotlib params
SAVE_NOT_SHOW = True

if SAVE_NOT_SHOW:
    # Use to following to output latex friendly pictures. Note plt.show() will no longer work
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

FIG_WIDTH = 3.4
FIG_HIGHT = 2.7
FONT_SIZE = 'small'


# Load the sim data
sim_data = pkl.load(open("simout.p", "rb"))

# Sim params
MAX_STEPS = 10
SKIP_FIRST = 2
TIME_BETWEEN_PLOTS = 2



"""
88888888888
    888
    888
    888  888d888 8888b.   .d8888b .d88b.  .d8888b
    888  888P"      "88b d88P"   d8P  Y8b 88K
    888  888    .d888888 888     88888888 "Y8888b.
    888  888    888  888 Y88b.   Y8b.          X88
    888  888    "Y888888  "Y8888P "Y8888   88888P'



"""
fig = plt.figure()
fig.set_size_inches(w=FIG_WIDTH, h=FIG_HIGHT)
ax = fig.add_subplot(111)
ax.set_title(r'FCI and SecFCI estimate traces')

# FCI estimates
split = list(zip(*sim_data['fusion_estimates']))
estimates = split[0]
errors = split[1]
error_traces = [np.trace(p) for p in errors]
ax.plot(error_traces, c='r', label=r'$tr(P_{FCI})$', marker='.')


# Secure FCI estimates
split = list(zip(*sim_data['secure_fusion_estimates']))
estimates = split[0]
errors = split[1]
error_traces = [np.trace(p) for p in errors]
ax.plot(error_traces, c='b', label=r'$tr(P_{SecFCI})$', marker='.')

plt.xlabel(r'Time')
plt.legend()
plt.tight_layout()
if SAVE_NOT_SHOW:
    plt.savefig('images/traces_cmp.pgf')
else:
    plt.show()
plt.close()



"""
 .d88888b.
d88P" "Y88b
888     888
888     888 88888b.d88b.   .d88b.   .d88b.   8888b.  .d8888b
888     888 888 "888 "88b d8P  Y8b d88P"88b     "88b 88K
888     888 888  888  888 88888888 888  888 .d888888 "Y8888b.
Y88b. .d88P 888  888  888 Y8b.     Y88b 888 888  888      X88
 "Y88888P"  888  888  888  "Y8888   "Y88888 "Y888888  88888P'
                                        888
                                   Y8b d88P
                                    "Y88P"
"""
fig = plt.figure()
fig.set_size_inches(w=FIG_WIDTH, h=FIG_HIGHT)
ax = fig.add_subplot(111)
ax.set_title(r'Difference in $\omega_i$ values')

LIMIT_POINTS = 10

split1 = list(zip(*sim_data['sensor_estimates'][0]))
errors1 = split1[1][:LIMIT_POINTS]
error_traces1 = [np.trace(p) for p in errors1][:LIMIT_POINTS]

split2 = list(zip(*sim_data['sensor_estimates'][1]))
errors2 = split2[1][:LIMIT_POINTS]
error_traces2 = [np.trace(p) for p in errors2][:LIMIT_POINTS]

trace_groups = list(zip(error_traces1, error_traces2))
omegas = [fci.omega_exact(ts) for ts in trace_groups]
omega_step_size = 0.1
approx_omegas = [fci.omega_estimates([[w*i for w in np.arange(0, 1+omega_step_size, omega_step_size)] for i in ts], omega_step_size) for ts in trace_groups]

split_omegas = list(zip(*omegas))
split_approx_omegas = list(zip(*approx_omegas))

ax.plot([i for i in range(len(split_omegas[0]))], split_omegas[0], c=(0.9,0,0), marker='.', label=r'FCI $\omega_0$')
ax.plot([i for i in range(len(split_omegas[1]))], split_omegas[1], c=(0.9,0.2,0.2), marker='.', label=r'FCI $\omega_1$')

ax.plot([i for i in range(len(split_approx_omegas[0]))], split_approx_omegas[0], c=(0,0,0.9), marker='.', label=r'SecFCI $\omega_0$')
ax.plot([i for i in range(len(split_approx_omegas[1]))], split_approx_omegas[1], c=(0.2,0.2,0.9), marker='.', label=r'SecFCI $\omega_1$')


diff = np.abs(np.array(split_omegas[0]) - np.array(split_approx_omegas[0])) + abs(np.array(split_omegas[1]) - np.array(split_approx_omegas[1]))
ax.plot([i for i in range(len(diff))], diff, c='grey', marker='.', label=r'Error')

plt.xlabel(r'Time')
plt.ylabel(r'Values of $\omega_i$')
plt.legend(loc=1)
plt.tight_layout()
if SAVE_NOT_SHOW:
    plt.savefig('images/omegas_cmp.pgf')
else:
    plt.show()
plt.close()


"""
8888888888                       8888888888 888 888 d8b
888                              888        888 888 Y8P
888                              888        888 888
8888888    888d888 888d888       8888888    888 888 888 88888b.  .d8888b   .d88b.  .d8888b
888        888P"   888P"         888        888 888 888 888 "88b 88K      d8P  Y8b 88K
888        888     888           888        888 888 888 888  888 "Y8888b. 88888888 "Y8888b.
888        888     888  d8b      888        888 888 888 888 d88P      X88 Y8b.          X88
8888888888 888     888  Y8P      8888888888 888 888 888 88888P"   88888P'  "Y8888   88888P'
                                                        888
                                                        888
                                                        888
"""

# Ensure this is the last plot as this fucks up the data (could copy it but meh)
gf_true = sim_data['ground_truth']
sim_data['ground_truth'] = [x for i,x in enumerate(sim_data['ground_truth']) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]
sim_data['measurements'][0] = [x for i,x in enumerate(sim_data['measurements'][0]) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]
sim_data['measurements'][1] = [x for i,x in enumerate(sim_data['measurements'][1]) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]
sim_data['sensor_estimates'][0] = [x for i,x in enumerate(sim_data['sensor_estimates'][0]) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]
sim_data['sensor_estimates'][1] = [x for i,x in enumerate(sim_data['sensor_estimates'][1]) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]
sim_data['fusion_estimates'] = [x for i,x in enumerate(sim_data['fusion_estimates']) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]
sim_data['secure_fusion_estimates'] = [x for i,x in enumerate(sim_data['secure_fusion_estimates']) if i%TIME_BETWEEN_PLOTS==0][SKIP_FIRST:SKIP_FIRST+MAX_STEPS]

# Also change the figure size for this plot
FIG_WIDTH = 3.4
FIG_HIGHT = 3.4


fig = plt.figure()
fig.set_size_inches(w=FIG_WIDTH, h=FIG_HIGHT)
ax = fig.add_subplot(111)
ax.set_title(r'FCI and SecFCI comparison')

# Ground truth
ax.plot(*zip(*[(x[0],x[2]) for x in gf_true[SKIP_FIRST*TIME_BETWEEN_PLOTS:(SKIP_FIRST+MAX_STEPS)*TIME_BETWEEN_PLOTS]]), c='lightgrey', marker='.')

# Measurements 1
ax.scatter(*zip(*[(x[0],x[1]) for x in sim_data['measurements'][0]]), c='limegreen', marker='x', label=r'Sensor 1')
# Lines from gt to measurements
m1 = list(zip(sim_data['ground_truth'], sim_data['measurements'][0]))
for i in range(len(m1)):
    ax.plot([m1[i][0][0], m1[i][1][0]], [m1[i][0][2], m1[i][1][1]], c='lightgrey', linestyle='--')

# Measurements 2
ax.scatter(*zip(*[(x[0],x[1]) for x in sim_data['measurements'][1]]), c='cornflowerblue', marker='x', label=r'Sensor 2')
# Lines from gt to measurements
m2 = list(zip(sim_data['ground_truth'], sim_data['measurements'][1]))
for i in range(len(m2)):
    ax.plot([m2[i][0][0], m2[i][1][0]], [m2[i][0][2], m2[i][1][1]], c='lightgrey', linestyle='--')

# FCI estimates
split = list(zip(*sim_data['fusion_estimates']))
estimates = split[0]
errors = split[1]

estimates2D = [np.array([e[0],e[2]]) for e in estimates]
errors2D = [np.array([[p[0,0], p[2,0]], [p[0,2], p[2,2]]]) for p in errors]

ax.scatter(None, None, c='r', marker='.', label=r'FCI estimate')
for i in range(len(estimates)):
    estimate = estimates2D[i]
    error = errors2D[i]

    ax.scatter(*estimate, c='r', marker='.')
    ax.add_artist(ph.get_cov_ellipse(error, estimate, 2, fill=False, linestyle='-', edgecolor='r'))


# Secure FCI estimates
split = list(zip(*sim_data['secure_fusion_estimates']))
estimates = split[0]
errors = split[1]

estimates2D = [np.array([e[0],e[2]]) for e in estimates]
errors2D = [np.array([[p[0,0], p[2,0]], [p[0,2], p[2,2]]]) for p in errors]

ax.scatter(None, None, c='b', marker='.', label=r'SecFCI estimate')
for i in range(len(estimates)):
    estimate = estimates2D[i]
    error = errors2D[i]

    ax.scatter(*estimate, c='b', marker='.')
    ax.add_artist(ph.get_cov_ellipse(error, estimate, 2, fill=False, linestyle='-', edgecolor='b'))

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.tight_layout()
if SAVE_NOT_SHOW:
    plt.savefig('images/fci_secfci_cmp.pgf')
else:
    plt.show()
plt.close()