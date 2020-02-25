import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# w_quant = 0.1
# w_steps = np.arange(0, 1+w_quant, w_quant)
# print("w_steps:", w_steps)
# print("1-w_steps:", 1-w_steps)

# trA = 10
# trB = 2

# A = w_steps * trA
# B = (1-w_steps) * trB

# print("trA_w_options:", A)
# print("trB_w_options:", B)

# cmp_list = A+B
# print(cmp_list)

# w_vals = w_steps*trA > (1-w_steps)*trB
# print(w_vals)

# plt.plot(w_steps, A, c='g')
# plt.scatter(w_steps, A, marker='o', c='g', label=r'$tr(A^{-1})\omega$')
# plt.plot(w_steps, B, c='c')
# plt.scatter(w_steps, B, marker='o', c='c', label=r'$tr(B^{-1})(1-\omega)$')
# plt.plot([2/12, 2/12],[0, 2/12*trA], linestyle='--', c='b')
# plt.scatter([2/12],[2/12*trA], marker='x', c='b', zorder=10)

# plt.ylim(bottom=0)

# plt.xlabel(r'$\omega$')
# plt.title('Fast covariance intersection')

# plt.legend()
# plt.show()

# ===== Muuuulti sensors! ===== #

# Thes sensor covar traces
A = 4
B = 7
C = 2

# Compute the real omega solution using paper formula (with division)
real_sol_w1 = (1/4)/sum([1/4,1/7,1/2])
real_sol_w2 = (1/7)/sum([1/4,1/7,1/2])
real_sol_w3 = 1 - real_sol_w1 - real_sol_w2
print('real solutions')
print('w1', '%.2f' % real_sol_w1)
print('w2', '%.2f' % real_sol_w2)
print('w3', '%.2f' % real_sol_w3)

# Discretisation and list of steps
w_step = 0.1
w_vals = np.arange(0, 1+w_step, w_step)

# What the sensors send
A_list = w_vals*A
B_list = w_vals*B
C_list = w_vals*C

# Helpful for finding intersections
B_list_rev = list(reversed(B_list))
C_list_rev = list(reversed(C_list))

#==============================================================================#
# Figure
fig = plt.figure(figsize=(12.8,9.2))
fig.suptitle(r'Visualisation of Fast Covariance Intersection $\omega_i$ computation, provided only values under an Order Revealing Encryption scheme')

# First pic is 3D for all discretised omegas
ax0 = fig.add_subplot(224, projection='3d')
ax0.set_xlabel(r'$\omega_1$')
ax0.set_ylabel(r'$\omega_2$')
ax0.set_zlabel(r'$\omega_3$')
ax0.view_init(elev=30, azim=65)
ax0.set_title(r'Intersection of partial solutions')
ps = [(a,b,c) for a in w_vals for b in w_vals for c in w_vals if np.isclose(a+b+c, 1)]

best_choice = 0#abs(real_sol_w1*A - real_sol_w2*B) + abs(real_sol_w2*B - real_sol_w3*C)
worst_choice = max(A, B) + max(B, C)
xs = []
ys = []
zs = []
cs = []
min_point = (None, None)
for p in ps:
    choice = abs(p[0]*A - p[1]*B) + abs(p[1]*B - p[2]*C)
    scaled_choice = (choice - best_choice)/(worst_choice - best_choice)
    xs.append(p[0])
    ys.append(p[1])
    zs.append(p[2])
    cs.append(scaled_choice)
    if min_point[1] == None:
        min_point = (p, choice)
    elif min_point[1] > choice:
        min_point = (p, choice)
ax0.scatter(xs, ys, zs, c=cs, marker='.', cmap='winter_r', label=r'$\omega$ plane')

ax0.plot([B/(A+B), 0], [A/(A+B), 0], [0, 1], c='m', linestyle='--', label=r'$\omega_1,\omega_2$ solutions')
ax0.plot([0, 1], [C/(B+C), 0], [B/(B+C), 0], c='c', linestyle='--', label=r'$\omega_2,\omega_3$ solutions')

# Mark best option
#ax0.scatter(min_point[0][0], min_point[0][1], min_point[0][2], c='r')
ax0.scatter(real_sol_w1, real_sol_w2, real_sol_w3, c='r', marker='x', label=r'Optimal solution')
ax0.plot([real_sol_w1, real_sol_w1], [real_sol_w2, real_sol_w2], [0, real_sol_w3], c='r', linestyle='--')
ax0.legend(loc=7)

#==============================================================================#

# Next we show how solving A and B equation for omega1/2 extrapolates to a line of solutions as we change omega3
ax1 = fig.add_subplot(221, projection='3d')
ax1.set_xlabel(r'$\omega_1$')
ax1.set_ylabel(r'$\omega_2$')
ax1.set_zlabel('')
ax1.view_init(elev=30, azim=65)
ax1.set_title(r'$\omega_1,\omega_2$ solution at $\omega_3=0$')

ax2 = fig.add_subplot(222, projection='3d')
ax2.set_xlabel(r'$\omega_1$')
ax2.set_ylabel(r'$\omega_2$')
ax2.set_zlabel('')
ax2.view_init(elev=30, azim=65)
ax2.set_title(r'$\omega_1,\omega_2$ solution at $\omega_3=0.5$')

ax3 = fig.add_subplot(223, projection='3d')
ax3.set_xlabel(r'$\omega_1$')
ax3.set_ylabel(r'$\omega_2$')
ax3.set_zlabel(r'$\omega_3$')
ax3.view_init(elev=30, azim=65)
ax3.set_title(r'Partial solutions over varied $\omega_3$')

# omega1*A - omega2*B = 0 equations
solution_points = []
for z in [0, 0.5]:
    # Plot on relevant picture
    if z == 0:
        ax = ax1
    elif z == 0.5:
        ax = ax2
    
    shift = int(z/w_step)
    As = []
    Bs = []
    plot_num = len(w_vals) - shift
    for i in range(plot_num):
        As.append(A_list[i])
        Bs.append(B_list_rev[i+shift])
    # plot (x,y,z) as (c discretised, omega, a and b solutions)
    ax.plot(w_vals[:-shift if shift!=0 else None], [0]*plot_num, As, c='g', marker='.', label=r'$tr(A)\omega_1$')
    ax.plot(w_vals[:-shift if shift!=0 else None], [0]*plot_num, Bs, c='c', marker='.', label=r'$tr(B)(1-\omega_1)$')
    ax.plot([1-z, 0],[0, 1-z],[0, 0], c='gray', label=r'$\omega$ plane')
    ax.plot([(1-z)*B/(A+B), (1-z)*B/(A+B), (1-z)*B/(A+B)], [0, 0, (1-z)*A/(A+B)], [A*(1-z)*B/(A+B), 0, 0], c='m', linestyle='--')
    ax.scatter((1-z)*B/(A+B), (1-z)*A/(A+B), 0, c='m', marker='x')
    ax.legend(loc=7)

    solution_points.append(((1-z)*B/(A+B), (1-z)*A/(A+B), z))

solution_points.append((0, 0, 1))
ax3.plot([x[0] for x in solution_points], [x[1] for x in solution_points], [x[2] for x in solution_points], c='m', marker='x', linestyle='--', label=r'$\omega_1,\omega_2$ solutions')

ps = [(a,b,c) for a in w_vals for b in w_vals for c in w_vals if np.isclose(a+b+c, 1)]
ax3.scatter([x[0] for x in ps], [x[1] for x in ps], [x[2] for x in ps], c='gray', marker='.', depthshade=False, label=r'$\omega$ plane')

ax3.legend(loc=7)



plt.show()