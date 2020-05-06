import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

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

    SAVE_PICS = True
    SHOW_PICS = False
else:
    SAVE_PICS = False
    SHOW_PICS = True

FIG_WIDTH = 2.33
FIG_HIGHT = 2.06

FONT_SIZE = 'small'


"""
 .d8888b.        .d8888b.
d88P  Y88b      d88P  Y88b
       888      Y88b.
     .d88P       "Y888b.    .d88b.  88888b.  .d8888b   .d88b.  888d888       .d8888b  8888b.  .d8888b   .d88b.
 .od888P"           "Y88b. d8P  Y8b 888 "88b 88K      d88""88b 888P"        d88P"        "88b 88K      d8P  Y8b
d88P"                 "888 88888888 888  888 "Y8888b. 888  888 888          888      .d888888 "Y8888b. 88888888
888"            Y88b  d88P Y8b.     888  888      X88 Y88..88P 888          Y88b.    888  888      X88 Y8b.
888888888        "Y8888P"   "Y8888  888  888  88888P'  "Y88P"  888           "Y8888P "Y888888  88888P'  "Y8888



"""
fig = plt.figure()
# Overwrite size on this plot
fig.set_size_inches(w=3.4, h=2)
ax = fig.add_subplot(111)

w_quant = 0.1
w_steps = np.arange(0, 1+w_quant, w_quant)
print("w_steps:", w_steps)
print("1-w_steps:", 1-w_steps)

trA = 7.6
trB = 2.4

A = w_steps * trA
B = (1-w_steps) * trB

print("trA_w_options:", A)
print("trB_w_options:", B)

cmp_list = A+B
print(cmp_list)

w_vals = w_steps*trA > (1-w_steps)*trB
print(w_vals)
l = r = 0
for i,b in enumerate(w_vals):
    if not b:
        l = r = i
    else:
        r = i
        break
l = w_steps[l]
r = w_steps[r]


ax.plot(w_steps, A, marker='.', c='g', label=r'$tr(P_1)\omega^{(x)}$', zorder=3)
ax.plot(w_steps, B, marker='.', c='b', label=r'$tr(P_2)(1-\omega^{(x)})$', zorder=3)
#ax.plot([trB/(trA+trB), trB/(trA+trB)],[0, trB/(trA+trB)*trA], linestyle='--', c='r')
#ax.scatter([trB/(trA+trB)],[trB/(trA+trB)*trA], marker='x', c='r', zorder=10)

ax.scatter([l,r],[0, 0], marker='x', c='grey', zorder=2, label=r'Solution limits')
ax.plot([l, l],[0, (1-l)*trB], linestyle='--', c='grey', zorder=2)
ax.plot([r, r],[0, r*trA], linestyle='--', c='grey', zorder=2)
ax.scatter([0.5*(l+r)],[0], marker='x', c='r', zorder=1, label=r'Approx. solution')

#plt.ylim(bottom=0)

plt.xlabel(r'$\omega^{(x)}$', fontsize=FONT_SIZE)
#ax.set_title('Fast covariance intersection')

plt.legend(fontsize=FONT_SIZE, numpoints=1)
ax.xaxis.set_tick_params(labelsize=FONT_SIZE)
ax.yaxis.set_tick_params(labelsize=FONT_SIZE)
plt.tight_layout()
if SAVE_PICS:
    plt.savefig('images/2_sensors.pgf')
if SHOW_PICS:
    plt.show()
plt.close()


"""
 d888        8888888b.                  888    d8b          888       .d8888b.           888
d8888        888   Y88b                 888    Y8P          888      d88P  Y88b          888
  888        888    888                 888                 888      Y88b.               888
  888        888   d88P 8888b.  888d888 888888 888  8888b.  888       "Y888b.    .d88b.  888
  888        8888888P"     "88b 888P"   888    888     "88b 888          "Y88b. d88""88b 888
  888        888       .d888888 888     888    888 .d888888 888            "888 888  888 888
  888        888       888  888 888     Y88b.  888 888  888 888      Y88b  d88P Y88..88P 888 d8b
8888888      888       "Y888888 888      "Y888 888 "Y888888 888       "Y8888P"   "Y88P"  888 Y8P



"""

fig = plt.figure()
fig.set_size_inches(w=FIG_WIDTH, h=FIG_HIGHT)
ax = fig.add_subplot(111, projection='3d')

# fig.suptitle(r'First partial solution')
# ax.set_title(r'First Partial solution')

ax.set_xlabel(r'$\omega_1$')
ax.set_ylabel(r'$\omega_2$')
ax.set_zlabel(r'$\omega_3$')
ax.view_init(elev=35, azim=2)
ax.dist = 12

# Solution plane
xy = np.array([[1,0],
               [0,0],
               [0,1]])
z = np.array([0,1,0])
trigs = [[0,1,2]]
triangles = mtri.Triangulation(xy[:,0], xy[:,1], triangles=trigs)
ax.plot_trisurf(triangles, z, color=(0.7,0.2,0.2,0.5), zorder=1)


# Partial solution 1
sol1 = 0.24
sol1Line, = ax.plot([sol1, 0],[1-sol1, 0],[0, 1], linestyle='--', c='g')
ax.scatter([sol1, 0],[1-sol1, 0],[0, 1], c='g', marker='x', depthshade=False)

solutionSurfaceFakeLine = mpl.lines.Line2D([0],[0], linestyle="none", c=(0.7,0.2,0.2), marker = 'o')
l = ax.legend([solutionSurfaceFakeLine, sol1Line], [r'$\omega_i$ solution space', r'$\omega_1$, $\omega_2$ partial solution'], numpoints=1, loc=1, fontsize=FONT_SIZE)

# Move the legend up slightly
bb = l.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
bb.y0 += 0.15
bb.y1 += 0.15
l.set_bbox_to_anchor(bb, transform = ax.transAxes)

ax.xaxis.set_tick_params(labelsize=FONT_SIZE)
ax.yaxis.set_tick_params(labelsize=FONT_SIZE)
ax.zaxis.set_tick_params(labelsize=FONT_SIZE)
ax.xaxis.set_ticks([0,0.5,1])
ax.yaxis.set_ticks([0,0.5,1])
ax.zaxis.set_ticks([0,0.5,1])
plt.autoscale()
if SAVE_PICS:
    plt.savefig('images/partial_sol1.pgf')
if SHOW_PICS:
    plt.show()
plt.close()


"""
888888b.            888    888           8888888b.                  888    d8b          888       .d8888b.           888
888  "88b           888    888           888   Y88b                 888    Y8P          888      d88P  Y88b          888
888  .88P           888    888           888    888                 888                 888      Y88b.               888
8888888K.   .d88b.  888888 88888b.       888   d88P 8888b.  888d888 888888 888  8888b.  888       "Y888b.    .d88b.  888 .d8888b
888  "Y88b d88""88b 888    888 "88b      8888888P"     "88b 888P"   888    888     "88b 888          "Y88b. d88""88b 888 88K
888    888 888  888 888    888  888      888       .d888888 888     888    888 .d888888 888            "888 888  888 888 "Y8888b.
888   d88P Y88..88P Y88b.  888  888      888       888  888 888     Y88b.  888 888  888 888      Y88b  d88P Y88..88P 888      X88 d8b
8888888P"   "Y88P"   "Y888 888  888      888       "Y888888 888      "Y888 888 "Y888888 888       "Y8888P"   "Y88P"  888  88888P' Y8P



"""

fig = plt.figure()
fig.set_size_inches(w=FIG_WIDTH, h=FIG_HIGHT)
ax = fig.add_subplot(111, projection='3d')

#fig.suptitle(r'All partial solutions')
#ax.set_title(r'All partial solutions')

ax.set_xlabel(r'$\omega_1$')
ax.set_ylabel(r'$\omega_2$')
ax.set_zlabel(r'$\omega_3$')
ax.view_init(elev=35, azim=2)
ax.dist = 12

# Solution plane
xy = np.array([[1,0],
               [0,0],
               [0,1]])
z = np.array([0,1,0])
trigs = [[0,1,2]]
triangles = mtri.Triangulation(xy[:,0], xy[:,1], triangles=trigs)
ax.plot_trisurf(triangles, z, color=(0.7,0.2,0.2,0.5), zorder=1)

# Partial solution 1
sol1 = 0.24
sol1Line, = ax.plot([sol1, 0],[1-sol1, 0],[0, 1], linestyle='--', c='g')
ax.scatter([sol1, 0],[1-sol1, 0],[0, 1], c='g', marker='x', depthshade=False)

# Partial solution 2
sol2 = 0.52
sol2Line, = ax.plot([0, 1],[sol2, 0],[1-sol2, 0], linestyle='--', c='b')
ax.scatter([0, 1],[sol2, 0],[1-sol2, 0], c='b', marker='x', depthshade=False)

solutionSurfaceFakeLine = mpl.lines.Line2D([0],[0], linestyle="none", c=(0.7,0.2,0.2), marker = 'o')
l = ax.legend([solutionSurfaceFakeLine, sol1Line, sol2Line], 
          [r'$\omega_i$ solution space', r'$\omega_1$, $\omega_2$ partial solution', r'$\omega_2$, $\omega_3$ partial solution'], numpoints=1, loc=1, fontsize=FONT_SIZE)

# Move the legend up slightly
bb = l.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
bb.y0 += 0.15
bb.y1 += 0.15
l.set_bbox_to_anchor(bb, transform = ax.transAxes)

ax.xaxis.set_tick_params(labelsize=FONT_SIZE)
ax.yaxis.set_tick_params(labelsize=FONT_SIZE)
ax.zaxis.set_tick_params(labelsize=FONT_SIZE)
ax.xaxis.set_ticks([0,0.5,1])
ax.yaxis.set_ticks([0,0.5,1])
ax.zaxis.set_ticks([0,0.5,1])
plt.autoscale()
if SAVE_PICS:
    plt.savefig('images/partial_sols.pgf')
if SHOW_PICS:
    plt.show()
plt.close()



"""
8888888b.  888
888   Y88b 888
888    888 888
888   d88P 888  8888b.  88888b.   .d88b.  .d8888b
8888888P"  888     "88b 888 "88b d8P  Y8b 88K
888        888 .d888888 888  888 88888888 "Y8888b.
888        888 888  888 888  888 Y8b.          X88
888        888 "Y888888 888  888  "Y8888   88888P'



"""


fig = plt.figure()
fig.set_size_inches(w=FIG_WIDTH, h=FIG_HIGHT)
ax = fig.add_subplot(111, projection='3d')

#fig.suptitle(r'Partial solutions as planes')
#ax.set_title(r'Partial solutions as planes')

ax.set_xlabel(r'$\omega_1$')
ax.set_ylabel(r'$\omega_2$')
ax.set_zlabel(r'$\omega_3$')
ax.view_init(elev=35, azim=2)
ax.dist = 12

# Solution plane
xy = np.array([[1,0],
               [0,0],
               [0,1]])
z = np.array([0,1,0])
trigs = [[0,1,2]]
triangles = mtri.Triangulation(xy[:,0], xy[:,1], triangles=trigs)

ax.plot_trisurf(triangles, z, color=(0.7,0.2,0.2,0.5), zorder=1)

# Partial solution 1
sol1 = 0.24
ax.plot([sol1, 0],[1-sol1, 0],[0, 1], linestyle='--', c='g')

# Partial solution 2
sol2 = 0.52
ax.plot([0, 1],[sol2, 0],[1-sol2, 0], linestyle='--', c='b')

# Partial solution plane 1
computed_point1 = np.array([sol1, 1-sol1, 0])
avg_known_points1 = np.array([0,0,1])
relevant_known_point1 = np.array([1,0,0])
to_project1 = relevant_known_point1 - avg_known_points1
to_project_onto1 = computed_point1 - avg_known_points1
projection1 = (np.dot(to_project1, to_project_onto1)/np.dot(to_project_onto1, to_project_onto1)) * to_project_onto1
norm_to_hyperplane1 = to_project1 - projection1
intercept1 = np.dot(norm_to_hyperplane1-avg_known_points1, computed_point1)
# Top point
eq = np.array([[1,1,0],
               [0,0,1],
               norm_to_hyperplane1])
so = np.array([1, 1, intercept1])
t1 = np.linalg.solve(eq, so)
#ax.scatter(*t)
# Bottom point
eq = np.array([[1,0,0],
               [0,0,1],
               norm_to_hyperplane1])
so = np.array([0, 0, intercept1])
b1 = np.linalg.solve(eq, so)
#ax.scatter(*b)
xyz = np.array([computed_point1,
                avg_known_points1,
                t1,
                b1])
trigs = [[0,1,2],[0,1,3]]
triangles = mtri.Triangulation(xyz[:,0], xyz[:,1], triangles=trigs)

ax.plot_trisurf(triangles, xyz[:,2], color=(0.2,0.7,0.2,0.5), zorder=1)

# Partial solution plane 2
computed_point2 = np.array([0, sol2, 1-sol2])
avg_known_points2 = np.array([1,0,0])
relevant_known_point2 = np.array([0,1,0])
to_project2 = relevant_known_point2 - avg_known_points2
to_project_onto2 = computed_point2 - avg_known_points2
projection2 = (np.dot(to_project2, to_project_onto2)/np.dot(to_project_onto2, to_project_onto2)) * to_project_onto2
norm_to_hyperplane2 = to_project2 - projection2
intercept2 = np.dot(norm_to_hyperplane2-avg_known_points2, computed_point2)
# Top point
eq = np.array([[1,1,0],
               [1,0,0],
               norm_to_hyperplane2])
so = np.array([1, 0, intercept2])
t2 = np.linalg.solve(eq, so)
#ax.scatter(*t)
# Bottom point
eq = np.array([[0,0,1],
               [1,0,0],
               norm_to_hyperplane2])
so = np.array([0, 0, intercept2])
b2 = np.linalg.solve(eq, so)
#ax.scatter(*b)
xyz = np.array([computed_point2,
                avg_known_points2,
                t2,
                b2])
trigs = [[0,1,2],[0,1,3]]
triangles = mtri.Triangulation(xyz[:,0], xyz[:,1], triangles=trigs)

ax.plot_trisurf(triangles, xyz[:,2], color=(0.2,0.2,0.7,0.5), zorder=1)

# TODO plotting multiple transparent planes is fucked up in matplotlib, need to split planes at intersection points so that the
# transparentcy is rendered properly. But that's a whole lotta effort...

# Complete solution point
eq = np.array([[1,1,1],
               norm_to_hyperplane1,
               norm_to_hyperplane2])
so = np.array([1, intercept1, intercept2])
i = np.linalg.solve(eq, so)
ax.scatter(*i, marker='o', c='r', zorder=10)

solutionSurfaceFakeLine = mpl.lines.Line2D([0],[0], linestyle="none", c=(0.7,0.2,0.2), marker = 'o')
partialSol1FakeLine = mpl.lines.Line2D([0],[0], linestyle="none", c=(0.2,0.7,0.2), marker = 'o')
partialSol2FakeLine = mpl.lines.Line2D([0],[0], linestyle="none", c=(0.2,0.2,0.7), marker = 'o')
l = ax.legend([solutionSurfaceFakeLine, partialSol1FakeLine, partialSol2FakeLine], 
          [r'$\omega_i$ solution space', r'$\omega_1$, $\omega_2$ solution plane', r'$\omega_2$, $\omega_3$ solution plane'], 
           numpoints=1,
           #loc=1,
           fontsize=FONT_SIZE)
# Fixes issue plotting planes over th legend
l.set_zorder(20)

# Move the legend up slightly
bb = l.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
bb.y0 += 0.15
bb.y1 += 0.15
l.set_bbox_to_anchor(bb, transform = ax.transAxes)

ax.xaxis.set_tick_params(labelsize=FONT_SIZE)
ax.yaxis.set_tick_params(labelsize=FONT_SIZE)
ax.zaxis.set_tick_params(labelsize=FONT_SIZE)
ax.xaxis.set_ticks([0,0.5,1])
ax.yaxis.set_ticks([0,0.5,1])
ax.zaxis.set_ticks([0,0.5,1])
plt.autoscale()
if SAVE_PICS:
    plt.savefig('images/partial_sol_planes.pgf')
if SHOW_PICS:
    plt.show()
plt.close()