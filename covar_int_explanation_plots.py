import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.set_xlabel(r'$\omega_0$')
ax.set_ylabel(r'$\omega_1$')
ax.set_zlabel(r'$\omega_2$')
ax.view_init(elev=30, azim=16)

ax.set_title(r'Partial solution of $\omega_0$ and $\omega_1$')

# Solution plane
xy = np.array([[1,0],
               [0,0],
               [0,1]])
z = np.array([0,1,0])
trigs = [[0,1,2]]
triangles = mtri.Triangulation(xy[:,0], xy[:,1], triangles=trigs)
ax.plot_trisurf(triangles, z, color=(0.5,0.5,0.5,0.5))

# Partial solution 1
sol1 = 0.24
ax.plot([sol1, 0],[1-sol1, 0],[0, 1], linestyle='--', c='g', marker='x')

# Partial solution 2
sol2 = 0.52
ax.plot([0, 1],[sol2, 0],[1-sol2, 0], linestyle='--', c='b', marker='x')

# Partial solution plane 1
computed_point = np.array([sol1, 1-sol1, 0])
avg_known_points = np.array([0,0,1])
relevant_known_point = np.array([1,0,0])
to_project = relevant_known_point - avg_known_points
to_project_onto = computed_point - avg_known_points
projection = (np.dot(to_project, to_project_onto)/np.dot(to_project_onto, to_project_onto)) * to_project_onto
norm_to_hyperplane = to_project - projection
intercept = np.dot(norm_to_hyperplane-avg_known_points, computed_point)
# Top point
eq = np.array([[1,1,0],
               [0,0,1],
               norm_to_hyperplane])
so = np.array([1, 1, intercept])
p = np.linalg.solve(eq, so)
ax.scatter(*p)
# Bottom point
eq = np.array([[1,0,0],
               [0,0,1],
               norm_to_hyperplane])
so = np.array([0, 0, intercept])
p = np.linalg.solve(eq, so)
ax.scatter(*p)

# Partial solution plane 2
computed_point = np.array([0, sol2, 1-sol2])
avg_known_points = np.array([1,0,0])
relevant_known_point = np.array([0,1,0])
to_project = relevant_known_point - avg_known_points
to_project_onto = computed_point - avg_known_points
projection = (np.dot(to_project, to_project_onto)/np.dot(to_project_onto, to_project_onto)) * to_project_onto
norm_to_hyperplane = to_project - projection
intercept = np.dot(norm_to_hyperplane-avg_known_points, computed_point)
# Top point
eq = np.array([[1,1,0],
               [0,0,1],
               norm_to_hyperplane])
so = np.array([1, 1, intercept])
p = np.linalg.solve(eq, so)
ax.scatter(*p)
# Bottom point
eq = np.array([[1,0,0],
               [0,0,1],
               norm_to_hyperplane])
so = np.array([0, 0, intercept])
p = np.linalg.solve(eq, so)
ax.scatter(*p)


plt.show()