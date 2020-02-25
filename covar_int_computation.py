import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PLOT = False

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def omega_exact(sensor_traces):
    inv_sum = sum((1/x for x in sensor_traces))
    weights = []
    for trace in sensor_traces:
        weights.append((1/trace)/inv_sum)
    return weights


def omega_estimates(sensor_lists, step):
    num_sensors = len(sensor_lists)
    norm_to_solution_hyperplane = np.array([1]*num_sensors)

    sensor_hyperplanes = []
    sensor_hyperplanes_eqns = [norm_to_solution_hyperplane]
    sensor_hyperplanes_intercepts = [1]

    # n sensors to loop
    for i in range(num_sensors-1):
        hyperplane_points = []

        # Get all partial solution hyperplane points that are known exactly. n^2 run time as that's the number of coordinates
        for j in range(num_sensors):
            if j == i or j == i+1:
                continue
            known_point = np.array([0]*num_sensors)
            known_point[j] = 1
            hyperplane_points.append(known_point)
        
        # Find the remaing unkonwn partial solution hyperplane point - when all omega values (except for current and next sensor) are 0. log(p) run time.
        list_a = sensor_lists[i]
        list_b = list(reversed(sensor_lists[i+1]))
        om = intersect_approx(list_a, list_b, step)
        computed_point = np.array([0]*i + [om, 1-om] + [0]*(num_sensors-2-i))

        # Get the normal to the hyperplane defined by all the known points, and the computed point above
        avg_known_points = np.mean(hyperplane_points, axis=0)
        relevant_known_point = np.array([0]*i + [1] + [0]*(num_sensors-1-i))
        to_project = relevant_known_point - avg_known_points
        to_project_onto = computed_point - avg_known_points
        projection = (np.dot(to_project, to_project_onto)/np.dot(to_project_onto, to_project_onto)) * to_project_onto
        norm_to_hyperplane = to_project - projection

        # Some debug plotting
        if PLOT and i==0:
            ax.plot(*zip(avg_known_points, relevant_known_point), c='b')
            ax.scatter(*avg_known_points, marker='^', c='b')

            ax.plot(*zip(avg_known_points, computed_point), c='r')
            ax.scatter(*computed_point, marker='^', c='r')
            
            ax.plot(*zip(computed_point, computed_point+norm_to_hyperplane), c='g')
            ax.scatter(*computed_point+norm_to_hyperplane, marker='^', c='g')

        # Add the computed point to the partial solution hyperplane point list, completing the list
        hyperplane_points.append(computed_point)

        # Compute the hyperplane equation in the form ax1 + bx2 + ... + intercept = 0. Store as vector ((a,b,...), intercept) for easier computing later
        intercept = np.dot(norm_to_hyperplane, computed_point)
        sensor_hyperplanes_intercepts.append(intercept)
        sensor_hyperplanes_eqns.append(norm_to_hyperplane)

        # Add to list of sensor planes
        sensor_hyperplanes.append(np.array(hyperplane_points))
    
    # Convert all hyperplane equations to numpy arrays for solving
    sensor_hyperplanes_eqns = np.array(sensor_hyperplanes_eqns)
    sensor_hyperplanes_intercepts = np.array(sensor_hyperplanes_intercepts).T

    # TODO Should handle case of multiple 0 traces, by equally weighting them all and making the rest 0
    # Solve intersection of all hyperplanes
    omegas = np.linalg.solve(sensor_hyperplanes_eqns, sensor_hyperplanes_intercepts)

    # Some debug printing and plotting
    #print(sensor_hyperplanes_eqns)
    #print(sensor_hyperplanes_intercepts)
    #print(sensor_hyperplanes)

    if PLOT:
        ax.scatter(*zip(*[(a,b,c) for a in np.arange(0,1.1,0.1) for b in np.arange(0,1.1,0.1) for c in np.arange(0,1.1,0.1) if np.isclose(a+b+c, 1)]), c='grey')
        plt.show()

    return omegas

def intersect_approx(increasing_cmp_list, decreasing_cmp_list, step):
    curr_om = 0
    found_om = 0
    # TODO make this run in log(p) instead of p steps
    for i in range(len(increasing_cmp_list)):
        if increasing_cmp_list[i] == decreasing_cmp_list[i]:
            found_om = curr_om
            break
        elif increasing_cmp_list[i] > decreasing_cmp_list[i]:
            found_om = curr_om - 0.5*step
            break
        curr_om += step
    
    # Debug printing
    # print('approx intersection')
    # print(['%1.4f'%i for i in increasing_cmp_list])
    # print(['%1.4f'%i for i in decreasing_cmp_list])
    # print(found_om)
    # print('exact intersection')
    # print(intersect_exact(increasing_cmp_list, decreasing_cmp_list, step))
    # print()

    return found_om

def intersect_exact(increasing_cmp_list, decreasing_cmp_list, step):
    # Don't need step for exact solution, pass it to keep signature akin to the approximation function
    inc = increasing_cmp_list[-1]
    dec = decreasing_cmp_list[0]
    return dec/(inc+dec)


sensor_traces = [4.254, 7.111, 6.234, 2.23, 8.23423, 9.1]
disc_step = 0.1
print('Exact solution: ', ['%1.4f'%i for i in omega_exact(sensor_traces)])
print('Approx solution:', ['%1.4f'%i for i in omega_estimates([[w*i for w in np.arange(0, 1+disc_step, disc_step)] for i in sensor_traces], disc_step)])