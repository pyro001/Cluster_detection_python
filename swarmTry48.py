import random as ran

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


# Need
def update_velocity_testfunction(p_best, g_best, pos):
    v = ([np.array([0, 0, 0]) for _ in range(particle_number)])
    nv = []
    print(p_best.shape, g_best.shape)
    for i in range(particle_number):
        nv.append((om * v[i]) + (c1 * ran.random()) * (p_best[i] - pos[i]) + (c2 * ran.random()) * (g_best - pos[i]))
        # Thresholding the Velocity
        # Thresholding along x-axis
        if (nv[i][0] > vmx):
            nv[i][0] = vmx
        elif (nv[i][0] < -vmx):
            nv[i][0] = -vmx

    return nv


# need
def update_position_testfunction(v, pos):
    new_pos = []
    for i in range(particle_number):
        new_pos.append(pos[i] + v[i])
    return new_pos


# Define the function
def functionGauss(peak, stddev, center, actual_x):
    return peak * np.exp(-np.power(actual_x - stddev, 2) / (2 * np.power(center, 2)))


def modelFunction(peak, stddev, center, x):
    return (peak) * np.power(stddev, 2) / (np.power(x - (center), 2) + np.power(stddev, 2))


# Error function
def calc_error(args):
    '''args is the parameter space arguments as a list. In this case a and b.'''
    # Initially there is no error
    e_2 = 0

    for p in data:
        # Calculate the function value for the (x,y) of the point and subtract the z value from this and then square this and add to the total error.
        e_2 += (modelFunction(args[0], args[1], args[2], p[1]) - p[0]) ** 2

    # Return the total error
    return e_2


# Need
def update_personal_best_testfunction(p_best, Par_Val, pos):
    for i in range(len(pos)):
        if (Par_Val[i] <= calc_error(p_best[i])):
            p_best[i] = pos[i]
    return p_best


# need
def update_global_best_testfunction(g_best, Par_Val, pos):
    for i in range(particle_number):
        if (Par_Val[i] <= calc_error(g_best)):
            g_best = pos[i]
    return g_best


# Need
def PSO_testfunction(n, ns, c1, c2, om, k, t, xmin, xmax):
    # Step 1a,
    # Initializing the Position
    xmax = int(xmax)
    pos = np.array(
        [np.array([ran.randrange(xmin, xmax), ran.randrange(xmin, xmax), ran.randrange(xmin, xmax)]) for _ in range(n)])

    # Step 1b,
    # Initializing the Particles best position as the initial assumptions
    p_best = pos

    # Step 1c
    # Calulating fitness of each particle
    Par_Val = []
    for i in range(n - 1):
        Par_Val.append(calc_error(pos[i]))

        # Finding the maximum value and then the points that give this maximum value and then setting these points as the new global best

    # indices = [i for i, x in enumerate(Par) if x == max(Par)] #Finds the max value, returns the indices of the best position that gives this value
    global_best_pos_index = (Par_Val.index(min(Par_Val)))

    # sets the global best position
    g_best = pos[global_best_pos_index]

    # Step 2
    # Iterating Until Stopping Criterion is met
    Iter = 0
    particles = [pos]
    while (Iter <= ns):
        # step 2a
        # Update Particle Velocity
        v = update_velocity_testfunction(p_best, g_best, pos)
        # Step 2b
        # Update Particle Position
        pos = update_position_testfunction(v, pos)
        # step 2c
        # Evaluate fitness
        for i in range(n):
            Par_Val.insert(i, calc_error(pos[i]))

        # Step 2d
        # Updating the personal best
        # if
        p_best = update_personal_best_testfunction(p_best, Par_Val, pos)
        particles.append(p_best)
        # Step 2e
        # Updating the global best
        # if
        g_best = update_global_best_testfunction(g_best, Par_Val, pos)
        g_best_value = calc_error(g_best)
        print("The best position is: ", g_best, " Value: ", g_best_value, " in iteration number ", Iter)
        Iter = Iter + 1

    return g_best, g_best_value, p_best


if __name__ == '__main__':
    global data
    # SETUP######
    data = np.loadtxt("./pictures/T001.txt", delimiter=',')

    particle_number = 150  # number of particles
    Iterations = 600  # number of iterations
    c1 = 2  # cognitive coefficient
    c2 = 2  # social coefficient
    om = 0.9  # inertia weight
    k = 1  # velocity restricting constant
    t = 0.2  # animation speed/delay after each iteration

    # search range of the gaussian
    xmin = 0
    xmax = (max(data[:, 1]))

    ymax = (max(data[:, 0]))
    # maximum velocity
    vmx = k * (xmax - xmin) / 2

    # ===================================================================================#

    fig = plt.figure(figsize=(12, 6))
    # fig.suptitle('Supplied Experimental Data & Function', size=16, y=0.95)

    plt.scatter(*data.T, marker='o', c=np.random.rand(len(data)))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data', pad=10)
    plt.show()
    x = np.linspace(0, 100, 200)
    index_est_center = np.where(data[:,0] == ymax)
    mean = index_est_center[0]
    try:
        mean=mean[1]
        print(mean)
    except IndexError:
        mean=mean



    Y = modelFunction(ymax, mean, mean, x)

    plt.scatter(x, Y, alpha=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function', pad=10)
    plt.show()
    # =================================================================================================#
    best_point, best_value, particles = PSO_testfunction(particle_number, Iterations, c1, c2, om, k, t, xmin, xmax)
    ## plot the stuff

    plt.title('Supplied Experimental Data & Function', size=16, y=0.95)
    plt.scatter(*data.T, marker='o')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    x = np.linspace(0, 2*int(xmax), 200)
    Y = modelFunction(best_point[0], best_point[1], best_point[2], x)

    plt.scatter(x, Y, alpha=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('optimized Data', pad=10)
    plt.show()
    #################################################################################################
    ######CROSS VERIFICATION WITH NELDER MEAD#####
    # x = [1]
    # Residual = minimize(calc_error, x, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    # print(Residual.x)
    #
    # fig = plt.figure(figsize=(12, 6))
    # fig.suptitle('Supplied Experimental Data & Function', size=16, y=0.95)
    # ax1 = fig.add_subplot()
    # ax1.scatter(*data.T, marker='o', c=np.random.rand(len(data)))
    #
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # ax1.set_title('Data', pad=10)
    #
    # x = np.linspace(-3, 6, 30)
    # Y = modelFunction(Residual.x[0], Residual.x[1], x)
    #
    # ax2 = fig.add_subplot()
    # ax2.plot_surface(x, Y, alpha=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_title('Function', pad=10)

    # plt.show()
