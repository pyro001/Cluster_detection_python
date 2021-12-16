import numpy as np
import csv
import numpy.linalg as la
import scipy.optimize as sopt
from mpl_toolkits.mplot3d import axes3d
import random as ran
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sympy.interactive.printing import init_printing
from sympy.matrices import *
from sympy.plotting import plot3d
from sympy import *
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from numpy.linalg import norm 

data = np.loadtxt("Experimental_Data_xy.txt", delimiter =',')

n = 15; # number of particles
ns = 20; #number of iterations
c1 = 2; # cognitive coefficient
c2 = 2; #social coefficient
om = 0.9; # inertia weight
k = 1; #velocity restricting constant
t = 0.2; #animation speed/delay after each iteration

#search range the Ackley function. to test the ackley function
xmin = -10
xmax = 10
ymin = -10
ymax = 10

#maximum velocity
vmx = k*(xmax-xmin)/2
vmy = k*(ymax-ymin)/2

def Function(pos):
    '''The Ackley Function (2D-implementation)
    
    The 2D input vector needs to be of the form: 
    np.array([[-2., 2., 2.],[2., 3., -2.]])
    '''
    x = pos[0]
    y = pos[1]
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum_sq_term = -a * np.exp(-b * np.sqrt(x*x + y*y) / 2)
    cos_term = -np.exp((np.cos(c*x) + np.cos(c*y)) / 2)
    Z = -(a + np.exp(1) + sum_sq_term + cos_term) 
    return Z


# Need 
def update_velocity_testfunction(p_best,g_best,pos):
    v = ([np.array([0, 0]) for _ in range(n)])
    nv = []
    for i in range(n):
        nv.append((om*v[i]) + (c1*ran.random()) * (p_best[i] - pos[i]) + (c2*ran.random()) * (g_best-pos[i]))
        #Thresholding the Velocity
        #Thresholding along x-axis
        if (nv[i][0]>vmx):
            nv[i][0] = vmx
        elif (nv[i][0]<-vmx):
            nv[i][0] = -vmx

        #Thresholding along y-axis
        if (nv[i][1]>vmy):
            nv[i][1] = vmy
        elif (nv[i][1]<-vmy):
            nv[i][1] = -vmy
    return nv

# need 
def update_position_testfunction(v,pos):
    new_pos=[]
    for i in range(n):
        new_pos.append(pos[i]+v[i])
    return new_pos

# Dont need 
def next_particle_set(particles):
    x= []
    y = []
    z = []
    for p in range(len(particles)):
        x.append(particles[p][0].tolist())
        y.append(particles[p][1].tolist())
        z.append(Function(particles[p]))
    return [x, y, z]

# dont need 
def func_MG(x_vec, y_vec, Function):
    '''Convert the 2D-FUNCTION TO THE MESHGRID 
    (might be possible in an easier way)'''
    F = np.zeros((x_vec.shape[0],y_vec.shape[0]))
    for cntx in range(x_vec.shape[0]):
        for cnty in range(y_vec.shape[0]):
            F[cntx,cnty] = Function([x_vec[cntx], y_vec[cnty]])        
    return(F)  

# Setup work
x_vec = np.arange(-10.,10.,0.1)
y_vec = x_vec

# Convert function to Meshgrid - for plotting purposes: 
F = func_MG(x_vec,y_vec, Function)
X, Y = np.meshgrid(x_vec,y_vec)

#Define the function
def f2(a, b, y, x): 
    c = 5
        #return (a- x)**2 + b*(y - x**2)**2
    return a*np.exp(-np.power(x - b, 2)/(2*np.power(c, 2))) - y
print(data[0])

fig = plt.figure(figsize=(12, 6))
fig.suptitle('Supplied Experimental Data & Function', size=16, y=0.95)
ax1 = fig.add_subplot(121,projection = '3d')
ax1.scatter(*data.T, marker= 'o', c=np.random.rand(len(data)))
    
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Data', pad=10)

x = np.linspace(-0.20, 0.20, 30)
y = np.linspace(-0.020, 0.020, 30)
X, Y = np.meshgrid(x, y)
Z = f2(1, 2, X, Y)

ax2 = fig.add_subplot(122,projection = '3d')
ax2.plot_surface(X, Y, Z, alpha=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x,y)')
ax2.set_title('Function', pad=10)

# Dont need? 
def calc_error(args):
    '''args is the parameter space arguments as a list. In this case a and b.'''
    #Initially there is no error
    e_2 = 0
    
    for p in data:
        #Calculate the function value for the (x,y) of the point and subtract the z value from this and then square this and add to the total error.
        e_2 += (f2(args[0], args[1], p[0], p[1]) - p[2])**2
    
    #Return the total error
    return e_2

# Need 
def update_personal_best_testfunction(p_best,Par_Val,pos):
    for i in range(len(pos)):
        if ( Par_Val[i] <=Function(p_best[i]) ):
            p_best[i] = pos[i]
    return p_best

# need 
def update_global_best_testfunction(g_best,Par_Val,pos):
    for i in range(n):
        if ( Par_Val[i]<= Function(g_best) ):
            g_best = pos[i]
    return g_best

# Need 
def PSO_testfunction(n,ns,c1,c2,om,k,t,xmin,xmax,ymin,ymax):
    

    #Step 1a, 
    #Initializing the Position
    #pos = []
    #for i in range(n):
        #pos.append([ran.randrange(xmin,xmax), ran.randrange(ymin,ymax)])
    pos = np.array([np.array([ran.randrange(xmin,xmax), ran.randrange(ymin,ymax)]) for _ in range(n)])

    #Step 1b,
    #Initializing the Particles best position as the initial assumptions
    p_best = pos

    #Step 1c
    #Calulating fitness of each particle
    Par_Val = []
    for i in range(n-1):
        Par_Val.append(calc_error(pos[i])) 

    #Finding the maximum value and then the points that give this maximum value and then setting these points as the new global best

    #indices = [i for i, x in enumerate(Par) if x == max(Par)] #Finds the max value, returns the indices of the best position that gives this value
    global_best_pos_index = (Par_Val.index(min(Par_Val)))

    #sets the global best position 
    g_best = pos[global_best_pos_index]

    #Step 2
    #Iterating Until Stopping Criterion is met
    Iter=0
    particles=[pos]
    while (Iter <= ns):
        #step 2a
        #Update Particle Velocity
        v = update_velocity_testfunction(p_best,g_best,pos)
        #Step 2b
        #Update Particle Position
        pos = update_position_testfunction(v,pos)
        #step 2c
        #Evaluate fitness
        for i in range(n):
            Par_Val.insert(i, calc_error(pos[i]))
        
        #Step 2d
        #Updating the personal best
        #if
        p_best = update_personal_best_testfunction(p_best,Par_Val,pos)
        particles.append(p_best)
        #Step 2e
        #Updating the global best
        #if
        g_best = update_global_best_testfunction(g_best, Par_Val,pos)
        g_best_value = calc_error(g_best)
        print("The best position is: ", g_best," Value: ",g_best_value, " in iteration number ", Iter)
        Iter = Iter+1
            
    return g_best, g_best_value, p_best

best_point, best_value,particles = PSO_testfunction(n,ns,c1,c2,om,k,t,xmin,xmax,ymin,ymax)

fig = plt.figure(figsize=(12, 6))
fig.suptitle('Supplied Experimental Data & Function', size=16, y=0.95)
ax1 = fig.add_subplot(121,projection = '3d')
ax1.scatter(*data.T, marker= 'o', c=np.random.rand(len(data)))
    
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Data', pad=10)

x = np.linspace(-3, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = f2(best_point[0],best_point[1], X, Y)

ax2 = fig.add_subplot(122,projection = '3d')
ax2.plot_surface(X, Y, Z, alpha=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x,y)')
ax2.set_title('Function', pad=10)

x=[1,2]
Resedual = minimize(calc_error, x, method='nelder-mead',options={'xatol': 1e-8, 'disp': True})
print(Resedual.x)

fig = plt.figure(figsize=(12, 6))
fig.suptitle('Supplied Experimental Data & Function', size=16, y=0.95)
ax1 = fig.add_subplot(121,projection = '3d')
ax1.scatter(*data.T, marker= 'o', c=np.random.rand(len(data)))
    
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Data', pad=10)

x = np.linspace(-3, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = f2(Resedual.x[0],Resedual.x[1], X, Y)

ax2 = fig.add_subplot(122,projection = '3d')
ax2.plot_surface(X, Y, Z, alpha=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x,y)')
ax2.set_title('Function', pad=10)

plt.show()
   


