# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:29:59 2023@author: Bijo
"""
import math
import cvxpy
import numpy as np
import matplotlib.pyplot as mp
import time

#params
DT = 0.1
horizon_length = 5 
angular_rate = 5.0 #rad/s
velocity_rate = 5 #m/s
number_of_states = 3
number_of_control_inputs = 2
R = np.diag([0.0, 0.0])  # input cost matrix
Q = np.diag([10.0, 10.0, 0.0])  # state cost matrix#MPC helper
windowSize = 5

#Plotter setup  
mp.close('all')
mp.ion()  
fig = mp.figure()
mp.axis([-15, 5, -5, 5])
    
def get_nparray_from_matrix(x):
    return np.array(x).flatten()

#The motion model for MPC
def get_linear_model_matrix(X_bar, U_bar):
    
    x = X_bar[0]
    y = X_bar[1]
    theta = X_bar[2]
    v = U_bar[0]
    w = U_bar[1]

    A = np.zeros((number_of_states, number_of_states))
    A[0, 0] =  1.0
    A[1, 1] =  1.0
    A[2, 2] =  1.0   
    A[0, 2] = -1.0*v*math.sin(theta)*DT
    A[1, 2] =  v*math.cos(theta)*DT
   
    B = np.zeros((number_of_states, number_of_control_inputs))
    B[0, 0] = math.cos(theta)*DT
    B[1, 0] = math.sin(theta)*DT
    B[2, 1] = DT
    
    C = np.zeros(number_of_states)
    C[0] =  v*math.sin(theta)*theta*DT
    C[1] = -1.0*v*math.cos(theta)*theta*DT
    return A, B, C


#The MPC implimentation
def mpc(X_ref, X_bar, U_bar):    
    
    #Create the optimsation variable x and u
    #Argument is the shape of the vector
    x = cvxpy.Variable((number_of_states, horizon_length + 1))
    u = cvxpy.Variable((number_of_control_inputs , horizon_length))    
    
    #set up costs
    cumulative_cost  = 0.0
    for t in range (horizon_length):        
        #Add up control cost
        cumulative_cost += cvxpy.quad_form(u[:, t], R)    
        
        #Add up state cost for updates cycles
        if t != 0:
            cumulative_cost += cvxpy.quad_form(X_ref[:, t] - x[:, t], Q)    
            
    #Add up state cost for last update cycle
    cumulative_cost += cvxpy.quad_form(X_ref[:, horizon_length] - x[:, horizon_length], Q)    
    
    #set up constraints
    constraint_vector = []    
    
    for t in range(horizon_length):        
        
        #Get updated model matrices
        A, B, C = get_linear_model_matrix(X_bar, U_bar)        
        
        #Add state evolution constraint
        constraint_vector += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]        
        
        #Add control constraint
        constraint_vector += [cvxpy.abs(u[0,t]) <= (velocity_rate)] 
        constraint_vector += [cvxpy.abs(u[1,t]) <= (angular_rate)]

    #initial condition
    constraint_vector += [x[:, 0] == X_bar]    
    
    #Formulate problem and solve
    prob = cvxpy.Problem(cvxpy.Minimize(cumulative_cost), constraint_vector)
    prob.solve(solver=cvxpy.ECOS, verbose=False)    
    
    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        OX = get_nparray_from_matrix(x.value[0, :])
        OY = get_nparray_from_matrix(x.value[1, :])
        OTheta = get_nparray_from_matrix(x.value[2, :])
        OV = get_nparray_from_matrix(u.value[0, :])
        OW = get_nparray_from_matrix(u.value[1, :])    
        
    else:
        print("Error: Cannot solve mpc..")
        OX, OY, OTheta = None, None, None
        OV, OW = None, None    
        
    return [OX, OY, OTheta, OV, OW]

# The test setup
def test_setup():    
    
    #Create variables
    X_ref_global = np.zeros((3,101))
    X_ref = np.zeros((number_of_states, horizon_length + 1))
    X_bar = np.zeros(number_of_states)
    U_bar = np.zeros(number_of_control_inputs) 
    
    #latch down initial pose
    X_bar[0] = 0.0
    X_bar[1] = 0.0
    X_bar[2] = math.pi/2.0
    #plot robot
    robo_fig_handle, = mp.plot(X_bar[0], X_bar[1], 'ro', ms = 5.0)
    
    #latch down initial controls
    U_bar = [0.0, 0.0]  
    
    #Create and plot global reference
    for i in range(101):
            X_ref_global[0,i] = 5.0*math.cos(i/10.0) - 5
            X_ref_global[1,i] = 2.0*math.sin(i/10.0)
            if i!=0:
                X_ref_global[2,i] = math.atan2(X_ref_global[1,i] - X_ref_global[1,i-1], X_ref_global[0,i] - X_ref_global[0,i-1] )
            else:
                X_ref_global[2,i] = math.pi/2.0
            #plot ref traj
            mp.plot(X_ref_global[0,i], X_ref_global[1,i], 'ko', ms = 1.0)
             
    for i in range(101 - horizon_length - 1 ):
        
        #Run MPC for local window
        OX, OY, OTheta, OV, OW = mpc(X_ref_global[:, i: i + horizon_length +1], X_bar, U_bar) 
        print(X_ref_global[:,i: i + horizon_length +1])
        print(X_bar)
        print(U_bar)
        
        print(OX)
        print(OY)
        print(OTheta)
        print(OV)
        print(OW)
        #Plot tracked traj
        robo_fig_handle.set_xdata(OX[1])
        robo_fig_handle.set_ydata(OY[1])
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)
        
        #Update states
        X_bar[0] = OX[1]
        X_bar[1] = OY[1]
        X_bar[2] = OTheta[1]
        
        #update controls
        U_bar = [OV[1], OW[1]] 
        
        print('completed iter',i)

    return

#run
if __name__ == '__main__':    
    test_setup()
    print ('Program ended')
