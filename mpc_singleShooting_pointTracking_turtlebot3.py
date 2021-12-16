#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import casadi as ca
from numpy import *
import time
import math 

pi = math.pi
t_start = time.time()


"""# variable parameters 
"""

n_states = 3
n_controls = 2
N =100                                                                                                                  #Prediction horizon(same as control horizon)
delta_T = 0.2                                                                                                           #timestamp bw two predictions

#if theta target is > pi, write as a negative angle
X_target = array([-5,-3,pi/2], dtype = 'f')                                                                            
error_allowed = 5e-2         

Q_x = 100                                                                                                               # gains to control error in x,y,theta during motion
Q_y = 100
Q_theta = 5
R1 = 300                                                                                                                # gains to control magnitude of V and omega                                                                                                           
R2 =75



"""# parameters that depend on simulator 
"""
n_bound_var = 2                                                                                                         # there are only two bounded variables, theta has no bound
x_bound_max = inf                                                                                                                                                                                        
x_bound_min = -inf                                                                                                                                                                                       
y_bound_max = inf                                                                                                                                                                                        
y_bound_min = -inf                                                                                                                                                                                    

v_max = 0.22                                                                                                            # Max speed and omega limits for Turtlebot3-burger bot
v_min = -v_max
omega_max = 2.84                                                
omega_min = -omega_max





global x,y,theta,vx,vy,qx,qy,qz,qw,V,omega                                                                              # (x,y,theta) will store the current position and orientation 
                                                                                                                        # qx,qy,qz,qw will store the quaternions of the bot position
                                                                                                                        # V and omega will store the inputs to the bot(Speed and Angular Velocity) 
     
    
    
def odomfunc(odom):

    global x,y,qx,qy,qz,qw,vx,vy,theta
    x = odom.pose.pose.position.x 
    y = odom.pose.pose.position.y 
    qx = odom.pose.pose.orientation.x                                                                                   # quaternions of location
    qy = odom.pose.pose.orientation.y
    qz = odom.pose.pose.orientation.z 
    qw = odom.pose.pose.orientation.w

    theta = math.atan2(2*(qx*qy+qw*qz),1-2*(qy*qy+qz*qz))                                                               # finding yaw from quaternions
    
def my_mainfunc():
    
    rospy.init_node('mpc_singleShooting_pointTracking_turtlebot3', anonymous=True)

    rospy.Subscriber('/odom', Odometry , odomfunc)    
      
    instance = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    rate = rospy.Rate(10) # 10hz    
    rate.sleep()                                                                                                       # rate.sleep() to run odomfunc once
    
    msg = Twist()

    """MPC"""

    x_casadi =ca.SX.sym('x')                                                                                          
    y_casadi = ca.SX.sym('y')
    theta_casadi = ca.SX.sym('theta')
    states =array([(x_casadi),(y_casadi),(theta_casadi)])                                                                                                                                                
    n_states = states.size                                                                                                                                                                               

    v_casadi =ca.SX.sym('v')
    omega_casadi = ca.SX.sym('omega')
    controls = array([v_casadi,omega_casadi])                                                                                                                                                           
    n_controls = controls.size                                                                                                                                                                           

    rhs = array([v_casadi*ca.cos(theta_casadi),v_casadi*ca.sin(theta_casadi),omega_casadi])                                                                                             

    f = ca.Function('f',[states,controls],[rhs])                                                                        # function to predict rhs using states and controls
                                                         
    U = ca.SX.sym('U', n_controls,N)                                                                                    # For storing predicted controls                                                                         
    X =ca.SX.sym('X', n_states, N+1)                                                                                    # For storing predicted states
    P = ca.SX.sym('P',1, n_states*2)                                                                                    # For storing odometry, next N path points and next N referance controls 

    pred_st = zeros((n_states,1))                                                                                                                                                                        #numpy
    for i in range(0,N+1):
        if i == 0:
            X[0:n_states,i] =  P[0:n_states]                                                                                                                                                             #numpy
        else:
            f_value = f(X[0:n_states,i-1],U[0:n_controls,i-1])                                                                                                                                           #numpy
            pred_st = X[0:n_states,i-1] + delta_T*f_value                                                                                                                                                #numpy
            X[0:n_states,i] = pred_st                                                                                                                                                                    #numpy

    ff = ca.Function('ff',[U,P],[X])
	
    obj = 0
    g = []

    Q = ca.diagcat(Q_x, Q_y, Q_theta)                                  

    R = ca.diagcat(R1, R2)                                             

    for i in range(0,N):                                                                                                  

        cost_pred_st = ca.mtimes(  ca.mtimes( (X[0:n_states,i] - P[n_states:n_states*2].reshape((n_states,1)) ).T , Q )  ,  (X[0:n_states,i] - P[n_states:n_states*2].reshape((n_states,1)) )  )  + ca.mtimes(  ca.mtimes( (U[0:n_controls,i]).T , R )  ,  U[0:n_controls,i]  )  
        obj = obj + cost_pred_st  
        
    obj = obj + ca.mtimes(  ca.mtimes( (X[0:n_states,N] - P[n_states:n_states*2].reshape((n_states,1)) ).T , Q )  ,  (X[0:n_states,N] - P[n_states:n_states*2].reshape((n_states,1)) )  )   # X has an extra column
                                                                                                                      



    for i in range(1,N+1):                                                                                          #first column of X is odom, no need to bound it
        for k in range(0,n_bound_var):
            g = ca.vertcat(g,X[k,i])

    OPT_variables = U.reshape((n_controls*N,1))                                                                                                                                                          #numpy


    nlp_prob ={
               'f':obj,
               'x':OPT_variables,
               'g':g,
               'p':P
              }

    opts = {
             'ipopt':
            {
              'max_iter': 100,
              'print_level': 0,
              'acceptable_tol': 1e-8,
              'acceptable_obj_change_tol': 1e-6
            },
             'print_time': 0
           }


    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    lbg = ca.DM.zeros((n_bound_var*N,1))                                                                            # bounds on g
    ubg = ca.DM.zeros((n_bound_var*N,1))

    lbg[0:n_bound_var*N:n_bound_var] = x_bound_min                                                                                                                                                                 #numpy
    ubg[0:n_bound_var*N:n_bound_var] = x_bound_max                                                                                                                                                                 #numpy
    lbg[1:n_bound_var*N:n_bound_var] = y_bound_min                                                                                                                                                                 #numpy
    ubg[1:n_bound_var*N:n_bound_var] = y_bound_max                                                                                                                                                                 #numpy

    lbx = ca.DM.zeros((n_controls*N,1))                                                                             # bounds on X 
    ubx = ca.DM.zeros((n_controls*N,1)) 

    lbx[0:n_controls*N:n_controls] = v_min                                                                                                                                                                        #numpy
    ubx[0:n_controls*N:n_controls] = v_max                                                                                                                                                                        #numpy
    lbx[1:n_controls*N:n_controls] = omega_min                                                                                                                                                                    #numpy
    ubx[1:n_controls*N:n_controls] = omega_max                                                                                                                                                                    #numpy






    X_init = array([x,y,theta], dtype = 'f')                                                                                                                                            

    P = concatenate((X_init, X_target))                                                                                                                                                                  #numpy
    initial_con = ca.DM.zeros((n_controls*N,1))                                                                      #initial search value of control matrix 

    n_iter = 0
    
    
    while ( ca.norm_2( P[0:n_states].reshape((n_states)) - X_target ) > error_allowed  ) :                           # norm_2 calculates dist. bw two points
        
        n_iter += 1 
        args = {
                'lbx':lbx,
                'lbg':lbg,	    
                'ubx':ubx,
                'ubg':ubg,
                'p':P,
                'x0':initial_con,                                      
               }
        
        sol = solver(
                        
                     x0=args['x0'],
                       
                     lbx=args['lbx'],
                     ubx=args['ubx'],
                    
                     lbg=args['lbg'],
                     ubg=args['ubg'],
                     p=args['p']
                          
                    )           
        
        U_sol = sol['x']	

        V = (U_sol[0].full())[0][0]
        omega = (U_sol[1].full())[0][0]
 

        #omega_left_wheel = (V - omega*robot_dia)/wheel_rad                          # differential drive kinematics (when global x cross y faces upward)
        #omega_right_wheel = (V + omega*robot_dia)/wheel_rad

        #omega_left_wheel = (V + omega*robot_dia)/wheel_rad                          # differential drive kinematics (when global x cross y faces downward)
        #omega_right_wheel = (V - omega*robot_dia)/wheel_rad
                                                                                                                     #linear.x and linear.y are velocities in local coordinates of bot
        msg.linear.x = V                                                                                             # linear.y always zero, linear.x is the speed of a diff. bot
        msg.linear.y = 0                            
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = omega
        
        instance.publish(msg)

 
        P[0:n_states] = [x,y,theta]                                                                                
        

        print ("Odometry = " , P[0:n_states])
        ff(U_sol.reshape((n_controls,N)) , P)

        for i in range(0,(N-1)*n_controls):                                                                          #initial control for next iteration should be the predicted one for that iteration
            initial_con[i] = U_sol[i+n_controls]                                                                              # and we will keep the last control as zero, hence N-1 range
                                                                 
        
        
        rate.sleep()
    
    print ("TARGET REACHED")
    print ("Total MPC iterations = " , n_iter)
    t_end = time.time()
    print ("Total Time taken = " , t_end - t_start)

    msg.linear.x = 0                                                                                                 # to stop the bot
    msg.linear.y = 0 
    msg.linear.z = 0 
    msg.angular.x = 0 
    msg.angular.y = 0 
    msg.angular.z = 0 
    instance.publish(msg)

    
if __name__ == '__main__':
   
    try:
        my_mainfunc()
    except rospy.ROSInterruptException:
        pass

