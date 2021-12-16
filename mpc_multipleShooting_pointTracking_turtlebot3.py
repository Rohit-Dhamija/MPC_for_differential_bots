#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from numpy import *
import time
import math 

pi = math.pi
t_start = time.time()



"""# variable parameters 
"""

n_states = 3
n_controls = 2
N = 100                                                                         #Prediction horizon(same as control horizon)
delta_T = 0.2                                                                   #timestamp bw two predictions
                      
# if theta target is > pi, write as a negative angle
X_target = array([2,1,pi], dtype = 'f')                                          
error_allowed = 5e-2

Q_x = 100                                                                       # gains to control error in x,y,theta during motion
Q_y = 100
Q_theta = 6
R1 = 300                                                                        # gains to control magnitude of V and omega                                                                                                           
R2 = 75

error_allowed_in_g = 1e-100                                                     # error in contraints (should be ~ 0)

"""# parameters that depend on simulator 
"""
n_bound_var = n_states                                                          #although theta will never have any bound but, we need to specify it because X is part of OPT_variables           
x_bound_max = inf                      
x_bound_min = -inf                     
y_bound_max = inf                     
y_bound_min = -inf                     
theta_bound_max = inf                     
theta_bound_min = -inf                     


v_max = 0.22
v_min = -v_max
omega_max = 2.84                                                
omega_min = -omega_max






global x,y,theta,vx,vy,qx,qy,qz,qw,V,omega                                                                            # (x,y,theta) will store the current position and orientation 
                                                                                                                      # qx,qy,qz,qw will store the quaternions of the bot position
                                                                                                                      # V and omega will store the inputs to the bot(Speed and Angular Velocity)
    
def odomfunc(odom):

    global x,y,qx,qy,qz,qw,vx,vy,theta
    x = odom.pose.pose.position.x 
    y = odom.pose.pose.position.y 
    qx = odom.pose.pose.orientation.x 
    qy = odom.pose.pose.orientation.y
    qz = odom.pose.pose.orientation.z 
    qw = odom.pose.pose.orientation.w

    theta = math.atan2(2*(qx*qy+qw*qz),1-2*(qy*qy+qz*qz))                                                             # finding yaw from quaternions

    
def my_mainfunc():
    
    rospy.init_node('mpc_multipleShooting_pointTracking_turtlebot3', anonymous=True)

    rospy.Subscriber('/odom', Odometry , odomfunc)    
   
    instance = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    rate = rospy.Rate(10) # 10hz    
    rate.sleep()                                                                                                       #rate.sleep() to run odomfunc once

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
    
    f = ca.Function('f',[states,controls],[rhs]) 
    
    U = ca.SX.sym('U', n_controls,N)
    P = ca.SX.sym('P',1, n_states*2)
    X =ca.SX.sym('X', n_states, N+1)
    
    
    obj = 0
    g = []
    
    Q = ca.diagcat(Q_x, Q_y, Q_theta)                                  
    
    R = ca.diagcat(R1, R2)    
    
  
    for i in range(0,N):                                                                                                  

        cost_pred_st = ca.mtimes(  ca.mtimes( (X[0:n_states,i] - P[n_states:n_states*2].reshape((n_states,1)) ).T , Q )  ,  (X[0:n_states,i] - P[n_states:n_states*2].reshape((n_states,1)) )  )  + ca.mtimes(  ca.mtimes( (U[0:n_controls,i]).T , R )  ,  U[0:n_controls,i]  )  
        obj = obj + cost_pred_st  
        
    obj = obj + ca.mtimes(  ca.mtimes( (X[0:n_states,N] - P[n_states:n_states*2].reshape((n_states,1)) ).T , Q )  ,  (X[0:n_states,N] - P[n_states:n_states*2].reshape((n_states,1)) )  )   # X has an extra column
    # no need to put objective function in the main while loop, casadi solver updates it own its own
                                                           
    
    
    
    pred_st = zeros((n_states,1))    
    for i in range(0,N+1):
        if i == 0:
    	    g = ca.vertcat( g,( X[0:n_states,i] - P[0:n_states].reshape((n_states,1)) )  )        
        else:
            f_value = f(X[0:n_states,i-1],U[0:n_controls,i-1])     
            pred_st = X[0:n_states,i-1] + delta_T*f_value           
            g = ca.vertcat( g,(X[0:n_states,i] - pred_st.reshape((n_states,1)) )  )     
    # no need to put g function in the main while loop, casadi solver updates it own its own
    #also no need of ff function as in single shooting because X is also an OPT_variable
    
    

    OPT_variables = X.reshape((n_states*(N+1),1))          
    OPT_variables = ca.vertcat( OPT_variables, U.reshape((n_controls*N,1)) )         
    
    
    
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
    
    
    
    lbg = ca.DM.zeros((n_states*(N+1),1)) 
    ubg = ca.DM.zeros((n_states*(N+1),1))
    
    
    lbg[0:n_states*(N+1)] = - error_allowed_in_g
    ubg[0:n_states*(N+1)] =  error_allowed_in_g
    
    lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N,1)) 
    ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N,1)) 
    
    lbx[0:n_bound_var*(N+1):3] = x_bound_min                      
    ubx[0:n_bound_var*(N+1):3] = x_bound_max                       
    lbx[1:n_bound_var*(N+1):3] = y_bound_min                      
    ubx[1:n_bound_var*(N+1):3] = y_bound_max                     
    lbx[2:n_bound_var*(N+1):3] = theta_bound_min                       
    ubx[2:n_bound_var*(N+1):3] = theta_bound_max                      
    
    lbx[n_bound_var*(N+1):(n_bound_var*(N+1)+n_controls*N):2] = v_min                      
    ubx[(n_bound_var*(N+1)):(n_bound_var*(N+1)+n_controls*N):2] = v_max                       
    lbx[(n_bound_var*(N+1)+1):(n_bound_var*(N+1)+n_controls*N):2] = omega_min                      
    ubx[(n_bound_var*(N+1)+1):(n_bound_var*(N+1)+n_controls*N):2] = omega_max                      
    


    X_init = array([x,y,theta], dtype = 'f')                                                                                                                                              

    P = concatenate((X_init, X_target))  

    initial_X = ca.DM.zeros((n_states*(N+1)))                          #all initial predicted states are X_init
    initial_X[0:n_states*(N+1):3] = X_init[0]                        
    initial_X[1:n_states*(N+1):3] = X_init[1]                        
    initial_X[2:n_states*(N+1):3] = X_init[2]                       
    
    initial_con = ca.DM.zeros((n_controls*N,1))                        #initial control should be zero 


    
    n_iter = 0
    while ( ca.norm_2( P[0:n_states].reshape((n_states)) - X_target ) > error_allowed  ) :                                        # norm_2 calculates dist. bw two points

        n_iter += 1 
        args = {
                'lbx':lbx,
                'lbg':lbg,	    
                'ubx':ubx,
                'ubg':ubg,
                'p':P,
                'x0':ca.vertcat(initial_X,initial_con),                                      
               }
        
        sol = solver(
                        
                     x0=args['x0'],
                       
                     lbx=args['lbx'],
                     ubx=args['ubx'],
                    
                     lbg=args['lbg'],
                     ubg=args['ubg'],
                     p=args['p']
                          
                    )           
    
        X_U_sol = sol['x']

        V = (X_U_sol[n_states*(N+1)].full())[0][0]      
        omega = (X_U_sol[n_states*(N+1)+1].full())[0][0]

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

    
        print ("Odometry = " , P[0:n_states-1],"  Theta = ",P[n_states-1]) 
    
    
        for i in range(0,N*n_states):                          #initial state for next iteration should be the predicted one for that iteration
            initial_X[i] = X_U_sol[i+n_states]                                      # and we will keep the last control as zero, hence N range
    
        for i in range(0,(N-1)*n_controls):                      #initial control for next iteration should be the predicted one for that iteration
            initial_con[i] = X_U_sol[n_states*(N+1)+i+n_controls]                          # and we will keep the last control as zero, hence N-1 range
    
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
    
    
    
    
