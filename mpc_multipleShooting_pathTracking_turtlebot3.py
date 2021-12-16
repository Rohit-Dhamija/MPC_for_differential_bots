#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path                                                                                                                            
import casadi as ca
import numpy as np                                                                                                    
import time
import math 
from scipy.spatial import KDTree                                                                                       

pi = math.pi
inf = np.inf
t_start = time.time()



"""# variable parameters 
"""

n_states = 3
n_controls = 2
N =100                                                                      # Prediction horizon(same as control horizon)
                                                                                                                
                      
error_allowed = 0.1

U_ref = np.array([0.22,0], dtype ='f')     				                    # Reference velocity and reference omega					                               		                                                                   

Q_x = 300                                                                   # gains to control error in x,y,theta during motion
Q_y = 300
Q_theta =  20                                                                                                                                            
R1 = 250                                                                    # gains to control magnitude of V and omega                                                                                                           
R2 = 80

error_allowed_in_g = 1e-100                                                 # error in contraints (should be ~ 0)




"""# parameters that depend on simulator 
"""
n_bound_var = n_states                                 
x_bound_max = inf                                                           # enter x and y bounds when limited world like indoor environments                   
x_bound_min = -inf                     
y_bound_max = inf                      
y_bound_min = -inf                     
theta_bound_max = inf                     
theta_bound_min = -inf                     


v_max = 0.22                                                                                                                                             
v_min = 0#-v_max                                                            # when we don't want to track the path on backward direction                                                                                                                                            
omega_max = 2.84                                                
omega_min = -omega_max



global x,y,theta,qx,qy,qz,qw,V,omega                                        # (x,y,theta) will store the current position and orientation 
                                                                            # qx,qy,qz,qw will store the quaternions of the bot position
                                                                            # V and omega will store the inputs to the bot(Speed and Angular Velocity)                                                                                                                                                                                        
  

global total_path_points                                                                                                                                        
total_path_points = 0                                                                                                                                                                            
global path                                                                                                                                         

    
    
def odomfunc(odom):

    global x,y,qx,qy,qz,qw,theta
    x = odom.pose.pose.position.x 
    y = odom.pose.pose.position.y 
    qx = odom.pose.pose.orientation.x                                       # quaternions of location
    qy = odom.pose.pose.orientation.y
    qz = odom.pose.pose.orientation.z 
    qw = odom.pose.pose.orientation.w

    theta = math.atan2(2*(qx*qy+qw*qz),1-2*(qy*qy+qz*qz))                   # finding yaw from quaternions



def pathfunc(Path):                                                                                                                                       

    global total_path_points,path
    if total_path_points == 0:        
        total_path_points = len(Path.poses)
        path = np.zeros((total_path_points,2))													
        
    for i in range(0,total_path_points):                                                                                                                
        path[i][0] = Path.poses[i].pose.position.x
        path[i][1] = Path.poses[i].pose.position.y   
    

def my_mainfunc():
    
    rospy.init_node('mpc_multipleShooting_pointTracking_turtlebot3', anonymous=True)

    rospy.Subscriber('/odom', Odometry , odomfunc)    
   
    rospy.Subscriber('/astroid_path', Path, pathfunc)                                                                                                                                                                   
    
    instance = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    rate = rospy.Rate(10)
    rate.sleep()                                                                #rate.sleep() to run odomfunc once 

    path_resolution =  ca.norm_2(path[0,0:2] - path[1,0:2])                                                                                 
    global delta_T                                                              #timestamp bw two predictions                                                                                                                

 
    delta_T = ( path_resolution / ((U_ref[0])) )/10                                                                                                           

    msg = Twist()


    """MPC"""

    x_casadi =ca.SX.sym('x')
    y_casadi = ca.SX.sym('y')
    theta_casadi = ca.SX.sym('theta')
    states =np.array([(x_casadi),(y_casadi),(theta_casadi)])
    n_states = states.size           
    
    v_casadi =ca.SX.sym('v')
    omega_casadi = ca.SX.sym('omega')
    controls = np.array([v_casadi,omega_casadi])
    n_controls = controls.size       
    
    rhs = np.array([v_casadi*ca.cos(theta_casadi),v_casadi*ca.sin(theta_casadi),omega_casadi])                         
    f = ca.Function('f',[states,controls],[rhs])                                                            # function to predict rhs using states and controls                                                                 

    
    U = ca.SX.sym('U', n_controls,N)                                                                        # For storing predicted controls                                                               
    X =ca.SX.sym('X', n_states, N+1)                                                                        # For storing predicted states
    P = ca.SX.sym('P',1, n_states + n_states*(N) + n_controls*(N) )                                         # For storing odometry, next N path points and next N referance controls    
    
    obj = 0
    g = []
    
    Q = ca.diagcat(Q_x, Q_y,Q_theta)                                                                                                                                              
    
    R = ca.diagcat(R1, R2)    
        
    for i in range(0,N):                                                                                                                                                                                                                            
        cost_pred_st = ca.mtimes(  ca.mtimes( (X[0:n_states,i] - P[n_states*(i+1) :n_states*(i+1) + n_states ].reshape((n_states,1)) ).T , Q )  ,  (X[0:n_states,i] - P[n_states*(i+1) :n_states*(i+1) + n_states ].reshape((n_states,1)) )  )  + ca.mtimes(  ca.mtimes( ( (U[0:n_controls,i]) - P[n_states*(N+1)+n_controls*(i):n_states*(N+1)+n_controls*(i) + n_controls].reshape((n_controls,1)) ).T , R )  ,  U[0:n_controls,i] - P[n_states*(N+1)+n_controls*(i):n_states*(N+1)+n_controls*(i) + n_controls].reshape((n_controls,1))  )  
        obj = obj + cost_pred_st  
           
    pred_st = np.zeros((n_states,1))     
    
    
    for i in range(0,N+1):                                                                                                      # adding contraints so the predictions are in sync with vehicle model
        if i == 0:
    	    g = ca.vertcat( g,( X[0:n_states,i] - P[0:n_states].reshape((n_states,1)) )  )                                                                                                     
        else:
            #f_value = f(X[0:n_states,i-1],U[0:n_controls,i-1])                   # euler method not used  
            #pred_st = X[0:n_states,i-1] + delta_T*f_value                                                                     

            K1 = f(X[0:n_states,i-1],U[0:n_controls,i-1])                         # Runge Kutta method of order 4 
            K2 = f(X[0:n_states,i-1] + np.multiply(K1,delta_T/2),U[0:n_controls,i-1])                                              
            K3 = f(X[0:n_states,i-1] + np.multiply(K2,delta_T/2),U[0:n_controls,i-1])                                            
            K4 = f(X[0:n_states,i-1] + np.multiply(K3,delta_T),U[0:n_controls,i-1])                                                  
            pred_st = X[0:n_states,i-1] + (delta_T/6)*(K1+2*K2+2*K3+K4)           # predicted state                                                          
             
            g = ca.vertcat( g,(X[0:n_states,i] - pred_st[0:n_states].reshape((n_states,1)) )  )                                                                
    
    
    
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


    lbg = ca.DM.zeros(((n_states)*(N+1),1))                                     # bounds on g                                                                                                      
    ubg = ca.DM.zeros(((n_states)*(N+1),1))                                                                                                         
       
    lbg[0:(n_states)*(N+1)] = - error_allowed_in_g                                                                                                  
    ubg[0:(n_states)*(N+1)] =  error_allowed_in_g                                                                                                    
    


    lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N,1))                        # bounds on X    
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
    
                               
    X_init = np.array([x,y,theta], dtype = 'f')                                                                                                                                                                                    
    X_target = np.array([ path[total_path_points-1][0], path[total_path_points-1][1], 0 ]  , dtype = 'f')   
   
    P = X_init                                                                                      

    close_index = KDTree(path).query(P[0:n_states-1])[1]                                      

    for i in range(0,N):                                                                           
        P = ca.vertcat(P,path[close_index+i,0:2])                                                         
        P = ca.vertcat(P, math.atan((path[close_index+i+1][1] - path[close_index+i][1])/(path[close_index+i+1][0] - path[close_index+i][0])) )        

    for i in range(0,N):                                                                             
        P = ca.vertcat(P, U_ref[0])                                                                 
        P = ca.vertcat(P, U_ref[1])                                                                   

    initial_X = ca.DM.zeros((n_states*(N+1)))                               #all initial predicted states are X_init    
    initial_X[0:n_states*(N+1):3] = X_init[0]
    initial_X[1:n_states*(N+1):3] = X_init[1]
    initial_X[2:n_states*(N+1):3] = X_init[2]
    
    initial_con = ca.DM.zeros((n_controls*N,1))                             #initial search value of control matrix
    
    n_iter = 0
    while ( ca.norm_2( P[0:n_states-1].reshape((n_states-1,1)) - X_target[0:n_states-1] ) > error_allowed  ) :                                                                                         

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


        msg.linear.x = V                                                             #linear.x and linear.y are velocities in local coordinates of bot                                                                                             
        msg.linear.y = 0                                                             # linear.y always zero, linear.x is the speed of a diff. bot
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = omega
        
        instance.publish(msg)

        P[0:n_states] = [x,y,theta]                                                                                                               


        close_index = KDTree(path).query(np.array([x,y]))[1]

        if N+(close_index-1) < total_path_points :                                                      # Updating P for next N path points and next N reference controls
            P[n_states:n_states*(N+1):n_states] = path[close_index:N+close_index,0] 
            P[n_states+1:n_states*(N+1):n_states] = path[close_index:N+close_index,1]
            for i in range(0,N):                                                                
                P[n_states*(i+1+1)-1] = math.atan( (path[i+close_index+1][1] - path[i+close_index][1])/(path[i+close_index+1][0] - path[i+close_index][0] + 1e-9) )         
            
            P[n_states*(N+1):n_states*(N+1)+n_controls*(N-1)]= P[n_states*(N+1)+n_controls:n_states*(N+1)+n_controls*(N)]                                                                                                                                                                                                             
            P[n_states*(N+1)+n_controls*(N-1):n_states*(N+1)+n_controls*(N)] = U_ref                                                                                     
        else:
            print (" The end point in inside horizon, slowing down")
            P[n_states:n_states*(N)] = P[n_states*2:n_states*(N+1)]                                                                  
            P[n_states*(N):n_states*(N+1)-1] = path[(total_path_points-1),0:2]                                                                                                                                                                                                                    
            P[n_states*(N+1)-1] = math.atan( (path[total_path_points-1][1] - path[total_path_points-1-1][1])/(path[total_path_points-1][0] - path[total_path_points-1-1][0]) )
                
            P[n_states*(N+1):n_states*(N+1)+n_controls*(N-1)]= P[n_states*(N+1)+n_controls:n_states*(N+1)+n_controls*(N)]                                                                                                                                                                                                                  
            P[n_states*(N+1)+n_controls*(N-1):n_states*(N+1)+n_controls*(N)] = np.array([0,0], dtype ='f')                  # we need to stop the bot at end, hence referance controls 0 at end 
          
        for i in range(0,N*n_states):                                           #initial search value of state for next iteration should be the predicted one for that iteration     
            initial_X[i] = X_U_sol[i+n_states]                 

        for i in range(0,(N-1)*n_controls):                                     #initial search value of control for next iteration should be the predicted one for that iteration
            initial_con[i] = X_U_sol[n_states*(N+1)+i+n_controls]
                
        rate.sleep()


    
            
    print ("PATH TRACKED")                                                                                                                   
    print ("Total MPC iterations = " , n_iter)
    t_end = time.time()
    print ("Total Time taken = " , t_end - t_start)
    

    msg.linear.x = 0                                                # stopping the bot                                      
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
