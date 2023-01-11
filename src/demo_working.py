import numpy as np
import matplotlib.pyplot as plt
import random
from functions import *
import random

import absl.logging #prevent checkpoint warnings while training
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
from keras import backend as K
from tensorflow.python.client import device_lib
from config import *
from motion_refiner_4D import Motion_refiner
from functions import *
from TF4D_mult_features import *

from scipy.spatial.transform import Rotation as R

reset_seed(seed = seed)

##########################
# Load preprocessed data #
##########################

# traj_n = 40
# mr = Motion_refiner(load_models=True ,traj_n = traj_n, locality_factor=True)
# feature_indices, obj_sim_indices, obj_poses_indices, traj_indices = mr.get_indices()
# embedding_indices = np.concatenate([feature_indices,obj_sim_indices, obj_poses_indices])

# # dataset_name = "4D_100k_scaping_factor"
# dataset_name = "latte_100k_lf"

# X,Y, data = mr.load_dataset(dataset_name, filter_data = True, base_path=base_folder+"../data/")
# X_train, X_test, X_valid, y_train, y_test, y_valid, indices_train, indices_test, indices_val = mr.split_dataset(X, Y, test_size=0.2, val_size=0.1)

# idx = random.choices(range(len(data)),k=1)
# data_sample = list(np.array(data)[idx])
# # show_data4D(data_sample)

# print(idx)

# #object matching accuracy
# mr.evaluate_obj_matching(data)

##############
# Load model #
##############
model_path = models_folder
model_name = "TF-num_layers_enc:1-num_layers_dec:5-d_model:400-dff:512-num_heads:8-dropout_rate:0.1-wp_d:4-num_emb_vec:4-bs:16-dense_n:512-num_dense:3-concat_emb:True-features_n:793-optimizer:adam-norm_layer:True-activation:tanh.h5"
model_file = model_path+model_name
model = load_model(model_file, delimiter ="-")

# x_test_new, y_test_new = mr.prepare_x(X_test), list_to_wp_seq(y_test,d=4)
# emb_test_new = X_test[:,embedding_indices]


# test_dataset = tf.data.Dataset.from_tensor_slices((prepare_x(X_test,traj_indices,obj_poses_indices),
#                                                 list_to_wp_seq(y_test,d=4),
#                                                 X_test[:,embedding_indices])).batch(3)
# x_t, y_t= next(generator(test_dataset))
# data_array = np.array(data)[indices_test[:3]]

# show_data4D(data_array, pred=x_t[0][:,6:,:], color_traj=False, plot_speed=False)

###########
# Testing #
###########
def interpolate_traj(wps,traj_n=40):
    #create spline function
    f, u = interpolate.splprep([wps[:,0],wps[:,1],wps[:,2]], s=0)
    xint,yint,zint= interpolate.splev(np.linspace(0, 1, traj_n), f)

    tck,u = interpolate.splprep([np.linspace(0,1,len(wps[:,3])), wps[:,3]])
    velint_x, velint = interpolate.splev(np.linspace(0, 1, traj_n), tck)

    traj = np.stack([xint,yint,zint,velint],axis=1)+offset
    return traj


# def norm_traj_and_objs(t, o, margin=0.45):
#     pts_ = np.concatenate([o,t])

#     vel = pts_[:,3:]
#     pts = pts_[:,:3]

#     vel_min = np.min(vel,axis = 0)
#     vel_max = np.max(vel,axis = 0)
#     vel_norm = np.max(np.abs(vel_max-vel_min))
#     if vel_norm > 1e-10:
#         vel = ((vel-(vel_max-vel_min)/2)/vel_norm)*(1-margin)

#     else:
#         vel = vel-vel_min

#     pts_min = np.min(pts,axis = 0)
#     pts_max = np.max(pts,axis = 0)
#     pts_norm = np.max(np.abs(pts_max-pts_min))

#     # pts  = ((pts-pts_min)/pts_norm)*(1-margin)+margin/2-0.5
#     pts  = ((pts-(pts_max-pts_min)/2)/pts_norm)*(1-margin)

#     pts_new= np.concatenate([pts,vel],axis=-1)
#     o_new = pts_new[:o.shape[0],:]
#     t_new = pts_new[o.shape[0]:,:]

#     return t_new, o_new, [pts_norm, (pts_max-pts_min)/2,vel_norm, (vel_max-vel_min)/2, margin]

# def rescale(pts_, factor_list):

#     vel = pts_[:,3:]
#     pts = pts_[:,:3]

#     pts_norm, pts_avr,vel_norm, vel_avr, margin = factor_list
#     pts = pts/(1-margin)*pts_norm+pts_avr
#     vel = vel/(1-margin)*vel_norm+vel_avr
#     pts_new= np.concatenate([pts,vel],axis=-1)

#     return pts_new

def norm_traj_and_objs(t, o, margin=0.40, rotation_degrees = 0, rotation_axis = np.array([0, 0, 1])):

    rotation_radians = np.radians(rotation_degrees)
    rotation_vector = rotation_radians * rotation_axis
    rotation = R.from_rotvec(rotation_vector)

    pts_ = np.concatenate([o,t])

    vel = pts_[:,3:]
    pts = pts_[:,:3]
    pts = rotation.apply(pts)

    vel_min = np.min(vel,axis = 0)
    vel_max = np.max(vel,axis = 0)
    vel_norm = np.max(np.abs(vel_max-vel_min))

    if vel_norm > 1e-10:
        vel = ((vel-vel_min)/vel_norm)*(1-margin)+margin/2-0.5 # old
        # vel = ((vel-(vel_max-vel_min)/2)/vel_norm)*(1-margin)

    else:
        vel = vel-vel_min

    pts_min = np.min(pts,axis = 0)
    pts_max = np.max(pts,axis = 0)
    pts_norm = np.max(np.abs(pts_max-pts_min))

    pts  = ((pts-pts_min)/pts_norm)*(1-margin)+margin/2-0.5 # old
    # pts  = ((pts-(pts_max-pts_min)/2)/pts_norm)*(1-margin)


    pts_new= np.concatenate([pts,vel],axis=-1)
    o_new = pts_new[:o.shape[0],:]
    t_new = pts_new[o.shape[0]:,:]

    return t_new, o_new, [pts_norm, pts_min,vel_norm, vel_min, margin,  rotation_degrees,  rotation_axis]


def rescale(pts_, factor_list):

    vel = pts_[:,3:]
    pts = pts_[:,:3]

    pts_norm, pts_min,vel_norm, vel_min, margin,  rotation_degrees,  rotation_axis = factor_list #old
    # pts_norm, pts_avr,vel_norm, vel_avr, margin = factor_list
    rotation_radians = np.radians(rotation_degrees)
    rotation_vector = -rotation_radians * rotation_axis
    rotation = R.from_rotvec(rotation_vector)

    pts = (pts+0.5-margin/2)/(1-margin)*pts_norm+pts_min # old
    if vel_norm > 1e-10:
        vel = (vel+0.5-margin/2)/(1-margin)*vel_norm+vel_min# old
    else:
        vel = vel+vel_min

    # pts = pts/(1-margin)*pts_norm+pts_avr
    # vel = vel/(1-margin)*vel_norm+vel_avr
    pts = rotation.apply(pts)

    pts_new= np.concatenate([pts,vel],axis=-1)

    return pts_new


traj_n = 40
mr = Motion_refiner(load_models=True ,traj_n = traj_n, locality_factor=True)

# original
obj_poses = np.array([[1,0,0.0,0],[0,1,0,0],[1,1,-0.5,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
obj_poses_offset = np.array([[0,0,-1,0],[0,0,-0.9,0],[0,0,0,0]])

# franka
# obj_poses = np.array([[0.1,-0.5,-0.10,0],[0.2,-0.8,0.0,0],[0.2,-0.5,-0.10,0],[0.6,-0.5,-0.60,0],[0.2,-0.5,-0.10,0],[0.2,-0.5,-0.10,0]])
# obj_poses_offset = np.array([0.0, 0.0,0,0])

# ours
# obj_poses = np.array([[0.55, -0.35, 0.04, 0.0],[0.55, -0.1, 0.07, 0.0],[0.44, 0.2, 0.07, 0.0]])

while True:
    # original
    base_wp = np.array([[0,0,0,0.3],[0.5,0.5,0,0.1],[1,1,0.3,0.0],[1.5,1.5,0.2,0.1],[0.5,1.7,0.1,0.2]])
    # demo - franka
    # base_wp = np.array([[0,0,0,0.0],[0.2,-0.25,0.2,0.1],[0.2,-0.5,0.3,0.2],[0.1,-0.75,0.2,0.2],[0.0,-0.8,0.1,0.1],[0.0,-1.0,0.0,0.0]])

    # base_wp = np.array([[ 0.40084913,  0.26196745,  0.41634452,  0.  ],
    #                     [ 0.45571071,  0.24231151,  0.44221422,  0.  ],
    #                     [ 0.51053351,  0.21376415,  0.45769837,  0.  ],
    #                     [ 0.56343371,  0.17458278,  0.46201563,  0.  ],
    #                     [ 0.61155039,  0.12356942,  0.45479456,  0.  ],
    #                     [ 0.65126449,  0.06049573,  0.43608987,  0.  ],
    #                     [ 0.67858505, -0.01357391,  0.40637881,  0.  ],
    #                     [ 0.6896528 , -0.09612258,  0.36653763,  0.  ],
    #                     [ 0.68129182, -0.18322225,  0.3178001 ,  0.  ],
    #                     [ 0.6515286 , -0.26978496,  0.2616998 ,  0.  ],
    #                     [ 0.59999996, -0.35000017,  0.19999978,  0.  ]])
    offset = np.array([0,0,0,0])
    init_pose =  np.array([0.382723920642, 0.5, 0.05,0])
    
    # base_wp = base_wp + init_pose
    # obj_poses = obj_poses+ init_pose

    obj_names = ["table", "bottle", "cup"]
    # obj_names = ["table"]

    text = input("Input:")     

    traj = interpolate_traj(base_wp, traj_n=traj_n)
    traj_, obj_poses_, factor_list = norm_traj_and_objs(traj, obj_poses)
    obj_poses_ = obj_poses_[:,:3]

    d = np2data(traj_, obj_names, obj_poses_, text)[0]
    pred, traj_in = mr.apply_interaction(model, d, text,  label=False, images=None)

    data_array = np.array([d])
    # %matplotlib qt
    show_data4D(data_array, pred=pred, color_traj=False)
    # 
    traj_new = rescale(pred[0], factor_list)