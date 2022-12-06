#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import json

import os

from config import *
from motion_refiner_4D import MAX_NUM_OBJS, Motion_refiner
from functions import *

from traj_utils import *
from TF4D_mult_features import *

import rospy
import rospkg
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from lang_corr_msgs.srv import GetEnvObj, LatteDeformTraj, LatteDeformTrajResponse


class LatteServer():
    def __init__(self):
        self.debug_traj_pub = rospy.Publisher("latte_trajectories", Marker, queue_size=5)
        self.latte_deform_srv = rospy.Service('latte_deformation', LatteDeformTraj, self.latte_deform_cb)
        self.img = None
        self.load_parameters()
        self.get_env_obj()
        if self.img is None:
            self.get_img()

    def load_parameters(self):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('latte')
        
        model_path = rospy.get_param("setup/model")
        model_name = rospy.get_param("setup/model_name")
        model_file = model_path + model_name
        
        base_path = pkg_path + rospy.get_param("setup/user_trajs_path")
        chomp_trajs_path = pkg_path + rospy.get_param("setup/chomp_trajs_path")
        self.img_file = pkg_path + rospy.get_param("setup/img_file")

        
        self.use_images = rospy.get_param("setup/use_images")
        self.traj_initial = None

        load_models = rospy.get_param("setup/load_models")
        if load_models:
            print("loading model...")
            self.model = load_model(model_file, delimiter="-")
            compile(self.model)
        
        self.mr = Motion_refiner(load_models=load_models, locality_factor=True)

    def get_env_obj(self):
        """
        Initializes the environment - object names, object poses (4D), and object bbox (2D)
        """
        self.object_poses = []
        self.object_bboxes = []
        try:
            srv = rospy.ServiceProxy("get_env_objects", GetEnvObj)
            resp = srv()
            self.object_names = resp.object_names
            for pos in resp.object_poses.poses:
                self.object_poses.append([pos.position.x, pos.position.y, pos.position.z, 0.0]) # in 4D

            for bbox in resp.bounding_boxes:
                self.object_bboxes.append([bbox.points[0].x, bbox.points[1].x,
                                           bbox.points[0].y, bbox.points[1].y])

        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s"%e)
            rospy.logerr("Using default objects to generate corpus")
            self.object_names = ['cup', 'bottle', 'laptop']
            self.object_poses.append([0.1, 0.5, 0.0, 0.0])
            self.object_poses.append([0.3, 0.2, 0.0, 0.0])
            self.object_poses.append([0.4, -0.2, 0.0, 0.0])
            self.img = cv2.imread(self.img_file)

            self.object_bboxes.append([1150, 1238, 627, 725])
            self.object_bboxes.append([1157, 1205, 473, 580])
            self.object_bboxes.append([1009, 1208, 297, 500])

        self.object_poses = np.array(self.object_poses)
        self.object_bboxes = np.array(self.object_bboxes)
        print(self.object_bboxes)

    def get_img(self):
        print("TODO. Get image from camera feed.")
        self.img = cv2.imread(self.img_file)


    def latte_deform_cb(self, req):
        rospy.loginfo("Received service request to deform trajectory with '%s'"%req.correction)
        self.traj_initial = req.seed_traj
        self.text = req.correction

        self.traj_deformed = self.modify_traj(self.mr)
        res = LatteDeformTrajResponse()

        res.deformed_traj = self.traj_deformed
        rospy.loginfo("Deformed trajectory with Latte")
        return res

    def modify_traj(self, mr, show=False):
        if self.traj_initial is None:
            rospy.logwarn("No initial traj provided. Setting a random base trajectory")
            self.base_wp = np.array([[0,0,0,0.3],[0.2,-0.25,0,0.1],[0.2,-0.5,0.0,0.0],[0.1,-0.75,0.1,0.1],[0.0,-1.0,0.1,0.2]])

            # add starting point
            offset = np.array([0.0, 0.0, 0.0, 0.0])
            self.base_wp = self.base_wp + offset
            
            self.base_wp = self.base_wp / 1.0     # -> shouldn't make a diff?

            init_pose =  np.array([0.382723920642, 0.5, 0.05, 0])
            self.base_wp = self.base_wp + init_pose     # transform entire initial traj by the offset?

            self.traj_initial = self.interpolate_traj(self.base_wp).copy()


        # TODO: why need to normalize?
        traj_, obj_poses_, factor_list = self.norm_traj_and_objs(self.traj_initial, self.object_poses)

        obj_poses_ = obj_poses_[:,:3]

        d = np2data(traj_, self.object_names, self.object_poses, self.text)[0]

        images = None
        if self.use_images:
            images = self.get_img_crops()
            # for img in images:
            #     cv2.imshow('cropped', img)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

        # TODO: what is the diff btwn pred and traj_in
        pred, traj_in = mr.apply_interaction(self.model, d, self.text, label=False, images=images)

        # %matplotlib qt
        if show:
            data_array = np.array([d])
            show_data4D(data_array, pred=pred, color_traj=False)

        # TODO: why need to rescale?
        traj_new = self.rescale(pred[0], factor_list)

        # publish_simple_traj(traj_new, self.object_poses + self.obj_poses_offset,new_traj_pub)
        # publish_simple_traj(traj,obj_poses+obj_poses_offset,traj_pub)

        return traj_new
    
    def interpolate_traj(self, wps, traj_n=40, offset=[0,0,0,0]):
        #create spline function
        f, u = interpolate.splprep([wps[:,0],wps[:,1],wps[:,2]], s=0)
        xint,yint,zint= interpolate.splev(np.linspace(0, 1, traj_n), f)

        tck,u = interpolate.splprep([np.linspace(0,1,len(wps[:,3])), wps[:,3]])
        velint_x, velint = interpolate.splev(np.linspace(0, 1, traj_n), tck)

        traj = np.stack([xint,yint,zint,velint],axis=1)+offset
        return traj
    
    def get_img_crops(self):
        """
        Gets images for each object based on the bounding boxes.
        Used to calculate the object CLIP embeddings

        Returns:
            images: object imgaes
        """
        images = []
        for b in self.object_bboxes:
            xmin, xmax, ymin, ymax = b
            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)
            images.append(self.img[ymin:ymax, xmin:xmax, :])
        return images

    def rescale(self, pts_, factor_list):
        vel = pts_[:,3:]
        pts = pts_[:,:3]

        pts_norm, pts_avr,vel_norm, vel_avr, margin = factor_list
        pts = pts/(1-margin)*pts_norm+pts_avr
        vel = vel/(1-margin)*vel_norm+vel_avr
        pts_new= np.concatenate([pts,vel],axis=-1)

        return pts_new

    def norm_traj_and_objs(self, t, o, margin=0.45):
        pts_ = np.concatenate([o,t])

        vel = pts_[:,3:]
        pts = pts_[:,:3]

        vel_min = np.min(vel,axis = 0)
        vel_max = np.max(vel,axis = 0)
        vel_norm = np.max(np.abs(vel_max-vel_min))
        if vel_norm > 1e-10:
            vel = ((vel-(vel_max-vel_min)/2)/vel_norm)*(1-margin)

        else:
            vel = vel-vel_min

        pts_min = np.min(pts,axis = 0)
        pts_max = np.max(pts,axis = 0)
        pts_norm = np.max(np.abs(pts_max-pts_min))

        # pts  = ((pts-pts_min)/pts_norm)*(1-margin)+margin/2-0.5
        pts  = ((pts-(pts_max-pts_min)/2)/pts_norm)*(1-margin)

        pts_new= np.concatenate([pts,vel],axis=-1)
        o_new = pts_new[:o.shape[0],:]
        t_new = pts_new[o.shape[0]:,:]

        return t_new, o_new, [pts_norm, (pts_max-pts_min)/2,vel_norm, (vel_max-vel_min)/2, margin]
            

if __name__ == "__main__":
    rospy.init_node('latte_trial', anonymous=True)

    ls = LatteServer()

    while not rospy.is_shutdown():
        ls.text = input("Key in command:")

        if ls.text == "q":
            exit()

        new_traj = ls.modify_traj(ls.mr, show=True)

        # update traj
        ls.traj_initial = new_traj.copy()
