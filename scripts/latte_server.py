#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import json

import os

from latte.motion_refiner_4D import MAX_NUM_OBJS, Motion_refiner
from latte.functions import *

from latte.traj_utils import *
from latte.TF4D_mult_features import *

import rospy
import rospkg
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from lang_corr_msgs.srv import GetEnvObj, LatteDeformTraj, LatteDeformTrajResponse
from lang_corr_msgs.srv import GetFeaturesFromLanguage, GetFeaturesFromLanguageRequest

NUM_WAYPTS = 20

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
        print(model_path)
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
        
        self.mr = Motion_refiner(traj_n = NUM_WAYPTS, load_models=load_models, locality_factor=True)

        print("Loaded parameters in latte server")

    def get_env_obj(self):
        """
        Initializes the environment - object names, object poses (4D), and object bbox (2D)
        """
        self.object_poses = []
        self.object_bboxes = []
        try:
            rospy.wait_for_service("get_env_objects", 5.0)
            srv = rospy.ServiceProxy("get_env_objects", GetEnvObj)
            resp = srv()
            self.object_names = resp.object_names
            for pos in resp.object_poses.poses:
                self.object_poses.append([pos.position.x, pos.position.y, pos.position.z, 0.0]) # in 4D

            for bbox in resp.bounding_boxes:
                self.object_bboxes.append([bbox.points[0].x, bbox.points[1].x,
                                           bbox.points[0].y, bbox.points[1].y])

        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s"%e)
            rospy.logerr("Using default objects to generate corpus")
            self.object_names = ['cup', 'bottle', 'laptop']
            self.object_poses.append([0.55, -0.35, 0.04, 0.0])
            self.object_poses.append([0.55, -0.1, 0.07, 0.0])
            self.object_poses.append([0.44, 0.2, 0.07, 0.0])
            self.img = cv2.imread(self.img_file)

            self.object_bboxes.append([1150, 1238, 627, 725])
            self.object_bboxes.append([1157, 1205, 473, 580])
            self.object_bboxes.append([1009, 1208, 297, 500])
            print("Objects:", self.object_names)

        self.object_poses = np.array(self.object_poses)
        self.object_bboxes = np.array(self.object_bboxes)

    def get_img(self):
        print("TODO. Get image from camera feed.")
        self.img = cv2.imread(self.img_file)

    def get_most_similar_text(self, text):
        try:
            client_get_features = rospy.ServiceProxy("get_features_from_language", GetFeaturesFromLanguage)
            resp = client_get_features(text)
            text = resp.feature
            if resp.confidence < 0.6:
                rospy.logwarn("Confidence less than 0.6 : %f" %resp.confidence)
            return text        
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s"%e)

    def latte_deform_cb(self, req):
        rospy.loginfo("Received service request to deform trajectory with '%s'"%req.correction)
        msg_traj = req.seed_traj.data
        msg_traj = np.array(msg_traj)

        self.traj_initial = msg_traj.reshape(int(len(msg_traj)/3), 3)
        vel_arr = np.zeros((int(len(msg_traj)/3), 1))
        self.traj_initial = np.hstack((self.traj_initial, vel_arr))

        assert self.traj_initial.shape[1] == 4

        # If modified version, find closest text embeddings first
        if req.modified:
            rospy.loginfo("Requesting most similar sentence...")
            try:
                self.text = self.get_most_similar_text(req.correction)
            
            except:
                rospy.logerr("Getting most similar text failed")

        else:
            self.text = req.correction
            
        # cannot set show=True here, qt only works in main thread
        self.traj_deformed = self.modify_traj(self.mr)
        res = LatteDeformTrajResponse()

        msg_deformed_traj = self.traj_deformed
        # remove the last element (velocity component)
        msg_deformed_traj = np.delete(msg_deformed_traj.copy(), -1, axis=1)
        res.deformed_traj.data = msg_deformed_traj.reshape(-1)

        rospy.loginfo("Deformed trajectory with Latte")
        self.publish_traj(msg_deformed_traj, self.debug_traj_pub, color="red")

        return res
    
    def publish_traj(self, traj_single, traj_pub, frame_id="map", ns=None, color="blue"):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = rospy.Time.now()
        if ns is None:
            m.ns = "traj_latte"
        else:
            m.ns = ns
        m.id = 0
        m.type = m.POINTS
        m.action = m.ADD
        if color == "blue":
            m.color.r = 0
            m.color.g = 0
            m.color.b = 1.0
        elif color == "yellow":
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 0
        else:   # red
            m.color.r = 1.0
            m.color.g = 0
            m.color.b = 0
        m.color.a = 0.5
        m.lifetime = rospy.Duration(0)  # displayed forever
        m.scale.x = 0.05
        m.scale.y = 0.05
        m.scale.z = 0.05

        for w in traj_single:
            pt = Point()
            pt.x = w[0]
            pt.y = w[1]
            pt.z = w[2]

            m.points.append(pt)

        traj_pub.publish(m)

    def modify_traj(self, mr, show=False):
        if self.traj_initial is None:
            rospy.logwarn("No initial traj provided. Setting a random base trajectory")
            self.base_wp = np.array([[ 0.40084913,  0.26196745,  0.41634452,  0.  ],
                                     [ 0.45571071,  0.24231151,  0.44221422,  0.  ],
                                     [ 0.51053351,  0.21376415,  0.45769837,  0.  ],
                                     [ 0.56343371,  0.17458278,  0.46201563,  0.  ],
                                     [ 0.61155039,  0.12356942,  0.45479456,  0.  ],
                                     [ 0.65126449,  0.06049573,  0.43608987,  0.  ],
                                     [ 0.67858505, -0.01357391,  0.40637881,  0.  ],
                                     [ 0.6896528 , -0.09612258,  0.36653763,  0.  ],
                                     [ 0.68129182, -0.18322225,  0.3178001 ,  0.  ],
                                     [ 0.6515286 , -0.26978496,  0.2616998 ,  0.  ],
                                     [ 0.59999996, -0.35000017,  0.19999978,  0.  ]])
            # self.base_wp = np.array([[0,0,0,0.3],[0.2,-0.25,0,0.1],[0.2,-0.5,0.0,0.0],[0.1,-0.75,0.1,0.1],[0.0,-1.0,0.1,0.2]])

            # # add starting point
            # offset = np.array([0.0, 0.0, 0.0, 0.0])
            # self.base_wp = self.base_wp + offset
            
            # self.base_wp = self.base_wp / 1.0     # -> shouldn't make a diff?

            # init_pose =  np.array([0.382723920642, 0.5, 0.05, 0])
            # self.base_wp = self.base_wp + init_pose     # transform entire initial traj by the offset?

            self.traj_initial = self.interpolate_traj(self.base_wp, traj_n = NUM_WAYPTS).copy()
        
        else:
            self.traj_initial = self.interpolate_traj(self.traj_initial, traj_n = NUM_WAYPTS).copy()

        # Normalize trajectory and object poses (change all to a range of -1 to 1)
        traj_, obj_poses_, factor_list = self.norm_traj_and_objs(self.traj_initial, self.object_poses)

        obj_poses_ = obj_poses_[:,:3]
        
        # Store all data into a dictionary form
        d = np2data(traj_, self.object_names, obj_poses_, self.text)[0]

        images = None
        if self.use_images:
            images = self.get_img_crops()
            # for img in images:
            #     cv2.imshow('cropped', img)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

        # Apply deformation
        # traj_in is the decoded trajectory input
        # pred is the decoded trajectory output
        # dec_only runs model.predict()
        # if dec_only is False, generate() decoded input?
        pred, traj_in = mr.apply_interaction(self.model, d, self.text, label=False, images=images, dec_only=False)

        # %matplotlib qt
        if show:
            data_array = np.array([d])
            show_data4D(data_array, pred=pred, color_traj=False)

        # Rescale since prediction is done in normalized form
        # factor_list stores the normalization params
        traj_new = self.rescale(pred[0], factor_list)

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
        """
        Normalizes trajectory and object poses to a range of -1 to 1

        Args:
            t (array): Trajectory with 4D waypoints (x,y,z,vel)
            o (array): Object poses (4D)
            margin (float, optional): Not sure why margin needed. Defaults to 0.45.

        Returns:
            t_new (array): normalized trajectory
            o_new (array): normalized object poses
            debug_info (array): pts_norm, pts_range/2, vel_norm, vel_range/2, margin
        """
        pts_ = np.concatenate([o,t])

        vel = pts_[:,3:]
        pts = pts_[:,:3]

        vel_min = np.min(vel,axis = 0)
        vel_max = np.max(vel,axis = 0)
        vel_norm = np.max(np.abs(vel_max-vel_min))
        if vel_norm > 1e-10:
            vel = ((vel-(vel_max-vel_min)/2)/vel_norm)*(1 - margin)

        else:
            vel = vel-vel_min

        pts_min = np.min(pts,axis = 0)
        pts_max = np.max(pts,axis = 0)
        pts_norm = np.max(np.abs(pts_max-pts_min))

        # pts  = ((pts-pts_min)/pts_norm)*(1-margin)+margin/2-0.5
        pts = ((pts - (pts_max - pts_min)/2) / pts_norm) * (1 - margin)

        pts_new = np.concatenate([pts,vel],axis=-1)
        o_new = pts_new[:o.shape[0],:]
        t_new = pts_new[o.shape[0]:,:]

        return t_new, o_new, [pts_norm, (pts_max-pts_min)/2, vel_norm, (vel_max-vel_min)/2, margin]
            
def print_help():
    print("""
    -------------------Keyboard Commands------------------------
    q: Quit 
    h: Print help
    r: Reset to initial seed trajectory
    m: Toggle between Latte and Modified-Latte

    Otherwise, key in command.
    -----------------------------------------------------------
    
    """)
if __name__ == "__main__":
    rospy.init_node('latte_server', anonymous=True)

    ls = LatteServer()
    rospy.loginfo("Initialized latte server")
    # rospy.spin()

    # this doesn't affect server, only the test script below
    mod_latte = True
    
    while not rospy.is_shutdown():
        cmd = input("Key in command:")

        if cmd == "q":
            exit()
        elif cmd == "r":
            ls.traj_initial = None
            print("Resetting trajectory")
            # reset base trajectory
            continue
        elif cmd == "h":
            print_help()
            continue
        elif cmd == "m":
            mod_latte = not mod_latte
            print("Using mod_latte:", mod_latte)
            continue
        elif cmd == "":
            continue
        
        print("Received command:", cmd)
        if mod_latte:
            ls.text = ls.get_most_similar_text(cmd)
        else:
            ls.text = cmd

        new_traj = ls.modify_traj(ls.mr, show=True)

        # update traj
        ls.traj_initial = new_traj.copy()
