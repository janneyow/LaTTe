import numpy as np
from tqdm import tqdm
import json
import random
from scipy import interpolate
import matplotlib.pyplot as plt

from labels_generator import Label_generator

OBJ_LIB_FILE = obj_library_file = "/home/janne/ros_ws/latte/src/LaTTe/imagenet1000_clsidx_to_labels.txt"

MAX_NUM_OBJS = 6
MIN_NUM_OBJS = 2

class DataGenerator():
    def __init__(self, change_types:dict, obj_lib_file=OBJ_LIB_FILE):
        
        self.change_types = change_types
        self.obj_library = {}
        self.margin = 0.4   # used in trajectory generation

        with open(obj_lib_file) as f:
            self.obj_library = json.load(f)
    
    def get_objs(self, num_objs:int):

        obj_classes = random.sample(self.obj_library.keys(),num_objs)
        obj_names = [random.choice(self.obj_library[o]) for o in obj_classes]

        obj_pt = np.random.random([num_objs,3])*(1-self.margin)+self.margin/2-0.5
        objs_dict  = {}
        for x,y,z,name,c in zip(obj_pt[:,0],obj_pt[:,1],obj_pt[:,2],obj_names, obj_classes):
            objs_dict[name] = {"value":{"obj_p":[x,y,z]}, "class":c}

        return obj_names,obj_classes, obj_pt, objs_dict
    
    def generate(self, maps=32, labels_per_map=4, N=100, n_int=10, plot=False, output_forces=False) -> dict:
        """Generates data

        Args:
            maps (int, optional): Size of dataset. Defaults to 32.
            labels_per_map (int, optional): Number of labels per map, used to deform trajectory. Defaults to 4.
            N ([int], optional): Used for trajectory generation. Initial number of sampled waypoints? Must be larger than 3. Defaults to 100.
            n_int ([int], optional): Used for trajectory generation. To examine difference vs n_wp. Defaults to 10.
            plot (bool, optional): _description_. Defaults to False.
            output_forces (bool, optional): _description_. Defaults to False.

        Returns:
            data (dict): Dictionary of data
        """
        data = []

        for mi in tqdm(range(maps)):
            num_objs = random.randint(MIN_NUM_OBJS, MAX_NUM_OBJS)
            obj_names,obj_classes, obj_pt, objs_dict = self.get_objs(num_objs)

            # Generate trajectory
            sample_done = True
            while(sample_done):
                try:
                    n_int_ = random.choice(range(n_int[0], n_int[1])) if not n_int is int else n_int
                    N_ = random.choice(range(N[0], N[1])) if not N is int else N
                    assert N_ > 3, ("N must be larger than 3")

                    pts = self.generate_traj(n_wp=40, N=N_, n_int=n_int_, margin=self.margin, show=False)
                    sample_done = False
                except Exception as e:
                    print("Error generating trajectory:", e)
                    # pass

            # Generate label for trajectory
            lg = Label_generator(objs_dict)
            lg_ct = random.choices(list(self.change_types.keys()), weights=list(self.change_types.values()))
            lg.generate_labels(lg_ct, shuffle=True)
            # print("Change type:", lg_ct, "  len = ", len(lg.labels))

            # Generate output trajectory
            for i, (text, map_cost_f) in enumerate(lg.sample_labels(labels_per_map)):
                print("text:", text)
                map_cost_f_list = [map_cost_f]

                #TODO: how this locality factor is calculated?
                locality_factor = np.random.random()*0.6 + 0.3

                # # Not used yet - for force changes
                # if output_forces:
                #     F_ext = np.zeros_like(pts)
                #     w = -0.02
                #     args = {"max_r": locality_factor}
                #     for func in map_cost_f_list:
                #         F_ext += func(pts,args)*w
                #     # pts_new = pts.copy()

                # Modifies the trajectory acording to the forces
                # TODO: locality factor is just randomly generated?
                pts_new = self.apply_force(pts, map_cost_f_list, locality_factor=locality_factor)[0][0] 

                # Store data as a dictionary
                obj = self.find_obj_in_text(obj_names,text)
                d = {"input_traj": pts,
                    "output_traj": pts_new,
                    "text": text,
                    "obj_names": obj_names,
                    "obj_classes": obj_classes,
                    "obj_pt": obj_pt,
                    "obj_in_text": obj,
                    "change_type": lg_ct[0],
                    "map_id": mi,
                    "locality_factor": locality_factor      # this is just randomly generated!
                    }
                data.append(d)

                # -- Plot one set of data -- 
                # show input, output traj and text
                fig = plt.figure(figsize=(8,13))
                ax = fig.add_subplot(1,1,1, projection='3d')
                # ax2 = fig.add_subplot(7,1,7)
                # plot initial traj
                ax.plot(pts[:,0], pts[:,1], pts[:,2], color="blue") 
                ax.plot(pts_new[:,0], pts_new[:,1], pts_new[:,2], color="red")
                # print(obj_names, obj_pt)
                for idx, pt in enumerate(obj_pt):
                    ax.scatter(pt[0], pt[1], pt[2], color="black")
                    ax.text(pt[0], pt[1], pt[2], obj_names[idx])
                
                # plots start and end point
                ax.scatter(pts[0][0], pts[0][1], pts[0][2], color="blue")
                ax.scatter(pts[-1][0], pts[-1][1], pts[-1][2], color="red")

                plt.show()
        
        return data
    
    def generate_traj(self, n_wp=20, N=100, n_int=10, margin=0.4, show=False):
        min_vel = 0.01

        R = (np.random.rand(N)*6).astype("int") #Randomly intializing the steps
        R_vel = (np.random.rand(N)*6).astype("int") #Randomly intializing the steps

        x = np.zeros(N) 
        y = np.zeros(N)
        z = np.zeros(N)
        vel = np.zeros(N)

        x[ R==0 ] = -1; x[ R==1 ] = 1 #assigning the axis for each variable to use
        y[ R==2 ] = -1; y[ R==3 ] = 1
        z[ R==4 ] = -1; z[ R==5 ] = 1
        vel[ R_vel==0 ] = -1; vel[ R_vel==1 ] = 1

        x = np.cumsum(x) #The cumsum() function is used to get cumulative sum over a DataFrame or Series axis i.e. it sums the steps across for eachaxis of the plane.
        y = np.cumsum(y)
        z = np.cumsum(z)
        vel = np.cumsum(vel)

        if show:
            fig = plt.figure(figsize=(8,13))
            ax = fig.add_subplot(1,1,1, projection='3d')
            ax2 = fig.add_subplot(7,1,7)
            # plots the vertices for spline generation
            ax.plot(x, y, z, alpha=0.5) #alpha sets the darkness of the path.
            
            # plots final point
            ax.scatter(x[-1],y[-1],z[-1], color="red")

        #create spline function
        f, u = interpolate.splprep([x, y, z], s=0)
        xint, yint, zint = interpolate.splev(np.linspace(0, 1, n_int), f)

        tck,u = interpolate.splprep([np.linspace(0,1,len(vel)), vel])
        velint_x, velint = interpolate.splev(np.linspace(0, 1, n_int), tck)


        if show:
            ax.plot(xint, yint, zint, alpha=0.5) 
            ax2.plot(np.linspace(0,1,len(velint)),velint)


        #create spline function
        f, u = interpolate.splprep([xint, yint, zint], s=0)
        xint, yint, zint = interpolate.splev(np.linspace(0, 1, n_wp), f)

        tck, u = interpolate.splprep([np.linspace(0,1,len(velint)), velint])
        velint_x, velint = interpolate.splev(np.linspace(0, 1, n_wp), tck)

        if show:
            ax.plot(xint, yint, zint, alpha=0.9) 
            ax2.plot(np.linspace(0,1,len(vel)),vel)
            ax2.plot(np.linspace(0,1,len(velint)),velint,color="red")
            plt.show()

        pts = np.stack([xint,yint,zint],axis=1)

        velint = np.expand_dims(velint,axis=-1)

        vel_min = np.min(velint,axis = 0)
        vel_max = np.max(velint,axis = 0)
        vel_norm = np.max(np.abs(vel_max-vel_min))
        vel = ((velint-vel_min)/vel_norm)*(1-margin)+margin/2-0.5

        pts_min = np.min(pts,axis = 0)
        pts_max = np.max(pts,axis = 0)
        norm = np.max(np.abs(pts_max-pts_min))
        pts  = ((pts-pts_min)/norm)*(1-margin)+margin/2-0.5

        if show:
            ax = plt.subplot(1,1,1, projection='3d')
            ax.plot(pts[:,0],pts[:,1],pts[:,2], alpha=0.9) 
            # ax.scatter(xint[-1],yint[-1],zint[-1])
            plt.show()

        return np.concatenate([pts,vel],axis = 1)

    def apply_force(self, pts_raw, map_cost_f, att_points=[], locality_factor=1.0):
        init_vel = pts_raw[-1:]

        pts = pts_raw.copy()
        wps = pts[:,:3]
        init_wps = wps.copy() # first 3 dim     
        
        # -- Getting Mean Distance --
        # Calculating distance between each waypoint
        wp_diff = wps[1:] - wps[:-1]  
        init_wp_dist = np.expand_dims(np.linalg.norm(wp_diff, axis=1), -1)
        
        # Calculating change? 
        # wp_diff_2: Calculates distance between w1 and w3 and finds the center
        # c: Calculates difference between wp_diff_2 and wp_diff
        wp_diff_2 = (wps[2:] - wps[:-2])/2
        c = wp_diff_2 - wp_diff[:-1]
        init_c_dist = np.expand_dims(np.linalg.norm(c, axis=1), axis=-1)

        # init_wp_ang_cos = np.array([np.dot(wp_dist[i-1,:]/init_wp_dist[i-1], wp_dist[i,:]/init_wp_dist[i]) for i in range(1,wp_diff.shape[0])])
        # init_wp_ang = np.arccos(init_wp_ang_cos)

        mean_dist = np.average(init_wp_dist)
        # print(mean_dist)
        
        k = -0.1
        k_ang = 0.5
        w = -0.02
        w_self = -0.05
        step = 1
        pts_H = []

        # Applying force. Magnitude of change affected by range of for loop??
        # TODO: what is the point of this for loop? affects the points?
        for i in range(1000):
        # for i in range(2):
            # print("--------------%d----------------------"%i)
            wps = pts[:,:3]
            wp_diff = wps[1:] - wps[:-1]
            wp_dist = np.expand_dims(np.linalg.norm(wp_diff, axis=1),-1)
            
            # getting vector (direction of motion)
            # dir_wp = wp_diff/np.expand_dims(np.linalg.norm(wp_dist,axis=1),-1)
            dir_wp = wp_diff / wp_dist

            # getting force at each waypoint?
            f_wp = (wp_dist - init_wp_dist) * k * dir_wp

            null_wp = [[0,0,0]]
            F_wp_l = np.concatenate([null_wp, f_wp], axis=0)      # adds null_wp at the start
            F_wp_r = np.concatenate([-f_wp, null_wp], axis=0)     # appends null_wp to the end

            F_wp = F_wp_l + F_wp_r
            F_wp = np.concatenate([F_wp, np.zeros(F_wp[:,-1:].shape)], axis=-1) # adds the speed component

            wp_diff_2 = (wps[2:] - wps[:-2])/2
            c = wp_diff_2 - wp_diff[:-1]
            c_dist = np.expand_dims(np.linalg.norm(c,axis=1),-1)
            
            c_dir = np.divide(c, c_dist, out = np.zeros_like(c), where = c_dist!=0)
            c_delta = (c_dist - init_c_dist)
            
            F_ang = c_delta * k_ang * c_dir
            F_ang = np.concatenate([null_wp, F_ang, null_wp],axis=0)    # no force at start and end waypoints
            # print("wp_diff shaoe:", wp_diff.shape)
            # print("wp_dist shape:", wp_dist.shape)
            # print("c shape:", c.shape)
            # print("c_delta shape:", c_delta.shape)
            # print("c_dir shape:", c_dir.shape)
            # print("F_ang shape:", F_ang.shape)
            F_ang = np.concatenate([F_ang, np.zeros(F_ang[:,-1:].shape)], axis=-1) # adds the speed component

            F_ext = np.zeros_like(pts_raw)
            # print("F_ext shape:", F_ext.shape)

            args = {"max_r": locality_factor}
            for func in map_cost_f:
                # print("locality factor:", args["max_r"])
                # if distance change, `repel` function used
                # gets distance and direction vector btwn each waypt and object pose
                # f = 0.5, -0.5 or 0
                # F = dir * f * w * base_w
                # TODO: check why multiply by w again here? - without this multiplication, modified trajectory is weird
                F_ext += func(pts, args) * w
            
            self_diff = wps - init_wps
            self_dist = np.expand_dims(np.linalg.norm(self_diff,axis=1), -1)
            dir_self = np.divide(self_diff, self_dist, out=np.zeros_like(self_diff), where=self_dist!=0)
            F_self = self_dist*dir_self*w_self
            F_self = np.concatenate([F_self, np.zeros(F_self[:,-1:].shape)], axis=-1) # adds the speed component

            delta = (F_ext + F_wp + F_self + F_ang) * mean_dist
            delta[0] = [0]*pts_raw.shape[-1]    # forces first waypoint to always be the same
            # delta[-1] = [0]*pts_raw.shape[-1]     # forces last waypoint to be the same
            pts = pts + delta * step
            # if i % 30 == 0:
            #     pts_H.append(pts)
        pts_H.append(pts)
        return pts_H, F_wp
    
    def find_obj_in_text(self, obj_names,text):
        obj = ""
        for o in obj_names:
            if o in text:
                return o
        return obj

if __name__ == "__main__":
    obj_library_file = "/home/janne/ros_ws/latte/src/LaTTe/imagenet1000_clsidx_to_labels.txt"

    # Note weights in change_types (1, 1, 1) in this case affects how likely the change type is selected
    # change_types = {'dist':1, 'cartesian':1, 'speed':1}
    change_types = {'cartesian': 1}
    dg = DataGenerator(change_types, obj_library_file)
    data = dg.generate(2, 4, [10, 20], [5, 8], plot=True)
    # print(len(data))