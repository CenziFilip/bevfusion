import numpy as np

# vehicle extent
veh_radius = {
    "vehicle.nissan.micra": 2.038,
    "vehicle.audi.a2": 2.057,
    "vehicle.lttm.bus01_c5": 5.328,
    "vehicle.mercedes.coupe_2020": 2.542,
    "vehicle.audi.tt": 2.316,
    "vehicle.ford.ambulance": 3.393,
    "vehicle.harley-davidson.low_rider": 1.178,
    "vehicle.bmw.grandtourer": 2.564,
    "vehicle.micro.microlino": 1.329,
    "vehicle.carlamotors.firetruck": 4.474,
    "vehicle.carlamotors.carlacola": 2.915,
    "vehicle.lttm.truck04_c7": 4.001,
    "vehicle.lincoln.mkz_2020": 2.688,
    "vehicle.chevrolet.impala": 2.865,
    "vehicle.ford.mustang": 2.542,
    "vehicle.citroen.c3": 2.198,
    "vehicle.nissan.patrol": 2.497,
    "vehicle.dodge.charger_police": 2.688,
    "vehicle.mini.cooper_s": 2.143,
    "vehicle.jeep.wrangler_rubicon": 2.155,
    "vehicle.dodge.charger_2020": 2.714,
    "vehicle.mercedes.coupe": 2.734,
    "vehicle.seat.leon": 2.285,
    "vehicle.lttm.bus01_c7": 5.328,
    "vehicle.lttm.truck02_c52": 4.574,
    "vehicle.lttm.truck02_c23": 4.574,
    "vehicle.toyota.prius": 2.470,
    "vehicle.lttm.truck04_c4": 4.001,
    "vehicle.yamaha.yzf": 1.105,
    "vehicle.kawasaki.ninja": 1.017,
    "vehicle.tesla.model3": 2.629,
    "vehicle.lttm.bus01_c3": 5.328,
    "vehicle.bh.crossbike": 0.744,
    "vehicle.gazelle.omafiets": 0.918,
    "vehicle.lttm.truck04_c8": 4.001,
    "vehicle.mercedes.sprinter": 3.120,
    "vehicle.diamondback.century": 0.821,
    "vehicle.tesla.cybertruck": 3.357,
    "vehicle.lttm.truck02_c21": 4.574,
    "vehicle.volkswagen.t2": 2.468,
    "vehicle.audi.etron": 2.632,
    "vehicle.lttm.bus01_c2": 5.328,
    "vehicle.lincoln.mkz_2017": 2.672,
    "vehicle.dodge.charger_police_2020": 2.821,
    "vehicle.lttm.bus02_c33": 5.879,
    "vehicle.vespa.zx125": 0.902,
    "vehicle.lttm.truck04_c9": 4.001,
    "vehicle.mini.cooper_s_2021": 2.506,
    "vehicle.nissan.patrol_2021": 2.983,
    "vehicle.lttm.bus01_c1": 5.328,
    "vehicle.lttm.bus02_c11": 5.879,
    "vehicle.lttm.truck04_c2": 4.001,
    "vehicle.lttm.train01": 14.932,
    "vehicle.lttm.bus02_c23": 5.879,
    "vehicle.lttm.truck04_c1": 4.001,
    "vehicle.lttm.train02": 12.671,
    "vehicle.lttm.truck02_c11": 4.574,
    "vehicle.lttm.bus01_c4": 5.328,
    "vehicle.lttm.truck02_c51": 4.574,
    "vehicle.lttm.truck04_c3": 4.001,
    "vehicle.lttm.truck04_c5": 4.001,
    "vehicle.lttm.truck02_c12": 4.574,
    "vehicle.lttm.truck02_c13": 4.574,
    "vehicle.lttm.truck02_c22": 4.574,
    "vehicle.lttm.truck02_c31": 4.574,
    "vehicle.lttm.truck02_c32": 4.574,
    "vehicle.lttm.truck02_c33": 4.574,
    "vehicle.lttm.truck02_c53": 4.574,
    "vehicle.lttm.bus01_c6": 5.328,
    "vehicle.lttm.truck04_c6": 4.001,
    "vehicle.lttm.bus01_c8": 5.328,
    "vehicle.lttm.bus02_c12": 5.879,
    "vehicle.lttm.bus02_c13": 5.879,
    "vehicle.lttm.bus02_c21": 5.879,
    "vehicle.lttm.bus02_c22": 5.879,
    "vehicle.lttm.bus02_c31": 5.879,
    "vehicle.lttm.bus02_c32": 5.879,
    "vehicle.lttm.truck02_c41": 4.574,
    "vehicle.lttm.truck02_c42": 4.574,
    "vehicle.lttm.truck02_c43": 4.574
}

# bbox volume < .5
mbikes = [
            'vehicle.harley-davidson.low_rider',
            'vehicle.yamaha.yzf',
            'vehicle.kawasaki.ninja',
            'vehicle.vespa.zx125'
          ]
# bbox volume < .5
bikes =  [
            'vehicle.bh.crossbike',
            'vehicle.gazelle.omafiets',
            'vehicle.diamondback.century'
         ]
two_wheels = bikes+mbikes

# produces an ordered set of corners
# which, when plotted, create a parallelogram
corners_xyz_coeffs = np.array([[-1,-1,-1],
                               [ 1,-1,-1],
                               [ 1, 1,-1],
                               [-1, 1,-1],
                               [-1,-1,-1], # front face
                               [-1,-1, 1],
                               [-1, 1, 1],
                               [-1, 1,-1], # left face
                               [-1, 1, 1],
                               [ 1, 1, 1],
                               [ 1, 1,-1], # top face
                               [ 1, 1, 1],
                               [ 1,-1, 1],
                               [ 1,-1,-1], # right face
                               [ 1,-1, 1],
                               [-1,-1, 1]], dtype=float)# bottom & back face

def make_rotation_matrix(pitch=0., yaw=0., roll=0., radians=False):
    if not radians:
        b, a, g = np.radians([pitch, yaw, roll])
    else:
        b, a, g = pitch, yaw, roll
        
    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    sg, cg = np.sin(g), np.cos(g)
        
    # as reported in: https://en.wikipedia.org/wiki/Rotation_matrix
    rot = np.array([[ca*cb, ca*sb*sg-sa*cg, ca*sb*cg-sa*sg],
                    [sa*cb, sa*sb*sg+ca*cg, sa*sb*cg-ca*sg],
                    [-sb, cb*sg, cb*cg]])
        
    return rot

# note that it expects points and camera intrinsics with the left-handed z-up convention
# meaning that the points height (and not the distance from the sensor) is encoded in the z position
# on the other hand, the y-direction corresponds to the left-right shift and the x is the distance
# returns the raw projection, so that we can feed it into a z-buffer
def project_bounding_box(center, extent, rotation, ego_position, ego_rotation, camera_position, camera_rotation, camera_intrinsics):
    ego_inverse = np.linalg.inv(make_rotation_matrix(**ego_rotation))
    camera_inverse = np.linalg.inv(make_rotation_matrix(**camera_rotation))
    
    corners = corners_xyz_coeffs*extent
    corners = np.matmul(make_rotation_matrix(**rotation), corners.T).T
    corners = center+corners-ego_position
    corners = np.matmul(ego_inverse, corners.T).T - camera_position
    corners = np.matmul(camera_inverse, corners.T)
    
    proj = np.matmul(camera_intrinsics, corners).T
    return proj
    
def project_boxes(bboxes, wdata, sdata, id_map=None, sensor="cam"):
    if sensor=="cam":
        s_tr, s_rot, K = np.array(sdata['translation']), sdata['rotation'], np.array(sdata['camera_intrinsic'])
        im_w, im_h = 2*K[0,0], 2*K[1,0]
        w_tr, w_rot = np.array([wdata['Location'][k] for k in wdata['Location']]), wdata['Rotation']
        
        to_render = []
        for bbox in bboxes:
            ext = np.array([bbox['extent'][k] for k in bbox['extent']])
            loc = np.array([bbox['location'][k] for k in bbox['location']])
            rot = bbox['rotation']
            lab = id_map[bbox['label']] if id_map is not None else bbox['label']
            
            proj = project_bounding_box(loc, ext, rot, w_tr, w_rot, s_tr, s_rot, K)
            # check if at least one corner is in front of the camera
            if np.any(proj[:,2]>0):
                min_z = np.min(proj[:,2])
                proj = proj[proj[:,2]>0,:]
                proj /= proj[:,2:3]
                # check if the projection is inside the 
                if np.any(np.logical_or(np.logical_and(proj[:,0]>=0,proj[:,0]<im_w), np.logical_and(proj[:,1]>=0, proj[:,1]<im_h))):
                    to_render.append((min_z, proj.copy(), lab))
        to_render.sort(key=lambda e: e[0])
        to_render = [(e[1], e[2]) for e in to_render]
        return to_render, im_w, im_h

    elif sensor=="lidar":
        s_tr, s_rot = np.array(sdata['translation']), sdata['rotation']
        w_tr, w_rot = np.array([wdata['Location'][k] for k in wdata['Location']]), wdata['Rotation']

        camera_position = s_tr
        camera_rotation = s_rot
        ego_position = w_tr
        ego_rotation = w_rot
        ego_inverse = np.linalg.inv(make_rotation_matrix(**ego_rotation))
        camera_inverse = np.linalg.inv(make_rotation_matrix(**camera_rotation))
        
        to_render = []
        for bbox in bboxes:
            ext = np.array([bbox['extent'][k]*2 for k in bbox['extent']])
            loc = np.array([bbox['location'][k] for k in bbox['location']])
            rot = bbox['rotation']
            lab = id_map[bbox['label']] if id_map is not None else bbox['label']
            instance = np.uint32(bbox['instance_id'])
            bp_id = bbox['bp_id']
            if (bbox['label']==104 or bbox['label']==105):
                loc[1] = loc[1]/2
                if abs(ext[1])<1e-9:
                    ext[1] = 2*veh_radius[bp_id]
                if abs(ext[0])<1e-9:
                    ext[0] = 2*veh_radius[bp_id]
                tmp = ext[1]
                ext[1] = ext[2]
                ext[2] = tmp
                try:
                    loc[2] = loc[2]+veh_radius[bp_id]/2
                    ext[2] = ext[2]
                except KeyError:
                    pass

            proj_loc = np.matmul(make_rotation_matrix(**ego_rotation), loc.T).T
            proj_loc = loc-ego_position
            proj_loc = np.matmul(ego_inverse, proj_loc.T).T - camera_position
            proj_loc = np.matmul(camera_inverse, proj_loc.T)

            bbox_rot = {angle:rot[angle]-(ego_rotation[angle]+camera_rotation.get(angle,0.0)) for angle in ["pitch", "yaw", "roll"]}

            to_render.append((proj_loc[2], proj_loc, make_rotation_matrix(**bbox_rot), ext, lab, instance, bp_id, bbox_rot["yaw"]))

        to_render.sort(key=lambda e: e[0])
        to_render = [(e[1], e[2], e[3], e[4], e[5], e[6], e[7]) for e in to_render]

        return to_render

    else:
        raise NotImplementedError("Only 'cam' or 'lidar' sensor are currently supported")
