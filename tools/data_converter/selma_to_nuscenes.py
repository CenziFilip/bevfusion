import numpy as np
import os
import json
import argparse
import glob
import tqdm
from plyfile import PlyData, PlyElement
import shutil
import math
from transforms3d.euler import euler2quat, quat2euler
from utils.bbox import project_boxes
from nuscenes.utils import splits
from random import shuffle

wanted_sensors = ["LIDAR_TOP", "CAM_LEFT", "CAM_RIGHT", "CAM_BACK", "CAM_FRONT"] #"CAM_DESK", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_FRONT"]

def euler_to_quaternion(r):
    # get r key from the dictionary
    r_key = list(r.keys())
    yaw, pitch, roll = 0, 0, 0
    for r_k in r_key:
        if r_k == 'yaw':
            yaw = math.radians(r['yaw'])
        elif r_k == 'pitch':
            pitch = math.radians(r['pitch'])
        elif r_k == 'roll':
            roll = math.radians(r['roll'])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qw, qx, qy, qz]

def euler_to_quaternion_v2(r):
    # get r key from the dictionary
    r_key = list(r.keys())
    yaw, pitch, roll = 0, 0, 0
    for r_k in r_key:
        if r_k == 'yaw':
            yaw = math.radians(r['yaw'])
        elif r_k == 'pitch':
            pitch = math.radians(r['pitch'])
        elif r_k == 'roll':
            roll = math.radians(r['roll'])
    quat = euler2quat(roll, pitch, yaw)
    return quat

def create_calibrated_sensor(dest_path, folders):
    print("Creating calibration sensor files...")
    sensors_stats = []
    for fold in tqdm.tqdm(folders):
        # get calibrated_sensor.json
        with open(fold + "/calibrated_sensor.json") as f:
            data = json.load(f)
        for d in data:
            if d["sensor_name"] in wanted_sensors:
                quat = euler_to_quaternion(d["rotation"])
                new_d = {"token": str(d["sensor_name"]) + "_SELMA",
                         "sensor_token": d["sensor_name"],
                         "translation": d["translation"],
                         "rotation": quat,
                         "camera_intrinsic": d["camera_intrinsic"]}
                sensors_stats.append(new_d)
        # save the new calibrated_sensor.json
        with open(dest_path + "/calibrated_sensor.json", "w") as f:
            json.dump(sensors_stats, f)
        break

def create_ego_pose(dest_path, folders):
    print("Creating ego pose files...")
    ego_list = []
    for fold in tqdm.tqdm(folders):
        town_envir = fold.split("/")[-1]
        # get ego_pose.json
        with open(fold + "/waypoints.json") as f:
            data = json.load(f)
        # get data keys
        data_keys = list(data.keys())
        for dk in data_keys:
            quat = euler_to_quaternion(data[dk]["Rotation"])
            xyz = []
            for axis in ["x", "y", "z"]:
                xyz.append(data[dk]["Location"][axis])
            ego_dict = {"token": town_envir + '_' + str(dk) + "_ego_pose_SELMA",
                        "timestamp": dk,
                        "rotation": quat,
                        "translation": xyz}
            ego_list.append(ego_dict)
        # save the new ego_pose.json
        with open(dest_path + "/ego_pose.json", "w") as f:
            json.dump(ego_list, f)

def output_category(categ):
    if categ == 40:
        lab = '1fa93b757fc74fb197cdd60001ad8abf'
    if categ == 100:
        lab = 'fd69059b62a3469fbaef25340c0eab7f'
    if categ == 101:
        lab = "6021b5187b924d64be64a702e5570edf"
    if categ == 102:
        lab = "fedb11688db84088883945752e480c2c"
    if categ == 103:
        lab = "003edbfb9ca849ee8a7496e9af3025d4"
    if categ == 104:
        lab = "dfd26f200ade4d24b540184e16050022"
    if categ == 105:
        lab = 'fc95c87b806f48f8a1faea2dcc2222a4'
    return lab

def create_instance(dest_path, folders):
    print("Creating instance files...")
    instance_list = []
    instances_for_count = []
    instances = []
    for fold in tqdm.tqdm(folders):
        town_name = fold.split("/")[-1].split("_")[0]
        for lab in tqdm.tqdm(glob.glob(fold + "/BBOX_LABELS/*.json")):
            with open(lab) as f:
                data = json.load(f)
            for d in data:
                instance_token = town_name + '_' + str(d["instance_id"])
                if instance_token not in instance_list:
                    instance_list.append(instance_token)
                instances_for_count.append(instance_token)

    for fold in tqdm.tqdm(folders):
        town_name = fold.split("/")[-1].split("_")[0]
        for lab in tqdm.tqdm(glob.glob(fold + "/BBOX_LABELS/*.json")):
            with open(lab) as f:
                data = json.load(f)
            for d in data:
                instance_token = town_name + '_' + str(d["instance_id"])
                if instance_token in instance_list:
                    instance_list.remove(instance_token)
                    instances.append({"token": instance_token,
                                      "category_token": output_category(d["label"]),
                                      "nbr_annotations": 0,
                                      "first_annotation_token": "",
                                      "last_annotation_token": ""})
    # count unique instances with its frequency
    instan = np.unique(instances_for_count, return_counts=True)
    instances_count = {}
    for i in range(len(instan[0])):
        instances_count[instan[0][i]] = instan[1][i]
    instances_v2 = []
    for inst in instances:
        instances_v2.append({"token": str(inst["token"]),
                             "category_token": inst["category_token"],
                             "nbr_annotations": int(instances_count[inst["token"]]),
                             "first_annotation_token": "",
                             "last_annotation_token": ""})
    # save the new instance.json
    with open(dest_path + "/instance.json", "w") as f:
        json.dump(instances_v2, f)
        
def create_log(dest_path, folders):
    print("Creating log files...")
    log_list = []
    towns = []
    for fold in tqdm.tqdm(folders):
        town_place = fold.split("/")[-1].split("_")[0]
        if town_place not in towns:
            towns.append(town_place)
    for to in towns:
        log_list.append({"token": to + "_SELMA",
                         "logfile": "logfile",
                         "vehicle": "SelmaCar",
                         "date_captured": "2020-01-01",
                         "location": to + "_SELMA"})
    # save the new log.json
    with open(dest_path + "/log.json", "w") as f:
        json.dump(log_list, f)

def create_scene(dest_path, folders, imageset):
    print("Creating scene files...")
    scene_list = []
    for scene in tqdm.tqdm(imageset):
        town_place = scene['Location'].split("_")[0]
        envir = scene['Location'].split("_")[-1]
        # count files in the folder
        num_files = len(scene['files'])
        bbox_list = scene['files']
        scene_list.append({"token": scene['Scene'],
                           "log_token": town_place + "_SELMA",
                            "nbr_samples": num_files,
                            "first_sample_token": bbox_list[0] + "_SELMA",
                            "last_sample_token": bbox_list[-1] + "_SELMA",
                            "name": scene['Scene'],
                            "description": envir})
    # save the new scene.json
    with open(dest_path + "/scene.json", "w") as f:
        json.dump(scene_list, f)

def create_sensors(dest_path, folders):
    print("Creating sensor files...")
    sensors_list = []
    for ws in wanted_sensors:
        if ws.split("_")[0] == "LIDAR":
            mod = "lidar"
        elif ws.split("_")[0] == "CAM":
            mod = "camera"
        sensors_list.append({"token": ws,
                             "channel": ws,
                             "modality": mod})

    # save the new sensor.json
    with open(dest_path + "/sensor.json", "w") as f:
        json.dump(sensors_list, f)

def create_sample(dest_path, folders, imageset):
    print("Creating sample files...")
    sample_list = []
    for scene in tqdm.tqdm(imageset):
        bbox_list = scene['files']
        next_token_sample = ''
        prev_token_sample = ''
        num_samples = len(bbox_list)
        for sample_i, lab in zip(range(num_samples), bbox_list):
            timestamp = lab.split("/")[-1].split(".")[0].split("_")[-1]
            token_stamp = lab.split("/")[-1].split(".")[0]
            token_sample = token_stamp + "_SELMA"
            if sample_i != 0:
                 prev_token_sample = bbox_list[sample_i-1] + "_SELMA"
            if sample_i != num_samples-1:
                 next_token_sample = bbox_list[sample_i+1] + "_SELMA"
            else:
                 next_token_sample = ""
            sample_list.append({"token": token_sample,
                                "timestamp": int(timestamp),
                                "next": next_token_sample,
                                "prev": prev_token_sample,
                                "scene_token": scene['Scene']})
    # save the new sample.json
    with open(dest_path + "/sample.json", "w") as f:
        json.dump(sample_list, f)
            
def create_sample_data(dest_path, folders):
    print("Creating sample_data files...")
    sample_data_list = []
    for fold in tqdm.tqdm(folders):
        town_envir = fold.split("/")[-1]
        for lab in glob.glob(fold + "/BBOX_LABELS/*.json"):
            timestamp = lab.split("/")[-1].split(".")[0].split("_")[-1]
            token_stamp = lab.split("/")[-1].split(".")[0]
            for ws in wanted_sensors:
                filename = "samples/" + ws + "/" + token_stamp
                if ws.split("_")[0] == "LIDAR":
                    ext = ".pcd.bin"
                    f_format = "pcd"
                    height = 0
                    width = 0
                elif ws.split("_")[0] == "CAM":
                    ext = ".jpg"
                    f_format = "jpg"
                    height = 1600
                    width = 900
                sample_data_list.append({"token": token_stamp + "_" + ws + "_SELMA",
                                            "sample_token": token_stamp + "_SELMA",
                                            "ego_pose_token":  token_stamp + "_ego_pose_SELMA",
                                            "calibrated_sensor_token": ws + "_SELMA",
                                            "timestamp": int(timestamp),
                                            "fileformat": f_format,
                                            "is_key_frame": True,
                                            "height": height,
                                            "width": width,
                                            "filename": filename + ext,
                                            "prev": "",
                                            "next": ""})
    # save the new sample_data.json
    with open(dest_path + "/sample_data.json", "w") as f:
        json.dump(sample_data_list, f)

def create_sample_annotation(dest_path, folders):
    print("Creating sample_annotation files...")
    sample_annotation_list = []
    for fold in tqdm.tqdm(folders):
        town_name = fold.split("/")[-1].split("_")[0]
        rootdir = fold
        sensor = 'LIDAR_TOP'
        with open(os.path.join(rootdir, 'calibrated_sensor.json'), 'r') as f:
            sensors = json.load(f)
        snames = [s['sensor_name'] for s in sensors]
        sidx = snames.index(sensor)
        sdata = sensors[sidx]

        with open(os.path.join(rootdir, 'waypoints.json'), 'r') as f:
            waypoints = json.load(f)

        for lab in tqdm.tqdm(glob.glob(fold + "/BBOX_LABELS/*.json")):
            # FROM SELMA CONVERTER #
            fname = lab.split('/')[-1].split('.')[0]

            pc_path = os.path.join(rootdir, sensor, fname+'.ply')
            data = PlyData.read(pc_path)
            instances = np.array([i for _,_,_,i,_ in data['vertex']])

            wdata = waypoints[fname.split('_')[-1]]
            with open(lab, 'r') as f:
                bboxes = json.load(f)

            t = project_boxes(bboxes, wdata, sdata, None, sensor="lidar")

            label_txt = ''
            pt_threshold = 5
            distance_threshold = 54
            # FROM SELMA CONVERTER #

            token_stamp = lab.split("/")[-1].split(".")[0]
            sample_token = token_stamp + "_SELMA"
            with open(lab, "r") as f:
                data = json.load(f)
            obj_count = 0
            for obj in data:
                close_obj = False
                for bb in t:
                    if str(bb[-3]) == obj["instance_id"]:
                        if (np.sqrt(bb[0][0]**2 + bb[0][1]**2) < distance_threshold) and (np.sum(instances==bb[-3])>pt_threshold):
                            close_obj = True
                            obj_count += 1
                quat = euler_to_quaternion(obj['rotation'])
                loc_xyz = []
                ext_xyz = []
                for axis in ["x", "y", "z"]:
                    loc_xyz.append(obj["location"][axis])
                    ext_xyz.append(obj["extent"][axis])
                #invert ext x with y
                ext_xyz[0], ext_xyz[1] = ext_xyz[1], ext_xyz[0]
                # multiply ext by 2
                ext_xyz = [2*ext for ext in ext_xyz]
                if close_obj and ext_xyz[0] != 0 and ext_xyz[1] != 0 and ext_xyz[2] != 0:
                    if(ext_xyz[0] == 0 or ext_xyz[1] == 0 or ext_xyz[2] == 0):
                        print("SIZE EQUAL 0!!!!")
                    sample_annotation_list.append({"token": token_stamp + "_instance_" + obj["instance_id"] + "_SELMA",
                                                    "sample_token": sample_token,
                                                    "instance_token": town_name + "_" + obj["instance_id"],
                                                    "visibility_token": "4",
                                                    "attribute_tokens": "",
                                                    "translation": loc_xyz,
                                                    "size": ext_xyz,
                                                    "rotation": quat,
                                                    "prev": "",
                                                    "next": "",
                                                    "num_lidar_pts": 5,
                                                    "num_radar_pts": 0})

    # save the new sample_annotation.json
    with open(dest_path + "/sample_annotation.json", "w") as f:
        json.dump(sample_annotation_list, f)

def test_lidars(dest_path, nuscenes_path, sample_dest_path):
    for ws in wanted_sensors:
        if ws.split("_")[0] == "LIDAR":
            # for lidar_scan in glob.glob(nuscenes_path + "/samples/" + ws + "/*.pcd.bin"):
            #     print(lidar_scan)
            #     # get the file
            #     scan = np.fromfile(lidar_scan, dtype=np.float32)
            #     print(scan)
            #     points = scan.reshape((-1, 5))[:, :4]
            #     print(points)
            print('TESTING LIDARS')
            for lidar_scan in glob.glob(sample_dest_path + ws + "/*.pcd.bin"):
                # get the file
                scan = np.fromfile(lidar_scan, dtype=np.float32)
                points = scan.reshape((-1, 5))[:, :4]
                print(points)
                print(scan)
                return 


def get_lidars(sample_dest_path, folders):
    print("Creating lidar files...")
    for fold in tqdm.tqdm(folders):
        for ws in wanted_sensors:
            if ws.split("_")[0] == "LIDAR":
                final_dest = sample_dest_path + ws + '/'
                if not os.path.exists(final_dest):
                    os.makedirs(final_dest)
                lidar_files = glob.glob(fold + '/' +  ws + "/*.ply")
                for lidar_scan in tqdm.tqdm(lidar_files):
                    file_name = lidar_scan.split("/")[-1].split(".")[0]
                    # get the ply file
                    points = PlyData.read(lidar_scan)
                    xyz = np.array([[x,y,z] for x,y,z,_,_ in points['vertex']])
                    # add 2 columns for intensity and ring
                    pcd = np.zeros((xyz.shape[0], 5))
                    pcd[:, :3] = xyz
                    # make pcd a single array with points ordered
                    pcd = pcd.reshape(-1)

                    # save into bin format file float32
                    pcd.astype('float32').tofile(final_dest + file_name + ".pcd.bin")

def get_cameras(sample_dest_path, folders):
    print("Creating camera files...")
    for fold in tqdm.tqdm(folders):
        for ws in wanted_sensors:
            final_dest = sample_dest_path + ws + '/'
            if not os.path.exists(final_dest):
                os.makedirs(final_dest)
            cam_files = glob.glob(fold + '/' +  ws + "/*.jpg")
            for cam in cam_files:
                file_name = cam.split("/")[-1].split(".")[0]
                shutil.copyfile(cam, final_dest + file_name + ".jpg")

def get_standard_files(dest_path, nuscences_v1mini_path):
    print("Creating standard files...")
    shutil.copyfile(nuscences_v1mini_path + 'v1.0-mini/attribute.json', dest_path + '/attribute.json')
    shutil.copyfile(nuscences_v1mini_path + 'v1.0-mini/category.json', dest_path + '/category.json')
    shutil.copyfile(nuscences_v1mini_path + 'v1.0-mini/map.json', dest_path + '/map.json')
    shutil.copyfile(nuscences_v1mini_path + 'v1.0-mini/visibility.json', dest_path + '/visibility.json')
    sweeps_dir = dest_path.split('v1')[0] + 'sweeps'
    if not os.path.exists(sweeps_dir):
        os.makedirs(sweeps_dir)
    if not os.path.exists(dest_path.split('v1')[0] + 'maps/'):
        shutil.copytree(nuscences_v1mini_path + 'maps/', dest_path.split('v1')[0] + 'maps/')

    with open(nuscences_v1mini_path + 'v1.0-mini/map.json') as f:
        map_data = json.load(f)
    log_tok = map_data[0]['log_tokens']
    town_available = ['01', '02', '03', '04', '05', '06', '07', '10']
    for ta in town_available:
        log_tok.append('Town'+ ta +'_SELMA')
    map_data[0]['log_tokens'] = log_tok

    with open(dest_path + '/map.json', "w") as f:
        json.dump(map_data, f)

def create_imagesets(dest_path, folders):
    #sample_per_scene = 100
    scene_numbers = ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077',
                     'scene-1094', 'scene-1100', 'scene-0103', 'scene-0916']
    train_scenes = splits.train
    val_scenes = splits.val
    #xscene_numbers = [*train_scenes,*val_scenes]

    count_files = 0
    for fold in folders:
        count_files += len(glob.glob(fold + '/BBOX_LABELS/*.json'))

    print(count_files)
    sample_per_scene = int(count_files/2/(len(scene_numbers)-2)) + 1
    sample_per_scene_val = int(count_files/2/2) + 1
    print(sample_per_scene)
    print(sample_per_scene_val)
    all_scene_json = []
    i = 0
    all_labs = []
    for fold in folders:
        location = fold.split('/')[-1]
    #    print(location)
    #    lab_files = []
        lab_list = glob.glob(fold + '/BBOX_LABELS/*.json')
    #    num_lab = len(lab_list)
        for ind, lab in enumerate(lab_list):
    #        lab_name = lab.split('/')[-1].split('.')[0]
    #        lab_files.append(lab_name)
            all_labs.append(lab)
    #        if (i % sample_per_scene == 0 and i != 0) or (ind == num_lab - 1):
    #            scene_json = {"Scene": scene, "Location": location, "files": lab_files}
    #            all_scene_json.append(scene_json)
    #            lab_files = []

    #        scene = scene_numbers[int(i/sample_per_scene)]
    #        i += 1
    shuffle(all_labs)
    scene_number = 0
    i = 0
    lab_files = []
    for ind, al in enumerate(all_labs):
          lab_name = al.split('/')[-1].split('.')[0]
          location = al.split('/')[-3]
          lab_files.append(lab_name)
          if (i % sample_per_scene == 0 and i != 0) and scene_number < 8:
              scene_json = {"Scene": scene, "Location": location, "files": lab_files}
              all_scene_json.append(scene_json)
              #print("len lab:",len(lab_files))
              #print("ind:",ind)
              lab_files = []
              scene_number += 1
          if ((i % sample_per_scene_val == 0 and i != 0 and i != np.ceil(count_files/2)) or ind == len(all_labs)-1) and scene_number >= 8:
              scene_json = {"Scene": scene, "Location": location, "files": lab_files}
              all_scene_json.append(scene_json)
              #print("len lab:",len(lab_files))
              #print("ind:",ind)
              lab_files = []
              scene_number += 1
          if len(all_labs)-1 == ind:
              break
          if scene_number < 8:
              scene = scene_numbers[scene_number]
              #print(int(i/sample_per_scene))
          else:
              scene = scene_numbers[scene_number]
    #          print(int(i/sample_per_scene))
          i += 1
          if i == count_files/2:
              i = 0
    #      print(ind)
    #     print(al)
    if os.path.exists(dest_path + '/imagesets.json'):
         os.remove(dest_path + '/imagesets.json')
    # save the json file
    with open(dest_path + '/imagesets.json', "w") as f:
        json.dump(all_scene_json, f)

def arg_parse():
    parser = argparse.ArgumentParser(description='SELMA to nuscenes converter')

    parser.add_argument("--path", dest = 'path', help =
                        "Path to the SELMA dataset",
                        default = "/hdd/SCANLAB/data/bevfusion/bevf/download_SELMA/CV/dataset/", type = str)
                        #default = "/hdd/SCANLAB_public/SELMA/bev_fusion_data/CV/dataset/", type = str)
    parser.add_argument("--dest", dest = 'dest', help =
                        "Path to the destination folder",
                        default = "/hdd/SCANLAB/data/bevfusion/", type = str)
                        #default = "/hdd/SCANLAB_public/catoad_datasets/", type = str)
    return parser.parse_args()

def main():
    args = arg_parse()

    selma_path = args.path
    path = args.dest
    dest_path = path + "data/selma_into_nuscenes/v1.0-mini"
    nuscences_v1mini_path = path + "data/nuscenes/"
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    sample_dest_path = path + "data/selma_into_nuscenes/samples/"
    if not os.path.exists(sample_dest_path):
        os.makedirs(sample_dest_path)

    num_of_towns = 5
    folders = glob.glob(selma_path + "*")
    fol = []
    for fld in folders:
        try:
            town_num = int(fld.split("/")[-1][4:6])
        except:
            town_num = 11
        if town_num <= num_of_towns: # and fld.split("/")[-1] == "Town01_Opt_ClearSunset": # for testing
            fol.append(fld)
    folders = fol
    #folders = folders[0:3] # for testing

    create_imagesets(selma_path, folders)

    # load the imagesets.json file
    with open(selma_path + '/imagesets.json') as f:
        imageset = json.load(f)

    create_calibrated_sensor(dest_path, folders)

    create_ego_pose(dest_path, folders)

    create_instance(dest_path, folders)

    create_log(dest_path, folders)

    create_scene(dest_path, folders, imageset)

    create_sensors(dest_path, folders)

    create_sample(dest_path, folders, imageset)

    create_sample_data(dest_path, folders)

    create_sample_annotation(dest_path, folders)

    get_lidars(sample_dest_path, folders)

    get_cameras(sample_dest_path, folders)

    test_lidars(dest_path, nuscences_v1mini_path, sample_dest_path)

    get_standard_files(dest_path, nuscences_v1mini_path)

if __name__ == "__main__":
    main()
