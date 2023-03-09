import numpy as np
import os
import glob
import shutil
from plyfile import PlyData, PlyElement
import tqdm
import json

lidar_path = "/hdd/SCANLAB_public/SELMA/bev_fusion_data/CV/dataset/"
town_path = glob.glob(lidar_path + "*")
path_to_use = []

lidar_top = "LIDAR_TOP"
main_sensor = "LIDAR_TOP_ORIGINAL"
sensors = ["LIDAR_FRONT_RIGHT","LIDAR_FRONT_LEFT"]

num_of_lidars = 1

for tp in town_path:
	try:
		if int(tp.split('/')[-1][4:6]) == 1:
			print(tp)
			path_to_use.append(tp)
	except:
		continue

for path in path_to_use:
	if not os.path.exists(path + "/" + main_sensor):
		shutil.copytree(path + "/LIDAR_TOP", path + "/" + main_sensor)
		print("LIDAR_TOP copy executed")
	else:
		print("LIDAR_TOP copy already exists")

	sensor_path = path + '/' + main_sensor
	lidar_files = glob.glob(sensor_path + "/*.ply")

	calibration_file = path + "/calibrated_sensor.json"
	with open(calibration_file) as f:
                calib = json.load(f)
	for calib_sen in calib:
		if calib_sen["sensor_name"] == "LIDAR_TOP":
			main_translation = calib_sen["translation"]
	#print(main_translation)

	for ply_file in tqdm.tqdm(lidar_files):
		file_name = os.path.basename(ply_file)

		points = PlyData.read(ply_file)
		#print(points)
		xyzil = np.array([[x,y,z,i,l] for x,y,z,i,l in points['vertex']])
		for sen in sensors:
			points_to_add = PlyData.read(path + '/' + sen + '/' + file_name)

			for calib_sen in calib:
				if calib_sen["sensor_name"] == sen:
					translation = calib_sen["translation"]
			#print(translation)
			translation_diff = [x-y for x,y in zip(main_translation,translation)]
			#print(translation_diff)
			translation_diff = np.array([-translation_diff[1], translation_diff[0], translation_diff[2]])
			#print(translation_diff)
			#print(points_to_add["vertex"].data)
			for p in points_to_add["vertex"].data:
				p[0] = p[0] - translation_diff[0]
				p[1] = p[1] - translation_diff[1]
				p[2] = p[2] - translation_diff[2]
			#print(points_to_add["vertex"].data)

			points["vertex"].data = np.r_[points["vertex"].data, points_to_add["vertex"].data]

		#print(points)
		if num_of_lidars == 1:
			points = PlyData.read(ply_file)

		save_path = path + "/" + lidar_top + "/" + file_name
		#print(save_path)
		points.write(save_path)


