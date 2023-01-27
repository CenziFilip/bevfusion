import numpy as np
import os
import glob
import shutil

path = "/hdd/SCANLAB_public/SELMA/bev_fusion_data/CV/dataset/Town01_Opt_ClearNight"
bb_v1_path = path + "/BBOX_LABELS_v1"
bb_path = path + "/BBOX_LABELS"

lab_names = []
for lab in glob.glob(bb_v1_path + "/*.json"):
	lab_names.append(lab.split('/')[-1].split('.')[0])
	n = lab.split('/')[-1].split('.')[0].split('_')[-1]
	new_name = '/Town01_Opt_ClearNight_' + n + '.json'
	new_full_name = bb_path + new_name
	shutil.copyfile(lab, new_full_name)

#lidar_names = []
#for ld in glob.glob(path + '/LIDAR_TOP/*.ply'):
#	lidar_names.append(ld.split('/')[-1].split('.')[0])

#count = 0
#for lab in lab_names:
#	for lid in lidar_names:
#		if lab.split('_')[-1] == lid.split('_')[-1]:
#			print(lab)
#			print(lid)
#			count += 1

#print(count)
