import numpy as np
import os
import glob
import shutil
import json

script_path = '/hdd/SCANLAB_public/SELMA/bev_fusion_data'
with open(script_path + "/dataset_original.json") as f:
	data = json.load(f)
scenes = []
for d in data:
	scenes.append(d['fullPath'].split('/')[4])
unique_scenes = np.unique(scenes)

sensors_to_add = ["CAM_DESK"]
data_copy = data.copy()
data_to_add = []
for us in unique_scenes:
	for d in data:
		if (us + '/LIDAR_TOP') in d['fullPath']:
			for sen in sensors_to_add:
				fullPath = d['fullPath'].split('/')
				del fullPath[-1]
				fullPath.append(sen)
				fullPath = '/'.join(fullPath)

				relPath = d['relativePath'].split('/')
				del relPath[-1]
				relPath.append(sen)
				relPath = '/'.join(relPath)

				files = []
				files.append(sen + '.zip')
				for fi in d['files'][1:]:
					files.append(fi)
				d_copy = {'fullPath': fullPath, 'relativePath': relPath, 'files': files}
				data_to_add.append(d_copy)

final_data = []

for dc in data_copy:
	if 'LIDAR_FRONT_LEFT' not in dc['fullPath'] and 'LIDAR_FRONT_RIGHT' not in dc['fullPath']: #int(dc['fullPath'].split('/')[4][4:6]) == 1 and 
		final_data.append(dc)

for dt in data_to_add:
	final_data.append(dt)

data_v2 = []
for d in final_data:
	item_1 = d['fullPath'].replace('ClearNight', 'ClearNoon')
	item_2 = d['relativePath'].replace('ClearNight', 'ClearNoon')
	item_3 = []
	for el in d['files']:
		item_3.append(el.replace('ClearNight', 'ClearNoon'))
	data_v2.append({'fullPath': item_1, 'relativePath': item_2, 'files': item_3})

for d in data_v2:
	print(d)

if os.path.exists(script_path + "/dataset.json"):
	os.remove(script_path + "/dataset.json")

with open(script_path + "/dataset.json", "w") as f:
	json.dump(data_v2, f)
