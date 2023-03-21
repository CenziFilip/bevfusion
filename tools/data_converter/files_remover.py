import numpy as np
import os
import glob
import random


path = "/hdd/SCANLAB/data/bevfusion/bevf/download_SELMA/CV/dataset/"
folds = glob.glob(path + "*")
folds = [f for f in folds if "imagesets" not in f]
fold_names = ["LIDAR_TOP","LIDAR_FRONT_LEFT","LIDAR_FRONT_RIGHT","CAM_FRONT","CAM_RIGHT","CAM_LEFT","CAM_DESK",
		"CAM_BACK","BBOX_LABELS"]
formats = {"LIDAR":".ply", "CAM":".jpg", "BBOX":".json"}

for f in folds:
	f2 = glob.glob(f + "/*")
	elems = []
	for ff in f2:
		#if ff.split("/")[-1] in fold_names:
		#	print(ff)
		if ff.split("/")[-1] == "BBOX_LABELS":
			files = glob.glob(ff + "/*")
			files_names = []
			for file in files:
				f_name = file.split("/")[-1].split(".")[0]
				files_names.append(f_name)
			amount = len(files)
			if amount > 1700:
				del_am = amount-1700
				elems = random.sample(files_names, del_am)
				print(len(elems))
				elems = np.unique(elems)
				print(len(elems))
		if ff.split("/")[-1] in fold_names:
			format = formats[ff.split("/")[-1].split("_")[0]]
			for el in elems:
				#print(ff + "/" + el + format)
				if el != "BBOX_LABELS":
					os.remove(ff + "/" + el + format)
