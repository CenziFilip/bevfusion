import numpy as np
import os

import torch

from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes import NuScenes
from nuscenes.eval.detection.data_classes import DetectionBox

from mmdet3d.core.bbox.iou_calculators import bbox_overlaps_nearest_3d

from transforms3d.euler import euler2quat, quat2euler

from .iou_3D_python import get_3d_box, box3d_iou

from sklearn.metrics import average_precision_score

class_names = ["car", "pedestrian"]
iou_th = {"car":0.5, "pedestrian":0.4}

def get_coordinates(box):
	_,_,yaw = quat2euler(box.rotation)
	bb = []
	for c in box.translation:
		bb.append(c)
	for s in box.size:
		bb.append(s)
	bb.append(yaw)

	return bb

def accumulate_v2(gt_boxes, pred_boxes, class_name, iou_thresh):
	print("@@@@@@@@",class_name, "@@@@@@@@")

	npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])

	pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
	pred_confs = [box.detection_score for box in pred_boxes_list]

	sam_toks = []
	for pb in pred_boxes_list:
		sam_toks.append(pb.sample_token)
	sam_toks = np.array(sam_toks)
	sam_toks = np.unique(sam_toks)
#	print(sam_toks)
#	print(len(sam_toks))

	tp = []  # Accumulator of true positives.
	fp = []  # Accumulator of false positives.
	conf = []  # Accumulator of confidences.

	for tok in sam_toks:
		boxes_pred = []
		boxes_gt = []
		for pred_box in pred_boxes[tok]:
			if pred_box.detection_name == class_name:
				p_box = get_coordinates(pred_box)
				boxes_pred.append(p_box)
				conf.append(pred_box.detection_score)
		boxes_pred = torch.as_tensor(boxes_pred)

		for gt_box in gt_boxes[tok]:
			if gt_box.detection_name == class_name:
				g_box = get_coordinates(gt_box)
				boxes_gt.append(g_box)
		boxes_gt = torch.as_tensor(boxes_gt)
#		print("len pred",len(boxes_pred))
#		print("len_gt",len(boxes_gt))
		if len(boxes_pred) > 0 and len(boxes_gt) > 0:
			ious = bbox_overlaps_nearest_3d(boxes_pred, boxes_gt)
			ious = ious.numpy()
		else:
			ious = np.zeros([len(boxes_pred), 2])
		for iou in ious:
#			print(iou)
			max_iou = max(iou)
			if max_iou > iou_th[class_name]:
				tp.append(1)
				fp.append(0)
			else:
				tp.append(0)
				fp.append(1)

	conf, tp, fp = map(list, zip(*sorted(zip(conf, tp, fp), reverse=True)))
#	print(conf, tp, fp)

	ap3 = average_precision_score(tp, conf)
	print(ap3)

	# Accumulate.
	tp = np.cumsum(tp).astype(float)
	fp = np.cumsum(fp).astype(float)
	conf = np.array(conf)

	# Calculate precision and recall.
	prec = tp / (fp + tp)
	rec = tp / float(npos)

	rec_interp = np.linspace(0, 1, 40) #, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
	prec = np.interp(rec_interp, rec, prec, right=0)
	conf = np.interp(rec_interp, rec, conf, right=0)
	rec = rec_interp

#	print(prec)
#	print(rec)

	return prec, rec

def _evaluate_single_v2(result_path, nusc_version, nusc_data_root, detection_config):
	verbose = True
	eval_set = "mini_val"
	print("\n\n################################################################")
	nusc = NuScenes(version=nusc_version, dataroot=nusc_data_root, verbose=True)

	pred_boxes, meta = load_prediction(result_path, detection_config.max_boxes_per_sample, DetectionBox, verbose=verbose)
	gt_boxes = load_gt(nusc, eval_set, DetectionBox, verbose=verbose)

	pred_boxes = add_center_dist(nusc, pred_boxes)
	gt_boxes = add_center_dist(nusc, gt_boxes)

	class_range = detection_config.class_range
	print(class_range)

	class_range["pedestrian"] = 50
	print(class_range)

	pred_boxes = filter_eval_boxes(nusc, pred_boxes, class_range, verbose=verbose)
	gt_boxes = filter_eval_boxes(nusc, gt_boxes, class_range, verbose=verbose)

	sample_tokens = gt_boxes.sample_tokens

	for class_name in class_names:
		prec, rec = accumulate_v2(gt_boxes, pred_boxes, class_name, iou_th[class_name])
		ap1 = float(np.mean(prec))
		print("AP1:",ap1)
		prec = prec[round(100*detection_config.min_recall) + 1:]
		prec -= detection_config.min_precision
		prec[prec < 0] = 0
		ap = float(np.mean(prec)) / (1.0 - detection_config.min_precision)
		print("AP2:",ap)

	print("################################################################\n\n")
