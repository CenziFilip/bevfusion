import numpy as np
import torch
import os

print("##########################MODEL 1################################")
model1 = torch.load("../../pretrained/swint-nuimages-pretrained.pth")
print(model1.keys())
model1_keys = model1["state_dict"].keys()
print(model1["state_dict"]["norm1.bias"])

print("##########################MODEL 2################################")
model2 = torch.load("../../runs/camera-only-43ep/latest.pth")
print(model2.keys())
model2_keys = model2["state_dict"].keys()

print(len(model1_keys))
print(len(model2_keys))
model3 = model1

count_not_found = 0
for k1 in model1_keys:
	corr = False
	k2_corr = ""
	for k2 in model2_keys:
		if k1 in k2:
			corr = True
			k2_corr = k2
			model3["state_dict"][k1] = model2["state_dict"][k2]
#	if corr:
#		print(k1)
#		print(k2_corr, "\n")
#	else:
#		print(k1)
#		print("NOT FOUND\n")
#		count_not_found += 1

print(count_not_found)

print(len(model3["state_dict"].keys()))
model3["state_dict"].pop("norm0.weight", None)
model3["state_dict"].pop("norm0.bias", None)
print(len(model3["state_dict"].keys()))

torch.save(model3, "../../runs/camera-only.pth")
