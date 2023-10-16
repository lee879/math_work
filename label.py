from data import ptu,util,vis,wing
import numpy as np
import matplotlib.pyplot as plt
from data import write_label,label_vis

file_name = r"D:\pj\math\math_work\data\train_label\label.txt"

LABEL = []
label = []
data_vis, name_vis = vis.data_vis(category="MOR_1A")
begain_0,end_0 = (3848,4025)
begain_1,end_1 = (4088,5753)
data_vis_cut_0 = data_vis[begain_0:end_0]
data_vis_cut_1 = data_vis[begain_1:end_1]
data_all = np.vstack((data_vis_cut_0,data_vis_cut_1))


inx = []
c = 0
for i in range(5760):
    inx.append([c])
    c += 15
inx = np.array(inx)
z = np.hstack([inx,data_vis])



for i,data_simple in enumerate(data_vis_cut_0):
     if i % 1 == 0:
         label.append(data_simple)

for i,data_simple in enumerate(data_vis_cut_1):
     if i % 8 == 0:
         label.append(data_simple)

# #train_label.append(data_vis[5752])
label = np.array(label)

#
for data_label in label:
    if (data_label >= 1000):
        LABEL.append([0])
    elif (data_label >= 500) & (data_label < 1000):
        LABEL.append([1])
    elif (data_label >= 200) & (data_label < 500):
        LABEL.append([2])
    elif (data_label >= 100) & (data_label < 200):
        LABEL.append([3])
    else:
        LABEL.append([4])


LABEL = np.array(LABEL)
# write_label.label_write(LABEL,file_name)
label_vis.label_vision(LABEL)

