
from data import ptu, vis, wing
import matplotlib.pyplot as plt
import numpy as np
import q1_data_vision
import matplotlib.pyplot as plt
from data_pca import data_dea_pca,data_error


x = ["PAINTS","TEMP","RH","WS2A"]

PAINTS = []
TEMP = []
RH = []
WS2A = []

data_ptu, name_ptu = ptu.data_ptu()
data_vis, name_vis = vis.data_vis()
data_wing, name_wing = wing.data_wing()

data_vis_1440 = []
data_wing_1440 = []

for i,data in enumerate(data_vis):
    if i % 4 == 0:
        data_vis_1440.append(data)
data_vis_1440 = np.array(data_vis_1440)

for i,data in enumerate(data_wing):
    if i % 4 == 0:
        data_wing_1440.append(data)
data_wing_1440 = np.array(data_wing_1440)

x = np.hstack((np.hstack((data_ptu[:,0].reshape(-1,1),data_ptu[:,3:5])),data_wing_1440[:,0].reshape(-1,1)))

RVR_1A = []
for i,data in enumerate(data_vis_1440):
        RVR_1A.append(data)
y = np.array(RVR_1A)

# 异常检测与处理
X = np.copy(x)

Y = np.copy(y)
error_indx = data_dea_pca.detect_outliers_with_pca(X,4,threshold=8)
X = data_error.replace_outliers_with_mean(X,error_indx)
#X = np.log(1 + (X / np.max(X, axis=0))) # 数据的归一化

# print("error:",error_indx)

Z = np.hstack((Y,X))  #["RVR_1A ":"PAINS","TEMP","RH","WS2A"]

label = ["RVR_1A ","PAINS","TEMP","RH","WS2A"]
indx = [(0,0),(0,1),(1,0),(1,1)]
sf = [10,50,10,10]
x_cut = [(1017,1019),(7.8,12),(80,100),(0,2)]
if_x_cut = False
color = ["b","g","r","y"]

q1_data_vision.data_vision(label,indx,sf,x_cut,color,Z,if_x_cut)


