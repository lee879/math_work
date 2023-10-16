import numpy as np
import glob
import matplotlib.path as plt


def data_wing():
    data_indx = [288, 2421, 3237, 4238, 4844]
    path = r"D:\pj\math\math_work\data\AMOS20200313\WIND_R06_12.his"
    ALL_data = []

    with open(path, 'r') as file:
        # 跳过文件的头部信息
        lines = file.readlines()[2:]

        # 创建一个空的列表，用于存储提取的数据
        extracted_data = []

        # 遍历文件的每一行
        for line in lines:
            # 使用制表符分割每一行的数据
            data = line.split('\t')

            # 提取关键数据列的值
            createdate = data[0]
            localdate_beijing = data[1]
            size = data[2]
            WSINS = data[3]
            WS2M = data[4]
            WS2A = data[5] # this 1
            WS2X = data[6]
            WS10M = data[7]
            WS10A = data[8]
            st_1 = data[9]
            WS10X = data[10]
            WDINS = data[11]
            WD2M = data[12]
            WD2A = data[13] # this 2
            WD2X = data[14]
            WD10M = data[15]
            WD10A = data[16]
            ST_2 = data[17]
            WD10X = data[18]
            CW2A = data[22] # this 3

            ALL_data.append([WS2A,WD2A,CW2A])

    all_data = np.array(ALL_data).astype(np.float32)
    for indx in data_indx:
        all_data = np.insert(all_data, indx + 1, np.mean(all_data[indx:indx + 2, :],axis=0,keepdims=True), axis=0)

    return all_data,["WS2A","WD2A","CW2A"]
# data_wing()

