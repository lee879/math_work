import numpy as np
import glob
import numpy as np
import math

def data_ptu():
    path = "D:\pj\math\math_work\data\AMOS20200313\PTU_R06_12.his"
    ALL_data = []

    with open(path, 'r') as file:
        # 跳过文件的头部信息
        lines = file.readlines()[2:]

        # 遍历文件的每一行
        for line in lines:
            # 使用制表符分割每一行的数据
            data = line.split('\t')

            # 提取关键数据列的值
            createdate = data[0]
            localdate_beijing = data[1]
            size = data[2]
            pains = data[3]
            qnh_aerodrome_hpa = data[4] # this 4
            st_0 = data[5]
            qfe_r06_hpa = data[6] #this 2
            st_1 = data[7]
            qfe_r24_hpa = data[8] #this 3
            st_2 = data[9]
            qff_aer0drome_hpa = data[10]
            trend = data[11]
            tendency = data[12]
            temp = data[13]   # this 1
            RH = data[14]
            dewpoint = data[15] # this 5

            ALL_data.append([pains,qnh_aerodrome_hpa,qfe_r06_hpa,temp,np.array(RH),dewpoint])


    # 打印提取的数据
    return np.array(ALL_data).astype(np.float32), ["PAINS","QNH","QFE","TEMP","RH","DEWPOINT"]