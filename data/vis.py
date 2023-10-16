import numpy as np
import glob
import numpy as np



def data_vis(category="RVR_1A"):
    data_indx = [288, 2420, 3235, 4235, 4840]
    path = r"D:\pj\math\math_work\data\AMOS20200313\VIS_R06_12.his"

    ALL_data = []
    TIME_DATA = []
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
            RVR1M = data[3]
            RVR_1A = data[4]
            st_0 = data[5]
            RVR1X = data[6]
            RVR_10M = data[7]
            RVR_10A = data[8]
            st_1 = data[9]
            RVR_10X = data[10]
            RVR_TEND = data[11]
            MOR_1A = data[12] # this 1
            MOR_10M = data[13]
            MOR_10A = data[14]
            MOR_10X= data[15]
            VIS1K = data[16]
            VIS1A = data[17]
            VIS10A = data[18]
            BL1A = data[19]
            if category == "RVR_1A":
                ALL_data.append([RVR_1A])
                TIME_DATA.append(localdate_beijing)
            elif category == "MOR_1A":
                ALL_data.append([MOR_1A])
                TIME_DATA.append(localdate_beijing)
            else:
                print("to select the existing data|")
                return None
                break
    all_data = np.array(ALL_data).astype(np.float32)

    for indx in data_indx:
        all_data = np.insert(all_data, indx + 1, np.mean(all_data[indx:indx+2,:],keepdims=True).reshape(1, 1), axis=0) #

   # new_data = np.vstack((data, new_row))

    return all_data,[category]
# data_vis()
