import numpy as np
import math
import os
import re

def sort_by_column(data, column_index, reverse=False):
    data_temp = np.copy(data)
    """
    对二维数据按照指定列进行排序

    参数:
        data (list[list]): 包含子列表的数据
        column_index (int): 要用于排序的列的索引
        reverse (bool): 是否按降序排序，默认为升序

    返回:
        list[list]: 排序后的数据
    """
    sorted_data = sorted(data_temp, key=lambda x: x[column_index], reverse=reverse)
    return np.array(sorted_data)

def rename_images(folder_path,folder_path_ol):
    # 获取文件夹中所有的文件名
    file_list = os.listdir(folder_path)
    sorted_files = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group(0)))

    # 遍历排序后的文件列表
    for i, filename in enumerate(sorted_files):
        # 构造新的文件名
        new_filename = str(i+177).zfill(3) + ".jpg"  # 使用zfill填充0，确保三位数命名

        # 构造完整的旧文件路径和新文件路径
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path_ol, new_filename)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
#
# 调用函数，传递包含图片的文件夹路径
folder_path = r"D:\pj\math\math_work\data\vision_data_1"
folder_path_ol = r"D:\pj\math\math_work\data\train_data"
rename_images(folder_path,folder_path_ol)




































