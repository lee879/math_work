
# # 指定要写入的文件名
# file_name = "D:\pj\math\math_work\data\train_label\train_label.txt"
def label_write(LABEL,path):
    # 打开文件以写入数据
    with open(path, "w") as file:
        # 遍历数组中的每一行并将其写入文件
        for row in LABEL:
            file.write(str(row[0]) + "\n")
    print("数据已写入到", path)
