import tensorflow as tf
import os
import numpy as np

class DataToTFRecordConverter:
    '''
    用于创建一个二进制文件，为了加快数据的读入时间。
    '''
    def __init__(self, image_folder, label_file, tfrecord_filename):
        self.image_folder = image_folder
        self.label_file = label_file
        self.tfrecord_filename = tfrecord_filename

    def _bytes_feature(self, value):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value]))

    def convert_to_tfrecord(self):
        image_paths = os.listdir(self.image_folder)
        labels = []

        with open(self.label_file, 'r') as f:
            labels = [int(line.strip()) for line in f]

        with tf.io.TFRecordWriter(self.tfrecord_filename) as writer:
            for image_path, label in zip(image_paths, labels):
                image_path = os.path.join(self.image_folder, image_path)
                image_raw = open(image_path, 'rb').read()

                feature = {
                    'image': self._bytes_feature(image_raw),
                    'label': self._int64_feature(label)
                }

                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

if __name__ == "__main__":
    image_folder = r'D:\pj\math\math_work\data\train_data'
    label_file = r'D:\pj\math\math_work\data\train_label\label.txt'
    tfrecord_filename = r'D:\pj\math\math_work\data\data_tf\output.tfrecord'

    converter = DataToTFRecordConverter(image_folder, label_file, tfrecord_filename)
    converter.convert_to_tfrecord()

# # 创建一个TFRecordDataset对象
# tfrecord_filename = 'your_tfrecord_file.tfrecord'
# dataset = tf.data.TFRecordDataset(tfrecord_filename)

