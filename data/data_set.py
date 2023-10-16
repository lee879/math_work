from sklearn.model_selection import train_test_split
import tensorflow as tf


def set_data(images,labels):
    x_train,x_test,y_train,y_test = train_test_split(images, labels, test_size=0.2, random_state=42,shuffle=1000)
    return x_train,x_test,y_train,y_test

def tarin_data(data_train,data_test,label_train,label_test,train_batchs,test_batchs):

    datas_train = tf.data.Dataset.from_tensor_slices((data_train,label_train)).batch(train_batchs)
    datas_train = datas_train.shuffle(10000)

    datas_test = tf.data.Dataset.from_tensor_slices((data_test,label_test)).batch(test_batchs)
    datas_test = datas_test.shuffle(10000)

    return datas_train,datas_test