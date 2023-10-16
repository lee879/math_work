from data import read_record
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data import image_read,data_set
import glob
from model import a001,HRNET
import os
from loss import loss_ce
import pandas as pd

'''
a gpu larger than rtx4090 is recommended to build models
'''

image_path = r"D:\pj\math\math_work\data\train_data\*.jpg"
label_path = r"D:\pj\math\math_work\data\train_label\label.txt"
excel_path = r"./result/training_info.xlsx"

epochs = 2000
lr_max = 0.000001
lr_min = 0.00000001
train_batchs = 32
test_batchs = 4
catag = 5

#导入模型
#model = HRNET.HRNet(out_channels=catag)
model = a001.GL_net(out=catag)
#model.load_weights()

#使用tensorboard进行可视化（导入路径）
summary_writer = tf.summary.create_file_writer(r"./log")

# 模型保存路径的加载
checkpoint_dir = './ckpt'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
best_weights_checkpoint_path_d = os.path.join(checkpoint_dir, 'best_d.hd5')
end_weights_checkpoint_path_d = os.path.join(checkpoint_dir, 'end_d.hd5')

# 加载数据
image_list = glob.glob(image_path)
image_data = image_read.read_images(image_list)
label_data = np.array(np.loadtxt(label_path))
data_train,data_test,label_train,label_test = data_set.set_data(image_data,label_data)
train_dataset,test_dataset = data_set.tarin_data(data_train,data_test,label_train,label_test,train_batchs,test_batchs)
# 计算进度条
trian_sps = len(train_dataset)
test_sps = len(test_dataset)

sp = 0
val_sp = 0
training_info = []
loss_best = []
loss_best_temp = 1

lr_schedule = loss_ce.CosineAnnealingSchedule(lr_max=lr_max, lr_min=lr_min,T=epochs)
#训练部分
for epoch in range(epochs):
    lr_cos = lr_schedule(epoch)
    pbar = tqdm(range(trian_sps), desc=f"Epoch {epoch}")
    dataset_iter = iter(train_dataset)
    loss_temp = []
    acc_train = []
    for step in pbar:
        train_data,train_label = next(dataset_iter)
        with tf.GradientTape() as tape:
            predict = model(train_data)
            loss = loss_ce.sparse_categorical_crossentropy(predict,train_label)
            ac_train = np.mean(np.array(train_label) == np.argmax(np.array(predict), axis=-1))
        gp = tape.gradient(loss, model.trainable_variables)
        # tf.keras.optimizers.RMSprop(learning_rate=0.000001, momentum=0.9, epsilon=1e-8).apply_gradients(zip(gp, model.trainable_variables))
        tf.keras.optimizers.Adam(learning_rate=lr_cos, epsilon=1e-8).apply_gradients(zip(gp, model.trainable_variables))

        acc_train.append(ac_train)
        loss_temp.append(loss)
        loss_best.append(loss)
        with summary_writer.as_default():

            tf.summary.scalar('loss', float(loss), step=sp)
            tf.summary.scalar('acc', float(ac_train), step=sp)
            tf.summary.scalar('lr', float(lr_cos), step=sp)
            pbar.set_postfix({"Train_Loss": float(loss), "acc ":float(ac_train),"Step": sp,"Lr":float(lr_cos)})

            training_info.append({
                "Epoch": epoch,
                "Step": sp,
                "Loss": float(loss),
                "LearningRate": float(lr_cos)
            })
            # 每一百次迭代，保存相关的参数
            # 每一百次迭代，保存相关的参数。
            if sp % 100 == 0:

                tf.summary.image("O_img", np.expand_dims(train_data[0, :, :, :], axis=0), step=sp)  # 原图

                tf.keras.backend.clear_session()  # 防止出现内存溢出

                if np.mean(np.array(loss_best)) <= loss_best_temp:
                    print("to keep the best model..........")
                    model.save_weights(best_weights_checkpoint_path_d)
                    loss_best_temp = np.mean(np.array(loss_best))
                loss_best = []
                print("to keep the end model.........")
                model.save_weights(end_weights_checkpoint_path_d)
                # 保存训练参数到excel文件中
                training_df = pd.DataFrame(training_info)
                # excel_path = "./training_info.xlsx"
                if os.path.exists(excel_path):
                    existing_df = pd.read_excel(excel_path)
                    updated_df = pd.concat([existing_df, training_df], ignore_index=True)
                    updated_df.to_excel(excel_path, index=False)
                else:
                    training_df.to_excel(excel_path, index=False)
                training_info = []
                print("Training information saved up to epoch:", epoch)
            sp += 1
    print("epoch: ",int(epoch)," losss: ",float(np.mean(np.array(loss_temp))),"acc_train",float(np.mean(np.array(acc_train))))


    #print(predict.numpy)
    if epoch % 5 == 0:
        pbar_test = tqdm(range(test_sps), desc=f"Val_Epoch {int(epoch % 10)}")
        test_iter = iter(test_dataset)
        loss_test_temp = []
        acc_val = []
        for step in pbar_test:
            test_data, test_label = next(test_iter)
            val_predict = model(test_data)
            val_loss = loss_ce.sparse_categorical_crossentropy(val_predict,test_label)
            loss_test_temp.append(val_loss)
            ac_val =np.mean(np.array(test_label) == np.argmax(np.array(val_predict),axis=-1))
            acc_val.append(ac_val)
            with summary_writer.as_default():
                tf.summary.scalar('val_loss', float(val_loss), step=val_sp)
                tf.summary.scalar('val_acc', float(ac_val), step=val_sp)
                val_sp += 1
        accuracy_percentage = np.mean(np.array(acc_val)) * 100
        print("val_loss: ",float(np.mean(np.array(loss_test_temp))),f"{accuracy_percentage:.2f}%")




