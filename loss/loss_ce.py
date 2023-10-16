import tensorflow as tf
import numpy as np

#使用余弦退火降低学习率
class CosineAnnealingSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_max, lr_min, T):
        super(CosineAnnealingSchedule, self).__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T = T
    def __call__(self, step):
        t = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos((step/self.T) * np.pi))
        return t

def sparse_categorical_crossentropy(y_pred,y_true ):
    # Convert integer labels to one-hot encoded labels
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

    # Clip y_pred to prevent log(0) errors
    y_pred = tf.clip_by_value(y_pred, clip_value_min=1e-7, clip_value_max=1 - 1e-7)

    # Calculate cross-entropy loss
    cross_entropy = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

    # Compute mean loss across batch
    mean_loss = tf.reduce_mean(cross_entropy)

    return mean_loss