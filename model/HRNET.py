import tensorflow as tf
from tensorflow.python.keras.layers import Dense,Conv2D,Conv2DTranspose,UpSampling2D,MaxPooling2D,Layer,Activation
from tensorflow.python.keras import Sequential,Model

class CustomPaddingLayer(tf.keras.layers.Layer):
    def __init__(self, padding):
        super(CustomPaddingLayer, self).__init__()
        self.padding = padding

    def call(self, inputs):
        paddings = tf.constant([[0, 0], [self.padding[0], self.padding[1]],
                                [self.padding[2], self.padding[3]], [0, 0]])
        return tf.pad(inputs, paddings, "CONSTANT")

class CBR(Layer):
    def __init__(self,out_channels,k=3,s=1):
        super(CBR, self).__init__()
        self.conv = Sequential([
            Conv2D(out_channels,kernel_size=k,strides=s,padding="same"),
            tf.keras.layers.BatchNormalization(),
            Activation(tf.nn.relu)
        ])

    def call(self, inputs, *args, **kwargs):

        return self.conv(inputs)


class UP_N(Layer):
    def __init__(self,filters,up_num = 2):
        super(UP_N, self).__init__()
        self.up_num = up_num
        self.up_2 = Sequential([
            Conv2D(filters, 1, 1, "same"),
            tf.keras.layers.BatchNormalization(),
            UpSampling2D(up_num)])

    def call(self, inputs, *args, **kwargs):

        return self.up_2(inputs)

class DOWN_2(Layer):
    def __init__(self,filters):
        super(DOWN_2, self).__init__()

        self.down = Sequential([
            Conv2D(filters, 3, 2, "same"),
            tf.keras.layers.BatchNormalization()
        ])


    def call(self, inputs, *args, **kwargs):

        return self.down(inputs)


class DOWN_4(Layer):
    def __init__(self, filters):
        super(DOWN_4, self).__init__()
        self.down=Sequential([
            Conv2D(filters,3,2,"same"),
            Conv2D(filters, 3, 2, "same"),
            tf.keras.layers.BatchNormalization()
        ])

    def call(self, inputs, *args, **kwargs):
        return self.down(inputs)


class DOWN_8(Layer):
    def __init__(self, filters):
        super(DOWN_8, self).__init__()

        self.down = Sequential([
            Conv2D(filters, 3, 2, "same"),
            Conv2D(filters, 3, 2, "same"),
            Conv2D(filters, 3, 2, "same"),
            tf.keras.layers.BatchNormalization()
        ])

    def call(self, inputs, *args, **kwargs):
        return self.down(inputs)

class Basic_block(Layer):
    def __init__(self, filters, stride=1,t=2):
        super(Basic_block, self).__init__()

        self.conv1 = Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = Activation(tf.nn.relu)

        self.conv2 = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        # 如果步幅不为1，则添加一个适当的卷积层，以便将输入匹配到输出的维度
        if t != 1:
            self.match_conv = Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='same')
            self.match_bn = tf.keras.layers.BatchNormalization()
        else:
            self.match_conv = None
            self.match_bn = None

    def call(self, inputs, *args, **kwargs):
        residual = inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # 如果有步幅不为1的情况，将输入也进行相应的卷积和标准化
        if self.match_conv is not None:
            residual = self.match_conv(residual)
            residual = self.match_bn(residual)

        # 残差连接
        x += residual
        x = self.relu(x)

        return x

class Basic_block_n(Layer):
    def __init__(self,filters):
        super(Basic_block_n, self).__init__()

        self.layer = Sequential([
            Basic_block(filters=filters,stride=1,t=2),
            Basic_block(filters=filters,stride=1,t=1),
            Basic_block(filters=filters,stride=1,t=1),
            Basic_block(filters=filters,stride=1,t=1)
        ])

    def call(self, inputs, *args, **kwargs):

        return self.layer(inputs)


class backbone_block_0(Model):
    def __init__(self):
        super(backbone_block_0, self).__init__()
        self.conv = Sequential([
            CBR(64,k=3,s=2),
            CBR(64,k=3,s=2)
        ])
        self.layer = Basic_block_n(filters=256)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        out = self.layer(x)
        return out


class backbone_block_1(Model):
    def __init__(self):
        super(backbone_block_1, self).__init__()
        self.conv_0 = CBR(out_channels=32,k=3,s=1)
        self.conv_1 = CBR(out_channels=64,k=3,s=2)
        self.block_0 = Basic_block_n(filters=32)
        self.block_1 = Basic_block_n(filters=64)
        self.down_2 = DOWN_2(filters=64)
        self.up_2 = UP_N(up_num=2,filters=32)
        self.ac = Activation(tf.nn.relu)

        #self.p_0 = CustomPaddingLayer([0,0,0,1])
    def call(self, inputs, training=None, mask=None):
        x_0 = self.conv_0(inputs)
        y_0 = self.conv_1(inputs)
        x_1 = self.block_0(x_0)
        y_1 = self.block_1(y_0)

        x_2 = self.down_2(x_1)
        y_2= self.up_2(y_1)

        out_x = self.ac(tf.add(x_1,y_2))
        out_y = self.ac(tf.add(y_1,x_2))

        return out_x,out_y

class Stage_3(Layer):
    def __init__(self):
        super(Stage_3, self).__init__()
        self.block_32 = Basic_block_n(32)
        self.block_64 = Basic_block_n(64)
        self.block_128 = Basic_block_n(128)
        self.up_2_0 = UP_N(up_num=2,filters=32)
        self.up_2_1 = UP_N(up_num=2,filters=64)
        self.up_4_0 = UP_N(up_num=4,filters=32)

        self.down_2_0 = DOWN_2(filters=64)
        self.down_2_1 = DOWN_2(filters=128)
        self.down_4_0 = DOWN_4(filters=128)
        self.ac = Activation(tf.nn.relu)

    def call(self, inputs_0,inputs_1,inputs_2, *args, **kwargs):
        x_0 = self.block_32(inputs_0)
        x_1 = self.block_64(inputs_1)
        x_2 = self.block_128(inputs_2)

        out_0 = tf.add(tf.add(self.up_2_0(x_1),self.up_4_0(x_2)),x_0)
        out_1 = tf.add(tf.add(self.down_2_0(x_0),self.up_2_1(x_2)),x_1)
        out_2 = tf.add(tf.add(self.down_4_0(x_0),self.down_2_1(x_1)),x_2)

        return self.ac(out_0),self.ac(out_1),self.ac(out_2)


class backbone_block_2(Model):
    def __init__(self):
        super(backbone_block_2, self).__init__()
        self.conv = CBR(128,k=3,s=2)
        self.stage = [Stage_3() for _ in range(1)]

    def call(self, inputs_0,inputs_1, training=None, mask=None):
        x_0 = inputs_0
        y_0 = inputs_1
        z_0 = self.conv(inputs_1)

        x_1, y_1, z_1 = self.stage[0](x_0, y_0, z_0)
        # x_2, y_2, z_2 = self.stage[1](x_1, y_1, z_1)
        # x_3, y_3, z_3 = self.stage[2](x_2, y_2, z_2)
        # x_4, y_4, z_4 = self.stage[3](x_3, y_3, z_3)

        return x_1,y_1,z_1

class Stage_4(Layer):
    def __init__(self):
        super(Stage_4, self).__init__()

        self.block_32 = Basic_block_n(32)
        self.block_64 = Basic_block_n(64)
        self.block_128 = Basic_block_n(128)
        self.block_256 = Basic_block_n(256)

        self.up_2_0 = UP_N(up_num=2,filters=32)
        self.up_2_1 = UP_N(up_num=2,filters=64)
        self.up_2_2 = UP_N(up_num=2,filters=128)

        self.up_4_0 = UP_N(up_num=4,filters=32)
        self.up_4_1 = UP_N(up_num=4,filters=64)

        self.up_8_0 = UP_N(up_num=8,filters=32)

        self.down_2_0 = DOWN_2(filters=64)
        self.down_2_1 = DOWN_2(filters=128)
        self.down_2_2 = DOWN_2(filters=256)

        self.down_4_0 = DOWN_4(filters=128)
        self.down_4_1 = DOWN_4(filters=256)

        self.down_8_0 = DOWN_8(filters=256)

        self.ac = Activation(tf.nn.relu)

    def call(self, inputs_0,inputs_1,inputs_2,inputs_3, *args, **kwargs):


        x0 = self.block_32(inputs_0)
        x1 = self.block_64(inputs_1)
        x2 = self.block_128(inputs_2)
        x3 = self.block_256(inputs_3)

        out_0 = tf.add(tf.add(tf.add(self.up_8_0(x3),self.up_4_0(x2)),self.up_2_0(x1)),x0)
        out_1 = tf.add(tf.add(tf.add(self.down_2_0(x0),self.up_2_1(x2)),self.up_4_1(x3)),x1)
        out_2 = tf.add(tf.add(tf.add(self.down_4_0(x0),self.down_2_1(x1)),self.up_2_2(x3)),x2)
        out_3 = tf.add(tf.add(tf.add(self.down_8_0(x0),self.down_4_1(x1)),self.down_2_2(x2)),x3)

        return self.ac(out_0),self.ac(out_1),self.ac(out_2),self.ac(out_3)

class backbone_block_3(Model):
    def __init__(self,out_channels):
        super(backbone_block_3, self).__init__()
        self.conv = CBR(256,k=3,s=2)
        self.stage = [Stage_4() for _ in range(2)]
        self.basic_block_32 = Basic_block_n(32)
        self.basic_block_64 = Basic_block_n(64)
        self.basic_block_128 = Basic_block_n(128)
        self.basic_block_256 = Basic_block_n(256)

        self.ac = Activation(tf.nn.relu)

        self.up_2 = UP_N(up_num=2,filters=32)
        self.up_4 = UP_N(up_num=4,filters=32)
        self.up_8 = UP_N(up_num=8,filters=32)

       # self.out = Conv2D(out_channels,1,1,"same")
        self.out = Sequential([
            tf.keras.layers.Flatten(),
            Dense(out_channels,activation=tf.nn.softmax)
        ])

    def call(self, inputs_0, inputs_1, inputs_2, training=None, mask=None):
        x_0 = inputs_0
        y_0 = inputs_1
        z_0 = inputs_2
        h_0 = self.conv(inputs_2)

        x_1, y_1, z_1, h_1 = self.stage[0](x_0, y_0, z_0, h_0)
        x_2, y_2, z_2, h_2 = self.stage[1](x_1, y_1, z_1, h_1)

        x_3 = self.basic_block_32(x_2)
        y_3 = self.basic_block_64(y_2)
        z_3 = self.basic_block_128(z_2)
        h_3 = self.basic_block_256(h_2)

        x_4 = x_3
        y_4 = self.up_2(y_3)
        z_4 = self.up_4(z_3)
        h_4 = self.up_8(h_3)

        out = self.ac(tf.add(tf.add(tf.add(h_4,z_4),y_4),x_4))
        return self.out(out)

class HRNet(Model):
    def __init__(self,out_channels):
        super(HRNet, self).__init__()
        self.block_0 = backbone_block_0()
        self.block_1 = backbone_block_1()
        self.block_2 = backbone_block_2()
        self.block_3 = backbone_block_3(out_channels=out_channels)

    def call(self, inputs, training=None, mask=None):
        x_0 = self.block_0(inputs)
        x_1,y_1 = self.block_1(x_0)
        x_2,y_2,z_2 = self.block_2(x_1,y_1)
        heatmap = self.block_3(x_2,y_2,z_2 )

        return heatmap

# # 256 192
# x = tf.random.normal(shape=(3,480,336,3))
#
# model = HRNet(out_channels=17)
#
# y = model(x)
#
# model.summary()
# print(y.shape)





