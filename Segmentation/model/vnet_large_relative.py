import tensorflow as tf
import inspect
from Segmentation.model.vnet_build_blocks import Conv3D_Block, Up_Conv3D


class VNet_Large_Relative(tf.keras.Model):

    def __init__(self,
                 num_channels,
                 num_classes,
                 num_conv_layers=3,
                 start_kernel_size=(5, 5, 5),
                 kernel_size=(3, 3, 3),
                 nonlinearity='relu',
                 use_batchnorm=True,
                 dropout_rate=0.25,
                 use_spatial_dropout=True,
                 data_format='channels_last',
                 merge_connections=True,
                 action='add',
                 output_activation=None,
                 use_res_connect=False,
                 use_stride_2=False,
                 name="vnet_large_relative"):
        self.params = str(inspect.currentframe().f_locals)
        super(VNet_Large_Relative, self).__init__(name=name)
        self.merge_connections = merge_connections
        self.num_classes = num_classes
        self.action = action
        assert self.action in ['add', 'multiply'], f"{self.action} not in action list"
        self.use_res_connect = use_res_connect
        self.use_stride_2 = use_stride_2

        self.conv_1 = Conv3D_Block(num_channels, num_conv_layers, start_kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c1")
        if self.use_stride_2:
            self.conv_1_stride = tf.keras.layers.Conv3D(num_channels, kernel_size=kernel_size, strides=2, activation="selu", padding="same")
        self.conv_2 = Conv3D_Block(num_channels * 2, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c2")
        if self.use_stride_2:
            self.conv_2_stride = tf.keras.layers.Conv3D(num_channels, kernel_size=kernel_size, strides=2, activation="selu", padding="same")
        self.conv_3 = Conv3D_Block(num_channels * 4, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c3")
        if self.use_stride_2:
            self.conv_3_stride = tf.keras.layers.Conv3D(num_channels, kernel_size=kernel_size, strides=2, activation="selu", padding="same")
        self.conv_4 = Conv3D_Block(num_channels * 8, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c4")
        if self.use_stride_2:
            self.conv_4_stride = tf.keras.layers.Conv3D(num_channels, kernel_size=kernel_size, strides=2, activation="selu", padding="same")                                   
        self.conv_5 = Conv3D_Block(num_channels * 16, num_conv_layers, kernel_size,
                                   nonlinearity, use_batchnorm=use_batchnorm,
                                   data_format=data_format, name="c5")
        self.up_5 = Up_Conv3D(num_channels * 8, (2, 2, 2), nonlinearity,
                              use_batchnorm=use_batchnorm, data_format=data_format, name="cu5")
        self.up_4 = Up_Conv3D(num_channels * 4, (2, 2, 2), nonlinearity,
                              use_batchnorm=use_batchnorm, data_format=data_format, name="cu4")
        self.up_3 = Up_Conv3D(num_channels * 2, (2, 2, 2), nonlinearity,
                              use_batchnorm=use_batchnorm, data_format=data_format, name="cu3")
        self.up_2 = Up_Conv3D(num_channels, (2, 2, 2), nonlinearity,
                              use_batchnorm=use_batchnorm, data_format=data_format, name="cu2")
        self.up_conv4 = Conv3D_Block(num_channels * 8, num_conv_layers, kernel_size, nonlinearity,
                                     use_batchnorm=use_batchnorm, data_format=data_format, name="upc4")
        self.up_conv3 = Conv3D_Block(num_channels * 4, num_conv_layers, kernel_size, nonlinearity,
                                     use_batchnorm=use_batchnorm, data_format=data_format, name="upc3")
        self.up_conv2 = Conv3D_Block(num_channels * 2, num_conv_layers, kernel_size, nonlinearity,
                                     use_batchnorm=use_batchnorm, data_format=data_format, name="upc2")
        self.up_conv1 = Conv3D_Block(num_channels, num_conv_layers, kernel_size, nonlinearity,
                                     use_batchnorm=use_batchnorm, data_format=data_format, name="upc1")

        self.pos_dense_1 = tf.keras.layers.Dense(16, activation="relu")
        self.pos_dense_2 = tf.keras.layers.Dense(4, activation="relu")
        self.pos_dense_3 = tf.keras.layers.Dense(1, activation="tanh")

        # convolution num_channels at the output
        self.conv_output = tf.keras.layers.Conv3D(num_classes, kernel_size, activation=nonlinearity, padding='same',
                                                  data_format=data_format)
        if output_activation is None:
            output_activation = 'sigmoid'
            if num_classes > 1:
                output_activation = 'softmax'
        if num_classes == 1:
            self.conv_1x1_binary = tf.keras.layers.Conv3D(num_classes, (1, 1, 1), activation='sigmoid',
                                                          padding='same', data_format=data_format)
        else:
            self.conv_1x1 = tf.keras.layers.Conv3D(num_classes, kernel_size=(1, 1, 1), activation='softmax',
                                                   padding='same', data_format=data_format)

    def call(self, inputs, training=True):
        image_inputs, pos_inputs = inputs

        pos_x = self.pos_dense_1(pos_inputs)
        pos_x = self.pos_dense_2(pos_x)
        pos_x = self.pos_dense_3(pos_x)

        # down x 1
        x1 = self.conv_1(image_inputs)

        # down x 2
        if self.use_stride_2:
            x2_in = self.conv_1_stride(x1)
        else:
            x2_in = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x1)
        x2 = self.conv_2(x2_in)
        if self.use_res_connect:
            x2 = tf.keras.layers.add([x2, x2_in])

        # down x 4
        if self.use_stride_2:
            x3_in = self.conv_2_stride(x2)
        else:
            x3_in = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x2)
        x3 = self.conv_3(x3_in)
        if self.use_res_connect:
            x3 = tf.keras.layers.add([x3, x3_in])

        # down x 8
        if self.use_stride_2:
            x4_in = self.conv_3_stride(x3)
        else:
            x4_in = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x3)
        x4 = self.conv_4(x4_in)
        if self.use_res_connect:
            x4 = tf.keras.layers.add([x4, x4_in])

        # down x 16
        if self.use_stride_2:
            x5_in = self.conv_4_stride(x4)
        else:
            x5_in = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x4)
        x5 = self.conv_5(x5_in)
        if self.use_res_connect:
            x5 = tf.keras.layers.add([x5, x5_in])

        # down x 8
        u5_in = self.up_5(x5)
        u5 = u5_in
        if self.action == 'add':
            u5 = tf.keras.layers.add([u5, pos_x])
        elif self.action == 'multiply':
            u5 = tf.keras.layers.multiply([u5, pos_x])
        if self.merge_connections:
            u5 = tf.keras.layers.concatenate([x4, u5], axis=-1)
        u5 = self.up_conv4(u5)
        if self.use_res_connect:
            u5 = tf.keras.layers.add([u5, u5_in])

        # down x 4
        u4_in = self.up_4(u5)
        u4 = u4_in
        if self.merge_connections:
            u4 = tf.keras.layers.concatenate([x3, u4], axis=-1)
        u4 = self.up_conv3(u4)
        if self.use_res_connect:
            u4 = tf.keras.layers.add([u4, u4_in])

        # down x 2
        u3_in = self.up_3(u4)
        u3 = u3_in
        if self.merge_connections:
            u3 = tf.keras.layers.concatenate([x2, u3], axis=-1)
        u3 = self.up_conv2(u3)
        if self.use_res_connect:
            u3 = tf.keras.layers.add([u3, u3_in])

        # down x 1
        u2_in = self.up_2(u3)
        u2 = u2_in
        if self.merge_connections:
            u2 = tf.keras.layers.concatenate([x1, u2], axis=-1)
        u2 = self.up_conv1(u2)
        if self.use_res_connect:
            u2 = tf.keras.layers.add([u2, u2_in])
        output = self.conv_output(u2)

        if self.num_classes == 1:
            output = self.conv_1x1_binary(output)
        else:
            output = self.conv_1x1(output)
        return output
