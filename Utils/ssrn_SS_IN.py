import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling2D,
    MaxPooling3D,
    AveragePooling3D,
    Conv2D,
    Conv3D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.layers.core import Reshape
from keras import regularizers
from keras.layers.merge import add

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def _bn_relu_spc(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu_spc(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", regularizers.l2(1.e-4))
    def f(input):
        # conv = Convolution3D(nb_filter=nb_filter, kernel_dim1=kernel_dim1, kernel_dim2=kernel_dim2,kernel_dim3=kernel_dim3, subsample=subsample,
        #                      init=init, W_regularizer=W_regularizer)(input)
        conv = Conv3D(kernel_initializer=init,strides=subsample,kernel_regularizer= W_regularizer, filters=nb_filter, kernel_size=(kernel_dim1,kernel_dim2,kernel_dim3))(input)
        # conv = Conv3D(kernel_initializer="he_normal", strides=(1,1,2), kernel_regularizer=regularizers.l2(1.e-4), filters=32,
        #               kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3))
        return _bn_relu_spc(conv)

    return f


def _bn_relu_conv_spc(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1,1,1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu_spc(input)
        return Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                          filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3), padding=border_mode)(activation)

    return f


def _shortcut_spc(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_dim1 = 1
    stride_dim2 = 1
    stride_dim3 = (input._keras_shape[CONV_DIM3]+1) // residual._keras_shape[CONV_DIM3]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    print("input shape:", input._keras_shape)
    # 1 X 1 conv if shape is different. Else identity.
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
        shortcut = Convolution3D(nb_filter=residual._keras_shape[CHANNEL_AXIS],
                                 kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
                                 subsample=(stride_dim1, stride_dim2, stride_dim3),
                                 init="he_normal", border_mode="valid",
                                 W_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block_spc(block_function, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (1, 1, 2)
            input = block_function(
                    nb_filter=nb_filter,
                    init_subsample=init_subsample,
                    is_first_block_of_first_layer=(is_first_layer and i == 0)
                )(input)
        return input

    return f


def basic_block_spc(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    扩充样本的光谱卷积模块：传入卷积核返回残差卷积块函数，此处的所有卷积都是same模式
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            # conv1 = Convolution3D(nb_filter=nb_filter,
            #                       kernel_dim1=1, kernel_dim2=1, kernel_dim3=7,
            #                      subsample=init_subsample,
            #                      init="he_normal", border_mode="same",
            #                      W_regularizer=l2(0.0001))(input)
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample, kernel_regularizer=regularizers.l2(0.0001),
                          filters=nb_filter, kernel_size=(1, 1, 7), padding='same')(input)
        else:
            conv1 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7, subsample=init_subsample)(input)

        residual = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7)(conv1)
        return _shortcut_spc(input, residual)   # 增加跳跃连接

    return f


def bottleneck_spc(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of nb_filter * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Convolution3D(nb_filter=nb_filter,
                                 kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="same",
                                 W_regularizer=l2(0.0001))(input)
        else:
            conv_1_1 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1, subsample=init_subsample)(input)

        conv_3_3 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv_1_1)
        residual = _bn_relu_conv_spc(nb_filter=nb_filter * 4, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1)(conv_3_3)
        return _shortcut_spc(input, residual)

    return f

def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", regularizers.l2(1.e-4))

    def f(input):
        # conv = Convolution3D(nb_filter=nb_filter, kernel_dim1=kernel_dim1, kernel_dim2=kernel_dim2,kernel_dim3=kernel_dim3, subsample=subsample,
        #                      init=init, W_regularizer=W_regularizer)(input)
        conv = Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                          filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3))(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1,1,1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", regularizers.l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        # return Convolution3D(nb_filter=nb_filter, kernel_dim1=kernel_dim1, kernel_dim2=kernel_dim2,kernel_dim3=kernel_dim3, subsample=subsample,
        #                      init=init, border_mode=border_mode, W_regularizer=W_regularizer)(activation)
        return  Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                          filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3), padding=border_mode)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_dim1 = (input._keras_shape[CONV_DIM1]+1) // residual._keras_shape[CONV_DIM1]
    stride_dim2 = (input._keras_shape[CONV_DIM2]+1) // residual._keras_shape[CONV_DIM2]
    stride_dim3 = (input._keras_shape[CONV_DIM3]+1) // residual._keras_shape[CONV_DIM3]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    print("input shape:", input._keras_shape)
    # 1 X 1 conv if shape is different. Else identity.
    # if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
    #     shortcut = Convolution3D(nb_filter=residual._keras_shape[CHANNEL_AXIS],
    #                              kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
    #                              subsample=(stride_dim1, stride_dim2, stride_dim3),
    #                              init="he_normal", border_mode="valid",
    #                              W_regularizer=l2(0.0001))(input)
    shortcut = Conv3D(kernel_initializer="he_normal", strides=(stride_dim1, stride_dim2, stride_dim3), kernel_regularizer=regularizers.l2(0.0001),
                          filters=residual._keras_shape[CHANNEL_AXIS], kernel_size=(1, 1, 1), padding='valid')(input)

    return add([shortcut, residual])


def _residual_block(block_function, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2, 1)
            input = block_function(
                    nb_filter=nb_filter,
                    init_subsample=init_subsample,
                    is_first_block_of_first_layer=(is_first_layer and i == 0)
                )(input)
        return input

    return f


def basic_block(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    扩充样本的空谱卷积模块：传入卷积核返回残差卷积块函数
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            # conv1 = Convolution3D(nb_filter=nb_filter,
            #                       kernel_dim1=3, kernel_dim2=3, kernel_dim3=1,
            #                      subsample=init_subsample,
            #                      init="he_normal", border_mode="same",
            #                      W_regularizer=l2(0.0001))(input)
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample, kernel_regularizer=regularizers.l2(0.0001),
                          filters=nb_filter, kernel_size=(3, 3, 1), padding='same')(input)
        else:
            conv1 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1, subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of nb_filter * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Convolution3D(nb_filter=nb_filter,
                                 kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="same",
                                 W_regularizer=l2(0.0001))(input)
        else:
            conv_1_1 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1, subsample=init_subsample)(input)

        conv_3_3 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv_1_1)
        residual = _bn_relu_conv(nb_filter=nb_filter * 4, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1)(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 3 #4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn_spc, block_fn, repetitions1, repetitions2):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2],input_shape[3], input_shape[0])

        # Load function from str if needed.
        block_fn_spc = _get_block(block_fn_spc)
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        print("input shape:", input._keras_shape[3])

        conv1_spc = _conv_bn_relu_spc(nb_filter=24, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7, subsample=(1, 1, 2))(input)

        block_spc = conv1_spc
        nb_filter = 24
        for i, r in enumerate(repetitions1):    # 构建多层残差块,通过repetition1参数进行层数控制
            block_spc = _residual_block_spc(block_fn_spc, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(block_spc)
            nb_filter *= 2

        # Last activation
        block_spc = _bn_relu_spc(block_spc)

        block_norm_spc = BatchNormalization(axis=CHANNEL_AXIS)(block_spc)
        block_output_spc = Activation("relu")(block_norm_spc)

        conv_spc_results = _conv_bn_relu_spc(nb_filter=128,kernel_dim1=1,kernel_dim2=1,kernel_dim3=block_output_spc._keras_shape[CONV_DIM3])(block_output_spc)

        print("conv_spc_result shape:", conv_spc_results._keras_shape)

        # 将(None, 7, 7, 1, 128)转换成(None, 7, 7, 128, 1)
        conv2_spc = Reshape((conv_spc_results._keras_shape[CONV_DIM1],conv_spc_results._keras_shape[CONV_DIM2],conv_spc_results._keras_shape[CHANNEL_AXIS],1))(conv_spc_results)

        # 用24个（3，3，128）的核进行卷积，padding模式为valid，得到（5，5，1，24）的卷积结果
        conv1 = _conv_bn_relu(nb_filter=24, kernel_dim1=3, kernel_dim2=3, kernel_dim3=128,
                              subsample=(1, 1, 1))(conv2_spc)
        #conv1 = _conv_bn_relu(nb_filter=32, kernel_dim1=3, kernel_dim2=3, kernel_dim3=input._keras_shape[3], subsample=(1, 1, 1))(input)
        #pool1 = MaxPooling3D(pool_size=(3, 3, 1), strides=(2, 2, 1), border_mode="same")(conv1)
        #conv1 = Convolution3D(nb_filter=32, kernel_dim1=3, kernel_dim2=3, kernel_dim3=176,subsample=(1,1,1))(input)
        print("conv1 shape:", conv1._keras_shape)   #(None, 5, 5, 1, 24)

        block = conv1
        nb_filter = 24
        for i, r in enumerate(repetitions2):    # 针对（5，5，1，24）的数据反复进行多层残差块连接，padding均为same
            block = _residual_block(block_fn, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(block)
            nb_filter *= 2

        # Last activation
        block = _bn_relu(block)

        block_norm = BatchNormalization(axis=CHANNEL_AXIS)(block)
        block_output = Activation("relu")(block_norm)

        # Classifier block 得到(None, 1, 1, 1, 24)的pool结果
        pool2 = AveragePooling3D(pool_size=(block._keras_shape[CONV_DIM1],
                                            block._keras_shape[CONV_DIM2],
                                            block._keras_shape[CONV_DIM3],),
                                 strides=(1, 1, 1))(block_output)

        flatten1 = Flatten()(pool2)     # 将(None, 1, 1, 1, 24)flatten为(None, 24)

        drop1 = Dropout(0.5)(flatten1)
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(drop1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def gaborcnn(input_shape, num_outputs):
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[3])

        input = Input(shape=input_shape)
        print("input shape:", input._keras_shape)

        conv1 = Conv2D(filters=32, kernel_size=(1, 1), kernel_initializer='he_normal', padding='valid',
                       kernel_regularizer=regularizers.l2(1.e-4), strides=(1, 1))(input)
        print(conv1._keras_shape)
        br1 = _bn_relu_spc(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(br1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(pool1)
        print(pool2._keras_shape)

        conv2 = Conv2D(filters=48, kernel_size=(5, 5), kernel_initializer='he_normal', padding='valid',
                       kernel_regularizer=regularizers.l2(1.e-4), strides=(1, 1))(pool2)
        print(conv2._keras_shape)

        br2 = _bn_relu_spc(conv2)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(br2)
        print(pool3._keras_shape)
        drop2 = Dropout(0.5)(pool3)

        conv3 = Conv2D(filters=64, kernel_size=(4, 4), kernel_initializer='he_normal', padding='valid', activation='relu',
                       kernel_regularizer=regularizers.l2(1.e-4), strides=(1, 1))(drop2)
        print(conv3._keras_shape)
        drop3 = Dropout(0.5)(conv3)

        conv4 = Conv2D(filters=num_outputs, kernel_size=(1, 1), kernel_initializer='he_normal', padding='valid',
                       data_format='channels_last', kernel_regularizer=regularizers.l2(1.e-4), strides=(1, 1))(drop3)
        print(conv4._keras_shape)

        flatten = Flatten()(conv4)
        print(flatten._keras_shape)
        output = Activation("softmax")(flatten)
        # dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(flatten)
        # print(dense._keras_shape)

        model = Model(inputs=input, outputs=output)
        # get_layer_output = K.function(inputs=[model.layers[0].input], outputs=[model.layers[3].output])
        return model

    @staticmethod
    def pfcnn(input_shape, num_outputs):
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[3])

        input = Input(shape=input_shape)
        print("input shape:", input._keras_shape)

        conv1 = Conv2D(filters=64, kernel_size=(1, 1), kernel_initializer='he_normal', padding='valid',
                       kernel_regularizer=regularizers.l2(1.e-4), strides=(1, 1))(input)
        print(conv1._keras_shape)
        br1 = _bn_relu_spc(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(br1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(pool1)
        print(pool2._keras_shape)

        conv2 = Conv2D(filters=96, kernel_size=(5, 5), kernel_initializer='he_normal', padding='valid',
                       kernel_regularizer=regularizers.l2(1.e-4), strides=(1, 1))(pool2)
        print(conv2._keras_shape)

        br2 = _bn_relu_spc(conv2)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(br2)
        print(pool3._keras_shape)
        drop2 = Dropout(0.5)(pool3)

        conv3 = Conv2D(filters=128, kernel_size=(4, 4), kernel_initializer='he_normal', padding='valid',
                       activation='relu',
                       kernel_regularizer=regularizers.l2(1.e-4), strides=(1, 1))(drop2)
        print(conv3._keras_shape)
        drop3 = Dropout(0.5)(conv3)

        conv4 = Conv2D(filters=num_outputs, kernel_size=(1, 1), kernel_initializer='he_normal', padding='valid',
                       data_format='channels_last', kernel_regularizer=regularizers.l2(1.e-4), strides=(1, 1))(drop3)
        print(conv4._keras_shape)

        flatten = Flatten()(conv4)
        print(flatten._keras_shape)
        output = Activation("sigmoid")(flatten)
        # dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(flatten)
        # print(dense._keras_shape)

        model = Model(inputs=input, outputs=output)
        # out_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

        return model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block_spc, basic_block, [1],[1])      #[2, 2, 2, 2]

    @staticmethod
    def build_resnet_12(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block_spc, basic_block, [2], [2])

def main():
    model = ResnetBuilder.build_resnet_8((1, 7, 7, 200), 16)            # IN DATASET model = ResnetBuilder.build_resnet_18((3, 224, 224), 1000)
    #model = ResnetBuilder.build_resnet_6((1,7,7,176), 13)               # KSC DATASET
    #model = ResnetBuilder.build_resnet_6((1, 7, 7, 103), 9)             # UP DATASET
    #model = ResnetBuilder.build_resnet_34((1, 27, 27, 103), 9)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()

if __name__ == '__main__':
    main()