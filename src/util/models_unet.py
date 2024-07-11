from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization, Dense
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras import regularizers
from keras.models import *


def DenseBlock(channels, inputs, use_l2_reg=False, l2_reg=1e-4):
    conv1_1 = Conv2D(channels, (1, 1), activation=None, padding='same', kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(inputs)
    conv1_1 = BatchActivate(conv1_1)
    conv1_2 = Conv2D(channels//4, (3, 3), activation=None, padding='same', kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(conv1_1)
    conv1_2 = BatchActivate(conv1_2)

    conv2 = concatenate([inputs, conv1_2])
    conv2_1 = Conv2D(channels, (1, 1), activation=None, padding='same', kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(conv2)
    conv2_1 = BatchActivate(conv2_1)
    conv2_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same', kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(conv2_1)
    conv2_2 = BatchActivate(conv2_2)

    conv3 = concatenate([inputs, conv1_2, conv2_2])
    conv3_1 = Conv2D(channels, (1, 1), activation=None, padding='same', kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(conv3)
    conv3_1 = BatchActivate(conv3_1)
    conv3_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same', kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(conv3_1)
    conv3_2 = BatchActivate(conv3_2)

    conv4 = concatenate([inputs, conv1_2, conv2_2, conv3_2])
    conv4_1 = Conv2D(channels, (1, 1), activation=None, padding='same', kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(conv4)
    conv4_1 = BatchActivate(conv4_1)
    conv4_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same', kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(conv4_1)
    conv4_2 = BatchActivate(conv4_2)
    result = concatenate([inputs, conv1_2, conv2_2, conv3_2, conv4_2])
    return result


def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def DenseUNet(input_size, channel=1, start_neurons=16, keep_prob=0.9, block_size=7, lr=1e-3, use_l2_reg=False, l2_reg=1e-4):
    inputs = Input(shape=input_size)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(inputs)
    conv1 = BatchActivate(conv1)
    conv1 = DenseBlock(start_neurons * 1, conv1, use_l2_reg=use_l2_reg, l2_reg=l2_reg)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = DenseBlock(start_neurons * 2, pool1, use_l2_reg=use_l2_reg, l2_reg=l2_reg)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = DenseBlock(start_neurons * 4, pool2, use_l2_reg=use_l2_reg, l2_reg=l2_reg)
    pool3 = MaxPooling2D((2, 2))(conv3)

    convm = DenseBlock(start_neurons * 8, pool3, use_l2_reg=use_l2_reg, l2_reg=l2_reg)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same", kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(start_neurons * 4, (1, 1), activation=None, padding="same", kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(uconv3)
    uconv3 = BatchActivate(uconv3)
    uconv3 = DenseBlock(start_neurons * 4, uconv3, use_l2_reg=use_l2_reg, l2_reg=l2_reg)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same", kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(start_neurons * 2, (1, 1), activation=None, padding="same", kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(uconv2)
    uconv2 = BatchActivate(uconv2)
    uconv2 = DenseBlock(start_neurons * 2, uconv2, use_l2_reg=use_l2_reg, l2_reg=l2_reg)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same", kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(start_neurons * 1, (1, 1), activation=None, padding="same", kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(uconv1)
    uconv1 = BatchActivate(uconv1)
    uconv1 = DenseBlock(start_neurons * 1, uconv1, use_l2_reg=use_l2_reg, l2_reg=l2_reg)

    output_layer_noActi = Conv2D(channel, (1, 1), padding="same", activation=None, kernel_regularizer=regularizers.l2(l2_reg) if use_l2_reg else None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    model = Model(inputs=inputs, outputs=output_layer)
    return model


def Attention_gate(g, s, num_filters):
    Wg = Conv2D(num_filters, 1, padding="same")(g)
    Wg = BatchNormalization()(Wg)

    Ws = Conv2D(num_filters, 1, padding="same")(s)
    Ws = BatchNormalization()(Ws)

    w = Add()([Wg,Ws])

    out = Activation("relu")(w)
    out = Conv2D(num_filters, 1, padding="same")(out)
    out = Activation("sigmoid")(out)
    return Multiply()([out,s])


def DenseUNet_AttGate(input_size, channel=1, start_neurons=16, keep_prob=0.9,block_size=7,lr=1e-3):

    inputs = Input(shape=input_size)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(inputs)
    conv1 = BatchActivate(conv1)
    conv1 = DenseBlock(start_neurons * 1, conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = DenseBlock(start_neurons * 2, pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = DenseBlock(start_neurons * 4, pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)


    convm = DenseBlock(start_neurons * 8, pool3)


    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    Att3 = Attention_gate(conv3,deconv3,start_neurons * 4)
    uconv3 = concatenate([deconv3, Att3])
    uconv3 = Conv2D(start_neurons * 4, (1, 1), activation=None, padding="same")(uconv3)
    uconv3 = BatchActivate(uconv3)
    uconv3 = DenseBlock(start_neurons * 4, uconv3)


    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    Att2 = Attention_gate(conv2,deconv2,start_neurons * 2)
    uconv2 = concatenate([deconv2, Att2])
    uconv2 = Conv2D(start_neurons * 2, (1, 1), activation=None, padding="same")(uconv2)
    uconv2 = BatchActivate(uconv2)
    uconv2 = DenseBlock(start_neurons * 2, uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    Att1 = Attention_gate(conv1,deconv1,start_neurons * 1)
    uconv1 = concatenate([deconv1, Att1])
    uconv1 = Conv2D(start_neurons * 1, (1, 1), activation=None, padding="same")(uconv1)
    uconv1 = BatchActivate(uconv1)
    uconv1 = DenseBlock(start_neurons * 1, uconv1)

    output_layer_noActi = Conv2D(channel, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    model = Model(inputs=inputs, outputs=output_layer)

    return model



def ConvBlock(in_fmaps, num_fmaps):
    conv1 = Conv2D(num_fmaps, (3, 3), activation='relu', padding='same')(in_fmaps)
    conv_out = Conv2D(num_fmaps, (3, 3), activation='relu', padding='same')(conv1)
    return conv_out

def Network():

    input = Input(shape=input_size)

    conv1 = ConvBlock(input, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = ConvBlock(pool1, 32)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = ConvBlock(pool2, 64)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = ConvBlock(pool3, 64)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = ConvBlock(pool4, 128)

    up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = ConvBlock(up6, 64)

    up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = ConvBlock(up7, 64)

    up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = ConvBlock(up8, 32)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = ConvBlock(up9, 32)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs = input, outputs = conv10)

    return model


def build_model(backbone, lr=0.003, starting_layer_name='conv5_block3_3_conv'):
    start_training = False
    for layer in backbone.layers:
        if layer.name == starting_layer_name:
            start_training = True
        layer.trainable = start_training
 
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=backbone.input, outputs=output)
 
    model.compile(
        loss='binary_crossentropy',  
        optimizer=Adam(learning_rate=lr), 
        metrics=['accuracy']
    )
 
    return model