from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, UpSampling2D

class generator():
    # define the standalone generator model
    def __init__(self):
        pass
    def __call__(self, latent_dim):
        model = Sequential()
        # foundation for 4x4 image
        n_nodes = 256 * 8 * 8
        model.add(Dense(n_nodes, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((8, 8, 256)))
        # upsample to 16x16
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 32x32
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 64x64
        model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        # #sadeghi: upsample to 96*96
        # model.add(Conv2DTranspose(128, (6,6), strides=(3,3), padding='same'))
        # model.add(LeakyReLU(alpha=0.2))

        #sadeghi: upsample to 128*128
        model.add(Conv2DTranspose(512, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        #sadeghi: upsample to 128*128
        # model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
        # model.add(LeakyReLU(alpha=0.2))

        # output layer
        model.add(Conv2D(3, (5,5), activation='tanh', padding='same'))
        print(model.summary)
        return model


