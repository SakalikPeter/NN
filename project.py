import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
from numpy.random import randn
from numpy import zeros, ones
from matplotlib import pyplot

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, MaxPooling2D, Flatten


def build_discriminator(in_shape):
    model = Sequential()
    model.add(Input(shape=in_shape))

    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same')) #128x128
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2))) #64x64

    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same')) #32x32
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2))) #16x16

    model.add(Conv2D(32, (3,3), strides=(2,2), padding='same')) #8x8
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2))) #4x4

    model.add(Flatten())
    model.add(Dense(16))
    model.add(LeakyReLU())

    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def build_generator(latency_shape = 100):
    model = Sequential()

    n_units = 8*8*256

    model.add(Input(shape=(latency_shape,)))
    model.add(Dense(units=n_units))
    model.add(LeakyReLU())

    model.add(Reshape((8, 8, 256)))

    model.add(Conv2DTranspose(256, (5,5), strides=(2,2), padding='same')) #16x16
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, (5,5), strides=(2,2), padding='same')) #32x32
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5,5), strides=(2,2), padding='same')) #64x64
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(32, (5,5), strides=(2,2), padding='same')) #128x128
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', activation='tanh')) #256x256
    
    return model


def build_gan(g_model, d_model):
	d_model.trainable = False

	model = Sequential()
	model.add(g_model)
	model.add(d_model)

	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model


def load_real_data():
    folder = 'data/pokemon/image_names.csv'
    df = pd.read_csv(folder, header=None)

    return df


def generate_real_samples(df, n_samples):
    batch = df.sample(n=n_samples)
    im_array = []

    for _, row in batch.iterrows():
        path = row[0]
        im = Image.open('data/pokemon/'+path)
        im = im.resize((256, 256))
        im_array.append(np.array(im))

    im_array = np.array(np.float32(im_array))
    im_array = (im_array - 127.5) / 127.5

    y = ones((n_samples, 1))

    return im_array, y


def generate_latent_points(latent_dim, n_samples):
	x_input = randn(latent_dim*n_samples)
	z_input = x_input.reshape(n_samples, latent_dim)

	return z_input
 

def generate_fake_samples(generator, latent_dim, n_samples):
	z_input = generate_latent_points(latent_dim, n_samples)
	images = generator.predict(z_input)
	y = zeros((n_samples, 1))

	return images, y


# generate samples and save as a plot and save the model
def summarize_performance(examples, n):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0])
	pyplot.savefig('output.png')


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=6, n_batch=128):
    batch_per_epoch = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    # manually enumerate epochs
    for i in range(n_epochs):
        for j in range(int(batch_per_epoch)):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

            # print(X_real.shape, X_fake.shape)
            d_loss_real = d_model.train_on_batch(X_real, y_real)
            d_loss_fake = d_model.train_on_batch(X_fake, y_fake)

            z_input = generate_latent_points(latent_dim, n_batch)
            y_gan = zeros((n_batch, 1))

            g_loss = gan_model.train_on_batch(z_input, y_gan)
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, j+1, batch_per_epoch, d_loss_real[0], d_loss_fake[0], g_loss))

        if (i+1) % (2) == 0:
            latent_points = generate_latent_points(100, 9)
            X  = g_model.predict(latent_points)
            X = (X + 1) / 2.0
            summarize_performance(X, 3)


latent_dim = 100
df = load_real_data()

g_model = build_generator()
d_model = build_discriminator((256,256,3))
gan_model = build_gan(g_model, d_model)

train(g_model, d_model, gan_model, df, latent_dim)