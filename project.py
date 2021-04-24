import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
from numpy.random import randn
from numpy import zeros, ones
from matplotlib import pyplot

from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Dense, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Flatten


def build_discriminator(in_shape):
	model = Sequential()
	# normal
	model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# classifier
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model


def build_generator(latent_dim = 100):
	model = Sequential()
	# foundation for 4x4 image
	n_nodes = 256 * 4 * 4
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((4, 4, 256)))
	# upsample to 8x8
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 16x16
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 32x32
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# output layer
	model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
	return model


def build_gan(g_model, d_model):
    d_model.trainable = False

    model = Sequential()
    model.add(g_model)
    model.add(d_model)

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def load_real_data():
    folder = 'data/pokemon/image_names.csv'
    df = pd.read_csv(folder, header=None)

    return df


def generate_real_samples(df, iter, n_samples):
    batch = df[iter*n_samples:iter*n_samples+n_samples]
    im_array = []

    for _, row in batch.iterrows():
        path = row[0]
        im = Image.open('data/pokemon/'+path)
        im = im.resize((32, 32))
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
		pyplot.imshow(examples[i])
	pyplot.savefig('output.png')


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):
    batch_per_epoch = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    # manually enumerate epochs
    for i in range(n_epochs):
        for j in range(int(batch_per_epoch)):
            X_real, y_real = generate_real_samples(dataset, j, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

            d_loss_real = d_model.train_on_batch(X_real, y_real)
            d_loss_fake = d_model.train_on_batch(X_fake, y_fake)

            z_input = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))

            g_loss = gan_model.train_on_batch(z_input, y_gan)

            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, j+1, batch_per_epoch, d_loss_real[0], d_loss_fake[0], g_loss))

        if (i+1) % (1) == 0:
            latent_points = generate_latent_points(100, 9)
            X  = g_model.predict(latent_points)
            X = (X + 1) / 2.0
            summarize_performance(X, 3)


latent_dim = 100
df = load_real_data()

g_model = build_generator()
print(g_model.summary())
d_model = build_discriminator((32,32,3))
print(d_model.summary())
gan_model = build_gan(g_model, d_model)

train(g_model, d_model, gan_model, df, latent_dim)
