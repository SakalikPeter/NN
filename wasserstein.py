import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
from numpy.random import randn
from numpy import ones
from matplotlib import pyplot
import wandb

from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras import backend
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Flatten, BatchNormalization, UpSampling2D, Input, Activation

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


# ########################################
# LOSSES
# ########################################
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


# ########################################
# critic
# ########################################
def define_critic(in_shape):
	model = Sequential()
	# downsample
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=config['lrelu']))
	# downsample
	model.add(Conv2D(128, (4,4), strides=(2,2), padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=config['lrelu']))
	# downsample
	model.add(Conv2D(256, (4,4), strides=(2,2), padding='same'))
	model.add(Conv2D(256, (3,3), strides=(1,1), padding="same"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=config['lrelu']))

	model.add(Conv2D(256, (3,3), strides=(1,1), padding="same"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=config['lrelu']))
	# downsample
	model.add(Conv2D(512, (5,5), strides=(2,2), padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=config['lrelu']))

	model.add(Conv2D(1, (1,1), strides=(1,1), padding="same"))
	# classifier
	model.add(Flatten())
	model.add(Dense(1))
	return model
 
# ########################################
# GENERATOR
# ########################################
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 4x4 image
	n_nodes = 512 * 8 * 8
	model.add(Dense(n_nodes, input_shape=(1,1,latent_dim)))
	model.add(Reshape((8, 8, 512)))	
	
	# upsample to 16x16
	model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=config['lrelu']))

	model.add(Conv2D(256, (3,3), strides=(1,1), padding="same"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=config['lrelu']))

	model.add(Conv2D(256, (3,3), strides=(1,1), padding="same"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=config['lrelu']))

	model.add(Conv2D(256, (3,3), strides=(1,1), padding="same"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=config['lrelu']))

	# upsample to 32x32
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=config['lrelu']))

	model.add(Conv2D(128, (3,3), strides=(1,1), padding="same"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=config['lrelu']))

	model.add(Conv2D(128, (3,3), strides=(1,1), padding="same"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=config['lrelu']))

	# upsample to 64x64
	model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=config['lrelu']))

	model.add(Conv2D(64, (3,3), strides=(1,1), padding="same"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=config['lrelu']))

	model.add(Conv2D(64, (3,3), strides=(1,1), padding="same"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=config['lrelu']))

	model.add(Conv2D(64, (3,3), strides=(1,1), padding="same"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=config['lrelu']))

	# upsample to 64x64
	model.add(Conv2D(3, (3,3), strides=(1,1), padding='same', activation='tanh'))
	return model


# ########################################
# GAN
# ########################################
class WGAN(tf.keras.Model):
	def __init__(
		self,
		discriminator,
		generator,
		latent_dim,
		dataset,
		batch_size,
		discriminator_extra_steps=3,
		gp_weight=10.0,
	):
		super(WGAN, self).__init__()
		self.discriminator = discriminator
		self.generator = generator
		self.latent_dim = latent_dim
		self.d_steps = discriminator_extra_steps
		self.gp_weight = gp_weight
		self.dataset = dataset
		self.batch_size = batch_size

	def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
		super(WGAN, self).compile()
		self.d_optimizer = d_optimizer
		self.g_optimizer = g_optimizer
		self.d_loss_fn = d_loss_fn
		self.g_loss_fn = g_loss_fn

	def gradient_penalty(self, batch_size, real_images, fake_images):
		""" Calculates the gradient penalty.

		This loss is calculated on an interpolated image
		and added to the discriminator loss.
		"""
		# Get the interpolated image
		alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
		diff = fake_images - real_images
		interpolated = real_images + alpha * diff

		with tf.GradientTape() as gp_tape:
			gp_tape.watch(interpolated)
			# 1. Get the discriminator output for this interpolated image.
			pred = self.discriminator(interpolated, training=True)

		# 2. Calculate the gradients w.r.t to this interpolated image.
		grads = gp_tape.gradient(pred, [interpolated])[0]
		# 3. Calculate the norm of the gradients.
		norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
		gp = tf.reduce_mean((norm - 1.0) ** 2)
		return gp

	def train_step(self, data):

		for i in range(self.d_steps):
			# Get the latent vector
			X_real = generate_real_samples(self.dataset, self.batch_size)
			random_latent_vectors = tf.random.normal(
				shape=(self.batch_size, self.latent_dim)
			)
			# X_fake = generate_fake_samples(g_model, self.latent_dim, half_batch)
			

			with tf.GradientTape() as tape:
				# Generate fake images from the latent vector
				fake_images = self.generator(random_latent_vectors, training=True)
				# Get the logits for the fake images
				fake_logits = self.discriminator(fake_images, training=True)
				# Get the logits for the real images
				real_logits = self.discriminator(X_real, training=True)

				# Calculate the discriminator loss using the fake and real image logits
				d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
				# Calculate the gradient penalty
				gp = self.gradient_penalty(self.batch_size, X_real, fake_images)
				# Add the gradient penalty to the original discriminator loss
				d_loss = d_cost + gp * self.gp_weight

			# Get the gradients w.r.t the discriminator loss
			d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
			# Update the weights of the discriminator using the discriminator optimizer
			self.d_optimizer.apply_gradients(
				zip(d_gradient, self.discriminator.trainable_variables)
			)
		
		# Train the generator
		# Get the latent vector
		random_latent_vectors = tf.random.normal(
			shape=(self.batch_size, self.latent_dim)
		)
		# X_fake = generate_fake_samples(g_model, self.latent_dim, self.batch_size)
		with tf.GradientTape() as tape:
			# Generate fake images using the generator
			generated_images = self.generator(random_latent_vectors, training=True)
			# Get the discriminator logits for fake images
			gen_img_logits = self.discriminator(generated_images, training=True)
			# Calculate the generator loss
			g_loss = self.g_loss_fn(gen_img_logits)

		# Get the gradients w.r.t the generator loss
		gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
		# Update the weights of the generator using the generator optimizer
		self.g_optimizer.apply_gradients(
			zip(gen_gradient, self.generator.trainable_variables)
		)
		return {"d_loss": d_loss, "g_loss": g_loss}


class GANMonitor(tf.keras.callbacks.Callback):
	def __init__(self, num_img=16, latent_dim=128):
		self.num_img = num_img
		self.latent_dim = latent_dim

	def on_epoch_end(self, epoch, logs=None):
		n = int(self.num_img**(1/2))
		random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
		generated_images = self.model.generator(random_latent_vectors)
		generated_images = (generated_images + 1) / 2.0

		if (epoch+1)% 10 == 0:
			for i in range(n * n):
				# define subplot
				pyplot.subplot(n, n, 1 + i)
				# turn off axis
				pyplot.axis('off')
				# plot raw pixel data
				pyplot.imshow(generated_images[i])
			pyplot.savefig(f'output_file/output_{epoch+1}.png')

def load_real_data():
    folder = 'data/pokemon/image_names.csv'
    df = pd.read_csv(folder, header=None)
    return df


def generate_real_samples(df, n_samples):
	batch = df.sample(n_samples)
	im_array = []

	for _, row in batch.iterrows():
		path = row[0]
		im = Image.open('data/pokemon/'+path)
		im = im.resize(config['input_shape2'])
		im_array.append(np.array(im))

	im_array = np.array(np.float32(im_array))
	im_array = (im_array - 127.5) / 127.5
	im_array = backend.constant(im_array)

	return im_array


def generate_latent_points(latent_dim, n_samples):
	# x_input = randn(latent_dim*n_samples)
	x_input = np.random.normal(0,1,latent_dim*n_samples)
	z_input = x_input.reshape(n_samples, latent_dim)

	return z_input
 

def generate_fake_samples(generator, latent_dim, n_samples):
	z_input = generate_latent_points(latent_dim, n_samples)
	images = generator.predict(z_input)

	return images



def summarize_accuracy(dataset, n_samples, d_model, g_model):
	X_real, y_real = generate_real_samples(dataset, n_samples)
	acc_real = d_model.evaluate(X_real, y_real, verbose=0)

	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)

	print(acc_real, acc_fake)

	if is_wandb:
		wandb.log({
			"Disc real Acc": acc_real,
			"Disc fake Acc": acc_fake
		})
                


is_wandb = False
size = 64


config = {
    "epochs": 10000,
    "batch_size": 64,
	"input_shape1": (size,size,3),
	"input_shape2": (size,size),
	"smooth": 0,
	"lrelu": 0.2,
	"dropout": 0.4,
	"batch_norm": False,
    "loss_function": "binary_crossentropy",
    "d_optimizer": "RMSprop",
	"d_learning_rate": 0.0005,
	"g_learning_rate": 0.0005,
	"d_beta1": 0.5,
	"g_beta1": 0.5,
	"g_optimizer": "RMSprop",
    "dataset": "Pokemon",
	"comment": "Aplikovanie WGAN s batchnorm."
}

if is_wandb:
	wandb.login()
	run = wandb.init(project='zadanie3', entity='nsfitt-pa')
	wandb.config.update(config)

latent_dim = 100
df = load_real_data()

cbk = GANMonitor(num_img=16, latent_dim=latent_dim)

g_model = define_generator(latent_dim)
print(g_model.summary())
d_model = define_critic(config['input_shape1'])
print(d_model.summary())
# g_model, d_model = define_gan(g_model, d_model, latent_dim)

# Instantiate the optimizer for both networks
# (learning_rate=0.0002, beta_1=0.5 are recommended)
generator_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)

# Instantiate the WGAN model.
wgan = WGAN(
    discriminator=d_model,
    generator=g_model,
    latent_dim=latent_dim,
	dataset=df,
	batch_size=config['batch_size'],
    discriminator_extra_steps=3,
)

wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)

wgan.fit(df, epochs=config['epochs'], callbacks=[cbk])

if is_wandb:
	summarize_accuracy(df, config['batch_size'], d_model, g_model)
	run.finish()