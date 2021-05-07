import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot
import pandas as pd
import time
from PIL import Image
import numpy as np
from numpy.random import randn
from numpy import zeros, ones

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

EPOCHS = 800
noise_dim = 100
BATCH_SIZE = 256

# ########################################
# GENERATOR
# ########################################

def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16*16*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 256)))   
    assert model.output_shape == (None,16,16,256)
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None,16,16,128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())             

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None,32,32,64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())             

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None,64,64,3)                                            
    
    return model

generator = generator_model()

# noise = tf.random.normal([1, 100])
# generated_image = generator(noise, training=False)

# pyplot.imshow((generated_image[0, :, :, :3] + 1)/2)
# pyplot.savefig(f'output_file/fero.png')

# ########################################
# DISCRIMINATOR
# ########################################

def discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))      # 32*32*32

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))      # 16*16*64
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = discriminator_model()
# decision = discriminator(generated_image)
# print(decision)

# ########################################
# LOSSES
# ########################################

def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) 

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

# ########################################
# OPTIMIZERS
# ########################################

generator_optimizer = tf.keras.optimizers.RMSprop(5e-5)
discriminator_optimizer = tf.keras.optimizers.RMSprop(5e-5)

# ########################################
# STEP
# ########################################

def load_real_data():
    folder = 'data/pokemon_upsampled/image_names.csv'
    df = pd.read_csv(folder, header=None)

    return df

def generate_real_samples(df, n_samples):
    batch = df.sample(n_samples)
    im_array = []

    for _, row in batch.iterrows():
        path = row[0]
        im = Image.open('data/pokemon_upsampled/'+path)
        im = im.resize((64,64))
        im_array.append(np.array(im))

    im_array = np.array(np.float32(im_array))
    im_array = (im_array - 127.5) / 127.5

    y = ones((n_samples, 1))

    return im_array, y

# generate samples and save as a plot and save the model
def summarize_performance(examples, n, iter):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i])
	pyplot.savefig(f'output_file/output_{iter}.png')


def train_step(dataset, n_samples):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        images, y_real = generate_real_samples(dataset, n_samples)
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    for idx, grad in enumerate(gradients_of_discriminator):
        gradients_of_discriminator[idx] = tf.clip_by_value(grad, -0.01, 0.01)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# ########################################
# TRAIN
# ########################################

epoch_loss_avg_gen = tf.keras.metrics.Mean('g_loss')
epoch_loss_avg_disc = tf.keras.metrics.Mean('d_loss')

g_loss_results = []
d_loss_results = []

def train(dataset, epochs=8000, n_batch=256):

    batch_per_epoch = int(dataset[0].shape[0] / n_batch)

    for epoch in range(epochs):
        start = time.time()
        
        for j in range(int(batch_per_epoch)):
            g_loss, d_loss = train_step(df, n_batch)
            epoch_loss_avg_gen(g_loss)
            epoch_loss_avg_disc(d_loss)

        g_loss_results.append(epoch_loss_avg_gen.result())
        d_loss_results.append(epoch_loss_avg_disc.result())
        
        epoch_loss_avg_gen.reset_states()
        epoch_loss_avg_disc.reset_states()

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print('epoch:', epoch + 1, 'g_loss:', g_loss.numpy(), 'd_loss:', d_loss.numpy())

        if (epoch + 1)%5 == 0:
            noise = tf.random.normal([16, 100])
            X = generator(noise, training=False)
            X = (X + 1) / 2.0
            summarize_performance(X, 4, epoch+1)
            

df = load_real_data()
train(df)       