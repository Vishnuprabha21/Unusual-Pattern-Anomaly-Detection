import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load the dataset from the .npy files
x_train = np.load('train.npy')
x_test = np.load('test.npy')

# Normalize the data (assuming the images are in the range [0, 255])
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add batch dimension to the data
x_train = np.expand_dims(x_train, axis=0)
x_test = np.expand_dims(x_test, axis=0)

# Check the shape of your data
print(f"Training data shape after expansion: {x_train.shape}")
print(f"Testing data shape after expansion: {x_test.shape}")

# Determine the input shape based on your data
input_shape = x_train.shape[1:]  # e.g., (32, 32, 3) if using CIFAR-10-like data

# Step 1: Build the Encoder
latent_dim = 64  # Latent space size


def build_encoder():
    encoder_inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(encoder_inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    return models.Model(encoder_inputs, [z_mean, z_log_var], name='encoder')


encoder = build_encoder()
encoder.summary()


# Continue with building the decoder, VAE model, and training process as in the previous code...


# Step 2: Build the Decoder
def build_decoder():
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 64, activation='relu')(decoder_inputs)
    x = layers.Reshape((8, 8, 64))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
    decoder_outputs = layers.Conv2DTranspose(input_shape[2], (3, 3), activation='sigmoid', padding='same')(x)

    return models.Model(decoder_inputs, decoder_outputs, name='decoder')


decoder = build_decoder()
decoder.summary()


# Step 3: Sampling Layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Step 4: Build the VAE Model
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = Sampling()([z_mean, z_log_var])
        return self.decoder(z)


vae = VAE(encoder, decoder)
vae.summary()


# Step 5: Define Loss Function
def vae_loss(x, x_reconstructed, z_mean, z_log_var):
    reconstruction_loss = tf.keras.losses.binary_crossentropy(x, x_reconstructed)
    reconstruction_loss *= input_shape[0] * input_shape[1] * input_shape[2]  # e.g., 32 * 32 * 3
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return tf.reduce_mean(reconstruction_loss + kl_loss)


vae.compile(optimizer='adam', loss=lambda x, y: vae_loss(x, vae(x), encoder(x)[0], encoder(x)[1]))

# Step 6: Train the VAE Model
vae.fit(x_train, x_train, epochs=30, batch_size=128, validation_data=(x_test, x_test))


# Step 7: Generate New Samples and Plot Them
def plot_generated_images(decoder, num_images=10):
    random_latent_vectors = np.random.normal(size=(num_images, latent_dim))
    generated_images = decoder.predict(random_latent_vectors)

    plt.figure(figsize=(20, 4))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.show()


plot_generated_images(decoder)
