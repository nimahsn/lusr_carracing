import tensorflow as tf
import constants as c
import matplotlib.pyplot as plt

class Encoder(tf.keras.Model):
    """
    Encoder network for VAE.
    """

    def __init__(self, mu_only):
        super(Encoder, self).__init__()
        self.mu_only = mu_only
        self.main = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(128, 4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(256, 4, strides=2, activation='relu'),
            tf.keras.layers.Flatten(),
        ])
        self.mu = tf.keras.layers.Dense(c.CONTENT_LATENT_SIZE)
        self.logvar = tf.keras.layers.Dense(c.CONTENT_LATENT_SIZE)
        self.domain_code = tf.keras.layers.Dense(c.DOMAIN_SPECIFIC_LATENT_SIZE)

    def call(self, x):
        """
        Forward pass of the encoder network. If mu_only is True, only the mean of the latent distribution is returned.
        """

        x = self.main(x)
        mu = self.mu(x)
        if self.mu_only:
            return mu
        logsigma = self.logvar(x)
        domain_code = self.domain_code(x)
        return mu, logsigma, domain_code
    
class Decoder(tf.keras.Model):
    """
    Decoder network for VAE.
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.main = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(c.DOMAIN_SPECIFIC_LATENT_SIZE + c.CONTENT_LATENT_SIZE,)),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Reshape((1, 1, 1024)),
            tf.keras.layers.Conv2DTranspose(128, 5, strides=2, activation='relu'),
            tf.keras.layers.Conv2DTranspose(64, 5, strides=2, activation='relu'),
            tf.keras.layers.Conv2DTranspose(32, 6, strides=2, activation='relu'),
            tf.keras.layers.Conv2DTranspose(3, 6, strides=2, activation='sigmoid')
        ])

    def call(self, x):
        """
        Forward pass of the decoder network.
        """
        x = self.main(x)
        return x
    
class ActorNetwork(tf.keras.Model):
    """
    Actor network for PPO.
    """

    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.main = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(c.CONTENT_LATENT_SIZE,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(c.NUM_ACTIONS, activation='softmax')
        ])

    def call(self, x):
        """
        Forward pass of the actor network.
        """
        x = self.main(x)
        return x
    
class CriticNetwork(tf.keras.Model):
    """
    Critic network for PPO.
    """

    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.main = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(c.CONTENT_LATENT_SIZE,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, x):
        """
        Forward pass of the critic network.
        """
        x = self.main(x)
        return x
    
class DisentangleVAE(tf.keras.Model):
    """
    Disentangled VAE.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(DisentangleVAE, self).__init__()
        assert encoder.mu_only == False, "Encoder must return mu, logsigma, domain_code."
        self.encoder = encoder
        self.decoder = decoder
        self.content_latent_size = c.CONTENT_LATENT_SIZE
        self.domain_specific_latent_size = c.DOMAIN_SPECIFIC_LATENT_SIZE
        self.kl_loss_weight = tf.Variable(c.KL_LOSS_WEIGHT, trainable=False, dtype=tf.float32)

    @tf.function
    def reparameterize(self, mu, logsigma):
        """
        Reparameterization trick to sample from a Gaussian distribution.
        """
        eps = tf.random.normal(shape=tf.shape(mu))

        return mu + tf.exp(logsigma / 2) * eps
    
    @tf.function
    def encode(self, x):
        """
        Encode an image to its latent representation.
        """
        mu, logsigma, domain_code = self.encoder(x)
        return mu, logsigma, domain_code
    
    @tf.function
    def encode_concat(self, x):
        """
        Encode an image to its latent representation. different from encode in that it returns the concatenatenation of mu, logsigma and domain_code.
        """
        mu, logsigma, domain_code = self.encoder(x)
        return tf.concat([mu, logsigma, domain_code], axis=1)
    
    @tf.function
    def decode(self, z, domain_code):
        """
        Decode a latent representation to an image.
        """
        return self.decoder(tf.concat([z, domain_code], axis=1))
    
    @tf.function
    def forward_cycle(self, x):
        """
        Forward cycle of the VAE.

        args:
            x: a batch of images from the same domain.
        """

        mu, logsigma, domain_code = self.encode(x)
        z = self.reparameterize(mu, logsigma)
        shuffled_domain_code = tf.random.shuffle(domain_code)
        
        x_recon = self.decode(z, domain_code)
        x_recon_shuffled = self.decode(z, shuffled_domain_code)

        mse_reconstruction = tf.reduce_mean(tf.square(x - x_recon))
        mse_reconstruction_shuffled = tf.reduce_mean(tf.square(x - x_recon_shuffled))
        kl_divergence = -0.5 * tf.reduce_mean(1 + logsigma - tf.square(mu) - tf.exp(logsigma))
        
        return mse_reconstruction + mse_reconstruction_shuffled + self.kl_loss_weight * kl_divergence

    @tf.function
    def reverse_cycle(self, x):
        """
        Reverse cycle of the VAE.

        args:
            x: a batch of images from all domains with shape (batch_size * num_domains, height, width, channels).
        """

        # no gradient through the encoder
        out = tf.stop_gradient(self.encode_concat(x))
        mu, logsigma, domain_code = tf.split(out, [self.content_latent_size, self.content_latent_size, self.domain_specific_latent_size], axis=1)
        mu_random = tf.random.normal(shape=tf.shape(mu))
        shuffled_domain_code = tf.random.shuffle(domain_code)

        x_recon = tf.stop_gradient(self.decode(mu_random, domain_code))
        x_recon_shuffled = tf.stop_gradient(self.decode(mu_random, shuffled_domain_code))
        mu_refeed, logsigma_refeed, _ = self.encode(x_recon)
        mu_refeed_shuffled, logsigma_refeed_shuffled, _ = self.encode(x_recon_shuffled)
        mu_refeed = self.reparameterize(mu_refeed, logsigma_refeed)
        mu_refeed_shuffled = self.reparameterize(mu_refeed_shuffled, logsigma_refeed_shuffled)

        return tf.reduce_mean(tf.abs(mu - mu_refeed))

    @tf.function
    def train_step(self, data):
        """
        Train step for the VAE.
        """
        with tf.GradientTape() as tape:
            x = data # tuple of size (num_domains) with each element of shape (batch_size, height, width, channels)
            x = tf.stack(x, axis=0) # shape (num_domains, batch_size, height, width, channels)
            
            # forward cycle
            forward_cycle_loss = 0.0
            for i in range(c.NUM_DOMAINS):
                forward_cycle_loss += self.forward_cycle(x[i])
            forward_cycle_loss /= c.NUM_DOMAINS

            # reverse cycle
            x = tf.reshape(x, [-1, 64, 64, 3]) # shape (num_domains * batch_size, height, width, channels)
            reverse_cycle_loss = self.reverse_cycle(x)

            # total loss
            loss = forward_cycle_loss + reverse_cycle_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {"loss": loss, "forward_cycle_loss": forward_cycle_loss, "reverse_cycle_loss": reverse_cycle_loss}