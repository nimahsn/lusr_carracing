import tensorflow as tf
import constants as c

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
            tf.keras.layers.Conv2DTranspose(3, 6, strides=2, activation='sigmoid'),
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
    
    