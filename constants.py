# General constants for the project
DOMAIN_SPECIFIC_LATENT_SIZE = 8
CONTENT_LATENT_SIZE = 16
NUM_DOMAINS = 5 # 1 source, others are target
INPUT_SHAPE = (64, 64, 3)

# PPO constants
NUM_ACTIONS = 5
NUM_WORKERS_PPO = 4

# VAE constants
VAE_EPOCHS = 2
INIT_LR = 1e-4
BATCH_SIZE = 20
KL_LOSS_WEIGHT = 100.0
REVERSE_LOSS_WEIGHT = 20.0
FORWARD_RECONS_LOSS_WEIGHT = 1.0
FORWARD_RECONS_SHUFFLE_LOSS_WEIGHT = 1.0
