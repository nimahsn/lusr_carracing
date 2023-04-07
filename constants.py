# General constants for the project
DOMAIN_SPECIFIC_LATENT_SIZE = 8
CONTENT_LATENT_SIZE = 32
NUM_DOMAINS = 4 # 1 source, others are target
INPUT_SHAPE = (64, 64, 3)

# PPO constants
NUM_ACTIONS = 5
NUM_WORKERS_PPO = 4

# VAE constants
VAE_EPOCHS = 2
INIT_LR = 1e-4
BATCH_SIZE = 32
KL_LOSS_WEIGHT = 10.0
