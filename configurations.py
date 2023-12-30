# Define general variables
#------------------------------------------------------------------------------
generate_model_graph = True
use_mixed_precision = True
use_tensorboard = False
XLA_acceleration = False

# Define variables for the training
#------------------------------------------------------------------------------
seed = 42
training_device = 'GPU'
epochs = 50
learning_rate = 0.001
batch_size = 25
embedding_dims = 128
num_heads = 3


# define variables for data processing
#------------------------------------------------------------------------------
picture_size = (96, 96)
num_channels = 3
image_shape = picture_size + (num_channels,)
pad_value = 0
num_samples = 800
test_size = 0.25





