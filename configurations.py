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
epochs = 10
learning_rate = 10e-05
batch_size = 20
embedding_dims = 64
num_heads = 4

# define variables for data processing
#------------------------------------------------------------------------------
picture_size = (96, 96)
num_channels = 1
image_shape = picture_size + (num_channels,)
pad_value = 0
num_samples = 2000
test_size = 0.1





