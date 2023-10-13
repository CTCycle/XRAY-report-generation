import os
import sys
import pandas as pd
import tensorflow as tf
import visualkeras
import PIL
from keras.utils import plot_model
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# [IMPORT MODULES AND CLASSES]
#==============================================================================
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.components.data_classes import XRAYDataSet
from modules.components.training_classes import PreProcessing, TrainingTools, RealTimeHistory, DataGenerator, ModelXREP
import modules.global_variables as GlobVar

# [LOAD DATA AND ADD IMAGES PATHS TO DATASET]
#==============================================================================
# Load the csv with data and transform the tokenized text column to convert the
# strings into a series of integers
#==============================================================================
print('''
-------------------------------------------------------------------------------
XRAYREP training
-------------------------------------------------------------------------------
XRAYREP model will be trained on the preprocessed data...
''')

dataworker = XRAYDataSet()
file_loc = os.path.join(GlobVar.data_path, 'dataset_training_XRAY.csv') 
df_train = pd.read_csv(file_loc, encoding = 'utf-8', sep = (';' or ',' or ' ' or  ':'), low_memory=False)
file_loc = os.path.join(GlobVar.data_path, 'dataset_test_XRAY.csv') 
df_test = pd.read_csv(file_loc, encoding = 'utf-8', sep = (';' or ',' or ' ' or  ':'), low_memory=False)

# assign paths to images in the dataset
#------------------------------------------------------------------------------
df_train = dataworker.images_pathfinder(GlobVar.images_path, df_train, 'id')
df_test = dataworker.images_pathfinder(GlobVar.images_path, df_test, 'id')

# [DEFINE ACTUAL BATCH SIZE AND LOAD TOKENIZERS]
#==============================================================================
# Load the csv with data and transform the tokenized text column to convert the
# strings into a series of integers
#==============================================================================
preprocessor = PreProcessing()

# define batch size based on sample size
#------------------------------------------------------------------------------
true_batch_size = preprocessor.batch_size_finder(GlobVar.batch_size, df_train.shape[0])

# load tokenizer to get padding length and vocabulary size
#------------------------------------------------------------------------------
tokenizer_path = os.path.join(GlobVar.data_path, 'Tokenizers')
tokenizer = preprocessor.load_tokenizer(tokenizer_path, 'word_tokenizer')
vocab_size = len(tokenizer.word_index)


# [ESTABLISH DATA GENERATOR]
#==============================================================================
# module for the selection of different operations
#==============================================================================

# transform string sequence into list of elements
#------------------------------------------------------------------------------
train_datagen = DataGenerator(df_train, GlobVar.batch_size, GlobVar.pic_size[:-1], shuffle=True)
test_datagen = DataGenerator(df_test, GlobVar.batch_size, GlobVar.pic_size[:-1], shuffle=True)

# define the output signature of the generator using tf.TensorSpec
#------------------------------------------------------------------------------
x_batch, y_batch = train_datagen.__getitem__(0)
img_shape = x_batch[0].shape
tokenseq_shape = x_batch[1].shape
caption_shape = y_batch.shape
output_signature = ((tf.TensorSpec(shape=img_shape, dtype=tf.float32),
                     tf.TensorSpec(shape=tokenseq_shape, dtype=tf.float32)),
                     tf.TensorSpec(shape=caption_shape, dtype=tf.float32))

# generate tf.dataset from generator and set prefetch
#------------------------------------------------------------------------------
train_dataset = tf.data.Dataset.from_generator(lambda : train_datagen, output_signature=output_signature)
test_dataset = tf.data.Dataset.from_generator(lambda : test_datagen, output_signature=output_signature)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# [PRINT STATISTICAL REPORT]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
Number of train samples: {df_train.shape[0]}
Number of test samples:  {df_test.shape[0]}
Target batch size:       {GlobVar.batch_size}
Actual batch size:       {true_batch_size}
Vocabulary size:         {vocab_size}
Caption length:          {caption_shape[1]} 
-------------------------------------------------------------------------------
''')

# [BUILD XREPORT MODEL]
#==============================================================================
# ....
#==============================================================================
XREP_model = ModelXREP(GlobVar.learning_rate, GlobVar.features_vector, 
                  GlobVar.pic_size, GlobVar.embedding_dims, 
                  GlobVar.RNN_dims, vocab_size, true_batch_size)

# compile after initialization (due to model being defined as class)
#------------------------------------------------------------------------------
XREP_model.compile(optimizer=XREP_model.optimizer, loss=XREP_model.loss_object)
XREP_model(x_batch[0], x_batch[1])

# generate graphviz plot fo the model layout
#------------------------------------------------------------------------------
model_savepath = preprocessor.model_savefolder(GlobVar.model_path, 'XREP')
plot_path = os.path.join(model_savepath, 'XREP_scheme.png')       
plot_model(XREP_model, to_file = plot_path, show_shapes = True, 
           show_layer_names = True, show_layer_activations = True, 
           expand_nested = True, rankdir = 'TB', dpi = 400) 

plot_path = os.path.join(model_savepath, 'XREP_model_3D.png')
visualkeras.layered_view(XREP_model, to_file = plot_path) 


# [TRAINING XREPORT MODEL]
#==============================================================================
# Setting callbacks and training routine for the features extraction model. 
# use command prompt on the model folder and (upon activating environment), 
# use the bash command: python -m tensorboard.main --logdir = tensorboard/
#==============================================================================

# define callbacks
#------------------------------------------------------------------------------
log_path = os.path.join(model_savepath, 'tensorboard')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
RTH_callback = RealTimeHistory(model_savepath, validation=False)

# training loop and model saving at end
#------------------------------------------------------------------------------
print(f'''Start model training for {GlobVar.epochs} epochs and batch size of {true_batch_size}
       ''')
trainworker = TrainingTools(device = 'GPU')
steps_per_epoch = int(df_train.shape[0]/true_batch_size)
training = XREP_model.fit(train_dataset, steps_per_epoch=steps_per_epoch,
                            validation_data = test_dataset, epochs = GlobVar.epochs, 
                             callbacks = [RTH_callback, tensorboard_callback],
                             workers = 6, use_multiprocessing=True)                          

XREP_model.save(model_savepath)


# [FEXT MODEL VALIDATION]
#==============================================================================
# Training the LSTM model using the functions specified in the designated class.
# The model is saved using keras saving procedures in order to store weights and
# other information
#==============================================================================
# validator = ModelValidation(FEXT_model)
# num_val_images = 10
# validation_subset = dataset_XRAYREP.sample(n=num_val_images)

# # extract batch of real and reconstructed images and perform visual validation
# #------------------------------------------------------------------------------
# val_generator = trainworker.FeatEXT_generator(validation_subset, 'images_path', GlobVar.FEXT_pic_size[:-1],
#                                               num_val_images, transform=False, shuffle=False)
# original_images, y_val = val_generator.next()
# recostructed_images = list(FEXT_model.predict(original_images))
# validator.FeatEXT_validation(original_images, recostructed_images, model_savepath)


