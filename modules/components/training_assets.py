import os
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Input, Reshape, Embedding, MultiHeadAttention
from keras import layers 
from keras.utils import plot_model
from tqdm import tqdm

# set environment variables
#------------------------------------------------------------------------------
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'

    
# [CALLBACK FOR REAL TIME TRAINING MONITORING]
#==============================================================================
# Real time history callback
#==============================================================================
class RealTimeHistory(keras.callbacks.Callback):
    
    """ 
    A class including the callback to show a real time plot of the training history. 
      
    Methods:
        
    __init__(plot_path): initializes the class with the plot savepath       
    
    """   
    def __init__(self, plot_path, validation=True):        
        super().__init__()
        self.plot_path = plot_path
        self.epochs = []
        self.loss_hist = []
        self.metric_hist = []
        self.loss_val_hist = []        
        self.metric_val_hist = []
        self.validation = validation            
    #--------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs = {}):
        if epoch % 10 == 0:                    
            self.epochs.append(epoch)
            self.loss_hist.append(logs[list(logs.keys())[0]])
            self.metric_hist.append(logs[list(logs.keys())[1]])
            if self.validation==True:
                self.loss_val_hist.append(logs[list(logs.keys())[2]])            
                self.metric_val_hist.append(logs[list(logs.keys())[3]])
        if epoch % 50 == 0:            
            #------------------------------------------------------------------
            fig_path = os.path.join(self.plot_path, 'training_history.jpeg')
            plt.subplot(2, 1, 1)
            plt.plot(self.epochs, self.loss_hist, label='training loss')
            if self.validation==True:
                plt.plot(self.epochs, self.loss_val_hist, label='validation loss')
                plt.legend(loc='best', fontsize = 8)
            plt.title('Loss plot')
            plt.ylabel('Categorical Crossentropy')
            plt.xlabel('epoch')
            plt.subplot(2, 1, 2)
            plt.plot(self.epochs, self.metric_hist, label='train metrics') 
            if self.validation==True: 
                plt.plot(self.epochs, self.metric_val_hist, label='validation metrics') 
                plt.legend(loc='best', fontsize = 8)
            plt.title('metrics plot')
            plt.ylabel('Categorical accuracy')
            plt.xlabel('epoch')       
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches='tight', format='jpeg', dpi = 300)
            plt.close()    

# [CALLBACK FOR LEARNING RATE SCHEDULER]
#==============================================================================
# learning rate scheduler callback
#==============================================================================
class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress

        return tf.cond(global_step < warmup_steps, lambda: warmup_learning_rate,
                       lambda: self.post_warmup_learning_rate)


# [CUSTOM DATA GENERATOR FOR TRAINING]
#==============================================================================
# Generate data on the fly to avoid memory burdening
#==============================================================================
class DataGenerator(keras.utils.Sequence):

    def __init__(self, dataframe, batch_size=6, image_size=(244, 244), channels=3, shuffle=True):        
        self.dataframe = dataframe
        self.path_col='images_path'        
        self.label_col='tokenized_text'
        self.num_of_samples = dataframe.shape[0]
        self.num_channels = channels
        self.image_size = image_size       
        self.batch_size = batch_size  
        self.batch_index = 0              
        self.shuffle = shuffle
        self.on_epoch_end()       

    # define length of the custom generator      
    #--------------------------------------------------------------------------
    def __len__(self):
        length = int(np.floor(self.num_of_samples)/self.batch_size)

        return length
    
    # define method to get X and Y data through custom functions, and subsequently
    # create a batch of data converted to tensors
    #--------------------------------------------------------------------------
    def __getitem__(self, idx): 
        path_batch = self.dataframe[self.path_col][idx * self.batch_size:(idx + 1) * self.batch_size]        
        label_batch = self.dataframe[self.label_col][idx * self.batch_size:(idx + 1) * self.batch_size]
        x1_batch = [self.__images_generation(image_path) for image_path in path_batch]
        x2_batch = [self.__labels_generation(label_id) for label_id in label_batch] 
        y_batch = [self.__labels_generation(label_id) for label_id in label_batch]
        X1_tensor = tf.convert_to_tensor(x1_batch)
        X2_tensor = tf.convert_to_tensor(x2_batch)
        Y_tensor = tf.convert_to_tensor(y_batch)

        return (X1_tensor, X2_tensor), Y_tensor
    
    # define method to perform data operations on epoch end
    #--------------------------------------------------------------------------
    def on_epoch_end(self):        
        self.indexes = np.arange(self.num_of_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # define method to load images and perform data augmentation    
    #--------------------------------------------------------------------------
    def __images_generation(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=self.num_channels)
        resized_image = tf.image.resize(image, self.image_size)
        if self.num_channels==3:
            resized_image = tf.reverse(resized_image, axis=[-1])
        norm_image = resized_image/255.0              
        pp_image = tf.keras.preprocessing.image.random_shift(norm_image, 0.2, 0.3)
        pp_image = tf.image.random_flip_left_right(pp_image)        

        return pp_image       
    
    # define method to load labels    
    #--------------------------------------------------------------------------
    def __labels_generation(self, sequence):
        pp_sequence = np.array(sequence.split(' '), dtype=np.float32) 

        return pp_sequence
    
    # define method to call the elements of the generator    
    #--------------------------------------------------------------------------
    def next(self):
        next_index = (self.batch_index + 1) % self.__len__()
        self.batch_index = next_index

        return self.__getitem__(next_index)

  
# [IMAGE ENCODER MODEL]
#==============================================================================
# Custom encoder model
#==============================================================================    
class ImageEncoder(layers.Layer):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.conv1 = Conv2D(64, 6, strides=1, padding='same', 
                            activation='relu', kernel_initializer='he_uniform')        
        self.conv2 = Conv2D(128, 6, strides=1, padding='same', 
                            activation='relu', kernel_initializer='he_uniform')        
        self.conv3 = Conv2D(256, 6, strides=1, padding='same', 
                            activation='relu', kernel_initializer='he_uniform')  
        self.conv4 = Conv2D(256, 6, strides=1, padding='same', 
                            activation='relu', kernel_initializer='he_uniform')        
        self.conv5 = Conv2D(512, 6, strides=1, padding='same', 
                            activation='relu', kernel_initializer='he_uniform') 
        self.conv6 = Conv2D(512, 6, strides=1, padding='same', 
                            activation='relu', kernel_initializer='he_uniform')        
        self.maxpool1 = MaxPooling2D((2, 2), padding='same')
        self.maxpool2 = MaxPooling2D((2, 2), padding='same')
        self.maxpool3 = MaxPooling2D((2, 2), padding='same')
        self.maxpool4 = MaxPooling2D((2, 2), padding='same')
        self.reshape = Reshape((-1, 512))   

    # implement encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, x):        
        layer = self.conv1(x)                  
        layer = self.maxpool1(layer) 
        layer = self.conv2(layer)                     
        layer = self.maxpool2(layer)        
        layer = self.conv3(layer)  
        layer = self.conv4(layer)                        
        layer = self.maxpool3(layer)                
        layer = self.conv5(layer) 
        layer = self.conv6(layer)               
        layer = self.maxpool4(layer) 
        layer = self.reshape(layer)       
        
        return layer

# [POSITIONAL EMBEDDING]
#==============================================================================
# Custom positional embedding layer
#==============================================================================
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embedding_dims, mask_zero=True):
        super(PositionalEmbedding, self).__init__()
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embedding_dims
        self.mask_zero = mask_zero
        self.token_embeddings = Embedding(input_dim=vocab_size, output_dim=embedding_dims)        
        self.position_embeddings = Embedding(input_dim=sequence_length, output_dim=embedding_dims)        
        self.embedding_scale = tf.math.sqrt(tf.cast(embedding_dims, tf.float32))
        
    
    # implement positional embedding through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = embedded_tokens * self.embedding_scale
        embedded_positions = self.position_embeddings(positions)
        full_embedding = embedded_tokens + embedded_positions        
        if self.mask_zero==True:
            mask = tf.math.not_equal(inputs, 0)
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)            
            full_embedding = full_embedding * mask

        return full_embedding
    
    # compute the mask for padded sequences  
    #--------------------------------------------------------------------------
    def compute_mask(self, inputs, mask=None):

        return tf.math.not_equal(inputs, 0)

    

# [TRANSFORMER ENCODER]
#==============================================================================
# Custom transformer encoder
#============================================================================== 
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embedding_dims, num_heads):
        super(TransformerEncoderBlock, self).__init__()
        self.embedding_dims = embedding_dims       
        self.num_heads = num_heads        
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.embedding_dims)
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dense = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')

    # implement transformer encoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, training):        
        inputs = self.layernorm1(inputs)
        inputs = self.dense(inputs)       
        attention_output = self.attention(query=inputs, value=inputs, key=inputs,
                                          attention_mask=None, training=training)
        
        output = self.layernorm2(inputs + attention_output)

        return output

# [TRANSFORMER DECODER]
#==============================================================================
# Custom transformer decoder
#============================================================================== 
class TransformerDecoderBlock(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embedding_dims, num_heads):
        super(TransformerDecoderBlock, self).__init__()
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims        
        self.num_heads = num_heads          
        self.MHA_1 = MultiHeadAttention(num_heads=num_heads, key_dim=self.embedding_dims, dropout=0.1)
        self.MHA_2 = MultiHeadAttention(num_heads=num_heads, key_dim=self.embedding_dims, dropout=0.1)
        self.FFN_1 = Dense(512, activation='relu')
        self.FFN_2 = Dense(self.embedding_dims)
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.layernorm3 = layers.LayerNormalization()        
        self.outmax = layers.Dense(self.vocab_size, activation='softmax')
        self.dropout1 = layers.Dropout(0.3)
        self.dropout2 = layers.Dropout(0.5)  

    # implement transformer decoder through call method  
    #--------------------------------------------------------------------------
    def call(self, inputs, encoder_outputs, training, mask):
        
        causal_mask = self.get_causal_attention_mask(inputs)
        
        padding_mask = None
        combined_mask = None

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)            
        
        attention_output1 = self.MHA_1(query=inputs, value=inputs, key=inputs,
                                       attention_mask=combined_mask, training=training)
        output1 = self.layernorm1(inputs + attention_output1)                       
        attention_output2 = self.MHA_2(query=output1, value=encoder_outputs,
                                       key=encoder_outputs, attention_mask=padding_mask,
                                       training=training)
        output2 = self.layernorm2(output1 + attention_output2)
        ffn_out = self.FFN_1(output2)
        ffn_out = self.dropout1(ffn_out, training=training)
        ffn_out = self.FFN_2(ffn_out)
        ffn_out = self.layernorm3(ffn_out + output2, training=training)
        ffn_out = self.dropout2(ffn_out, training=training)
        print(ffn_out)
        preds = self.outmax(ffn_out)

        return preds

    # generate causal attention mask   
    #--------------------------------------------------------------------------
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
                          axis=0)
        
        return tf.tile(mask, mult)
    

 

# [XREP CAPTIONING MODEL]
#==============================================================================
# Custom captioning model
#==============================================================================  
class XREPCaptioningModel(keras.Model):    
    def __init__(self, pic_shape, sequence_length, vocab_size, embedding_dims, num_heads,
                 learning_rate, XLA_state):   
        super(XREPCaptioningModel, self).__init__()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.pic_shape = pic_shape
        self.sequence_length = sequence_length 
        self.learning_rate = learning_rate
        self.XLA_state = XLA_state 
        self.posembedding = PositionalEmbedding(sequence_length, vocab_size, embedding_dims, mask_zero=True)                 
        self.image_encoder = ImageEncoder()        
        self.encoder = TransformerEncoderBlock(embedding_dims, num_heads)
        self.decoder = TransformerDecoderBlock(sequence_length, vocab_size, embedding_dims, num_heads) 

    # calculate loss
    #--------------------------------------------------------------------------
    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask) 
    
    # calculate accuracy
    #--------------------------------------------------------------------------
    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    # calculate the caption loss and accuracy
    #--------------------------------------------------------------------------
    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        encoder_out = self.encoder(img_embed, training=training)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(batch_seq_inp, encoder_out, training=training, mask=mask)
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

        return loss, acc
    
    # define train step
    #--------------------------------------------------------------------------
    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0       
        img_embed = self.image_encoder(batch_img)
        with tf.GradientTape() as tape:
            loss, acc = self._compute_caption_loss_and_acc(img_embed, batch_seq, training=True)
            batch_loss += loss
            batch_acc += acc           
            train_vars = (self.encoder.trainable_variables + self.decoder.trainable_variables)           
            grads = tape.gradient(loss, train_vars)
            self.optimizer.apply_gradients(zip(grads, train_vars))      
        
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        return {"loss": self.loss_tracker.result(),
                "acc": self.acc_tracker.result()}

    # define test step
    #--------------------------------------------------------------------------
    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0
        img_embed = self.image_encoder(batch_img)
        with tf.GradientTape() as tape:
            loss, acc = self._compute_caption_loss_and_acc(img_embed, batch_seq, training=False)
            batch_loss += loss
            batch_acc += acc            
       
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)
        
        return {"loss": self.loss_tracker.result(),
                "acc": self.acc_tracker.result()}

    @property
    def metrics(self):        
        return [self.loss_tracker, self.acc_tracker]
 
    # implement captioning model through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs, training):
        images, sequences = inputs 
        mask = tf.math.not_equal(sequences, 0)      
        image_features = self.image_encoder(images)
        sequences = self.posembedding(sequences)
        encoder = self.encoder(image_features, training)
        decoder = self.decoder(sequences, encoder, training, mask)

        return decoder
        
    
    # track metrics and losses  
    #--------------------------------------------------------------------------
    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]
        
        
    
    # compile the model
    #--------------------------------------------------------------------------
    def compile(self):
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)  
        metric = keras.metrics.SparseCategoricalAccuracy()  
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)          
        super(XREPCaptioningModel, self).compile(optimizer=opt, loss=loss, metrics=metric, jit_compile=self.XLA_state)

    # print summary
    #--------------------------------------------------------------------------
    def summary(self):
        image_input = Input(shape=self.pic_shape)    
        seq_input = Input(shape=(self.sequence_length, ))
        model = Model(inputs=[image_input, seq_input], 
                      outputs = self.call([image_input, seq_input], 
                      training=False))
        
        model.summary()

    # plot model as 2D schematics
    #--------------------------------------------------------------------------
    def plot_model(self, model_savepath):
        image_input = Input(shape=self.pic_shape)    
        seq_input = Input(shape=(self.sequence_length, ))
        model = Model(inputs=[image_input, seq_input], 
                      outputs = self.call([image_input, seq_input], 
                      training=False))
        plot_path = os.path.join(model_savepath, 'XREP_scheme.png')       
        plot_model(model, to_file = plot_path, show_shapes = True, 
               show_layer_names = True, show_layer_activations = True, 
               expand_nested = True, rankdir='TB', dpi = 400)  



    

   

    

# [TRAINING OPTIONS]
#==============================================================================
# Custom training operations
#==============================================================================
class ModelTraining:    
       
    def __init__(self, device='default', seed=42, use_mixed_precision=False):                            
        np.random.seed(seed)
        tf.random.set_seed(seed)         
        self.available_devices = tf.config.list_physical_devices()
        print('-------------------------------------------------------------------------------')        
        print('The current devices are available: ')
        print('-------------------------------------------------------------------------------')
        for dev in self.available_devices:
            print()
            print(dev)
        print()
        print('-------------------------------------------------------------------------------')
        if device == 'GPU':
            self.physical_devices = tf.config.list_physical_devices('GPU')
            if not self.physical_devices:
                print('No GPU found. Falling back to CPU')
                tf.config.set_visible_devices([], 'GPU')
            else:
                if use_mixed_precision == True:
                    policy = keras.mixed_precision.Policy('mixed_float16')
                    keras.mixed_precision.set_global_policy(policy) 
                tf.config.set_visible_devices(self.physical_devices[0], 'GPU')                 
                print('GPU is set as active device')
            print('-------------------------------------------------------------------------------')
            print()        
        elif device == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            print('CPU is set as active device')
            print('-------------------------------------------------------------------------------')
            print()        
    
    #-------------------------------------------------------------------------- 
    def model_parameters(self, parameters_dict, savepath):

        '''
        Saves the model parameters to a JSON file. The parameters are provided 
        as a dictionary and are written to a file named 'model_parameters.json' 
        in the specified directory.

        Keyword arguments:
            parameters_dict (dict): A dictionary containing the parameters to be saved.
            savepath (str): The directory path where the parameters will be saved.

        Returns:
            None       

        '''
        path = os.path.join(savepath, 'model_parameters.json')      
        with open(path, 'w') as f:
            json.dump(parameters_dict, f)         
    
    
    
# [TOOLKIT TO USE THE PRETRAINED MODEL]
#==============================================================================
# Custom training operations
#==============================================================================
class InferenceTools:

    #--------------------------------------------------------------------------
    def load_pretrained_model(self, path, load_parameters=True):

        '''
        Load pretrained keras model (in folders) from the specified directory. 
        If multiple model directories are found, the user is prompted to select one,
        while if only one model directory is found, that model is loaded directly.
        If `load_parameters` is True, the function also loads the model parameters 
        from the target .json file in the same directory. 

        Keyword arguments:
            path (str): The directory path where the pretrained models are stored.
            load_parameters (bool, optional): If True, the function also loads the 
                                              model parameters from a JSON file. 
                                              Default is True.

        Returns:
            model (keras.Model): The loaded Keras model.

        '''        
        model_folders = []
        for entry in os.scandir(path):
            if entry.is_dir():
                model_folders.append(entry.name)
        if len(model_folders) > 1:
            model_folders.sort()
            index_list = [idx + 1 for idx, item in enumerate(model_folders)]     
            print('Please select a pretrained model:') 
            print()
            for i, directory in enumerate(model_folders):
                print(f'{i + 1} - {directory}')        
            print()               
            while True:
                try:
                    dir_index = int(input('Type the model index to select it: '))
                    print()
                except:
                    continue
                break                         
            while dir_index not in index_list:
                try:
                    dir_index = int(input('Input is not valid! Try again: '))
                    print()
                except:
                    continue
            self.model_path = os.path.join(path, model_folders[dir_index - 1])

        elif len(model_folders) == 1:
            self.model_path = os.path.join(path, model_folders[0])            
        
        model = keras.models.load_model(self.model_path)
        if load_parameters==True:
            path = os.path.join(self.model_path, 'model_parameters.json')
            with open(path, 'r') as f:
                self.model_configuration = json.load(f)            
        
        return model   

    #--------------------------------------------------------------------------    
    def generate_caption(self, model, paths, num_channels, image_size,
                         max_length):
        
        for pt in tqdm(paths):
            image = tf.io.read_file(pt)
            image = tf.image.decode_image(image, channels=num_channels)
            image = tf.image.resize(image, image_size)
            if self.num_channels==3:
                image = tf.reverse(image, axis=[-1])
            image = image/255.0  

            input_image = tf.expand_dims(image, 0)
            features = model.image_encoder(input_image)            
            encoded_img = model.encoder(features, training=False)

            decoded_caption = '[START]'
            for i in range(max_length):
                tokenized_caption = vectorization([decoded_caption])[:, :-1]
                mask = tf.math.not_equal(tokenized_caption, 0)
                predictions = caption_model.decoder(
                    tokenized_caption, encoded_img, training=False, mask=mask
                )
                sampled_token_index = np.argmax(predictions[0, i, :])
                sampled_token = index_lookup[sampled_token_index]
                if sampled_token == "<end>":
                    break
                decoded_caption += " " + sampled_token

        decoded_caption = decoded_caption.replace("<start> ", "")
        decoded_caption = decoded_caption.replace(" <end>", "").strip()
        print("Predicted Caption: ", decoded_caption)



    
# [VALIDATION OF PRETRAINED MODELS]
#==============================================================================
# Validation and evaluation of model performances
#==============================================================================
class ModelValidation:

    def __init__(self, model):
        
        self.model = model

    # sequential model as generator with Keras module
    #========================================================================== 
    def XREPORT_validation(self, real_images, predicted_images, path):
        
        num_pics = len(real_images)
        fig_path = os.path.join(path, 'FEXT_validation.jpeg')
        fig, axs = plt.subplots(num_pics, 2, figsize=(4, num_pics * 2))
        for i, (real, pred) in enumerate(zip(real_images, predicted_images)):            
            axs[i, 0].imshow(real)
            if i == 0:
                axs[i, 0].set_title('Original picture')
            axs[i, 0].axis('off')
            axs[i, 1].imshow(pred)
            if i == 0:
                axs[i, 1].set_title('Reconstructed picture')
            axs[i, 1].axis('off')
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches='tight', format='jpeg', dpi=400)
        plt.show(block=False)
        plt.close()
