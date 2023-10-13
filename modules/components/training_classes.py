import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import layers 



# [PREPROCESSING PIPELINE]
#==============================================================================
#==============================================================================
#==============================================================================
class PreProcessing:
    
    def __init__(self): 
        pass      
        
    #==========================================================================
    def batch_size_finder(self, target, samples):

        divisors = []
        for i in range(1, samples+1):
            if samples % i == 0:
                divisors.append(i)
        delta_bs = [abs(target - x) for x in divisors]
        min_index = delta_bs.index(min(delta_bs))
        refined_bs = divisors[min_index]       

        return refined_bs     

    #==========================================================================
    def text_preparation(self, strings):

        '''
        text_preparation(strings)

        Prepares a list of strings for tokenization by converting them to lowercase, 
        adding spaces around punctuation symbols and delimiting the strings with start
        and end sequence tokens 

        Keyword arguments:
            strings (list): A list of strings to be prepared.

        Returns:
            prepared_strings (list): A list of prepared strings.
        
        '''
        symbols = ['.', ',', ';', ':', '"', '-']       
        prepared_strings = []
        for st in strings:
            string = st.lower()        
            for sym in symbols:
                string = string.replace(sym, f' {sym} ')
            delimited_str = 'START ' + string + ' END'
            prepared_strings.append(delimited_str)

        return prepared_strings
    
    #==========================================================================
    def text_tokenization(self, text, savepath, matrix_output=False):

        '''
        text_tokenization(text, savepath, matrix_output=False)

        Tokenizes a list of texts and saves the tokenizer to a specified path.

        Keyword arguments:
            text (list): A list of texts to be tokenized.
            savepath (str): The path to save the tokenizer as a JSON file.
            matrix_output (bool): Whether to return the tokenized texts as a matrix 
            or a list of sequences. If True, the function returns a matrix of TF-IDF scores. 
            If False, the function returns a list of sequences of token indices.

        Returns:
            tokenized_text (list or numpy.ndarray): The tokenized texts in the specified output format.
        
        '''
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(text)
        tokenized_text = self.tokenizer.texts_to_sequences(text)
        if matrix_output == True:
            tokenized_text = self.tokenizer.texts_to_matrix(text, mode = 'tfidf')
        self.vocabulary = self.tokenizer.word_index
        self.vocabulary_size = len(self.vocabulary)

        token_directory = os.path.join(savepath, 'Tokenizers')
        if not os.path.exists(token_directory):
            os.mkdir(token_directory) 

        tokenizer_json = self.tokenizer.to_json()          
        json_path = os.path.join(token_directory, 'word_tokenizer.json')
        with open(json_path, 'w', encoding = 'utf-8') as f:
            f.write(tokenizer_json)

        return tokenized_text
 
    #==========================================================================
    def sequence_padding(self, sequences, pad_value, pad_length, output = 'array'):

        '''
        sequence_padding(sequences, pad_value, pad_length, output='array')

        Pads a list of sequences to a specified length with a specified value.

        Keyword arguments:
            sequences (list): A list of sequences to be padded.
            pad_value (int): The value to use for padding.
            pad_length (int): The length to pad the sequences to.
            output (str): The format of the output. If 'array', the function returns a list of 
            padded sequences as numpy arrays. If 'string', the function returns a list of padded sequences as strings.

        Returns:
            padded_text (list): A list of padded sequences in the specified output format.
        
        '''
        padded_text = pad_sequences(sequences, maxlen = pad_length, value = pad_value, 
                                    dtype = 'int32', padding = 'post')
        if output == 'string':
            padded_text_str = []
            for x in padded_text:
                x_string = ' '.join(str(i) for i in x)
                padded_text_str.append(x_string)
            padded_text = padded_text_str
          
        
        return padded_text    
    
    
    #==========================================================================
    def model_savefolder(self, path, model_name):

        '''
        Creates a folder with the current date and time to save the model.
    
        Keyword arguments:
            path (str):       A string containing the path where the folder will be created.
            model_name (str): A string containing the name of the model.
    
        Returns:
            str: A string containing the path of the folder where the model will be saved.
        
        '''        
        raw_today_datetime = str(datetime.now())
        truncated_datetime = raw_today_datetime[:-10]
        today_datetime = truncated_datetime.replace(':', '').replace('-', '').replace(' ', 'H') 
        model_name = f'{model_name}_{today_datetime}'
        model_savepath = os.path.join(path, model_name)
        if not os.path.exists(model_savepath):
            os.mkdir(model_savepath)               
            
        return model_savepath 

    #==========================================================================
    def load_tokenizer(self, path, filename):  

        json_path = os.path.join(path, f'{filename}.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            json_string = f.read()
            tokenizer = tokenizer_from_json(json_string)

        return tokenizer
    
# [CALLBACK FOR REAL TIME TRAINING MONITORING]
#==============================================================================
#==============================================================================
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

        if epoch % 1 == 0:            
            self.epochs.append(epoch)
            self.loss_hist.append(logs['loss'])
            self.metric_hist.append(logs['accuracy'])
            if self.validation==True:
                self.loss_val_hist.append(logs['val_loss'])            
                self.metric_val_hist.append(logs['val_accuracy'])
        if epoch % 1 == 0:            
            #------------------------------------------------------------------
            fig_path = os.path.join(self.plot_path, 'training_history.jpeg')
            plt.subplot(2, 1, 1)
            plt.plot(self.epochs, self.loss_hist, label = 'training loss')
            if self.validation==True:
                plt.plot(self.epochs, self.loss_val_hist, label = 'validation loss')
                plt.legend(loc = 'best', fontsize = 8)
            plt.title('Loss plot')
            plt.ylabel('Binary crossentropy')
            plt.xlabel('epoch')
            plt.subplot(2, 1, 2)
            plt.plot(self.epochs, self.metric_hist, label = 'train metrics') 
            if self.validation==True: 
                plt.plot(self.epochs, self.metric_val_hist, label = 'validation metrics') 
                plt.legend(loc = 'best', fontsize = 8)
            plt.title('metrics plot')
            plt.ylabel('Accuracy')
            plt.xlabel('epoch')       
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches = 'tight', format = 'jpeg', dpi = 300)
            plt.show(block = False)
            plt.close()      



# [CUSTOM DATA GENERATOR FOR TRAINING]
#==============================================================================
#==============================================================================
#==============================================================================
class DataGenerator(keras.utils.Sequence):

    def __init__(self, dataframe, batch_size=6, image_size=(244, 244), shuffle=True):        
        self.dataframe = dataframe
        self.path_col = 'images_path'        
        self.label_col = 'tokenized_text'
        self.num_of_samples = dataframe.shape[0]
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
        image = tf.image.decode_image(image, channels=3)
        resized_image = tf.image.resize(image, self.image_size)
        rgb_image = tf.reverse(resized_image, axis=[-1])
        norm_image = rgb_image / 255.0              
        pp_image = tf.keras.preprocessing.image.random_shift(norm_image, 0.2, 0.3)
        pp_image = tf.image.random_flip_left_right(pp_image)
        pp_image = tf.image.random_flip_up_down(pp_image)

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

# [CUSTOM ATTENTION MODEL]
#==============================================================================
#==============================================================================
#==============================================================================   
class BahdanauAttention(Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features, hidden):
        
        hidden_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_time_axis)))
        score = self.V(attention_hidden_layer)        
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# [CUSTOM ENCODER MODEL]
#==============================================================================
#==============================================================================
#============================================================================== 
class EncoderXREP(Model):
    def __init__(self, features_size, picture_size):
        super(EncoderXREP, self).__init__()

        self.conv1 = layers.Conv2D(128, (3, 3), strides=1, activation='relu',input_shape=picture_size)
        self.maxpool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(256, (3, 3), strides=1, activation='relu')
        self.maxpool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(512, (3, 3), strides=1, activation='relu')
        self.conv4 = layers.Conv2D(512, (3, 3), strides=1, activation='relu')
        self.maxpool3 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(features_size, activation='relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x
    
# [CUSTOM DECODER MODEL]
#==============================================================================
#==============================================================================
#============================================================================== 
class DecoderXREP(Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(DecoderXREP, self).__init__()
        self.units = units

        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc1 = layers.Dense(self.units)
        self.fc2 = layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)    
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)       
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
    
# [CUSTOM MODEL]
#==============================================================================
#==============================================================================
#============================================================================== 
class ModelXREP(Model):
    def __init__(self, learning_rate, features_size, picture_size, embedding_dim, units, vocab_size, batch_size):
        super(ModelXREP, self).__init__()
        self.encoder = EncoderXREP(features_size, picture_size)
        self.decoder = DecoderXREP(embedding_dim, units, vocab_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.hidden = self.decoder.reset_state(batch_size=batch_size)


    def call(self, img_tensor, target):
        features = self.encoder(img_tensor)
        output, self.weight_matrix = self.decoder(target, features, self.hidden)

        return output

    

    
# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]
#==============================================================================
#==============================================================================
#==============================================================================
class TrainingTools:
    
    """     
    A class for training operations. Includes many different methods that can 
    be used in sequence to build a functional
    preprocessing pipeline.
      
    Methods:
        
    __init__(df_SC, df_BN): initializes the class with the single component 
                            and binary mixrture datasets
    
    training_logger(path, model_name):     write the training session info in a txt file
    prevaluation_model(model, n_features): evaluates the model with dummy datasets 
              
    """    
    def __init__(self, device = 'default'): 
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)              
        np.random.seed(42)
        tf.random.set_seed(42)         
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
            tf.config.set_visible_devices(self.physical_devices[0], 'GPU')
            print('GPU is set as active device')
            print('-------------------------------------------------------------------------------')
            print()        
        elif device == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            print('CPU is set as active device')
            print('-------------------------------------------------------------------------------')
            print()    
    

    
# [VALIDATION OF PRETRAINED MODELS]
#==============================================================================
#==============================================================================
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
