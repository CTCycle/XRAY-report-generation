import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
tqdm.pandas()



# [CONSOLE USER OPERATIONS]
#==============================================================================
# Perform operations to control the console
#==============================================================================
class UserOperations:   
    
    #--------------------------------------------------------------------------
    def menu_selection(self, menu):
        
        '''        
        menu_selection(menu)
        
        Presents a menu to the user and returns the selected option.
        
        Keyword arguments:                      
            menu (dict): A dictionary containing the options to be presented to the user. 
                         The keys are integers representing the option numbers, and the 
                         values are strings representing the option descriptions.
        
        Returns:            
            op_sel (int): The selected option number.
        
        '''        
        indexes = [idx + 1 for idx, val in enumerate(menu)]
        for key, value in menu.items():
            print(f'{key} - {value}')       
        print()
        while True:
            try:
                op_sel = int(input('Select the desired operation: '))
            except:
                continue           
            while op_sel not in indexes:
                try:
                    op_sel = int(input('Input is not valid, please select a valid option: '))
                except:
                    continue
            break
        
        return op_sel    
    


    
# [PREPROCESSING PIPELINE]
#==============================================================================
# Preprocess data
#==============================================================================
class PreProcessing: 


    #--------------------------------------------------------------------------
    def images_pathfinder(self, path, dataframe, id_col):

        images_paths = {}
        for pic in os.listdir(path):
            pic_name = pic.split('.')[0]
            pic_path = os.path.join(path, pic)                        
            path_pair = {pic_name : pic_path}        
            images_paths.update(path_pair)
        
        dataframe['images_path'] = dataframe[id_col].map(images_paths)
        dataframe = dataframe.dropna(subset=['images_path']).reset_index(drop = True)

        return dataframe 

    #--------------------------------------------------------------------------
    def load_images(self, paths, num_channels, image_size):
        
        images = []
        for pt in tqdm(paths):
            image = tf.io.read_file(pt)
            image = tf.image.decode_image(image, channels=num_channels)
            image = tf.image.resize(image, image_size)
            if num_channels==3:
                image = tf.reverse(image, axis=[-1])
            image = image/255.0 
            images.append(image) 

        return images    
        

    #--------------------------------------------------------------------------
    def text_preparation(self, strings):

        '''
        text_preparation(strings)

        Prepares a list of strings for tokenization by converting them to lowercase, 
        adding spaces around punctuation symbols and delimiting the strings with start
        and end sequence tokens 

        Keyword arguments:
            strings (list): A list of strings to be prepared.

        Returns:
            processed_strings (list): A list of prepared strings.
        
        '''
        symbols = ['.', ',', ';', ':', '"', '-', '=', '/']       
        processed_strings = []
        for st in strings:
            string = st.lower()        
            for sb in symbols:
                string = string.replace(sb, '')
            delimited_str = '[START] ' + string + ' [END]'
            processed_strings.append(delimited_str)

        return processed_strings
    
    #--------------------------------------------------------------------------
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
 
    #--------------------------------------------------------------------------
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
    
    
    #--------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------
    def load_tokenizer(self, path, filename):  

        json_path = os.path.join(path, f'{filename}.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            json_string = f.read()
            tokenizer = tokenizer_from_json(json_string)

        return tokenizer






        
      
