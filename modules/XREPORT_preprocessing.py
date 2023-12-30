import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add modules path to sys
#------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  

# import modules and classes
#------------------------------------------------------------------------------
from modules.components.data_classes import PreProcessing, XREPDataSet
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD TEXT DATA]
#==============================================================================
# module for the selection of different operations
#==============================================================================
file_loc = os.path.join(GlobVar.data_path, 'XREP_dataset.csv') 
dataset = pd.read_csv(file_loc, encoding = 'utf-8', sep = (';' or ',' or ' ' or  ':'), low_memory = False)

# [ADD PATH TO XRAY DATASET AND SPLIT DATASET]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print('''
-------------------------------------------------------------------------------
XRAY data preprocessing
-------------------------------------------------------------------------------
The XRAY dataset must be preprocessed before feeding it to the training model.
The preprocessing procedure comprises the tokenization and padding on the text sequences
''')

# select only a part of the main dataset
#------------------------------------------------------------------------------
dataworker = XREPDataSet()
dataset = dataworker.images_pathfinder(GlobVar.images_path, dataset, 'id')
dataset = dataset.sample(n=cnf.num_samples)

# split data into train and test dataset and start preprocessor
#------------------------------------------------------------------------------
preprocessor = PreProcessing()
train_data, test_data = train_test_split(dataset, test_size=cnf.test_size, random_state=cnf.seed)

# [TOKENIZE TEXT]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print('''STEP 1 ----> Text tokenization using word tokenizer
''')

# extract text sequences as list perform tokenization
#------------------------------------------------------------------------------
train_text = train_data['text'].to_list()
test_text = test_data['text'].to_list()
total_text = train_text + test_text
train_text = preprocessor.text_preparation(train_text)
test_text = preprocessor.text_preparation(test_text)
tokenized_train_text = preprocessor.text_tokenization(train_text, GlobVar.data_path, matrix_output=False)
tokenized_test_text = preprocessor.text_tokenization(test_text, GlobVar.data_path, matrix_output=False)

# [PAD SEQUENCES]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print('''STEP 2 ----> Sequence padding to equalize sequence length
''')
vocabulary_size = preprocessor.vocabulary_size
pad_length = max([len(x) for x in total_text])

# perform padding of sequences
#------------------------------------------------------------------------------
padded_train_text = preprocessor.sequence_padding(tokenized_train_text, cnf.pad_value, pad_length, output = 'string')
padded_test_text = preprocessor.sequence_padding(tokenized_test_text, cnf.pad_value, pad_length, output = 'string')
train_data['tokenized_text'] = padded_train_text
test_data['tokenized_text'] = padded_test_text

# [SAVE CSV DATA]
#==============================================================================
# module for the selection of different operations
#==============================================================================
file_loc = os.path.join(GlobVar.data_path, 'XREP_train.csv')  
train_data.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')
file_loc = os.path.join(GlobVar.data_path, 'XREP_test.csv')  
test_data.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')


