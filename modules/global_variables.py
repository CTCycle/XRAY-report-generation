import os
import sys
import pandas as pd

#==============================================================================
if getattr(sys, 'frozen', False):
    data_path = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'dataset')
    model_path = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'models')   
else:
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')    

images_path = os.path.join(data_path, 'XRAY images') 
pretrained_path =  os.path.join(model_path, 'pretrained models')


if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(images_path):
    os.mkdir(images_path)
if not os.path.exists(pretrained_path):
    os.mkdir(pretrained_path)



#==============================================================================
df_train = pd.DataFrame()
df_validation = pd.DataFrame()

#==============================================================================
seed = 42

data_size = 0.3
test_size = 0.2

pic_size = (160, 160, 3)
embedding_dims = 212
features_vector = 256
RNN_dims = 256
batch_size = 2
epochs = 10
learning_rate = 0.00000001

