import os

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
images_path = os.path.join(data_path, 'XRAY images') 
pretrained_path = os.path.join(model_path, 'pretrained models')

if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(images_path):
    os.mkdir(images_path)
if not os.path.exists(pretrained_path):
    os.mkdir(pretrained_path)



