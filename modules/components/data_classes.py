import os
from datetime import datetime
from tqdm import tqdm
tqdm.pandas()



# ...
#==============================================================================
#==============================================================================
#==============================================================================
class UserOperations:
    
    """    
    A class for user operations such as interactions with the console, directories 
    and files cleaning and other maintenance operations.
      
    Methods:
        
    menu_selection(menu):         console menu management
    clear_all_files(folder_path): cleaning files and directories 
   
    """
    
    #==========================================================================
    def menu_selection(self, menu):
        
        """        
        menu_selection(menu)
        
        Presents a menu to the user and returns the selected option.
        
        Keyword arguments:                      
            menu (dict): A dictionary containing the options to be presented to the user. 
                         The keys are integers representing the option numbers, and the 
                         values are strings representing the option descriptions.
        
        Returns:            
            op_sel (int): The selected option number.
        
        """
        
        indexes = [idx + 1 for idx, val in enumerate(menu)]
        for key, value in menu.items():
            print('{0} - {1}'.format(key, value))            
        
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
    
    
               
    #==========================================================================
    def datetime_fetching(self):
        
        raw_today_datetime = str(datetime.now())
        truncated_datetime = raw_today_datetime[:-7]
        today_datetime = truncated_datetime
        for rep in ('-', ':', ' '):
            today_datetime = today_datetime.replace(rep, '_')
            
        return today_datetime            


# define the class for inspection of the input folder and generation of files list.
#==============================================================================
#==============================================================================
#==============================================================================
class XRAYDataSet:
    
    """    
    A class collecting methods to build the NIST adsorption dataset from collected
    data. Methods are meant to be used sequentially as they are self-referring 
    (no need to specify input argument in the method). 
      
    Methods:
        
    extract_molecular_properties(df_mol):  retrieve molecular properties from ChemSpyder
    split_by_mixcomplexity():              separates single component and binary mixture measurements
    extract_adsorption_data():             retrieve molecular properties from ChemSpyder
    
    
    
    """
   
    
    #==========================================================================
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
    
# define the class for inspection of the input folder and generation of files list.
#==============================================================================
#==============================================================================
#==============================================================================
class DataStorage: 

    #==========================================================================
    def JSON_serializer(self, object, filename, path, mode='SAVE'):

        if mode == 'SAVE':
            object_json = object.to_json()          
            json_path = os.path.join(path, f'{filename}.json')
            with open(json_path, 'w', encoding = 'utf-8') as f:
                f.write(object_json)
        elif mode == 'LOAD':
            pass




        
      
