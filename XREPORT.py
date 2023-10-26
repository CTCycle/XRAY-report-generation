import sys

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# import modules and classes
#------------------------------------------------------------------------------
from modules.components.data_classes import UserOperations

# [MAIN MENU]
# =============================================================================
# Starting DITK analyzer, checking for dictionary presence and perform conditional
# import of modules
# =============================================================================
print('''
-------------------------------------------------------------------------------
XRAY REPORTER
-------------------------------------------------------------------------------
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed euismod, velit vel 
tincidunt luctus, justo velit ultricies nisl, vel lacinia velit justo sed sapien. 
Vivamus euismod, velit vel tincidunt luctus, justo velit ultricies nisl, vel 
lacinia velit justo sed sapien. Nam euismod, velit vel tincidunt luctus, justo velit 
ultricies nisl, vel lacinia velit justo sed sapien. Mauris euismod, velit vel tincidunt luctus, 
justo velit ultricies nisl, vel lacinia velit justo sed sapien. 
''')
user_operations = UserOperations()
operations_menu = {'1': 'Preprocess XRAY dataset', 
                   '4': 'Pretrain XRAYREP model',                   
                   '5': 'Generate reports based on images',
                   '6': 'Exit and close'}

while True:
    print('------------------------------------------------------------------------')
    print('MAIN MENU')
    print('------------------------------------------------------------------------')
    op_sel = user_operations.menu_selection(operations_menu)
    print()
    
    if op_sel == 1:
        import modules.XREPORT_preprocessing
        del sys.modules['modules.XREPORT_preprocessing']    
    
    elif op_sel == 2:
        import modules.XREPORT_training
        del sys.modules['modules.XREPORT_training']

    elif op_sel == 3:
        break

    elif op_sel == 4:
        break


