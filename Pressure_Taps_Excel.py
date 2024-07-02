import pandas as pd
import numpy as np
import scipy as sp





if __name__ == "__main__":
    
    xlsx_file_name='Pressure Tap Layout Summary.xlsx'
    xlsx_full_file=r'data\{}'.format(xlsx_file_name)
    xlsx_sheet_names=['North Wall','West Wall','South Wall','East Wall']


    xlsx=pd.read_excel(xlsx_full_file,sheet_name=xlsx_sheet_names,header=None)

    with open(r'data\Tap_Summary.txt','w') as f:
        for s in xlsx_sheet_names:
            xlsx_removed=xlsx[s].drop(0,axis=0)
            xlsx_removed=xlsx_removed.values.flatten()
            f.write(s+':')
            f.write(str(xlsx_removed.tolist()))
            f.write('\n')

 


