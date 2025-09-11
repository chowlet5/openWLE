import numpy as np
from scipy.signal import welch, csd
from scipy.stats import weibull_min
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



class GeneralFunctions:

    def spectra_value(self, spectra:np.ndarray, frequency:np.ndarray, lookup_frequency:np.ndarray|float ) -> np.ndarray:
        
        if not isinstance(lookup_frequency, np.ndarray):
            lookup_frequency = np.array([lookup_frequency])
        if spectra.ndim == 1:
            spectra = np.expand_dims(spectra,axis = 0)

        return np.array(list(map(lambda sp, f: np.interp(f,frequency,sp), spectra, lookup_frequency)))


    def spectral_density(self, time_history:np.ndarray, sampling_freq:float = 1.0) -> tuple[np.ndarray, np.ndarray]:

        mean_value = np.mean(time_history,axis = -1)
        if time_history.ndim == 1:
            fluctuating_time_history = time_history - mean_value
        else:
            mean_value = np.array([mean_value]) if not isinstance(mean_value,np.ndarray) else mean_value
            fluctuating_time_history = time_history - mean_value[:, None]
        f, spectra = welch(fluctuating_time_history, fs=sampling_freq, nperseg = 2048, nfft=fluctuating_time_history.shape[-1], detrend=False,axis = -1)
        
        return spectra, f
    
    def spectra_intergration(self, time_history:np.ndarray, sampling_freq:float = 1.0) -> float:

        spectra, f = self.spectral_density(time_history, sampling_freq)
        response = np.sqrt(np.trapz(spectra, f))

        return response


    def cross_spectral_density(self,time_history_1:np.ndarray, time_history_2:np.ndarray, sampling_freq:float) -> float:
        mean_value_1 = np.mean(time_history_1)
        fluctuating_time_history_1 = time_history_1 - mean_value_1
        
        mean_value_2 = np.mean(time_history_2)
        fluctuating_time_history_2 = time_history_2 - mean_value_2

        f_cross,cross_spectra = csd(fluctuating_time_history_1, fluctuating_time_history_2, fs=sampling_freq,nfft=min(len(time_history_1),len(time_history_2)),detrend=False)

        return cross_spectra, f_cross
    
    def correlation_coefficient(self,time_history_1:np.ndarray, time_history_2:np.ndarray, sampling_freq:float) -> float:
        spectra_1,f_1 = self.spectral_density(time_history_1,sampling_freq)
        spectra_2,f_2 = self.spectral_density(time_history_2,sampling_freq)
        cross_spectra,f_cross = self.cross_spectral_density(time_history_1,time_history_2,sampling_freq)
        correlation_coefficient = np.real(np.trapz(cross_spectra,f_cross))/(np.trapz(spectra_1,f_1)*np.trapz(spectra_2,f_2))

        return correlation_coefficient
    

def number_to_list(x) -> list:
    
    if isinstance(x,list):
        return x
    elif isinstance(x,(int,float)):
        return [x]
    else:
        raise TypeError('Value must be a list or int/float.')




class InputFileNaming:

    input_naming_type = {
        'int': int,
        'float': float,
        'str': str
    }

    def __init__(self, config:dict) -> None:
        
        self.config = config  
        self.naming_order = config['naming_order']
        self.naming_type = [self.input_naming_type[i] for i in config['naming_type']]
        
        self.sample_regex = config['sample_regex']
        self.naming_dict = config['naming_dict']

        if config['delimiter'] != '':
            self.delimiter = config['delimiter']
            self.delimiter_extraction_setup()
            self.extract = self.delimiter_extraction

        else:
            
            self.index_extraction_setup()
            self.extract = self.index_extraction
    
    def add_building_config(self,name_dict:dict) -> dict:

        
        name_dict['building_config'] = self.naming_dict['building_config'] 

        return name_dict

        
    def add_exposure_config(self,name_dict:dict) -> dict:

        
        name_dict['exposure'] = self.naming_dict['exposure'] 

        return name_dict
    
    def add_angle_config(self,name_dict:dict) -> dict:

        name_dict['angle'] = self.naming_dict['angle'] 

        return name_dict

        
    def index_extraction_setup(self):
        
        index_list = self.find_text_group(self.sample_regex) 
        
        if len(index_list) != len(self.naming_order):
            raise ValueError('The number of indexes in the sample regex does not match the number of indexes in the naming dictionary')

        self.index_list = index_list

    def index_extraction(self,string:str) -> dict:

        index_list = self.index_list
        name_dict = {}
        if not "building_config" in [name.lower() for name in self.naming_order]:
            name_dict = self.add_building_config(name_dict)
        if not "exposure" in [name.lower() for name in self.naming_order]:
            name_dict = self.add_exposure_config(name_dict)
        if not "angle" in [name.lower() for name in self.naming_order] and "exposure_config" in [name.lower() for name in self.naming_order]:
            name_dict = self.add_angle_config(name_dict)
        naming_index = 0
        for index, name in zip(index_list,self.naming_order):
            func = self.naming_type[naming_index]
            sub_string = func(string[index[0]:index[1]])
            
            name_dict[name] = self.naming_dict[name][sub_string]
            naming_index += 1
        
        return name_dict

    def delimiter_extraction_setup(self):

        string_groups = self.sample_regex.split(self.delimiter)
        
        index_list = []
        for group in string_groups:
            if r'*' in group:
                index_values = self.find_text_group(group)
                if len(index_values) != 1:
                    raise ValueError('Mutiple indexes found in the sample regex group. Only one index should be found.')
                else:
                    index_list.append(index_values[0])
            else:
                index_list.append(None)
        self.index_list = index_list

    def delimiter_extraction(self,file_string:str) -> dict:
        
        index_list = self.index_list 
        name_dict = {}
        if not "building_config" in [name.lower() for name in self.naming_order]:
            name_dict = self.add_building_config(name_dict)
        if not "exposure" in [name.lower() for name in self.naming_order]:
            name_dict = self.add_exposure_config(name_dict)
        if not "angle" in [name.lower() for name in self.naming_order] and "exposure_config" in [name.lower() for name in self.naming_order]:
            name_dict = self.add_angle_config(name_dict)
        name_index = 0
        for index, string in zip(index_list,file_string.split(self.delimiter)):
            if index == None:
                continue
            else:
                name = self.naming_order[name_index]
                func = self.naming_type[name_index]
                sub_string = func(string[index[0]:index[1]])
                name_dict[name] = self.naming_dict[name][sub_string]
                name_index += 1
        
        return name_dict
        
    def find_text_group(self,string:str) -> list:
        
        index = np.array([ind for ind, character in enumerate(string) if character == '*'],dtype=int)
        
        check = np.diff(index)
        check = np.insert(check,0,0)
        start_index = index[np.where(check != 1)[0]]

        check = np.diff(index)
        check = np.insert(check,len(index)-1,0)
        end_index = index[np.where(check != 1)[0]] + 1

        index_list = [(start,end) for start,end in zip(start_index,end_index)]
        
        return index_list
    
    def extract_name(self,file_string:str) -> dict:
        return self.extract(file_string)
    

def plot_check(polygons:list) -> None:

    

    fig, ax = plt.subplots()
    for polygon in polygons:
        ax.plot(*polygon.exterior.xy)
    plt.show()
    
def plot_tap_check(polygons:list, points:list, ids:list) -> None:

    fig, ax = plt.subplots()
    for polygon in polygons:
        ax.plot(*polygon.exterior.xy)
    for point,id_value in zip(points,ids):
        ax.plot(*point,'ro')
        ax.text(point[0],point[1],f'{id_value}')
    plt.savefig('tap_check.png')



class BiModalWeibull:
    def __init__(self, shape1=None, scale1=None, shape2=None, scale2=None, weight=None):
        self.shape1 = shape1
        self.scale1 = scale1
        self.shape2 = shape2
        self.scale2 = scale2
        self.weight = weight

    def pdf(self, x):
        pdf1 =  self.shape1/self.scale1 * (x/self.scale1)**(self.shape1-1) * np.exp(-(x/self.scale1)**self.shape1)
        pdf2 =  self.shape2/self.scale2 * (x/self.scale2)**(self.shape2-1) * np.exp(-(x/self.scale2)**self.shape2)
        return self.weight * pdf1 + (1 - self.weight) * pdf2

    def cdf(self, x):
        cdf1 = 1 - np.exp(-(x/self.scale1)**self.shape1)
        cdf2 = 1 - np.exp(-(x/self.scale2)**self.shape2)
        return self.weight * cdf1 + (1 - self.weight) * cdf2

    def mean_values(self) -> tuple:

        mean1 = weibull_min(self.shape1, scale=self.scale1).mean()
        mean2 = weibull_min(self.shape2, scale=self.scale2).mean()
        return (mean1, mean2)

    def var_values(self) -> tuple:

        var1 = weibull_min(self.shape1, scale=self.scale1).var()
        var2 = weibull_min(self.shape2, scale=self.scale2).var()
        return (var1, var2)
    

    def std_values(self) -> tuple:

        std1 = weibull_min(self.shape1, scale=self.scale1).std()
        std2 = weibull_min(self.shape2, scale=self.scale2).std()
        return (std1, std2)
    

    def std(self):

        mean1, mean2 = self.mean_values()
        var1, var2 = self.var_values()
        binomial_std = np.sqrt(self.weight*var1 + (1 - self.weight) * var2 + self.weight * (1 - self.weight) * (mean1-mean2)**2)

        return binomial_std

    def fit(self, x_data, y_data, p0=None):
        """
        Fits the bi-modal Weibull distribution to data.

        Args:
            x_data: Independent variable data.
            y_data: Dependent variable data.
            p0: Initial guess for the parameters (shape1, scale1, shape2, scale2, weight).
                If not provided, default values are used.

        Returns:
            Tuple of optimized parameters (shape1, scale1, shape2, scale2, weight).
        """

        bounds = (0,[np.inf, np.inf, np.inf, np.inf, 1])

        def bimodal_weibull_func(x, k1, lam1, k2, lam2, w):
            func1 = np.exp(-(x/lam1)**k1)
            func2 = np.exp(-(x/lam2)**k2)
            return w * func1 + (1 - w) * func2
            

        if p0 is None:
            p0 = [2, 2, 5, 2, 0.5]  # Default initial guess

        popt, _ = curve_fit(bimodal_weibull_func, x_data, y_data, p0=p0, bounds=bounds)
        self.shape1, self.scale1, self.shape2, self.scale2, self.weight = popt

        return popt

        
class UpcrossingMethod:

    def __init__(self):

        pass


class StormPassageMethod:

    def __init__(self):

        pass