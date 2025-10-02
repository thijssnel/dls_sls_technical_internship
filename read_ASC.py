import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import numpy as np
import os


class read_asc:
    def __init__(self, path):

        self.Path = path
        self.data = self.ASC_2_dict(self.Path)

        self.Date = self.data['Date']
        self.Time = self.data['Time']

        self.Samplename = self.data['Samplename']
        self.SampleMemo = [self.data[f'SampMemo({i})'] for i in range(10)]

        self.Temperature_K = float(self.data['Temperature[K]'])
        self.Viscosity_cp = float(self.data['Viscosity[cp]'])
        self.Refractive_Index = float(self.data['RefractiveIndex'])
        self.Wavelength_nm = float(self.data['Wavelength[nm]'])
        self.Angle_deg = float(self.data['Angle[°]'])

        self.Duration_s = float(self.data['Duration[s]'])
        self.Runs = float(self.data['Runs'])
        self.Mode = self.data['Mode']

        self.MeanCR0_kHz = float(self.data['MeanCR0[kHz]'])
        self.MeanCR1_kHz = float(self.data['MeanCR1[kHz]'])

        self.Correlationx = np.array(self.data['Correlationx'])
        self.Correlationy = np.array(self.data['Correlationy'])  

        self.CountRatex = np.array(self.data['CountRatex'])
        self.CountRatey = np.array(self.data['CountRatey'])

        if 'StandardDeviationx' in self.data.keys():
            self.StandardDeviatiox = np.array(self.data['StandardDeviationx'])
            self.StandardDeviatioy = np.array(self.data['StandardDeviationy'])

    def ASC_2_dict(self, path):
        with open(path, encoding='unicode_escape') as f:
            data = f.read()
        
        word, j, words = str(), str(), []

        for i in data:
            if (((i == " ") and (i != j) ) or ((i == "\n") or (i == '\t'))):
                if word.count('"') == 1:
                    continue

                elif word == 'Index:':
                    words[-1] = 'RefractiveIndex'
                    words.append(":")
                    word = str()

                elif (word != ''):
                    word = word.replace('"','')
                    words.append(word)
                    word = str()
            
            elif (i != " ") and ((i != "\t") or (i != "\n")):
                word += i
            j = i

        cor_cou = 0
        dict = {}
        for i in range(len(words)):
            if (words[i] == ':') and (words[i-4] == ':'):
                dict[words[i-2]+words[i-1]] = words[i+1]

            elif words[i] == ':':
                dict[words[i-1]] = words[i+1]

            elif cor_cou == 1 and (words[i] != 'CountRate'):
                if len(dict['Correlationx']) <= len(dict['Correlationy']):
                    dict["Correlationx"].append(float(words[i]))
                else:
                    dict["Correlationy"].append(float(words[i]))

            elif cor_cou == 2 and (words[i] != 'StandardDeviation'):
                if len(dict['CountRatex']) <= len(dict['CountRatey']):
                    dict["CountRatex"].append(float(words[i]))
                else:
                    dict["CountRatey"].append(float(words[i]))

            elif cor_cou == 3:
                if len(dict['StandardDeviationx']) <= len(dict['StandardDeviationy']):
                    dict["StandardDeviationx"].append(float(words[i]))
                else:
                    dict["StandardDeviationy"].append(float(words[i]))

            if words[i] == 'Correlation':
                dict['Correlationx'] = []
                dict['Correlationy'] = []
                cor_cou = 1

            elif words[i] == 'CountRate':
                dict['CountRatex'] = []
                dict['CountRatey'] = []
                cor_cou = 2

            elif words[i] == 'StandardDeviation':
                dict['StandardDeviationx'] = []
                dict['StandardDeviationy'] = []
                cor_cou = 3          
            

        return dict
    
    def quick_plot_cor(self, title:str='name'):
        if title.lower() == 'angle':
            title_str = f'correlation of angle{self.Angle_deg}'
        elif title.lower() == 'name':
            title_str = f'correlation of {self.Samplename}'

    
        sns.set_theme()
        plt.plot(self.Correlationx, self.Correlationy)
        plt.title(title_str)
        plt.xscale('log')
        plt.xlabel('Tau (ms)')
        plt.ylabel('correlation')
        plt.show()
    
    def quick_plot_count(self, title:str='name'):
        if title.lower() == 'angle':
            title_str = f'countrate of angle {self.Angle_deg}'
        elif title.lower() == 'name':
            title_str = f'countrate of {self.Samplename}'

        sns.set_theme()
        plt.plot(self.CountRatex, self.CountRatey)
        plt.title(title_str)
        plt.xlabel('time (s)')
        plt.ylabel('rate (-)')
        plt.show()

    def g2_model(self, tau, beta, Gamma):
        return beta * np.exp(-2 * Gamma * tau)

    def fit(self):
        corx = self.Correlationx
        cory = self.Correlationy
        beta0 = max(cory)
        gamma0 = 1 
        # Initial guess: beta=0.5, Gamma=1000 (adjust depending on your data)
        popt, pcov = curve_fit(self.g2_model, corx, cory,
                               p0=[beta0, gamma0], bounds=(0, np.inf), maxfev=10000)
        beta, Gamma = popt
        return beta, Gamma
    
    def plot_fit(self):
        beta, Gamma = self.fit()
        tau_fit = np.logspace(np.log10(min(self.Correlationx)), np.log10(max(self.Correlationx)), 100)
        g2_fit = self.g2_model(tau_fit, beta, Gamma)
        radius = self.hydrodynamic_radius()
        print(f"Fitted parameters: beta = {beta:.3f}, Gamma = {Gamma:.1f}, Hydrodynamic Radius = {radius:.2f} nm")
        sns.set_theme()

        plt.figure(figsize=(8, 5))
        plt.plot(self.Correlationx, self.Correlationy, 'b.', label='Data')
        plt.plot(tau_fit, g2_fit, 'r-', label=f'Fit: beta={beta:.3f}, Gamma={Gamma:.1f}')
        plt.xscale('log')
        plt.xlabel('Tau (ms)')
        plt.ylabel('g-1')
        plt.title(f'Fit for {self.Samplename} at angle {self.Angle_deg}°')
        plt.legend()
        plt.show()
    
    def hydrodynamic_radius(self):
        beta, Gamma = self.fit()
        q = (4 * np.pi * self.Refractive_Index / (self.Wavelength_nm*10**-9)) * np.sin(np.radians(self.Angle_deg) / 2)
        D = Gamma / (q ** 2)
        T = self.Temperature_K
        eta = self.Viscosity_cp/1000  # Convert cp to Pa.s
        R_h = (1.38e-23 * T) / (6 * np.pi * eta * D) * 1e9  # in nm
        return R_h
    


class dls_sls_analysis:
    def __init__(self, dict_path):
        self.dict_path = dict_path
        self.sample_names = []
        self.sample_durations = []
        self.sample_angles = []
        self.sample_data = {}
        for dict in os.listdir(self.dict_path):
            if 'standard' in dict.lower():
                self.standard_kalibration = [read_asc(path=os.path.join(self.dict_path, dict, file)) for file in os.listdir(os.path.join(self.dict_path, dict)) if file.endswith('.ASC')]
            elif 'solvent' in dict.lower():
                self.solvent_kalibration = [read_asc(path=os.path.join(self.dict_path, dict, file)) for file in os.listdir(os.path.join(self.dict_path, dict)) if file.endswith('.ASC')]
            elif 'solution' in dict.lower():
                self.solution_data = [read_asc(path=os.path.join(self.dict_path, dict, file)) for file in os.listdir(os.path.join(self.dict_path, dict)) if file.endswith('.ASC')]
        
        for n, data in  self.get_data().values():
            if data.Samplename.lower() not in self.sample_names:
                self.sample_names.append(data.Samplename.lower())

            if data.Duration_s not in self.sample_durations:
                self.sample_durations.append(data.Duration_s)

            if round(data.Angle_deg) not in self.sample_angles:
                self.sample_angles.append(round(data.Angle_deg))

    
    def get_data(self, angle='all', data_type='all', sample_name='all', duration='all'):
        #control if varibles are valid
        search_vars = {'sample_name': sample_name, 'duration': duration,  'angle': angle}
        for var, val in search_vars.items():
            if (val != 'all') and (type(val) != list):
                if (var == 'sample_name') and (val.lower() not in self.sample_names):
                    raise ValueError(f"Sample name '{val}' not found. Available names: {self.sample_names}")
                if (var == 'duration') and (val not in self.sample_durations):
                    raise ValueError(f"Duration '{val}' not found. Available names: {self.sample_durations}")               
                if (var == 'angle') and (val not in self.sample_angles):
                    raise ValueError(f"Angle '{val}' not found. Available names: {self.sample_angles}")
            
            if (val != 'all') and (type(val) == list):
                if (var == 'sample_name') and any(value for value in val if value not in self.sample_names):
                    raise ValueError(f"Sample name '{[name for name in val if name not in self.sample_names]}' not found. Available names: {self.sample_names}")
                if (var == 'duration') and any(value for value in val if value not in self.sample_durations):
                    raise ValueError(f"Durations '{[dur for dur in val if dur not in self.sample_durations]}' not found. Available durations: {self.sample_durations}")               
                if (var == 'angle') and any(value for value in val if value not in self.sample_angles):
                    raise ValueError(f"Angles '{[ang  for ang in val if ang not in self.sample_angles]}' not found. Available angles: {self.sample_angles}")

        experiment = {}
        #filter type of samples 
        if data_type.lower() == 'all':
            data = self.solution_data + self.solvent_kalibration + self.standard_kalibration
        elif data_type.lower() == 'solution':
            data = self.solution_data
        elif data_type.lower() == 'solvent':
            data = self.solvent_kalibration
        elif data_type.lower() == 'standard':
            data = self.standard_kalibration
        
        i = 0
        #loop over elements 
        for exp in data:
            
            # angle filter
            if (angle == 'all'):
                angle_check = True
            elif type(angle) == int or type(angle) == float:
                angle_check = angle - 1 < exp.Angle_deg < angle + 1
            elif type(angle) == list:
                angle_check = any(ang - 1 < exp.Angle_deg < ang + 1 for ang in angle)
            

            # sample name filter
            if (sample_name == 'all'):
                name_check = True

            elif type(sample_name) == str:
                name_check = sample_name.lower() in exp.Samplename.lower()

            elif type(sample_name) == list:
                name_check = any(name.lower() in exp.Samplename.lower() for name in sample_name)
            

            # duration filter
            if (duration == 'all'):
                duration_check = True

            elif type(duration) == int or type(duration) == float:
                duration_check = duration - 1 < exp.Duration_s < duration + 1

            elif type(duration) == list:
                duration_check = any(dur - 1 < exp.Duration_s < dur + 1 for dur in duration)


            # combine all filters
            if angle_check and name_check and duration_check:
                experiment[i] = (f'{exp.Samplename}_angle {round(exp.Angle_deg)}_dur {round(exp.Duration_s)}', exp)
                i += 1
        return experiment
    

    def true_size(self, sol:dict = None, angle= 'all'):
        gamma_list = []
        for nam,val in self.get_data(data_type='solution',angle=angle):
            pass
