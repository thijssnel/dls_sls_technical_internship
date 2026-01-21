import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import numpy as np
import os
from operator import itemgetter
from jupyter_analysis_tools.utils import (isWindows, isMac, isLinux,
    isList, pushd, grouper, updatedDict)
from pathlib import Path
from functions import *



class read_asc:
    def __init__(self, path):
        self.Path = Path(path)
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
        self.angle_rad = float(np.deg2rad(self.Angle_deg))
        self.q = (4 * np.pi * self.Refractive_Index / (self.Wavelength_nm/1e9)) * np.sin(self.angle_rad/ 2)



        if 'Cumulant_1' in self.data.keys():
            self.MonitorDiode = float(self.data['Monitor diode'])
            self.Cumulant_1 = self.data['Cumulant_1']   
            self.Cumulant_2 = self.data['Cumulant_2']
            self.Cumulant_3 = self.data['Cumulant_3']


        self.Duration_s = float(self.data['Duration[s]'])
        self.Runs = float(self.data['Runs'])
        self.Mode = self.data['Mode']

        self.MeanCR0_kHz = float(self.data['MeanCR0[kHz]'])
        self.MeanCR1_kHz = float(self.data['MeanCR1[kHz]'])

        self.Correlationx = np.array([val * 1e-3 for val in self.data['Correlationx'][5:]])
        self.Correlationy = np.array([val for val in self.data['Correlationy'][5:]])
        self.Correlationy = self.Correlationy / max(self.Correlationy)

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

            elif cor_cou == 2 and (words[i] != 'Monitor') and (words[i] != 'StandardDeviation'):
                if len(dict['CountRatex']) <= len(dict['CountRatey']):
                    dict["CountRatex"].append(float(words[i]))
                else:
                    dict["CountRatey"].append(float(words[i]))

            elif words[i] == 'Monitor':
                dict["Monitor diode"] = words[i+2]
                    
            
            elif cor_cou == 3 and words[i] =='Cumulant1.Order':
                dict['Cumulant_1']['Fluc_Freq'] = float(words[i+3])
                dict['Cumulant_1']['Diff_Cof'] =float(words[i+6])
                dict['Cumulant_1']['Hydr_Rad'] = float(words[i+10])
            
            elif cor_cou == 3 and words[i] == 'Cumulant2.Order':
                dict['Cumulant_2']['Fluc_Freq'] = float(words[i+3])
                dict['Cumulant_2']['Diff_Cof'] = float(words[i+6])
                dict['Cumulant_2']['Hydr_Rad'] = float(words[i+10])
                dict['Cumulant_2']['Exp_Par_2'] = float(words[i+14])

            elif cor_cou == 3 and words[i] == 'Cumulant3.Order':
                dict['Cumulant_3']['Fluc_Freq'] = float(words[i+3])
                dict['Cumulant_3']['Diff_Cof'] = float(words[i+6])
                dict['Cumulant_3']['Hydr_Rad'] = float(words[i+10])
                dict['Cumulant_3']['Exp_Par_2'] = float(words[i+14])
                dict['Cumulant_3']['Exp_Par_3'] = float(words[i+18])

            elif cor_cou == 4:
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

            elif words[i] == 'Monitor':
                dict['Cumulant_1'] ={'Fluc_Freq':0,
                                     'Diff_Cof': 0,
                                     'Hydr_Rad': 0}

                dict['Cumulant_2'] = {'Fluc_Freq':0,
                                     'Diff_Cof': 0,
                                     'Hydr_Rad': 0,
                                     'Exp_Par_2' : 0}

                dict['Cumulant_3'] = {'Fluc_Freq':0,
                                     'Diff_Cof': 0,
                                     'Hydr_Rad': 0,
                                     'Exp_Par_2' : 0,
                                     'Exp_Par_3': 0
                                     }
                cor_cou = 3

                

            elif words[i] == 'StandardDeviation':
                dict['StandardDeviationx'] = []
                dict['StandardDeviationy'] = []
                cor_cou = 4       
            

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
        return beta * np.exp(- 2 * Gamma * tau)

    def fit(self):
        corx = self.Correlationx
        cory = self.Correlationy
        beta0 = cory[4]
        gamma0 = 1 / (corx[ np.argmax(cory < beta0*0.5) ])
        # Initial guess: beta=0.5, Gamma=1000 (adjust depending on your data)
        popt, pcov = curve_fit(self.g2_model, corx, cory,
                               p0=[beta0, gamma0], maxfev=200000)
        beta, Gamma = popt
        return beta, Gamma
    
    def plot_fit(self):
        beta, Gamma = self.fit()
        g2_fit = self.g2_model(self.Correlationx, beta, Gamma)
        radius = hydrodynamic_radius(self.Refractive_Index,
                                     self.Wavelength_nm/1e9,
                                     Gamma,
                                     self.Angle_deg,
                                     self.Temperature_K,
                                     self.Viscosity_cp/1e3)
        print(fr"Fitted parameters: beta = {beta:.3f}, Diff_coef (square micrometer per second) = {Gamma/(self.q**2)*1e12:.5f}, Hydrodynamic Radius = {radius:.2f} nm")
        sns.set_theme()

        plt.figure(figsize=(8, 5))
        plt.plot(self.Correlationx, self.Correlationy, 'b.', label='Data')
        plt.plot(self.Correlationx, g2_fit, 'r-', label=f'Fit: beta={beta:.3f}, Gamma={Gamma:.1f}')
        plt.xscale('log')
        plt.xlabel('Tau (ms)')
        plt.ylabel('g-1')
        plt.title(f'Fit for {self.Samplename} at angle {self.Angle_deg}°')
        plt.legend()
        plt.show()


class hide():
    def tikhonov_Phillips_fit(self):
        self.createFittingS_space(0.09,1e3,200)
        self.getBetaEstimate()
        self.getG1correlation()
        self.getInitialEstimates()
        self.getInitialEstimatesManyAlpha()
        self.getOptimalAlphaLcurve()
        self.getInitialEstimatesOptimalAlphaLcurve()
        self.getInitialEstimatesManyAlpha()
        self.predictAutocorrelationCurves()
    def createFittingS_space(self,lowHr,highHr,n):

        """

        Create the s (inverse of gamma decay rate) space that will be used for the fitting
        The limits are given by the minimum and maximum desired hydrodynamic radius (in nanometers)
        
        Run after getQ()!

        """
        n = int(n) # Convert n to integer type for the np.logspace function


        sUpLimitHigh  = s_inverse_decay_rate(
            diffusion_from_hydrodynamic_radius(highHr/1e9,self.Temperature_K,self.Viscosity_cp), self.q)

        sUpLimitLow   = s_inverse_decay_rate(
            diffusion_from_hydrodynamic_radius(lowHr/1e9,self.Temperature_K,self.Viscosity_cp), self.q)

        # Sequence in linear space! 10.0**start to 10**stop
        self.s_space     = np.logspace(np.log10(sUpLimitLow),np.log10(sUpLimitHigh), n) 

        self.ds          = diffusion_from_inverse_decay_rate(self.s_space,self.q)
        self.hrs         = hydrodynamic_radius(self.ds ,self.Temperature_K,self.Viscosity_cp)*1e9  # In nanometers

        return None
    
    def getBetaEstimate(self):

        """
        Fit a polynomial of degree 2 to the first 5 microseconds of data
        """

        self.betaGuess               = get_beta_prior(self.Correlationy,self.Correlationx) 

        return None
    

    def getG1correlation(self):

        """ 
        Calculate the first order autocorrelation function g1
        """
        
        self.g1              = np.array([g1_from_g2(self.Correlationy[i],self.betaGuess) for i in range(len(self.Correlationy))])

        return None


    def getInitialEstimates(self,alpha=0.1,timeLimit=1e8):

        """

        Obtain initial estimates for the relative contributions

        Run after createFittingS_space() !

        timeLimit should be given in microseconds! Default time is 100 seconds (all the autocorrelation curve).

        alpha can be one value (same for all curves) or a list of values (one value per curve)
    
        """
 
        selectedTimes = self.Correlationx < (timeLimit / 1e6)

        # Return the fitted contributions and residuals of the first order autocorrelation function
        self.contributionsGuess, self.residualsG1, _   = get_contributios_prior(
            self.g1[selectedTimes],self.Correlationx[selectedTimes],self.s_space,self.betaGuess,alpha) 

        return None
    

    def getInitialEstimatesManyAlpha(self, alphaVec=(5**np.arange(-6,2,0.1,dtype=float))**2,timeLimit=1e8):

        """
        Apply the Tikhonov Philips regularisation for a given set of different values of alpha
        Useful to get afterwards the optimal alpha according to the L-curve criteria

        Result:

            We add curvesResidualNorm, curvesPenaltyNorm & alphaVec to the class object

            curvesResidualNorm contains the norm of the fidelity     term 
            curvesPenaltyNorm  contains the norm of the penalization term 
            alphaVec           contains the explored values of alpha

        """

        selectedTimes = self.Correlationx < (timeLimit / 1e6)

        curvesResidualNorm, curvesPenaltyNorm = [],[]

        self.alphaVec           = alphaVec

        # Iterate over the vector with different values of alpha
        for alpha in alphaVec:

            _ , residualNorm, penaltyNorm = get_contributios_prior(
                self.g1[selectedTimes],self.Correlationx[selectedTimes],
                self.s_space,self.betaGuess,alpha) 
          
            curvesResidualNorm.append(residualNorm) # List (one element per alpha) of lists (one element per curve)
            curvesPenaltyNorm.append(penaltyNorm)   # List (one element per alpha) of lists (one element per curve)

        self.curvesResidualNorm = np.array(curvesResidualNorm) # One row per alpha, one column per curve
        self.curvesPenaltyNorm  = np.array(curvesPenaltyNorm)  # One row per alpha, one column per curve

        return None
    
    def getOptimalAlphaLcurve(self):

        """
        Apply the triangle method to find the corner of the L-curve criteria en return the 'optimal' alpha for each curve
        """

        alphaOptIdx = []

        # Iterate over the curves
 

        alphaOptIdx.append(find_Lcurve_corner(self.curvesResidualNorm[:],self.curvesPenaltyNorm[:]))

        self.alphaOptIdx = alphaOptIdx

        return None
    

    def getInitialEstimatesOptimalAlphaLcurve(self,timeLimit=1e8):

        """
        Use the 'optimal' alpha selected using the L-curve corner criteria and the triangle method
        to estimate the distribution of (inverse) decay rates
        """

        self.optimalAlpha = [self.alphaVec[idx] for idx in self.alphaOptIdx]

        self.getInitialEstimates(self.optimalAlpha,timeLimit)

        return None

    def predictAutocorrelationCurves(self):

        # Create list to store the predicted autocorrelation data
        
        self.autocorrelationPredicted    = []


            
        betaEst = self.betaGuess
        contEst = self.contributionsGuess

        # check that we estimated the contributions!
        if len(contEst) > 1:
                
            self.autocorrelationPredicted           =  g2_finite_aproximation(1 / self.s_space,self.Correlationx,betaEst,contEst)
            # self.autocorrelationPredicted.append(np.array(autocorrelationPredicted))

        else:
            # In the case we couldn't fit anything!
            self.autocorrelationPredicted.append(np.array(0))



        return None
    

    def getWeights(self):

        """
        Compare the fitted and experimental autocorrelation curve to get the residuals
        and assign weights to each point

        Caution: Not used in the Raynals online tool!
        """

        residuals      = np.subtract(self.Correlationy,self.autocorrelationPredicted)
        weights        = 1 / np.abs(residuals)
        self.weights   = weights / weights.max(axis=0)

        return None
    
    def getWeightedInitialEstimates(self,alpha=0.15,timeLimit=1e8):

        """

        Call after fitting the g2 correlation curves
        that is, after running self.predictAutocorrelationCurves()

        Caution: Not used in the Raynals online tool!
        """

        if self.weights is None:

            self.getWeights()

        selectedTimes = self.Correlationx < (timeLimit / 1e6)

        # Return the fitted contributions and residuals of the first order autocorrelation function
        self.contributionsGuess, self.residualsG1   = get_contributios_prior(
            self.g1[selectedTimes],self.Correlationx[selectedTimes],self.s_space,self.betaGuess,alpha,self.weights) 

        return None
    
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
        
        for data in  self.get_data().values():
            if data.Samplename.lower() not in self.sample_names:
                self.sample_names.append(data.Samplename.lower())

            if data.Duration_s not in self.sample_durations:
                self.sample_durations.append(data.Duration_s)

            if round(data.Angle_deg,2) not in self.sample_angles:
                self.sample_angles.append(round(data.Angle_deg,2))

    
    def get_data(self, angle='all', data_type='all', sample_name='all', duration='all', time='all', indices=None):
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
        self.get_sample_names = []
        self.get_sample_durations = []
        self.get_sample_angles = []
        self.get_sample_temperatures = []
        self.get_sample_times = []
        self.get_sample_viscosities = []
        self.get_sample_wavelengths = []
        self.get_sample_refractive_indeces = []

        #filter type of samples 
        if data_type.lower() == 'all':
            data = self.solution_data + self.solvent_kalibration + self.standard_kalibration
        elif data_type.lower() == 'solution':
            data = self.solution_data
        elif data_type.lower() == 'solvent':
            data = self.solvent_kalibration
        elif data_type.lower() == 'standard':
            data = self.standard_kalibration
        else:
            raise(ValueError('not valid data type, all, solution, solvent and standard are available'))
        
        i = 0
        #loop over elements 
        for exp in data:
            
            # angle filter
            if (angle == 'all'):
                angle_check = True
            elif type(angle) == int or type(angle) == float:
                angle_check = angle - 0.01 < exp.Angle_deg < angle + 0.01
            elif type(angle) == list:
                angle_check = any(ang - 0.01 < exp.Angle_deg < ang + 0.01 for ang in angle)
            

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

            #time filter
            if (time == 'all'):
                time_check = True
            elif type(time) == str:
                time_check = time in exp.Time
            elif type(time) == list:
                time_check = any(tim in exp.Time for tim in time)


            # combine all filters 
            if angle_check and name_check and duration_check and time_check:
                if exp.Samplename.lower() not in self.get_sample_names:
                    self.get_sample_names.append(exp.Samplename.lower())

                if exp.Duration_s not in self.get_sample_durations:
                    self.get_sample_durations.append(exp.Duration_s)

                if round(exp.Angle_deg) not in self.get_sample_angles:
                    self.get_sample_angles.append(round(exp.Angle_deg))

                if exp.Temperature_K not in self.get_sample_temperatures:
                    self.get_sample_temperatures.append(exp.Temperature_K)

                if exp.Viscosity_cp not in self.get_sample_viscosities:
                    self.get_sample_viscosities.append(exp.Viscosity_cp)

                if exp.Wavelength_nm not in self.get_sample_wavelengths:
                    self.get_sample_wavelengths.append(exp.Wavelength_nm)

                if exp.Refractive_Index not in self.get_sample_refractive_indeces:
                    self.get_sample_refractive_indeces.append(exp.Refractive_Index)
                
                if exp.Time not in self.get_sample_times:
                    self.get_sample_times.append(exp.Time)

            
                experiment[f'{exp.Samplename}, angle {round(exp.Angle_deg)}, dur {round(exp.Duration_s)}, {exp.Time}'] = exp
                i += 1

        


        if type(indices) == int:

            return (*[name for i, name in enumerate(experiment.keys())  if i == indices],itemgetter(*[name for i, name in enumerate(experiment.keys())  if i == indices])(experiment))

        elif type(indices) in (float, int, str, list, tuple):
            raise ValueError(f'expected int, not {type(indices)}')
        
        if len(experiment) == 1:
           return itemgetter(*[name for name in experiment.keys()])(experiment)


        return dict(sorted(experiment.items()))

    

class contin_fit(dls_sls_analysis):
    def __init__(self, path, angle='all', data_type='solution',
                 sample_name='all', duration='all', indices=None,
                 continConfig=None):

        super().__init__(path)

        if indices is not None:
            self.data = self.get_data(
                angle=angle,
                data_type=data_type,
                sample_name=sample_name,
                duration=duration,
                indices=indices
            )
        else:
            self.data = list(
                self.get_data(
                    angle=angle,
                    data_type=data_type,
                    sample_name=sample_name,
                    duration=duration
                ).items()
            )

        
        if continConfig is not None:
            self.contin_config = continConfig
        else:
            self.contin_config = dict(recalc=True,
                                    ptRangeSec=(5e-7, 1e0), fitRangeM=(200e-9, 300e-9), gridpts=500,
                                    transformData=True, baselineCoeffs=1, # N_L
                                    # weighs noise level of data points accordinly for photon correlation spectroscopy
                                    # where the variance of Y is proportional to (Y**2+1)/(4*Y**2)
                                    # (from contin.for, line 1430)
                                    weighResiduals=True)
          
        self.fitRangeCrop = 30
        if 0 < self.fitRangeCrop < 100:
            self.contin_config['fitRangeM'] = (
                self.contin_config['fitRangeM'][0] * (1 - self.fitRangeCrop/100),
                self.contin_config['fitRangeM'][1] * (1 + self.fitRangeCrop/100),
    )
    
    def run_contin(self):
        resultDirs, summary = runContinOverFiles(self.data, self.contin_config)
        self.resultDirs = resultDirs
        self.summary = summary
        return resultDirs, summary
    
    def filter_results(self, angle='all', sample_name='all', duration='all', time='all',filter=None):
        filtered_results = []

        if filter:
            angle = filter.get('angle', angle)
            sample_name = filter.get('sample_name', sample_name)
            duration = filter.get('duration', duration)
            time = filter.get('time', time)

        for dn in self.resultDirs:
            angle_check = (angle == 'all') or (f'angle {angle}' in dn.name.lower())
            name_check = (sample_name == 'all') or (sample_name.lower() in dn.name.lower())
            duration_check = (duration == 'all') or (f'dur {duration}' in dn.name.lower())
            time_check = (time == 'all') or (time in dn.name)
            if angle_check and name_check and duration_check and time_check:
                filtered_results.append(dn)
        if not filtered_results:
            raise ValueError("No results match the specified filters.")
        return filtered_results
    
    def get_contin_result(self, dn=None, filter:dict=None,Number=None, volume=None):
        if dn is None and filter is None:
            result= {}
            for dn in self.resultDirs:
                print(dn)
                dfDistrib, dfFit, varmap = getContinResults(dn)
                dfDistrib['radius(nm)'] = dfDistrib['decay'] * varmap["gamma"]*1e9
                r_min = min(dfDistrib['radius(nm)']) * (1 + self.fitRangeCrop/100)
                r_max = max(dfDistrib['radius(nm)']) * (1 - self.fitRangeCrop/100)

                dfDistrib = dfDistrib[
                    (dfDistrib['radius(nm)'] > r_min) &
                    (dfDistrib['radius(nm)'] < r_max)
                ][['radius(nm)', 'distrib', 'err']]
                if Number == False and volume == False:
                    raise ValueError("Please provide either Number=True or volume=True to get specific distribution.")
                elif Number == True:
                    dfDistrib['distrib'] = dfDistrib['distrib']/dfDistrib['radius(nm)']**6
                    
                elif volume == True:
                    dfDistrib['distrib'] = dfDistrib['distrib']/dfDistrib['radius(nm)']**3
                dfDistrib['distrib'] /= dfDistrib['distrib'].sum()

                result[dn.name] = (dfDistrib, dfFit, varmap)

                

            return result
        
        elif dn is not None:
            if dn not in self.resultDirs:
                raise ValueError(f"The {dn}is not in the resultDirs. Please run run_contin() first or provide existing path.")
            dfDistrib, dfFit, varmap = getContinResults(dn)
            dfDistrib['radius(nm)'] = dfDistrib['decay'] * varmap["gamma"]*1e9
            r_min = min(dfDistrib['radius(nm)']) * (1 + self.fitRangeCrop/100)
            r_max = max(dfDistrib['radius(nm)']) * (1 - self.fitRangeCrop/100)

            dfDistrib = dfDistrib[
                (dfDistrib['radius(nm)'] > r_min) &
                (dfDistrib['radius(nm)'] < r_max)
            ][['radius(nm)', 'distrib', 'err']]
            if Number == False and volume == False:
                raise ValueError("Please provide either Number=True or volume=True to get specific distribution.")
            elif Number == True:
                dfDistrib['distrib'] = dfDistrib['distrib']/dfDistrib['radius(nm)']**6
            
            elif volume == True:
                dfDistrib['distrib'] = dfDistrib['distrib']/dfDistrib['radius(nm)']**3
            dfDistrib['distrib'] /= dfDistrib['distrib'].sum()

            return dfDistrib, dfFit, varmap
        
        elif filter is not None:
            filtered_results = self.filter_results(filter = filter)
            result = {}
            for dn in filtered_results:
                dfDistrib, dfFit, varmap = getContinResults(dn)
                dfDistrib['radius(nm)'] = dfDistrib['decay'] * varmap["gamma"]*1e9
                r_min = min(dfDistrib['radius(nm)']) * (1 + self.fitRangeCrop/100)
                r_max = max(dfDistrib['radius(nm)']) * (1 - self.fitRangeCrop/100)

                dfDistrib = dfDistrib[
                    (dfDistrib['radius(nm)'] > r_min) &
                    (dfDistrib['radius(nm)'] < r_max)
                ][['radius(nm)', 'distrib', 'err']]

                result[dn.name] = (dfDistrib, dfFit, varmap)
                if Number == False and volume == False:
                    raise ValueError("Please provide either Number=True or volume=True to get specific distribution.")
                elif Number == True:
                    dfDistrib['distrib'] = dfDistrib['distrib']/dfDistrib['radius(nm)']**6
                elif volume == True:
                    dfDistrib['distrib'] = dfDistrib['distrib']/dfDistrib['radius(nm)']**3
                dfDistrib['distrib'] /= dfDistrib['distrib'].sum()

            return result
        
        else:
            raise ValueError("Please provide either dn or filter to get specific results.")
        
       
    
    def plot_contin_results(self, dn=None, filter:dict=None):
        if dn is None and filter is None:
            for dn, (dfDistrib, dfFit, varmap) in self.get_contin_result().items():
                sns.set_theme()

      
                plt.plot(dfDistrib['radius(nm)'], dfDistrib['distrib'], label=dn)
                plt.xlabel('Hydrodynamic Radius (nm)')
                plt.ylabel('Distribution')
                plt.xlim(dfDistrib['radius(nm)'].min()*(1+self.fitRangeCrop/100), dfDistrib['radius(nm)'].max()*(1-self.fitRangeCrop/100))
                plt.title(f'hdr {dfDistrib["radius(nm)"][dfDistrib["distrib"]==max(dfDistrib["distrib"])].values} nm')
                plt.errorbar(dfDistrib['radius(nm)'], dfDistrib['distrib'], yerr=dfDistrib['err'], fmt='o', alpha=0.5)
                plt.grid()
                plt.legend()
                plt.show()



        # dfDistrib, dfFit, varmap = self.get_contin_result(dn)

        # plt.plot(dfDistrib['radius'], dfDistrib['distrib'], label=dn.name)
        # plt.xlabel('Hydrodynamic Radius (nm)')
        # plt.ylabel('Distribution')
        # plt.title(f'hdr {dfDistrib["radius"][dfDistrib["distrib"]==max(dfDistrib["distrib"])].values} nm')
        # plt.errorbar(dfFit['tau'], dfFit['g2_fit'], yerr=dfFit['g2_fit_std'], fmt='o', label='Fitted g2', alpha=0.5)
        # plt.grid()
        # plt.legend()
        # plt.show()


        # varmap

        # print(dfDistrib['radius'][dfDistrib['distrib']==max(dfDistrib['distrib'])].values)
    



def hydrodynamic_radius(refractive_index, wavelength, Gamma, angle_deg, temperature, viscosity):
    q = (4 * np.pi * refractive_index / (wavelength)) * np.sin(np.radians(angle_deg) / 2)
    D = Gamma / (q ** 2)
    R_h = (1.38e-23 * temperature) / (6 * np.pi * viscosity * D)*1e9  # in nm
    return R_h

# def predict_gamma(refractive_index, wavelength, radius, angle_deg, temperature, viscosity):
#     q = (4 * np.pi * refractive_index / (wavelength)) * np.sin(np.radians(angle_deg) / 2)
#     gamma = (1.38e-23 * temperature) / (6 * np.pi * viscosity * radius * 1e-9)  # in 1/s
#     return gamma

# def ks_from_theta(theta):
#     """Assume phi = 0 plane (x-z plane). theta in radians."""
#     return np.array([np.sin(theta), 0.0, np.cos(theta)])

# def q_from_ki_ks(ki, ks):
#     cos_theta = np.clip(np.dot(ki, ks), -1.0, 1.0)
#     theta = np.arccos(cos_theta)
#     q = np.sin(theta/2.0)
#     return q

# class contin_analysis()
#     def __init__(self, path, filter,)