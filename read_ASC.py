import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import numpy as np
import os
from operator import itemgetter
from pathlib import Path
from functions import *



class read_asc:
    """
    Read and analyze DLS .ASC files.

    This class:
    - Parses the raw ASCII file produced by the DLS instrument
    - Extracts metadata (temperature, angle, viscosity, etc.)
    - Extracts correlation and count-rate data
    - Fits a single-exponential decay to obtain diffusion parameters
    """

    def __init__(self, path):
        """
        Parameters
        ----------
        path : str or Path
            Path to the .ASC file
        """

        # Store file path
        self.Path = Path(path)

        # Read and parse the ASCII file into a dictionary
        self.data = self.ASC_2_dict(self.Path)

        # Measurement date and time
        self.Date = self.data['Date']
        self.Time = self.data['Time']

        # Sample identification
        self.Samplename = self.data['Samplename']
        self.SampleMemo = [self.data[f'SampMemo({i})'] for i in range(10)]

        # Experimental conditions
        self.Temperature_K = float(self.data['Temperature[K]'])     # Temperature in Kelvin
        self.Viscosity_cp = float(self.data['Viscosity[cp]'])       # Viscosity in centipoise
        self.Refractive_Index = float(self.data['RefractiveIndex']) # Solvent refractive index
        self.Wavelength_nm = float(self.data['Wavelength[nm]'])     # Laser wavelength
        self.Angle_deg = float(self.data['Angle[°]'])               # Scattering angle

        # Convert angle to radians
        self.angle_rad = np.deg2rad(self.Angle_deg)

        # Scattering vector magnitude q
        self.q = (
            4 * np.pi * self.Refractive_Index / (self.Wavelength_nm / 1e9)
        ) * np.sin(self.angle_rad / 2)

        # Optional cumulant analysis data (if present)
        if 'Cumulant_1' in self.data:
            self.MonitorDiode = float(self.data['Monitor diode'])
            self.Cumulant_1 = self.data['Cumulant_1']
            self.Cumulant_2 = self.data['Cumulant_2']
            self.Cumulant_3 = self.data['Cumulant_3']

        # Measurement settings
        self.Duration_s = float(self.data['Duration[s]'])
        self.Runs = float(self.data['Runs'])
        self.Mode = self.data['Mode']

        # Mean count rates
        self.MeanCR0_kHz = float(self.data['MeanCR0[kHz]'])
        self.MeanCR1_kHz = float(self.data['MeanCR1[kHz]'])

        # Autocorrelation function g2(tau)
        # Skip first 5 points (typically noisy)
        # Convert time from microseconds to milliseconds
        self.Correlationx = np.array(self.data['Correlationx'][5:])* 1e-3
        self.Correlationy = np.array(self.data['Correlationy'][5:])

        # Normalize correlation to its maximum value
        self.Correlationy /= self.Correlationy.max()

        # Photon count rate vs time
        self.CountRatex = np.array(self.data['CountRatex'])
        self.CountRatey = np.array(self.data['CountRatey'])

        # Optional standard deviation data
        if 'StandardDeviationx' in self.data:
            self.StandardDeviatiox = np.array(self.data['StandardDeviationx'][5:])*1e-3
            self.StandardDeviatioy = np.array(self.data['StandardDeviationy'][5:])

    def ASC_2_dict(self, path):
        """
        Parse a DLS .ASC file into a Python dictionary.

        The .ASC file is space-separated and poorly structured,
        so this method manually tokenizes the file and reconstructs
        the data blocks (metadata, correlation, count rate, cumulants).

        Returns
        -------
        dict
            Dictionary containing all extracted data
        """

        # Read raw file content
        with open(path, encoding='unicode_escape') as f:
            data = f.read()
        
        # Tokenize the file word by word
        word, previous_char, words = str(), str(), []

        for char in data:
            # Split on whitespace, newline or tab
            if (((char == " ") and (char != previous_char) ) or ((char == "\n") or (char == '\t'))):
                # Ignore unfinished quoted strings
                if word.count('"') == 1:
                    continue

                # Fix "Index:" keyword naming
                elif word == 'Index:':
                    words[-1] = 'RefractiveIndex'
                    words.append(":")
                    word = str()

                # Store completed words
                elif (word != ''):
                    word = word.replace('"','')
                    words.append(word)
                    word = str()
            
            elif (char != " ") and ((char != "\t") or (char != "\n")):
                word += char
            previous_char = char

        # cor_cou controls which data block we are reading
        # 0 = metadata
        # 1 = correlation
        # 2 = count rate
        # 3 = cumulants
        # 4 = standard deviation
        cor_cou = 0
        result = {}

        for i in range(len(words)):

            # Standard key:value entries
            if words[i] == ':' and words[i - 4] == ':':
                result[words[i - 2] + words[i - 1]] = words[i + 1]

            elif words[i] == ':':
                result[words[i - 1]] = words[i + 1]

            # Correlation function data
            elif cor_cou == 1 and words[i] != 'CountRate':
                if len(result['Correlationx']) <= len(result['Correlationy']):
                    result['Correlationx'].append(float(words[i]))
                else:
                    result['Correlationy'].append(float(words[i]))

            # Count rate data
            elif cor_cou == 2 and words[i] not in ('Monitor', 'StandardDeviation'):
                if len(result['CountRatex']) <= len(result['CountRatey']):
                    result['CountRatex'].append(float(words[i]))
                else:
                    result['CountRatey'].append(float(words[i]))

            # Monitor diode value
            elif words[i] == 'Monitor':
                result['Monitor diode'] = words[i + 2]

            # Cumulant analysis blocks
            elif cor_cou == 3 and words[i] == 'Cumulant1.Order':
                result['Cumulant_1']['Fluc_Freq'] = float(words[i + 3])
                result['Cumulant_1']['Diff_Cof'] = float(words[i + 6])
                result['Cumulant_1']['Hydr_Rad'] = float(words[i + 10])

            elif cor_cou == 3 and words[i] == 'Cumulant2.Order':
                result['Cumulant_2']['Fluc_Freq'] = float(words[i + 3])
                result['Cumulant_2']['Diff_Cof'] = float(words[i + 6])
                result['Cumulant_2']['Hydr_Rad'] = float(words[i + 10])
                result['Cumulant_2']['Exp_Par_2'] = float(words[i + 14])

            elif cor_cou == 3 and words[i] == 'Cumulant3.Order':
                result['Cumulant_3']['Fluc_Freq'] = float(words[i + 3])
                result['Cumulant_3']['Diff_Cof'] = float(words[i + 6])
                result['Cumulant_3']['Hydr_Rad'] = float(words[i + 10])
                result['Cumulant_3']['Exp_Par_2'] = float(words[i + 14])
                result['Cumulant_3']['Exp_Par_3'] = float(words[i + 18])

            # Standard deviation data
            elif cor_cou == 4:
                if len(result['StandardDeviationx']) <= len(result['StandardDeviationy']):
                    result['StandardDeviationx'].append(float(words[i]))
                else:
                    result['StandardDeviationy'].append(float(words[i]))

            # Detect start of each data block
            if words[i] == 'Correlation':
                result['Correlationx'], result['Correlationy'] = [], []
                cor_cou = 1

            elif words[i] == 'CountRate':
                result['CountRatex'], result['CountRatey'] = [], []
                cor_cou = 2

            elif words[i] == 'Monitor':
                result['Cumulant_1'] = {'Fluc_Freq': 0, 'Diff_Cof': 0, 'Hydr_Rad': 0}
                result['Cumulant_2'] = {'Fluc_Freq': 0, 'Diff_Cof': 0, 'Hydr_Rad': 0, 'Exp_Par_2': 0}
                result['Cumulant_3'] = {'Fluc_Freq': 0, 'Diff_Cof': 0, 'Hydr_Rad': 0,
                                        'Exp_Par_2': 0, 'Exp_Par_3': 0}
                cor_cou = 3

            elif words[i] == 'StandardDeviation':
                result['StandardDeviationx'], result['StandardDeviationy'] = [], []
                cor_cou = 4

        return result

    # ------------------------------------------------------------------

    def quick_plot_cor(self,deviation=False, title='name'):
        """
        Plot the normalized autocorrelation function g2(tau)
        """

        title_str = (
            f'correlation of angle {self.Angle_deg}'
            if title.lower() == 'angle'
            else f'correlation of {self.Samplename}'
        )

        sns.set_theme()
        plt.plot(self.Correlationx, self.Correlationy)
        if deviation:
            plt.errorbar(self.StandardDeviatiox,self.Correlationy,self.StandardDeviatioy)
        plt.xscale('log')
        plt.xlabel('Tau (s)')
        plt.ylabel('Correlation')
        plt.title(title_str)
        plt.show()

    def quick_plot_count(self, title='name'):
        """
        Plot the photon count rate vs time
        """

        title_str = (
            f'countrate of angle {self.Angle_deg}'
            if title.lower() == 'angle'
            else f'countrate of {self.Samplename}'
        )

        sns.set_theme()
        plt.plot(self.CountRatex, self.CountRatey)
        plt.xlabel('Time (s)')
        plt.ylabel('Count rate')
        plt.title(title_str)
        plt.show()

    # ------------------------------------------------------------------

    def g2_model(self, tau, beta, Gamma):
        """
        Single-exponential model for g2(tau) - 1
        """
        return beta * np.exp(-2 * Gamma * tau)

    def fit(self):
        """
        Fit the correlation function with a single-exponential decay.

        Returns
        -------
        beta : float
            Intercept (coherence factor)
        Gamma : float
            Decay rate (1/s)
        """

        beta0 = self.Correlationy[4]
        gamma0 = 1 / self.Correlationx[np.argmax(self.Correlationy < beta0 * 0.5)]

        popt, _ = curve_fit(
            self.g2_model,
            self.Correlationx,
            self.Correlationy,
            p0=[beta0, gamma0],
            maxfev=200000
        )

        return popt

    def plot_fit(self):
        """
        Plot experimental data together with fitted correlation curve
        and report hydrodynamic radius.
        """

        beta, Gamma = self.fit()
        g2_fit = self.g2_model(self.Correlationx, beta, Gamma)

        radius = hydrodynamic_radius(
            self.Refractive_Index,
            self.Wavelength_nm / 1e9,
            Gamma,
            self.Angle_deg,
            self.Temperature_K,
            self.Viscosity_cp / 1e3
        )

        print(
            f"Fitted parameters:\n"
            f"  beta = {beta:.3f}\n"
            f"  Diffusion coefficient = {Gamma / self.q**2 * 1e12:.5f} µm²/s\n"
            f"  Hydrodynamic radius = {radius:.2f} nm"
        )

        sns.set_theme()
        plt.figure(figsize=(8, 5))
        plt.plot(self.Correlationx, self.Correlationy, 'b.', label='Data')
        plt.plot(self.Correlationx, g2_fit, 'r-', label='Fit')
        plt.xscale('log')
        plt.xlabel('Tau (s)')
        plt.ylabel('g₂')
        plt.title(f'{self.Samplename} at {self.Angle_deg}°')
        plt.legend()
        plt.show()

 
class dls_sls_analysis:
    """
    Class for loading, filtering, and organizing DLS/SLS experiment data
    from a directory structure containing standard, solvent, and solution data.
    """

    def __init__(self, dict_path):
        """
        Initialize the analysis object.

        Parameters
        ----------
        dict_path : str
            Path to the main directory containing subfolders with ASC files.
            Folder names must contain 'standard', 'solvent', or 'solution'.
        """
        self.dict_path = dict_path

        # Lists with unique metadata values found in the dataset
        self.sample_names = []
        self.sample_durations = []
        self.sample_angles = []

        # Container for loaded experiment data
        self.sample_data = {}

        # Load all ASC files and classify them based on folder name
        for dict in os.listdir(self.dict_path):
            if 'standard' in dict.lower():
                self.standard_kalibration = [read_asc(path=os.path.join(self.dict_path, dict, file)) for file in os.listdir(os.path.join(self.dict_path, dict)) if file.endswith('.ASC')]
            elif 'solvent' in dict.lower():
                self.solvent_kalibration = [read_asc(path=os.path.join(self.dict_path, dict, file)) for file in os.listdir(os.path.join(self.dict_path, dict)) if file.endswith('.ASC')]
            elif 'solution' in dict.lower():
                self.solution_data = [read_asc(path=os.path.join(self.dict_path, dict, file)) for file in os.listdir(os.path.join(self.dict_path, dict)) if file.endswith('.ASC')]
        
        # Collect all unique sample names, durations, and angles
        # These are used later for validation of filter inputs
        for data in  self.get_data().values():
            if data.Samplename.lower() not in self.sample_names:
                self.sample_names.append(data.Samplename.lower())

            if data.Duration_s not in self.sample_durations:
                self.sample_durations.append(data.Duration_s)

            if round(data.Angle_deg,2) not in self.sample_angles:
                self.sample_angles.append(round(data.Angle_deg,2))

    
    def get_data(self, angle='all', data_type='all', sample_name='all', duration='all', time='all', indices=None):
        """
        Filter and retrieve experiment data based on metadata.

        Parameters
        ----------
        angle : float, list, or 'all'
            Scattering angle(s) in degrees.
        data_type : str
            One of ['all', 'solution', 'solvent', 'standard'].
        sample_name : str, list, or 'all'
            Sample name(s) to filter.
        duration : float, list, or 'all'
            Measurement duration(s) in seconds.
        time : str, list, or 'all'
            Time stamp(s) to filter.
        indices : int or None
            If provided, return only the experiment at this index.

        Returns
        -------
        dict or object
            Dictionary of experiments or a single experiment object.
        """

        # -------------------------------
        # Validate input filter arguments
        # -------------------------------
        search_vars = {'sample_name': sample_name, 'duration': duration,  'angle': angle}
        
        for var, val in search_vars.items():
            
            # Single-value validation
            if (val != 'all') and (type(val) != list):
                
                if (var == 'sample_name') and (val.lower() not in self.sample_names):
                    raise ValueError(f"Sample name '{val}' not found. Available names: {self.sample_names}")
                
                if (var == 'duration') and (val not in self.sample_durations):
                    raise ValueError(f"Duration '{val}' not found. Available names: {self.sample_durations}")               
                
                if (var == 'angle') and (val not in self.sample_angles):
                    raise ValueError(f"Angle '{val}' not found. Available names: {self.sample_angles}")
            
            # List-based validation
            if (val != 'all') and (type(val) == list):
                
                if (var == 'sample_name') and any(value for value in val if value not in self.sample_names):
                    raise ValueError(f"Sample name '{[name for name in val if name not in self.sample_names]}' not found. Available names: {self.sample_names}")
                
                if (var == 'duration') and any(value for value in val if value not in self.sample_durations):
                    raise ValueError(f"Durations '{[dur for dur in val if dur not in self.sample_durations]}' not found. Available durations: {self.sample_durations}")               
                
                if (var == 'angle') and any(value for value in val if value not in self.sample_angles):
                    raise ValueError(f"Angles '{[ang  for ang in val if ang not in self.sample_angles]}' not found. Available angles: {self.sample_angles}")
        
        # ----------------------------------
        # Containers for filtered experiment
        # ----------------------------------
        experiment = {}
        
        # Metadata collected for the filtered selection
        self.get_sample_names = []
        self.get_sample_durations = []
        self.get_sample_angles = []
        self.get_sample_temperatures = []
        self.get_sample_times = []
        self.get_sample_viscosities = []
        self.get_sample_wavelengths = []
        self.get_sample_refractive_indeces = []

        # ---------------------------
        # Select data source(s)
        # ---------------------------
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
        

        # ---------------------------
        # Apply all filters
        # ---------------------------
        for exp in data:
            
           # Angle filter (with tolerance)
            if (angle == 'all'):
                angle_check = True
            
            elif type(angle) == int or type(angle) == float:
                angle_check = angle - 0.01 < exp.Angle_deg < angle + 0.01
            
            elif type(angle) == list:
                angle_check = any(ang - 0.01 < exp.Angle_deg < ang + 0.01 for ang in angle)
            

            # Sample name filter (substring match)
            if (sample_name == 'all'):
                name_check = True

            elif type(sample_name) == str:
                name_check = sample_name.lower() in exp.Samplename.lower()

            elif type(sample_name) == list:
                name_check = any(name.lower() in exp.Samplename.lower() for name in sample_name)
            

           # Duration filter (±1 s tolerance)
            if (duration == 'all'):
                duration_check = True

            elif type(duration) == int or type(duration) == float:
                duration_check = duration - 1 < exp.Duration_s < duration + 1

            elif type(duration) == list:
                duration_check = any(dur - 1 < exp.Duration_s < dur + 1 for dur in duration)

            # Time filter (string containment)
            if (time == 'all'):
                time_check = True
            elif type(time) == str:
                time_check = time in exp.Time
            elif type(time) == list:
                time_check = any(tim in exp.Time for tim in time)


            # ----------------------------------
            # Combine all filter conditions
            # ----------------------------------
            if angle_check and name_check and duration_check and time_check:
                
                # Collect unique metadata for the filtered result
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

                # Store experiment using a descriptive key
                experiment[f'{exp.Samplename}, angle {round(exp.Angle_deg)}, dur {round(exp.Duration_s)}, {exp.Time}'] = exp

        # ---------------------------
        # Optional index-based access
        # ---------------------------
        if type(indices) == int:

            return (*[name for i, name in enumerate(experiment.keys())  if i == indices],itemgetter(*[name for i, name in enumerate(experiment.keys())  if i == indices])(experiment))

        elif type(indices) in (float, int, str, list, tuple):
            raise ValueError(f'expected int, not {type(indices)}')
        
        # Return single experiment directly (not wrapped in dict)
        if len(experiment) == 1:
           return itemgetter(*[name for name in experiment.keys()])(experiment)
        
        return dict(sorted(experiment.items()))

    

class contin_fit(dls_sls_analysis):
    """
    Extension of dls_sls_analysis that runs CONTIN fits on selected DLS data
    and provides filtering, post-processing, and plotting utilities.
    """

    def __init__(self, path, angle='all', data_type='solution',
                 sample_name='all', duration='all', indices=None,
                 continConfig=None):
        """
        Initialize CONTIN fitting workflow.

        Parameters
        ----------
        path : str
            Path to the directory containing DLS/SLS data.
        angle, data_type, sample_name, duration :
            Filters passed to get_data() to select experiments.
        indices : int or None
            If provided, selects a single experiment by index.
        continConfig : dict or None
            Optional custom CONTIN configuration dictionary.
        """
        # Initialize parent class and load all experiment data
        super().__init__(path)

        # Retrieve data to be fitted
        # If indices is provided, a single experiment is returned
        if indices is not None:
            self.data = self.get_data(
                angle=angle,
                data_type=data_type,
                sample_name=sample_name,
                duration=duration,
                indices=indices
            )
        else:
            # Otherwise, retrieve all matching experiments as a list of (key, value)
            self.data = list(
                self.get_data(
                    angle=angle,
                    data_type=data_type,
                    sample_name=sample_name,
                    duration=duration
                ).items()
            )

        # CONTIN configuration:
        # either use user-provided settings or fall back to defaults
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

        # Percentage used to crop the fitted size range
        # This removes edge artifacts from the CONTIN solution   
        self.fitRangeCrop = 30

        # Expand fit range slightly before cropping
        if 0 < self.fitRangeCrop < 100:
            self.contin_config['fitRangeM'] = (
                self.contin_config['fitRangeM'][0] * (1 - self.fitRangeCrop/100),
                self.contin_config['fitRangeM'][1] * (1 + self.fitRangeCrop/100),
    )
    
    def run_contin(self):
        """
        Run CONTIN on all selected datasets.

        Returns
        -------
        resultDirs : list
            Directories containing CONTIN results.
        summary : object
            Summary output from runContinOverFiles().
        """
        resultDirs, summary = runContinOverFiles(self.data, self.contin_config)
        self.resultDirs = resultDirs
        self.summary = summary
        return resultDirs, summary
    
    def filter_results(self, angle='all', sample_name='all', duration='all', time='all',filter=None):
        """
        Filter CONTIN result directories based on metadata encoded
        in directory names.

        Parameters
        ----------
        angle, sample_name, duration, time :
            Filters applied to directory names.
        filter : dict or None
            Optional dictionary containing filter values.

        Returns
        -------
        list
            Filtered list of result directories.
        """
        filtered_results = []

        # Override individual filters if a filter dictionary is provided
        if filter:
            angle = filter.get('angle', angle)
            sample_name = filter.get('sample_name', sample_name)
            duration = filter.get('duration', duration)
            time = filter.get('time', time)

        # Apply string-based matching on directory names
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
        """
        Load and post-process CONTIN results.

        Parameters
        ----------
        dn : Path or None
            Specific result directory to load.
        filter : dict or None
            Filter applied to result directories.
        Number : bool
            Convert to number-weighted distribution if True.
        volume : bool
            Convert to volume-weighted distribution if True.

        Returns
        -------
        dict or tuple
            CONTIN results including size distribution, fit, and metadata.
        """

        # Case 1: return results for all CONTIN directories
        if dn is None and filter is None:
            result= {}
            
            for dn in self.resultDirs:
                dfDistrib, dfFit, varmap = getContinResults(dn)
                
                # Convert decay rates to hydrodynamic radius (nm)
                dfDistrib['radius(nm)'] = dfDistrib['decay'] * varmap["gamma"]*1e9
                
                # Crop edges of size distribution
                r_min = min(dfDistrib['radius(nm)']) * (1 + self.fitRangeCrop/100)
                r_max = max(dfDistrib['radius(nm)']) * (1 - self.fitRangeCrop/100)

                dfDistrib = dfDistrib[
                    (dfDistrib['radius(nm)'] > r_min) &
                    (dfDistrib['radius(nm)'] < r_max)
                ][['radius(nm)', 'distrib', 'err']]
                
                # Ensure exactly one weighting mode is chosen
                if Number == False and volume == False:
                    raise ValueError("Please provide either Number=True or volume=True to get specific distribution.")
               
                elif Number == True:
                    dfDistrib['distrib'] = dfDistrib['distrib']/dfDistrib['radius(nm)']**6
                    
                elif volume == True:
                    dfDistrib['distrib'] = dfDistrib['distrib']/dfDistrib['radius(nm)']**3
                
                # Normalize distribution
                dfDistrib['distrib'] /= dfDistrib['distrib'].sum()

                result[dn.name] = (dfDistrib, dfFit, varmap)

            return result
        
        # Case 2: return result for a specific directory
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
        
        # Case 3: return results matching a filter
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

                
                if Number == False and volume == False:
                    raise ValueError("Please provide either Number=True or volume=True to get specific distribution.")
                
                elif Number == True:
                    dfDistrib['distrib'] = dfDistrib['distrib']/dfDistrib['radius(nm)']**6
                
                elif volume == True:
                    dfDistrib['distrib'] = dfDistrib['distrib']/dfDistrib['radius(nm)']**3
                
                dfDistrib['distrib'] /= dfDistrib['distrib'].sum()
                
                result[dn.name] = (dfDistrib, dfFit, varmap)
            
            return result
        
        else:
            raise ValueError("Please provide either dn or filter to get specific results.")
        
       
    
    def plot_contin_results(self, dn=None, filter: dict = None):
        """
        Plot CONTIN size distributions.

        Parameters
        ----------
        dn : Path or None
            Specific CONTIN result directory to plot.
        filter : dict or None
            Dictionary of filters (angle, sample_name, duration, time).
        """

        # Retrieve CONTIN results depending on input
        if dn is not None:
            # Single result directory
            results = {dn.name: self.get_contin_result(dn=dn)}
        else:
            # All results or filtered results
            results = self.get_contin_result(filter=filter)

        sns.set_theme()

        for name, (dfDistrib, dfFit, varmap) in results.items():

            # Plot size distribution
            plt.plot(
                dfDistrib['radius(nm)'],
                dfDistrib['distrib'],
                label=name
            )

            # Add error bars
            plt.errorbar(
                dfDistrib['radius(nm)'],
                dfDistrib['distrib'],
                yerr=dfDistrib['err'],
                fmt='o',
                alpha=0.5
            )

            plt.xlabel('Hydrodynamic Radius (nm)')
            plt.ylabel('Distribution')

            # Plot range consistent with cropped fit range
            plt.xlim(
                dfDistrib['radius(nm)'].min() * (1 + self.fitRangeCrop / 100),
                dfDistrib['radius(nm)'].max() * (1 - self.fitRangeCrop / 100)
            )

            # Annotate peak hydrodynamic radius
            peak_radius = dfDistrib.loc[
                dfDistrib['distrib'].idxmax(), 'radius(nm)'
            ]
            plt.title(f'HDR peak ≈ {peak_radius:.2f} nm')

            plt.grid()
            plt.legend()
            plt.show()




    



def hydrodynamic_radius(refractive_index, wavelength, Gamma, angle_deg, temperature, viscosity):
    q = (4 * np.pi * refractive_index / (wavelength)) * np.sin(np.radians(angle_deg) / 2)
    D = Gamma / (q ** 2)
    R_h = (1.38e-23 * temperature) / (6 * np.pi * viscosity * D)*1e9  # in nm
    return R_h

