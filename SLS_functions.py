import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_ASC_a import *
from scipy.optimize import curve_fit

plt.rcParams.update({
    "font.family": "Calibri",        # Setting Font type
    "xtick.direction": "in",         # x-Ticks inside
    "ytick.direction": "in",         # y-Ticks inside
    "xtick.top": True,               # Ticks on top
    "ytick.right": True,             # Ticks on rightside
    "xtick.labelsize": 12,           # Equal label sizes
    "ytick.labelsize": 12,
    "axes.labelsize": 15,            # Axis-labels same size
    'svg.fonttype': 'none'           # SVG fonttype setting, for post processing
})

class SLS_analysis:
    """
    Multi-concentration Static Light Scattering analysis:
    Zimm, Berry, Guinier + Debye form factor
    """

    def __init__(self, measurement_folder, concentrations, dn_dc, x_max):
        self.measurement_directory = measurement_folder
        self.concentrations = np.asarray(concentrations, dtype=float)
        self.dn_dc = dn_dc
        self.x_max = x_max

        self.datasets = self.build_datasets()
        self.results = self.zimm_analysis_multi_concentration(self.datasets)
        self.form_factor = self.determine_form_factor()

    # ------------------------------------------------------------------
    # DATA EXTRACTION
    # ------------------------------------------------------------------
    
    def build_datasets(self):
        datasets = list()
        
        # Loop through only the folders directly inside the main directory
        for folder, concentration in zip(os.listdir(self.measurement_directory), self.concentrations):
            folder_path = os.path.join(self.measurement_directory, folder)
            
            # Check if it is a folder
            if os.path.isdir(folder_path):
                # Apply your analysis function directly on this folder
                measurement = dls_sls_analysis(folder_path)
                results = self.zimm_analysis_data_single_folder(measurement, concentration)
                datasets.append(results)
                print(f"Processed: {folder_path}") 
                  
            else:
                raise ValueError('Measurement directory contains a file that is not a folder, this directory should only contain folders')
            
        return datasets
    
    def zimm_analysis_data_single_folder(self, measurement_reader, concentration):
        records = []

        # solution data
        for _, obj in measurement_reader.get_data(data_type="solution").items():
            records.append({
                "angle_deg": round(obj.Angle_deg),
                "mean_countrate": np.mean(obj.CountRatey),
                "monitor_diode": obj.MonitorDiode,
                "wavelength": obj.Wavelength_nm * 1e-9,
                "refractive_index": obj.Refractive_Index,
                "q": obj.q
            })

        df_sol = pd.DataFrame(records)

        sol_mean = df_sol.groupby("angle_deg")["mean_countrate"].mean()
        diode_mean = df_sol.groupby("angle_deg")["monitor_diode"].mean()

        # solvent
        records_solvent = []
        for _, obj in measurement_reader.get_data(data_type="solvent").items():
            records_solvent.append({
                "angle_deg": round(obj.Angle_deg),
                "mean_countrate": np.mean(obj.CountRatey)
            })

        df_solv = pd.DataFrame(records_solvent)
        solv_mean = df_solv.groupby("angle_deg")["mean_countrate"].mean()

        # average by angle
        sol_mean = df_sol.groupby("angle_deg")["mean_countrate"].mean()
        diode_mean = df_sol.groupby("angle_deg")["monitor_diode"].mean()
        solv_mean = df_solv.groupby("angle_deg")["mean_countrate"].mean()

        # align common angles
        common_angles = sol_mean.index.intersection(solv_mean.index)

        if len(common_angles) < 5:
            raise ValueError("Too few overlapping angles for solvent subtraction")

        solution_mean = sol_mean.loc[common_angles].sort_index()
        diode_mean = diode_mean.loc[common_angles].sort_index()
        solvent_mean = solv_mean.loc[common_angles].sort_index()

        # physics
        N_A = 6.022e23
        r = 0.14 # distance from cuvet to photodetector in [m]
        V = 2.5e-6 # Volume of a cuvet in [m^3]

        I0 = diode_mean.values
        I = solution_mean.values - solvent_mean.values

        Rθ = (I / I0) * (r**2 / V)

        wavelength = df_sol["wavelength"].mean()
        n = df_sol["refractive_index"].mean()

        optical_constant = (4*np.pi**2 / (wavelength**4 * N_A)) * (n * self.dn_dc)**2
        
        theta = np.radians(common_angles.values)

        x = np.sin(theta/2)**2
        y = (optical_constant * concentration) / Rθ

        scattering_vector = (
            df_sol.groupby("angle_deg")["q"]
            .mean()
            .loc[common_angles]
            .values
        )

        return x, y, concentration, wavelength, scattering_vector, n
    
    # ------------------------------------------------------------------
    # ZIMM ANALYSIS
    # ------------------------------------------------------------------

    def zimm_analysis_multi_concentration(self, datasets):
        intercepts = []
        slopes_theta = []
        concentrations = []

        for x, y, c, _, _, _ in datasets:
            mask = (
                np.isfinite(x) & np.isfinite(y) &
                (x > 0) & (y > 0) &
                (x < self.x_max)
            )

            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            intercepts.append(intercept)
            slopes_theta.append(slope)
            concentrations.append(c)

        intercepts = np.asarray(intercepts)
        slopes_theta = np.asarray(slopes_theta)
        concentrations = np.asarray(concentrations)

        slope_c, intercept_c = np.polyfit(concentrations, intercepts, 1)

        Mw = 1.0 / intercept_c
        A2 = slope_c / 2.0
        Rg = np.sqrt(np.mean(3 * Mw * slopes_theta))

        return dict(Mw=Mw, Rg=Rg, A2=A2)
    
    # ------------------------------------------------------------------
    # Form Factor
    # ------------------------------------------------------------------
    
    def determine_form_factor(self, normalized:bool=False):
        Mw = self.results['Mw']
        _, y, _, _, q, _ = min(self.datasets, key=lambda d: d[2])
        P_exp = 1.0 / (Mw * y)
        
        if normalized:
            P_exp /= P_exp.max()
            return q, P_exp, normalized
        
        return q, P_exp

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------

    def plot_zimm(self):
        plt.figure()
        for x, y, c, concentration, _, _ in self.datasets:
            plt.scatter(x + 2*concentration, y, label=f"c={c:.1e}")
        plt.xlabel(r"$\sin^2(\theta/2)$ + 2c")
        plt.ylabel(r"$Kc/R_\theta$")
        plt.legend(title="Concentration in g/mL", fancybox=True)
        plt.grid(alpha=0.5)
        plt.title("Zimm plot")
        plt.show()


    def plot_berry(self):
        plt.figure()
        for x, y, c, concentration, _, _ in self.datasets:
            plt.scatter(x + 2*concentration, np.sqrt(y), label=f"c={c:.1e}")
        plt.xlabel(r"$\sin^2(\theta/2)$  + 2c")
        plt.ylabel(r"$(Kc/R_\theta)^{1/2}$")
        plt.legend(title="Concentration in g/mL", fancybox=True)
        plt.grid(alpha=0.5)
        plt.title("Berry plot")
        plt.show()


    def plot_guinier(self):
        plt.figure()
        for _, y, c, concentration, q, _ in self.datasets:
            plt.scatter(q**2 + 2*concentration, np.log(1 / y), label=f"c={c:.1e}")
        plt.xlabel(r"$q^2$  + 2c")
        plt.ylabel(r"$\ln[I_0 / I_q]$")
        plt.legend(title="Concentration in g/mL", fancybox=True)
        plt.grid(alpha=0.5)
        plt.title("Guinier plot")
        plt.show()
        
    
    def plot_kratochvil(self, x_max=None):
        """
        Kratochvíl plot: (Kc/Rθ)/c vs concentration

        Parameters
        ----------
        datasets : list of tuples
            Each tuple: (x, y, c, wavelength, q_vector)
        x_max : float, optional
            Maximum sin²(θ/2) value for low-angle restriction
        """

        c_vals = []
        y_vals = []

        for x, y, c, *_, _ in self.datasets:
            x = np.asarray(x)
            y = np.asarray(y)

            if x_max is not None:
                mask = x < x_max
                if np.sum(mask) < 3:
                    continue
                y0 = np.mean(y[mask])   # θ → 0 estimate
            else:
                y0 = np.mean(y)

            c_vals.append(c)
            y_vals.append(y0 / c)

        c_vals = np.asarray(c_vals)
        y_vals = np.asarray(y_vals)

        # Sort by concentration
        idx = np.argsort(c_vals)
        c_vals = c_vals[idx]
        y_vals = y_vals[idx]

        # Plot
        plt.scatter(c_vals, y_vals, zorder=3)
        plt.plot(c_vals, y_vals, lw=1)

        plt.xlabel("Concentration $c$")
        plt.ylabel(r"$\frac{1}{c}\frac{Kc}{R_\theta}$")
        
        plt.ticklabel_format(axis='x', style='sci', scilimits=(min(c_vals), max(c_vals)))
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()
        
        
    def plot_guinier_selection(self, qRg_max):
        def guinier_mask_from_x(x, q, Rg):
            """
            Determine Guinier regime mask using q * Rg criterion.
            """
            x = np.asarray(x)
            q = np.asarray(q)
            return (q * Rg) < qRg_max
        
        x, y, _, wavelength, q, _ = min(self.datasets, key=lambda d: d[2])
        Rg = self.results["Rg"]

        print("min(q*Rg):", np.min(q * self.results["Rg"]))
        print("max(q*Rg):", np.max(q * self.results["Rg"]))

        mask = guinier_mask_from_x(x, Rg, wavelength)

        plt.scatter(q, y, label="Excluded", alpha=0.4)
        plt.scatter(q[mask], y[mask], label="Guinier region", zorder=3)

        plt.xlabel(r"$q\ (\mathrm{m^{-1}})$")
        plt.ylabel(r"$Kc/R_\theta$")
        plt.legend(fancybox=True)
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()
            
        
    def plot_form_factor(self, fit_data=None, fit_output: bool = False, fit_label_name: str = ''):
        q, Pq = self.form_factor

        plt.plot(q, Pq, 'o', label='Experimental form factor')

        if fit_output:
            if fit_data is None:
                raise ValueError("fit_data must be provided when fit_output=True")

            plt.plot(q, fit_data, '-', label=fit_label_name or 'Fit')

        plt.xlabel('Scattering vector (1/m)')
        plt.ylabel('Form factor P(q)')
        plt.legend(fancybox=True)
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()