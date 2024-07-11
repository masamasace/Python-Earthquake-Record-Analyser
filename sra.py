from pathlib import Path
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import gc
import scipy.signal as ss
import matplotlib.dates as mdates


plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = "k"

def setup_figure(num_row=1, num_col=1, width=5, height=4, left=0.125, right=0.9, hspace=0.2, wspace=0.2):

    fig, axes = plt.subplots(num_row, num_col, figsize=(width, height), squeeze=False)   
    fig.subplots_adjust(left=left, right=right, hspace=hspace, wspace=wspace)
    return (fig, axes)

# JMA intergration method with use of reccurence formula
# CITE: https://www.data.jma.go.jp/eqev/data/kyoshin/kaisetsu/calc_wave.html
def acc2vel_jma(acc):

    G = 0.004937561699
    A1 = -2.974867761716
    A2 = 2.950050339269
    A3 = -0.975180618018
    B0 = 1.0
    B1 = -1.0
    B2 = -1.0
    B3 = 1.0
    
    v = np.zeros(len(acc))
    
    for i in range(3, len(acc)):
        v[i] = G * (B0 * acc[i] + B1 * acc[i-1] + B2 * acc[i-2] + B3 * acc[i-3]) - (A1 * v[i-1] + A2 * v[i-2] + A3 * v[i-3])
        
    return v

def acc2disp_jma(acc):

    H = 1.0
    C1 = -1.988438073558305
    C2 = 0.9885471048650272
    D0 = 0.00002485615736514583
    D1 = 0.00004971231473029166
    D2 = 0.00002485615736514583
    
    d = np.zeros(len(acc))
    
    for i in range(2, len(acc)):
        d[i] = H * (D0 * acc[i] + D1 * acc[i-1] + D2 * acc[i-2]) - (C1 * d[i-1] + C2 * d[i-2])
    
    return d   

# base class for seismic record
class SeismicRecord:
    def __init__(self, record_path, record_type="JMA", parzen_width=0, record_interval=0.01, h=0.05, 
                 integration_method="JMA", integration_fft_butterworth_cutoff=0.1, integration_fft_butterworth_order=4) -> None:
        
        self.record_type = record_type
        self.record_path = Path(record_path).resolve()
        self.parzen_width = parzen_width
        self.record_interval = record_interval
        self.h = h
        self.integration_method = integration_method
        self.integration_fft_butterworth_cutoff = integration_fft_butterworth_cutoff
        self.integration_fft_butterworth_order = integration_fft_butterworth_order
        
        # check intergration method
        if self.integration_method != "JMA" and self.integration_method != "fft":
            raise ValueError("Invalid integration method!")
        
        # create result folder
        self.result_folder = self.record_path.parent / "result"
        self.result_folder.mkdir(exist_ok=True, parents=True)
        
        print("File Path:", self.record_path)
        
        if self.record_type == "JMA":
            encoding_chr = "shift-jis"
            self.col_names = ["NS_acc", "EW_acc", "UD_acc"]
            self.record_data = pd.read_csv(self.record_path,
                                           encoding=encoding_chr, 
                                           names=self.col_names)
            
            # Load header info
            start_time_str = self.record_data.iloc[5, 0][14:]
            temp_start_time_format = "%Y %m %d %H %M %S"
            self.start_time = datetime.datetime.strptime(start_time_str, temp_start_time_format)
            
            # Delite header info
            self.record_data.drop(range(7), inplace=True)
            
            # Add time column
            self.record_data["Time"] = [self.start_time + datetime.timedelta(seconds=i * self.record_interval) for i in range(len(self.record_data))]

            self.record_data[self.col_names] = self.record_data[self.col_names].astype(float)
            self.record_data = self.record_data[["Time", "NS_acc", "EW_acc", "UD_acc"]]
            
                    
        elif self.record_type == "NIED":
            
            # prepare file path for each component
            temp_record_path_NS = self.record_path.parent / (self.record_path.stem + ".NS")
            temp_record_path_EW = self.record_path.parent / (self.record_path.stem + ".EW")
            temp_record_path_UD = self.record_path.parent / (self.record_path.stem + ".UD")
            
            temp_record_paths = [temp_record_path_NS, temp_record_path_EW, temp_record_path_UD]
            
            temp_record = []
            
            for temp_each_record_path in temp_record_paths:
                temp_each_record_header_data = pd.read_csv(temp_each_record_path, nrows=16, header=None)
                
                temp_start_time_format = "%Y/%m/%d %H:%M:%S"
                
                temp_scale_factor_bra_index = temp_each_record_header_data.iloc[13].values[0].find("(")
                temp_scale_factor_slash_index = temp_each_record_header_data.iloc[13].values[0].find("/")
                
                temp_each_record_header = {"start_time": datetime.datetime.strptime(temp_each_record_header_data.iloc[9].values[0][18:], 
                                                                                      temp_start_time_format),
                                           "sampling_freq": float(temp_each_record_header_data.iloc[10].values[0][18:21]),
                                           "scale_factor": float(temp_each_record_header_data.iloc[13].values[0][18:temp_scale_factor_bra_index]) /\
                                                           float(temp_each_record_header_data.iloc[13].values[0][temp_scale_factor_slash_index+1:])}
                
                colspecs = [(9*i, 9*i+8) for i in range(8)]
                temp_each_record_data = pd.read_fwf(temp_each_record_path, colspecs=colspecs, skiprows=17, header=None)
                temp_each_record_data = temp_each_record_data.values.T.flatten("F")
                temp_each_record_data = temp_each_record_data[~np.isnan(temp_each_record_data)] * temp_each_record_header["scale_factor"]
                
                temp_record.append([temp_each_record_header, temp_each_record_data])
            
            self.start_time = temp_record[0][0]["start_time"]
            self.record_interval = 1 / temp_record[0][0]["sampling_freq"]
            temp_record_time = np.array([self.start_time + datetime.timedelta(seconds=i * self.record_interval) for i in range(len(temp_record[0][1]))])
            temp_record_NS = temp_record[0][1]
            temp_record_EW = temp_record[1][1]
            temp_record_UD = temp_record[2][1]
            
            self.col_names = ["Time", "NS_acc", "EW_acc", "UD_acc"]
            self.record_data = pd.DataFrame(np.vstack([temp_record_time, temp_record_NS, temp_record_EW, temp_record_UD]).T, 
                                            columns=self.col_names)
            
        elif self.record_type == "HG":
            
            # load acceleration record
            encoding_chr = "shift-jis"
            self.col_names = ["NS_acc", "EW_acc", "UD_acc"]
            self.record_data = pd.read_csv(self.record_path,
                                           encoding=encoding_chr, 
                                           names=self.col_names,
                                           skiprows=37)
            
            # load start time
            with open(self.record_path, "r", encoding=encoding_chr) as f:
                for i, line in enumerate(f):
                    if i == 5:
                        start_time_str = line.split(",")[1]
                        temp_start_time_format = "%Y/%m/%d %H:%M:%S"
                        self.start_time = datetime.datetime.strptime(start_time_str, temp_start_time_format)
                        break        
            
            # Add time column
            self.record_data["Time"] = [self.start_time + datetime.timedelta(seconds=i * self.record_interval) for i in range(len(self.record_data))]

            self.record_data[self.col_names] = self.record_data[self.col_names].astype(float)
            self.record_data = self.record_data[["Time", "NS_acc", "EW_acc", "UD_acc"]] 
            
            self.col_names = self.record_data.columns.values           
            
        else:
            raise ValueError("Invalid record type!")
        
        self._calcurate_additional_parameter()

    
    def _calcurate_additional_parameter(self):
                
        # calcurate basic parameter
        self.record_length = len(self.record_data)
        
        # calcurate holizontal component
        self.record_data["H_acc"] = (self.record_data["NS_acc"] ** 2 + self.record_data["EW_acc"] ** 2) ** (1/2)
        
        # calcurate FFT
        self._calcurate_fft()
        
        # compute velocity and displacement
        self._calcurate_velocity_displacement()
        
        # calcurate absolute maximum value with sign in each component
        temp_record_data_abs_max = self.record_data.loc[self.record_data[self.col_names[1:]].abs().idxmax()]
        
        # create dictionary for each component
        self.record_data_abs_max = np.array([temp_record_data_abs_max.iloc[i, i+1] for i in range(0, len(temp_record_data_abs_max))])
        
        for i in range(len(self.fft_col_names[1:])):
                temp_col_name = self.fft_col_names[1 + i] + "_smoothed"       
                self.fft_record_data[temp_col_name] = self.fft_record_data[self.fft_col_names[1 + i]]
        
        # TODO:Viewwaveとの整合性が取れていない
        # apply parzen window
        if self.parzen_width > 0:
            number_of_points_parzen_width = int(self.parzen_width / self.freq_interval)
            parzen_window = ss.windows.parzen(number_of_points_parzen_width)
            parzen_window = parzen_window / sum(parzen_window)
            
            for i in range(len(self.fft_col_names[1:])):
                temp_col_name = self.fft_col_names[1 + i] + "_smoothed"       
                self.fft_record_data[temp_col_name] = np.convolve(np.abs(self.fft_record_data[self.fft_col_names[1 + i]]), 
                                                                                  parzen_window, mode="same") 
        
        print("finish apply Parzen window!")
        
        # compute response spectrum
        self.response_spectrum = pd.DataFrame({"nFreq": [], "NS_acc_resp": [], "EW_acc_resp": [], "UD_acc_resp": [], 
                                              "NS_vel_resp": [], "EW_vel_resp": [], "UD_vel_resp": [],
                                              "NS_disp_resp": [], "EW_disp_resp": [], "UD_disp_resp": []})
                                                
        self.response_spectrum["nFreq"] = np.logspace(np.log10(0.05), np.log10(20), 200)
        temp_omega = 2 * np.pi * self.fft_record_data["Freq"]
        
        for i, n_freq in enumerate(self.response_spectrum["nFreq"]):
            
            temp_n_omega = 2 * np.pi * n_freq
            
            # compute transfer function
            temp_H = 1 / (temp_omega**2 - 2j * temp_n_omega * self.h * temp_omega - temp_n_omega**2)
            
            # calcurate time-series response by inverse Fourier transform
            temp_ifft_NS_disp = np.fft.ifft(self.fft_record_data["NS_acc"] * temp_H, norm="backward")
            temp_ifft_EW_disp = np.fft.ifft(self.fft_record_data["EW_acc"] * temp_H, norm="backward")
            temp_ifft_UD_disp = np.fft.ifft(self.fft_record_data["UD_acc"] * temp_H, norm="backward")
            
            temp_ifft_NS_vel = np.fft.ifft(self.fft_record_data["NS_acc"] * temp_H * 1j * temp_omega, norm="backward")
            temp_ifft_EW_vel = np.fft.ifft(self.fft_record_data["EW_acc"] * temp_H * 1j * temp_omega, norm="backward")
            temp_ifft_UD_vel = np.fft.ifft(self.fft_record_data["UD_acc"] * temp_H * 1j * temp_omega, norm="backward")
            
            temp_ifft_NS_acc_abs = np.fft.ifft(self.fft_record_data["NS_acc"] * temp_H * -temp_omega**2, norm="backward")
            temp_ifft_EW_acc_abs = np.fft.ifft(self.fft_record_data["EW_acc"] * temp_H * -temp_omega**2, norm="backward")
            temp_ifft_UD_acc_abs = np.fft.ifft(self.fft_record_data["UD_acc"] * temp_H * -temp_omega**2, norm="backward")
            
            temp_ifft_NS_acc_rel = temp_ifft_NS_acc_abs + self.record_data["NS_acc"]
            temp_ifft_EW_acc_rel = temp_ifft_EW_acc_abs + self.record_data["EW_acc"]
            temp_ifft_UD_acc_rel = temp_ifft_UD_acc_abs + self.record_data["UD_acc"]
            
            # calcurate absolute maximum value of each response component
            self.response_spectrum.loc[i, "NS_acc_resp_abs"] = np.abs(temp_ifft_NS_acc_abs).max()
            self.response_spectrum.loc[i, "EW_acc_resp_abs"] = np.abs(temp_ifft_EW_acc_abs).max()
            self.response_spectrum.loc[i, "UD_acc_resp_abs"] = np.abs(temp_ifft_UD_acc_abs).max()
            
            self.response_spectrum.loc[i, "NS_acc_resp_rel"] = np.abs(temp_ifft_NS_acc_rel).max()
            self.response_spectrum.loc[i, "EW_acc_resp_rel"] = np.abs(temp_ifft_EW_acc_rel).max()
            self.response_spectrum.loc[i, "UD_acc_resp_rel"] = np.abs(temp_ifft_UD_acc_rel).max()
            
            self.response_spectrum.loc[i, "NS_vel_resp"] = np.abs(temp_ifft_NS_vel).max()
            self.response_spectrum.loc[i, "EW_vel_resp"] = np.abs(temp_ifft_EW_vel).max()
            self.response_spectrum.loc[i, "UD_vel_resp"] = np.abs(temp_ifft_UD_vel).max()
            
            self.response_spectrum.loc[i, "NS_disp_resp"] = np.abs(temp_ifft_NS_disp).max()
            self.response_spectrum.loc[i, "EW_disp_resp"] = np.abs(temp_ifft_EW_disp).max()
            self.response_spectrum.loc[i, "UD_disp_resp"] = np.abs(temp_ifft_UD_disp).max()
            
        print("finish calcurate Response Spectrum!")

        self.fft_col_names = self.fft_record_data.columns.values
    
    def _calcurate_fft(self):
        
        # calcurate Fourier Spectrum
        temp_fft_col_names = self.col_names[1:4]
        temp_record_data = self.record_data[temp_fft_col_names]
        
        temp_freq = np.fft.fftfreq(len(temp_record_data), d=self.record_interval)
        self.freq_interval = temp_freq[1] - temp_freq[0]
        
        temp_fft_record_data = np.fft.fft(temp_record_data, norm="backward", axis=0)
        
        # stack temp_freq and temp_fft_record_data
        temp_fft_record_data = np.vstack([temp_freq, temp_fft_record_data.T]).T
        
        # convert to DataFrame
        self.fft_col_names = np.append(["Freq"], temp_fft_col_names)
        self.fft_record_data = pd.DataFrame(temp_fft_record_data, columns=self.fft_col_names)
        
        # convert freq column to float
        self.fft_record_data["Freq"] = self.fft_record_data["Freq"].abs()
        
        print("finish calcurate Fourier Spectrum!")
    
    def _calcurate_velocity_displacement(self):
        
        if self.integration_method == "JMA":
            
            self.record_data["NS_vel"] = acc2vel_jma(self.record_data["NS_acc"].values)
            self.record_data["EW_vel"] = acc2vel_jma(self.record_data["EW_acc"].values)
            self.record_data["UD_vel"] = acc2vel_jma(self.record_data["UD_acc"].values)
            
            self.record_data["NS_disp"] = acc2disp_jma(self.record_data["NS_acc"].values)
            self.record_data["EW_disp"] = acc2disp_jma(self.record_data["EW_acc"].values)
            self.record_data["UD_disp"] = acc2disp_jma(self.record_data["UD_acc"].values)
        
        # TODO: FFT integration method
        elif self.integration_method == "fft":
            
            # apply highpass filter and calculate velocity and displacement
            temp_b, temp_a = ss.butter(self.integration_fft_butterworth_order,
                                        self.integration_fft_butterworth_cutoff,
                                        btype="high", analog=False, output="ba", fs=1/self.record_interval)
            
            _, temp_h = ss.freqz(temp_b, temp_a, worN=len(self.fft_record_data) // 2, fs=1/self.record_interval)
            
            temp_h_extended = np.append(temp_h, temp_h[-1])
            temp_h_extended = np.append(temp_h_extended, np.flip(temp_h)[:-1])
            
            temp_h_extended[0] = 0
            
            temp_vel_fft = self.fft_record_data.copy().values[:, 1:]
            temp_disp_fft = self.fft_record_data.copy().values[:, 1:]
            
            temp_omega = 2 * np.pi * self.fft_record_data["Freq"].values[1:]
            
            temp_vel_fft[1:, 0] = temp_vel_fft[1:, 0] / (1j * temp_omega) * temp_h_extended[1:]
            temp_vel_fft[1:, 1] = temp_vel_fft[1:, 1] / (1j * temp_omega) * temp_h_extended[1:]
            temp_vel_fft[1:, 2] = temp_vel_fft[1:, 2] / (1j * temp_omega) * temp_h_extended[1:]
            
            temp_disp_fft[1:, 0] = temp_disp_fft[1:, 0] / (-temp_omega**2) * temp_h_extended[1:]
            temp_disp_fft[1:, 1] = temp_disp_fft[1:, 1] / (-temp_omega**2) * temp_h_extended[1:]
            temp_disp_fft[1:, 2] = temp_disp_fft[1:, 2] / (-temp_omega**2) * temp_h_extended[1:]
            
            temp_vel_ifft = np.fft.ifft(temp_vel_fft, axis=0, norm="backward").real
            temp_disp_ifft = np.fft.ifft(temp_disp_fft, axis=0, norm="backward").real
            
            self.record_data["NS_vel"] = temp_vel_ifft[:, 0]
            self.record_data["EW_vel"] = temp_vel_ifft[:, 1]
            self.record_data["UD_vel"] = temp_vel_ifft[:, 2]
            
            self.record_data["NS_disp"] = temp_disp_ifft[:, 0]
            self.record_data["EW_disp"] = temp_disp_ifft[:, 1]
            self.record_data["UD_disp"] = temp_disp_ifft[:, 2]
            
        else:
            raise ValueError("Invalid integration method!")
            
        
        self.col_names = self.record_data.columns.values
        print("finish calcurate Velocity and Displacement!")
    
    # get record data
    def get_max_value(self):

        return self.record_data_abs_max
        
    # export time-series record     
    def export_time_series_record(self, xlim=[], ylim=[], second_locator=[0]) -> None:
        
        flag_set_xlim = False
        
        # format xlim and ylim 
        if type(xlim) == list:
            if len(xlim) == 0:
                flag_set_xlim = False
            elif len(xlim) == 2:
                try:
                    xlim = [datetime.datetime.strptime(xlim[0], "%H:%M:%S"), datetime.datetime.strptime(xlim[1], "%H:%M:%S")]
                    flag_set_xlim = True
                except:
                    raise ValueError("Invalid xlim!")
        else: 
            raise ValueError("Invalid xlim!")
        
        if len(ylim) == 0:
            ylim = [np.max(np.abs(self.record_data_abs_max[0:3])), np.max(np.abs(self.record_data_abs_max[4:7])), np.max(np.abs(self.record_data_abs_max[7:]))]
        elif len(ylim) != 3:
            raise ValueError("Invalid ylim!")
        
        # setup figure
        fig, axes = setup_figure(num_row=9, hspace=.125, width=8, height=12)
        
        # plot time-series record
        axes[0, 0].plot(self.record_data["Time"], self.record_data["NS_acc"], "r", linewidth=0.5)
        axes[1, 0].plot(self.record_data["Time"], self.record_data["EW_acc"], "g", linewidth=0.5)
        axes[2, 0].plot(self.record_data["Time"], self.record_data["UD_acc"], "b", linewidth=0.5)
        axes[3, 0].plot(self.record_data["Time"], self.record_data["NS_vel"], "r", linewidth=0.5)
        axes[4, 0].plot(self.record_data["Time"], self.record_data["EW_vel"], "g", linewidth=0.5)
        axes[5, 0].plot(self.record_data["Time"], self.record_data["UD_vel"], "b", linewidth=0.5)
        axes[6, 0].plot(self.record_data["Time"], self.record_data["NS_disp"], "r", linewidth=0.5)
        axes[7, 0].plot(self.record_data["Time"], self.record_data["EW_disp"], "g", linewidth=0.5)
        axes[8, 0].plot(self.record_data["Time"], self.record_data["UD_disp"], "b", linewidth=0.5)
        
        y_label = [r"NS Acc. (cm/s$^2$)", r"EW Acc. (cm/s$^2$)", r"UD Acc. (cm/s$^2$)",
                   r"NS Vel. (cm/s)", r"EW Vel. (cm/s)", r"UD Vel. (cm/s)",
                   r"NS Disp. (cm)", r"EW Disp. (cm)", r"UD Disp. (cm)"]
        
        temp_max_values = np.append(self.record_data_abs_max[:3], self.record_data_abs_max[4:])
        
        # change figure style
        for i in range(9):
            if flag_set_xlim:
                axes[i, 0].set_xlim(xlim)
            
            if i < 3:
                axes[i, 0].set_ylim(-ylim[0], ylim[0])
                temp_max_value = temp_max_values[i]
                axes[i, 0].text(0.95, 0.05, "Max: {:.1f}".format(temp_max_value) + " cm/s$^2$", 
                                transform=axes[i, 0].transAxes, verticalalignment="bottom", 
                                horizontalalignment="right", fontsize=8, color="k")
            elif i < 6:
                axes[i, 0].set_ylim(-ylim[1], ylim[1])
                temp_max_value = temp_max_values[i]
                axes[i, 0].text(0.95, 0.05, "Max: {:.2f}".format(temp_max_value) + " cm/s", 
                                transform=axes[i, 0].transAxes, verticalalignment="bottom", 
                                horizontalalignment="right", fontsize=8, color="k")
            else:
                axes[i, 0].set_ylim(-ylim[2], ylim[2])
                temp_max_value = temp_max_values[i]
                axes[i, 0].text(0.95, 0.05, "Max: {:.2f}".format(temp_max_value) + " cm", 
                                transform=axes[i, 0].transAxes, verticalalignment="bottom", 
                                horizontalalignment="right", fontsize=8, color="k")
                
            
            axes[i, 0].spines["top"].set_visible(False)
            axes[i, 0].spines["bottom"].set_linewidth(0.5)
            axes[i, 0].spines["right"].set_visible(False)
            axes[i, 0].spines["left"].set_linewidth(0.5)
            axes[i, 0].xaxis.set_tick_params(width=0.5)
            axes[i, 0].yaxis.set_tick_params(width=0.5)
            axes[i, 0].set_ylabel(y_label[i], fontsize=8)
            
            if i == 8:
                axes[i, 0].xaxis.set_major_locator(mdates.SecondLocator(bysecond=second_locator))
                axes[i, 0].tick_params(axis="x", which="major", labelsize=8)
                axes[i, 0].set_xlabel("Time")
                
            else:
                axes[i, 0].xaxis.set_ticklabels([])
        
        # annotate max value of holizontal component
        max_value_holizontal = self.record_data_abs_max[3]
        axes[0, 0].text(0.95, 0.95, f"Max of Hol. Comp.: {max_value_holizontal:.1f} gal", 
                        transform=axes[0, 0].transAxes, verticalalignment="top", 
                        horizontalalignment="right", fontsize=8, color="k")
        
        # set title
        title_str = self.start_time.strftime("%Y/%m/%d %H:%M:%S")
        axes[0, 0].set_title(title_str, fontsize=10)

        fig_name = self.result_folder / (self.record_path.stem + "_timeseries.png")
        
        fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
        print("Exported time-series record!")
        
        # clear figure
        plt.clf()
        plt.close()
        gc.collect()
    
    
    def export_fourier_spectrum(self, xlim=[0.05, 20], ylim=[0.1, 1000]) -> None:

        # setup figure        
        fig, axes = setup_figure(height=3.5)
        
        temp_index = self.fft_record_data["Freq"] > 0
        temp_fft_record_data = self.fft_record_data[temp_index]
        
        axes[0, 0].plot(temp_fft_record_data["Freq"], np.abs(temp_fft_record_data["NS_acc_smoothed"] / self.record_length / self.freq_interval), "r", linewidth=0.5, label="NS")
        axes[0, 0].plot(temp_fft_record_data["Freq"], np.abs(temp_fft_record_data["EW_acc_smoothed"] / self.record_length / self.freq_interval), "g", linewidth=0.5, label="EW")
        axes[0, 0].plot(temp_fft_record_data["Freq"], np.abs(temp_fft_record_data["UD_acc_smoothed"] / self.record_length / self.freq_interval), "b", linewidth=0.5, label="UD")
                
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_xlim(xlim)
        axes[0, 0].set_ylim(ylim)
        axes[0, 0].set_xlabel("Frequency (Hz)")
        axes[0, 0].set_ylabel("Fourier Spectrum (cm/s)")
        axes[0, 0].spines["top"].set_linewidth(0.5)
        axes[0, 0].spines["bottom"].set_linewidth(0.5)
        axes[0, 0].spines["right"].set_linewidth(0.5)
        axes[0, 0].spines["left"].set_linewidth(0.5)
        axes[0, 0].xaxis.set_tick_params(width=0.5)
        axes[0, 0].yaxis.set_tick_params(width=0.5)
        leg = axes[0, 0].legend()
        leg.get_frame().set_linewidth(0.5)
        axes[0, 0].grid(visible=True, which="major", axis="both", color="k", linewidth=0.25, linestyle="--")
        
        title_str = self.start_time.strftime("%Y/%m/%d %H:%M:%S") + " (Parzen Width:" + str(self.parzen_width) + " Hz)"
        axes[0, 0].set_title(title_str, fontsize=10)
        
        fig_name =  self.result_folder / (self.record_path.stem + "_fourierspectrum.png")
        fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
        print("Exported Fourier spectrum record!")
        
        
        plt.clf()
        plt.close()
        gc.collect()
        
        
    def export_response_spectrum(self, xlim=[0.05, 20], ylim=[2, 2000], export_type=["abs_acc", "rel_acc", "vel", "disp"]) -> None:
        
        # check type
        if type(export_type) != list:
            raise ValueError("Invalid export_type! It should be list containing 'abs_acc', 'rel_acc', 'vel', 'disp'!")
        
        for temp_export_type in export_type:
            fig, axes = setup_figure()
            
            if temp_export_type == "abs_acc":
                temp_base_col_name = "acc_resp_abs"
                temp_ylabel = r"Abs. Acc. Res. Spectrum (cm/s$^2$)"
            elif temp_export_type == "rel_acc":
                temp_base_col_name = "acc_resp_rel"
                temp_ylabel = r"Rel. Acc. Res. Spectrum (cm/s$^2$)"
            elif temp_export_type == "vel":
                temp_base_col_name = "vel_resp"
                temp_ylabel = "Vel. Res. Spectrum (cm/s)"
            elif temp_export_type == "disp":
                temp_base_col_name = "disp_resp"
                temp_ylabel = "Disp. Res. Spectrum (cm)"
            
            
            axes[0, 0].plot(self.response_spectrum["nFreq"], self.response_spectrum["NS_" + temp_base_col_name], "r", linewidth=0.5, label="NS")
            axes[0, 0].plot(self.response_spectrum["nFreq"], self.response_spectrum["EW_" + temp_base_col_name], "g", linewidth=0.5, label="EW")
            axes[0, 0].plot(self.response_spectrum["nFreq"], self.response_spectrum["UD_" + temp_base_col_name], "b", linewidth=0.5, label="UD")
        
            axes[0, 0].set_xscale("log")
            axes[0, 0].set_yscale("log")
            axes[0, 0].set_xlim(xlim)
            axes[0, 0].set_ylim(ylim)
            axes[0, 0].set_xlabel("Frequency (Hz)")
            axes[0, 0].set_ylabel(temp_ylabel)
            axes[0, 0].spines["top"].set_linewidth(0.5)
            axes[0, 0].spines["bottom"].set_linewidth(0.5)
            axes[0, 0].spines["right"].set_linewidth(0.5)
            axes[0, 0].spines["left"].set_linewidth(0.5)
            axes[0, 0].xaxis.set_tick_params(width=0.5)
            axes[0, 0].yaxis.set_tick_params(width=0.5)
            leg = axes[0, 0].legend()
            leg.get_frame().set_linewidth(0.5)
            axes[0, 0].grid(visible=True, which="major", axis="both", color="k", linewidth=0.25, linestyle="--")
            
            title_str = self.start_time.strftime("%Y/%m/%d %H:%M:%S") + " (h=" + str(self.h) + ")"
            axes[0, 0].set_title(title_str, fontsize=10)
            
            fig_name =  self.result_folder / (self.record_path.stem + "_" + temp_export_type + "_responsespectrum.png")
            fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
            print("Exported", temp_export_type, "response spectrum record!")
            
            plt.clf()
            plt.close()
            gc.collect()