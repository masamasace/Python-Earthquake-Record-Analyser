from pathlib import Path
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import gc
import scipy.signal as ss
import matplotlib.dates as mdates
import os


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


class SeismicRecord:
    def __init__(self, record_path, record_type="JMA", parzen_width=0, time_range=[0, 0]) -> None:
        
        self.record_type = record_type
        self.record_path = Path(record_path).resolve()
        self.parzen_width = parzen_width
        self.result_folder = self.record_path.parent / "result"
        self.time_range = time_range
        
        self.result_folder.mkdir(exist_ok=True, parents=True)
        
        print("File Path:", self.record_path)
        
        if self.record_type == "JMA":
            encoding_chr = "shift-jis"
            self.col_names = ["NS", "EW", "UD"]
            self.record_data = pd.read_csv(self.record_path,
                                           encoding=encoding_chr, 
                                           names=self.col_names)
            
            # Load header info
            initial_time_str = self.record_data.iloc[5, 0][14:]
            self.record_interval = 0.01
            temp_initial_time_format = "%Y %m %d %H %M %S"
            self.initial_time = datetime.datetime.strptime(initial_time_str, temp_initial_time_format)
            
            # Delite header info
            self.record_data.drop(range(7), inplace=True)
            
            # Add time column
            self.record_data["Time"] = [self.initial_time + datetime.timedelta(seconds=i * self.record_interval) for i in range(len(self.record_data))]

            self.record_data[self.col_names] = self.record_data[self.col_names].astype(float)
            self.record_data = self.record_data[["Time", "NS", "EW", "UD"]]
            
                    
        elif self.record_type == "NIED":
            
            # prepare file path for each component
            temp_record_path_NS = self.record_path.parent / (self.record_path.stem + ".NS")
            temp_record_path_EW = self.record_path.parent / (self.record_path.stem + ".EW")
            temp_record_path_UD = self.record_path.parent / (self.record_path.stem + ".UD")
            
            temp_record_paths = [temp_record_path_NS, temp_record_path_EW, temp_record_path_UD]
            
            temp_record = []
            
            for temp_each_record_path in temp_record_paths:
                temp_each_record_header_data = pd.read_csv(temp_each_record_path, nrows=16, header=None)
                
                temp_initial_time_format = "%Y/%m/%d %H:%M:%S"
                
                temp_scale_factor_bra_index = temp_each_record_header_data.iloc[13].values[0].find("(")
                temp_scale_factor_slash_index = temp_each_record_header_data.iloc[13].values[0].find("/")
                
                temp_each_record_header = {"initial_time": datetime.datetime.strptime(temp_each_record_header_data.iloc[9].values[0][18:], 
                                                                                      temp_initial_time_format),
                                           "sampling_freq": float(temp_each_record_header_data.iloc[10].values[0][18:21]),
                                           "scale_factor": float(temp_each_record_header_data.iloc[13].values[0][18:temp_scale_factor_bra_index]) /\
                                                           float(temp_each_record_header_data.iloc[13].values[0][temp_scale_factor_slash_index+1:])}
                
                colspecs = [(9*i, 9*i+8) for i in range(8)]
                temp_each_record_data = pd.read_fwf(temp_each_record_path, colspecs=colspecs, skiprows=17, header=None)
                temp_each_record_data = temp_each_record_data.values.T.flatten("F")
                temp_each_record_data = temp_each_record_data[~np.isnan(temp_each_record_data)] * temp_each_record_header["scale_factor"]
                
                temp_record.append([temp_each_record_header, temp_each_record_data])
            
            self.initial_time = temp_record[0][0]["initial_time"]
            self.record_interval = 1 / temp_record[0][0]["sampling_freq"]
            temp_record_time = np.array([self.initial_time + datetime.timedelta(seconds=i * self.record_interval) for i in range(len(temp_record[0][1]))])
            temp_record_NS = temp_record[0][1]
            temp_record_EW = temp_record[1][1]
            temp_record_UD = temp_record[2][1]
            
            self.col_names = ["Time", "NS", "EW", "UD"]
            self.record_data = pd.DataFrame(np.vstack([temp_record_time, temp_record_NS, temp_record_EW, temp_record_UD]).T, 
                                            columns=self.col_names)
        
        else:
            pass
        
        self._calcurate_additional_parameter(parzen_width)

    
    def _calcurate_additional_parameter(self, parzen_width=0):
        
        temp_initial_time_format = "%Y/%m/%d %H:%M:%S"
        if self.time_range[0] != 0:
            temp_start_time = datetime.datetime.strptime(self.time_range[0], temp_initial_time_format)
            self.record_data = self.record_data[self.record_data["Time"] >= temp_start_time]
        
        if self.time_range[1] != 0:
            temp_end_time = datetime.datetime.strptime(self.time_range[1], temp_initial_time_format)
            self.record_data = self.record_data[self.record_data["Time"] <= temp_end_time]         
        
        self.parzen_width = parzen_width
        
        # calcurate holizontal component
        self.record_data["H"] = (self.record_data["NS"] ** 2 + self.record_data["EW"] ** 2) ** (1/2)
        self.col_names = self.record_data.columns
        
        # calcurate Fourier Spectrum
        freq = np.fft.fftfreq(len(self.record_data), d=self.record_interval)
        
        fft_record_data_NS = np.fft.fft(self.record_data[self.col_names[1]], norm="backward")
        fft_record_data_EW = np.fft.fft(self.record_data[self.col_names[2]], norm="backward")
        fft_record_data_UD = np.fft.fft(self.record_data[self.col_names[3]], norm="backward")
        fft_record_data_H  = np.fft.fft(self.record_data[self.col_names[4]], norm="backward")
    
        
        fft_record_data = np.vstack([fft_record_data_NS, fft_record_data_EW, fft_record_data_UD, fft_record_data_H]).T
        
        norm_fft_record_data = np.abs(fft_record_data / (len(self.record_data)) / (freq[1] - freq[0]))
        
        max_index = (len(self.record_data) + 1) // 2
        
        freq = freq[:max_index]
        freq_reshaped = freq.reshape(-1, 1)
        norm_fft_record_data = norm_fft_record_data[:max_index]
        
        self.col_names_freq = np.insert(self.col_names[1:].values + "_fft", 0, "Freq")
        self.norm_fft_record_data = pd.DataFrame(np.hstack((freq_reshaped, norm_fft_record_data)), 
                                                 columns=self.col_names_freq)
    
        # apply parzen window
        if self.parzen_width > 0:
            number_of_points_parzen_width = int(self.parzen_width / (freq[1] - freq[0]))
            parzen_window = ss.windows.parzen(number_of_points_parzen_width)
            
            for i in range(len(self.col_names_freq[1:])):
                temp_col_name = self.col_names_freq[1 + i] + "_smoothed"       
                self.norm_fft_record_data.loc[:, temp_col_name] = ss.convolve(self.norm_fft_record_data[self.col_names_freq[1 + i]], 
                                                                            parzen_window / sum(parzen_window), mode="same")
        else:
            for i in range(len(self.col_names_freq[1:])):
                temp_col_name = self.col_names_freq[1 + i] + "_smoothed"       
                self.norm_fft_record_data.loc[:, temp_col_name] = self.norm_fft_record_data[self.col_names_freq[1 + i]]

        self.col_names_freq = self.norm_fft_record_data.columns.values
                
    def export_time_series_record(self, ylim=[-200, 200], second_locator=[0]) -> None:
        
        fig, axes = setup_figure(num_row=3, hspace=.125)
        
        axes[0, 0].plot(self.record_data["Time"], self.record_data["NS"], "k", linewidth=0.5)
        axes[1, 0].plot(self.record_data["Time"], self.record_data["EW"], "k", linewidth=0.5)
        axes[2, 0].plot(self.record_data["Time"], self.record_data["UD"], "k", linewidth=0.5)
        
        for i in range(3):
            axes[i, 0].set_ylim(ylim)
            axes[i, 0].spines["top"].set_visible(False)
            axes[i, 0].spines["bottom"].set_linewidth(0.5)
            axes[i, 0].spines["right"].set_visible(False)
            axes[i, 0].spines["left"].set_linewidth(0.5)
            axes[i, 0].xaxis.set_tick_params(width=0.5)
            axes[i, 0].yaxis.set_tick_params(width=0.5)
            axes[i, 0].set_ylabel(self.col_names[i + 1] + " Acc. (gal)")
            
            max_value = max(abs(self.record_data[self.col_names[i + 1]]))
            axes[i, 0].text(0.95, 0.05, f"Max : {max_value:.1f} gal", transform=axes[i, 0].transAxes, 
                            verticalalignment="bottom", horizontalalignment="right", fontsize=8, color="k")
            
            if i == 2:
                axes[i, 0].xaxis.set_major_locator(mdates.SecondLocator(bysecond=second_locator))
                axes[i, 0].tick_params(axis="x", which="major", labelsize=8)
                
            else:
                axes[i, 0].xaxis.set_ticklabels([])
                
        max_value_holizontal = max(abs(self.record_data["H"]))
        axes[0, 0].text(0.95, 0.95, f"Max of Hol. Comp.: {max_value_holizontal:.1f} gal", 
                        transform=axes[0, 0].transAxes, verticalalignment="top", 
                        horizontalalignment="right", fontsize=8, color="r")
        
        fig_name = self.result_folder / (self.record_path.stem + "_timeseries.png")
        
        fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
        print("Exported time-series record!")
        
        plt.clf()
        plt.close()
        gc.collect()
    
    
    def export_fourier_spectrum(self, xlim=[0.1, 10], ylim=[0.1, 1000]) -> None:

        # setup figure        
        fig, axes = setup_figure(height=3.5)
        line_style_list = ["solid", "dotted", "dashed"]
        
        for i in range(len(self.col_names_freq[5:-1])):
            axes[0, 0].plot(self.norm_fft_record_data.iloc[:, 0], 
                            self.norm_fft_record_data.loc[:, self.col_names_freq[i+5]], "k", 
                            linewidth=0.5, linestyle=line_style_list[i], label=self.col_names_freq[i+5][:2])
        
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
        axes[0, 0].grid(visible=True, which="major", axis="both", color="k", linewidth=0.25)
        
        fig_name =  self.result_folder / (self.record_path.stem + "_fourierspectrum.png")
        fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
        print("Exported Fourier spectrum record!")
        
        
        plt.clf()
        plt.close()
        gc.collect()
        
        
    