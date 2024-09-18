from pathlib import Path
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import gc
import scipy.signal as ss
import scipy.integrate as si
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import re
import json


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
                 adxl_params={"gain": 2 * 980.665 / 2 ** 19, "method": "serial"}, 
                 integration_method="JMA", integration_butterworth_cutoff=0.1, integration_butterworth_order=4,
                 flag_baseline_correction=True, baseline_count_indices= 100, flag_export_csv=True) -> None:
        
        self.record_type = record_type
        self.record_path = Path(record_path).resolve()
        self.parzen_width = parzen_width
        self.record_interval = record_interval
        self.h = h
        self.adxl_params = adxl_params
        self.integration_method = integration_method
        self.integration_butterworth_cutoff = integration_butterworth_cutoff
        self.integration_butterworth_order = integration_butterworth_order
        self.flag_baseline_correction = flag_baseline_correction
        self.baseline_count_indices = baseline_count_indices
        self.baseline_offset_values = pd.DataFrame()
        self.flag_export_csv = flag_export_csv
        
        # create result folder
        self.result_folder = self.record_path.parents[1] / "res" / self.record_path.name
        self.result_folder.mkdir(exist_ok=True, parents=True)
        
        print("File:", self.record_path.stem)
        
        # load acceleration record
        # self.startime and self.record_data must be defined
        if self.record_type == "JMA":
            encoding_chr = "shift-jis"
            
            # Load header info
            with open(self.record_path, "r", encoding=encoding_chr) as f:
                for i, line in enumerate(f):
                    if i == 5:
                        start_time_str = line.split("=")[1].replace(" ", "").replace("\n", "")
                        temp_start_time_format = "%Y%m%d%H%M%S"
                        self.start_time = datetime.datetime.strptime(start_time_str, temp_start_time_format)
                        break
            
            # load acceleration record
            temp_col_names = ["NS_acc", "EW_acc", "UD_acc"]
            self.record_data = pd.read_csv(self.record_path,
                                           encoding=encoding_chr, 
                                           names=temp_col_names,
                                           skiprows=7)
            
            # Add time column
            self.record_data["Time"] = [self.start_time + datetime.timedelta(seconds=i * self.record_interval) for i in range(len(self.record_data))]

            self.record_data[temp_col_names] = self.record_data[temp_col_names].astype(float)
            self.record_data = self.record_data[["Time", "NS_acc", "EW_acc", "UD_acc"]]
            
                    
        elif self.record_type == "NIED":

            # file extension check and define file path for each component
            if self.record_path.suffix in [".NS", ".EW", ".UD"]:
                temp_record_path_NS = self.record_path.parent / (self.record_path.stem + ".NS")
                temp_record_path_EW = self.record_path.parent / (self.record_path.stem + ".EW")
                temp_record_path_UD = self.record_path.parent / (self.record_path.stem + ".UD")

            # KiK-net format: suffix 1 means the sensor located in the ground
            elif self.record_path.suffix in [".NS1", ".EW1", ".UD1"]:
                temp_record_path_NS = self.record_path.parent / (self.record_path.stem + ".NS1")
                temp_record_path_EW = self.record_path.parent / (self.record_path.stem + ".EW1")
                temp_record_path_UD = self.record_path.parent / (self.record_path.stem + ".UD1")
            
            # KiK-net format: suffix 2 means the sensor located on the ground
            elif self.record_path.suffix in [".NS2", ".EW2", ".UD2"]:
                temp_record_path_NS = self.record_path.parent / (self.record_path.stem + ".NS2")
                temp_record_path_EW = self.record_path.parent / (self.record_path.stem + ".EW2")
                temp_record_path_UD = self.record_path.parent / (self.record_path.stem + ".UD2")
            else:
                raise ValueError("Invalid file extension!")
            
            temp_record_paths = [temp_record_path_NS, temp_record_path_EW, temp_record_path_UD]
            
            temp_record = []
            
            for temp_each_record_path in temp_record_paths:
                temp_each_record_header_data = pd.read_csv(temp_each_record_path, 
                                                           nrows=16, 
                                                           header=None)
                
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
            
            temp_col_names = ["Time", "NS_acc", "EW_acc", "UD_acc"]
            self.record_data = pd.DataFrame(np.vstack([temp_record_time, temp_record_NS, temp_record_EW, temp_record_UD]).T, 
                                            columns=temp_col_names)
            self.record_data[["NS_acc", "EW_acc", "UD_acc"]] = self.record_data[["NS_acc", "EW_acc", "UD_acc"]].astype(float)
        
        elif self.record_type == "HG":
            
            encoding_chr = "shift-jis"
            sensor_model = ""
            
            with open(self.record_path, "r", encoding=encoding_chr) as f:
                for i, line in enumerate(f):
                    if i == 0:
                        line = line[:7]
                        sensor_model = line
                    elif sensor_model == "SLP-SE6" and i == 4:
                        start_time_str = line.split(",")[1].replace(" ", "")
                        temp_start_time_format = "%y/%m/%d%H:%M:%S"
                        self.start_time = datetime.datetime.strptime(start_time_str, temp_start_time_format)
                        break
                    
                    elif sensor_model == "SLP-SE7" and i == 5:
                        start_time_str = line.split(",")[1]
                        temp_start_time_format = "%Y/%m/%d %H:%M:%S"
                        self.start_time = datetime.datetime.strptime(start_time_str, temp_start_time_format)
                        break
            
            # load acceleration record
            temp_col_names = ["UD_acc", "NS_acc", "EW_acc"]
            self.record_data = pd.read_csv(self.record_path,
                                           encoding=encoding_chr, 
                                           names=temp_col_names,
                                           skiprows=37)
            
            # Add time column
            self.record_data["Time"] = [self.start_time + datetime.timedelta(seconds=i * self.record_interval) for i in range(len(self.record_data))]

            self.record_data[temp_col_names] = self.record_data[temp_col_names].astype(float)
            self.record_data = self.record_data[["Time", "NS_acc", "EW_acc", "UD_acc"]] 
            
        # RTRI (Japan Railway Technical Research Institute) format for JR Takatori Record in 1995 Kobe EQ
        # Reference: http://wiki.arch.ues.tmu.ac.jp/KyoshinTebiki/index.php?%C3%F8%CC%BE%A4%CA%B6%AF%BF%CC%B5%AD%CF%BF#v13066f8
        
        elif self.record_type == "JR":
        
            encoding_chr = "utf-8"
            
            temp_record_path_NS = self.record_path.parent / (self.record_path.stem + ".001")
            temp_record_path_EW = self.record_path.parent / (self.record_path.stem + ".002")
            temp_record_path_UD = self.record_path.parent / (self.record_path.stem + ".003")
            
            temp_record_paths = [temp_record_path_NS, temp_record_path_EW, temp_record_path_UD]
            
            temp_start_time_list =[]
            temp_record = []
            
            for temp_each_record_path in temp_record_paths:
                
                with open(temp_each_record_path, "r", encoding=encoding_chr) as f:
                    for i, line in enumerate(f):
                        line = re.sub(r"\s+", " ", line)
                        
                        # Load start time info
                        if i == 0:
                            temp_start_date_str = line.split(" ")[0]
                            temp_start_time_str = line.split(" ")[2]
                            temp_start_date_time_str = temp_start_date_str + temp_start_time_str
                            temp_start_date_time_format = "%y%m%d%H.%M"
                            temp_start_time_list.append(datetime.datetime.strptime(temp_start_date_time_str, temp_start_date_time_format))
                        # Load gain info
                        elif i == 1:
                            temp_max_value = float(line.split(" ")[3])
                            break
                    
                # Load acceleration record
                colspecs = [(10*i, 10*i+10) for i in range(8)]
                temp_each_record_data = pd.read_fwf(temp_each_record_path, colspecs=colspecs, skiprows=2, header=None)
                temp_each_record_data = temp_each_record_data.dropna(axis=0)
                temp_each_record_data = temp_each_record_data.values.T.flatten("F")
                temp_each_record_data = temp_each_record_data.astype(float)
                temp_each_record_data_max = np.max(temp_each_record_data)
                temp_gain = temp_max_value / temp_each_record_data_max
                temp_each_record_data = temp_each_record_data * temp_gain
                
                temp_record.append(temp_each_record_data)
                                        
            # check if the start time is the same
            if len(set(temp_start_time_list)) != 1:
                raise Warning("Start time is not the same! Consider the start time of the first component.")
            
            self.start_time = temp_start_time_list[0]
            
            self.record_interval = 0.01
            temp_record_time = np.array([self.start_time + datetime.timedelta(seconds=i * self.record_interval) for i in range(len(temp_record[0]))])
            
            temp_record_NS = temp_record[0]
            temp_record_EW = temp_record[1]
            temp_record_UD = temp_record[2]
            
            self.col_names = ["Time", "NS_acc", "EW_acc", "UD_acc"]
            self.record_data = pd.DataFrame(np.vstack([temp_record_time, temp_record_NS, temp_record_EW, temp_record_UD]).T, 
                                            columns=self.col_names)
            self.record_data[["NS_acc", "EW_acc", "UD_acc"]] = self.record_data[["NS_acc", "EW_acc", "UD_acc"]].astype(float)
        
        elif self.record_type == "ADXL":

            encoding_chr = "utf-8"

            temp_record_path = self.record_path

            if self.adxl_params["method"] == "serial":

                # Time: 119990ms, X: 5715, Y: -3663, Z: -2671
                # Time: 120000ms, X: 5339, Y: -4559, Z: -2553
                # Time: 120010ms, X: 5437, Y: -4394, Z: -4510
                
                temp_col_names = ["Time", "NS_acc", "EW_acc", "UD_acc"]
                temp_record_data = pd.read_csv(temp_record_path,
                                                  encoding=encoding_chr, 
                                                  names=temp_col_names,
                                                  skiprows=0)
                
                temp_end_datetime = temp_record_path.stat().st_mtime
                temp_end_datetime = datetime.datetime.fromtimestamp(temp_end_datetime)
                
                temp_record_data["Time_delta"] = temp_record_data["Time"].str.extract(r"Time: (\d+)ms").astype(int)
                temp_record_data["Time"] = [temp_end_datetime - datetime.timedelta(milliseconds=i) for i in temp_record_data["Time_delta"][::-1]]
                temp_record_data["NS_acc"] = temp_record_data["NS_acc"].str.extract(r"X: (-?\d+)").astype(int) * self.adxl_params["gain"]
                temp_record_data["EW_acc"] = temp_record_data["EW_acc"].str.extract(r"Y: (-?\d+)").astype(int) * self.adxl_params["gain"]
                temp_record_data["UD_acc"] = temp_record_data["UD_acc"].str.extract(r"Z: (-?\d+)").astype(int) * self.adxl_params["gain"]

                temp_record_data = temp_record_data[["Time", "NS_acc", "EW_acc", "UD_acc"]]

                self.start_time = temp_record_data["Time"].iloc[0]
                self.record_data = temp_record_data


            elif self.adxl_params["method"] == "sd":
                raise ValueError("Not implemented yet!")
            
            else:
                raise ValueError("Invalid obtaining method!")
        
        elif self.record_type == "ASW":

            encoding_chr = "shift-jis"

            with open(self.record_path, "r", encoding=encoding_chr) as f:
                for i, line in enumerate(f):
                    if i == 2:
                        temp_start_date = line.split(",")[1].replace('"', "").replace(" ", "").replace("\n", "")
                        temp_start_time = line.split(",")[2].replace('"', "").replace(" ", "").replace("\n", "")
                        temp_start_time_format = "%Y/%m/%d %H:%M:%S"
                        self.start_time = datetime.datetime.strptime(temp_start_date + " " + temp_start_time, temp_start_time_format)
                        print(self.start_time)

            temp_col_names = ["Time", "NS_acc", "EW_acc", "UD_acc"]
            self.record_data = pd.read_csv(self.record_path,
                                           encoding=encoding_chr, 
                                           names=temp_col_names,
                                           skiprows=14)
            
            raise ValueError("Not implemented yet!")

        else:
            raise ValueError("Invalid record type!")
        
        self._validate_record_data()
        self._calcurate_additional_parameter()
        
        # export time series record.
        if self.flag_export_csv:

            temp_json_path = self.result_folder / (self.record_path.stem + "_params.json")
            temp_json = {"record_type": self.record_type,
                        "record_path": str(self.record_path),
                        "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "record_data_abs_max" : self.record_data_abs_max.to_dict(),
                        "parzen_width": self.parzen_width,
                        "record_interval": self.record_interval,
                        "h": self.h,
                        "adxl_params": self.adxl_params,
                        "integration_method": self.integration_method,
                        "integration_butterworth_cutoff": self.integration_butterworth_cutoff,
                        "integration_butterworth_order": self.integration_butterworth_order,
                        "flag_baseline_correction": self.flag_baseline_correction,
                        "baseline_count_indices": self.baseline_count_indices,
                        "baseline_offset_values": self.baseline_offset_values.to_dict(),
                        "flag_export_csv": self.flag_export_csv}
            
            with open(temp_json_path, "w") as f:
                json.dump(temp_json, f, indent=4)

            
            temp_time_series_csv_path = self.result_folder / (self.record_path.stem + "_timeseries.csv")
            self.record_data.to_csv(temp_time_series_csv_path, index=False)
            
            temp_fft_csv_path = self.result_folder / (self.record_path.stem + "_fft.csv")
            self.fft_record_data.abs().to_csv(temp_fft_csv_path, index=False)
            
            temp_response_spectrum_csv_path = self.result_folder / (self.record_path.stem + "_response_spectrum.csv")
            self.response_spectrum_data.to_csv(temp_response_spectrum_csv_path, index=False)

    def _validate_record_data(self):

        # check if the record data includes NaN
        if self.record_data.isnull().values.any():
            raise ValueError("Record data includes NaN!")
        
        # check if all the record is numeric
        if not self.record_data.map(np.isreal).all().all():
            raise ValueError("Record data includes non-numeric value!")
        
        print("finish validate record data!")
    
    def _calcurate_additional_parameter(self):
                
        # calcurate basic parameter
        self.record_length = len(self.record_data)
        
        # baseline correction
        if self.flag_baseline_correction:
            
            self.baseline_offset_values, (temp_NS, temp_EW, temp_UD) = self._baseline_correction()

            self.record_data["NS_acc_raw"] = self.record_data["NS_acc"].copy()
            self.record_data["EW_acc_raw"] = self.record_data["EW_acc"].copy()
            self.record_data["UD_acc_raw"] = self.record_data["UD_acc"].copy()
            
            self.record_data["NS_acc"] = temp_NS
            self.record_data["EW_acc"] = temp_EW
            self.record_data["UD_acc"] = temp_UD
            
            print("finish baseline correction!")
        
        else:
            self.record_data["NS_acc_raw"] = self.record_data["NS_acc"].copy()
            self.record_data["EW_acc_raw"] = self.record_data["EW_acc"].copy()
            self.record_data["UD_acc_raw"] = self.record_data["UD_acc"].copy()
        
        # reorganize column names
        self.record_data = self.record_data[["Time", "NS_acc", "EW_acc", "UD_acc", "NS_acc_raw", "EW_acc_raw", "UD_acc_raw"]]
        self.col_names_all = self.record_data.columns.values
        
        # calcurate FFT
        self._calcurate_fft()
        
        # compute velocity and displacement
        self._calcurate_velocity_displacement()

        # calcurate holizontal component
        self.record_data["H_acc"] = (self.record_data["NS_acc"] ** 2 + self.record_data["EW_acc"] ** 2) ** (1/2)
        self.record_data["3D_acc"] = (self.record_data["NS_acc"] ** 2 + self.record_data["EW_acc"] ** 2 + self.record_data["UD_acc"] ** 2) ** (1/2)
        self.record_data["H_vel"] = (self.record_data["NS_vel"] ** 2 + self.record_data["EW_vel"] ** 2) ** (1/2)
        self.record_data["3D_vel"] = (self.record_data["NS_vel"] ** 2 + self.record_data["EW_vel"] ** 2 + self.record_data["UD_vel"] ** 2) ** (1/2)
        self.record_data["H_disp"] = (self.record_data["NS_disp"] ** 2 + self.record_data["EW_disp"] ** 2) ** (1/2)
        self.record_data["3D_disp"] = (self.record_data["NS_disp"] ** 2 + self.record_data["EW_disp"] ** 2 + self.record_data["UD_disp"] ** 2) ** (1/2)
        
        # calcurate absolute maximum value with sign in each component
        # create col name list except time and _raw columns
        temp_col_name = [col_name for col_name in self.record_data.columns.values if "_raw" not in col_name] 
        temp_col_name = temp_col_name[1:]

        temp_record_data_abs_max = self.record_data.loc[self.record_data[temp_col_name].abs().idxmax(), temp_col_name]
        self.record_data_abs_max = temp_record_data_abs_max.copy()

        # keep only 1st row and drop subsequent rows
        self.record_data_abs_max = self.record_data_abs_max.iloc[0]

        for i in range(len(temp_col_name)):
            self.record_data_abs_max[temp_col_name[i]] = temp_record_data_abs_max.iloc[i, i]
    
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
        self.response_spectrum_data = pd.DataFrame({"nFreq": [], "nPeriod": [],
                                                    "NS_acc_resp_abs": [], "EW_acc_resp_abs": [], "UD_acc_resp_abs": [], "H_acc_resp_abs": [],
                                                    "NS_acc_resp_rel": [], "EW_acc_resp_rel": [], "UD_acc_resp_rel": [], "H_acc_resp_rel": [],
                                                    "NS_vel_resp": [], "EW_vel_resp": [], "UD_vel_resp": [], "H_vel_resp": [],
                                                    "NS_disp_resp": [], "EW_disp_resp": [], "UD_disp_resp": [], "H_disp_resp": []})
                                                
        self.response_spectrum_data["nFreq"] = np.logspace(np.log10(0.05), np.log10(20), 200)
        self.response_spectrum_data["nPeriod"] = 1 / self.response_spectrum_data["nFreq"]
        
        temp_omega = 2 * np.pi * self.fft_record_data["Freq"]
        
        for i, n_freq in enumerate(self.response_spectrum_data["nFreq"]):
            
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
            
            temp_ifft_NS_acc_rel = np.fft.ifft(self.fft_record_data["NS_acc"] * temp_H * -temp_omega**2, norm="backward")
            temp_ifft_EW_acc_rel = np.fft.ifft(self.fft_record_data["EW_acc"] * temp_H * -temp_omega**2, norm="backward")
            temp_ifft_UD_acc_rel = np.fft.ifft(self.fft_record_data["UD_acc"] * temp_H * -temp_omega**2, norm="backward")
            
            temp_ifft_NS_acc_abs = temp_ifft_NS_acc_rel + self.record_data["NS_acc"]
            temp_ifft_EW_acc_abs = temp_ifft_EW_acc_rel + self.record_data["EW_acc"]
            temp_ifft_UD_acc_abs = temp_ifft_UD_acc_rel + self.record_data["UD_acc"]

            temp_ifft_H_disp = (temp_ifft_NS_disp ** 2 + temp_ifft_EW_disp ** 2) ** (1/2)
            temp_ifft_H_vel = (temp_ifft_NS_vel ** 2 + temp_ifft_EW_vel ** 2) ** (1/2)
            temp_ifft_H_acc_rel = (temp_ifft_NS_acc_rel ** 2 + temp_ifft_EW_acc_rel ** 2) ** (1/2)
            temp_ifft_H_acc_abs = (temp_ifft_NS_acc_abs ** 2 + temp_ifft_EW_acc_abs ** 2) ** (1/2)
            
            # calcurate absolute maximum value of each response component
            self.response_spectrum_data.loc[i, "NS_acc_resp_abs"] = np.abs(temp_ifft_NS_acc_abs).max()
            self.response_spectrum_data.loc[i, "EW_acc_resp_abs"] = np.abs(temp_ifft_EW_acc_abs).max()
            self.response_spectrum_data.loc[i, "UD_acc_resp_abs"] = np.abs(temp_ifft_UD_acc_abs).max()
            self.response_spectrum_data.loc[i, "H_acc_resp_abs"] = np.abs(temp_ifft_H_acc_abs).max()
            
            self.response_spectrum_data.loc[i, "NS_acc_resp_rel"] = np.abs(temp_ifft_NS_acc_rel).max()
            self.response_spectrum_data.loc[i, "EW_acc_resp_rel"] = np.abs(temp_ifft_EW_acc_rel).max()
            self.response_spectrum_data.loc[i, "UD_acc_resp_rel"] = np.abs(temp_ifft_UD_acc_rel).max()
            self.response_spectrum_data.loc[i, "H_acc_resp_rel"] = np.abs(temp_ifft_H_acc_rel).max()
            
            self.response_spectrum_data.loc[i, "NS_vel_resp"] = np.abs(temp_ifft_NS_vel).max()
            self.response_spectrum_data.loc[i, "EW_vel_resp"] = np.abs(temp_ifft_EW_vel).max()
            self.response_spectrum_data.loc[i, "UD_vel_resp"] = np.abs(temp_ifft_UD_vel).max()
            self.response_spectrum_data.loc[i, "H_vel_resp"] = np.abs(temp_ifft_H_vel).max()
            
            self.response_spectrum_data.loc[i, "NS_disp_resp"] = np.abs(temp_ifft_NS_disp).max()
            self.response_spectrum_data.loc[i, "EW_disp_resp"] = np.abs(temp_ifft_EW_disp).max()
            self.response_spectrum_data.loc[i, "UD_disp_resp"] = np.abs(temp_ifft_UD_disp).max()
            self.response_spectrum_data.loc[i, "H_disp_resp"] = np.abs(temp_ifft_H_disp).max()
            
        print("finish calcurate Response Spectrum!")

        self.fft_col_names = self.fft_record_data.columns.values
    
    def _baseline_correction(self):
        
        # calcurate baseline
        temp_record_data = self.record_data[["NS_acc", "EW_acc", "UD_acc"]]
        temp_baseline_offset_values = temp_record_data.iloc[:self.baseline_count_indices].mean()

        # apply baseline correction
        temp_NS = self.record_data["NS_acc"] - temp_baseline_offset_values["NS_acc"]
        temp_EW = self.record_data["EW_acc"] - temp_baseline_offset_values["EW_acc"]
        temp_UD = self.record_data["UD_acc"] - temp_baseline_offset_values["UD_acc"]

        return (temp_baseline_offset_values, (temp_NS, temp_EW, temp_UD))
        
        
    def _calcurate_fft(self):
        
        # calcurate Fourier Spectrum
        temp_fft_col_names = ["NS_acc", "EW_acc", "UD_acc"]
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
        self.fft_record_data["Freq"] = np.real(self.fft_record_data["Freq"])
        
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
            
            raise ValueError("Not implemented yet!")
        
        elif self.integration_method == "cumtrapz":
        
            # apply highpass filter and calculate velocity and displacement
            sos = ss.butter(self.integration_butterworth_order,
                            self.integration_butterworth_cutoff,
                            btype="highpass", analog=False, output="sos", fs=1/self.record_interval)
            
            
            temp_NS, temp_EW, temp_UD = self._baseline_correction()
            temp_acc = np.vstack([temp_NS, temp_EW, temp_UD]).T            
            temp_acc_filtered = ss.sosfilt(sos, temp_acc, axis=0)
            
            temp_vel = si.cumulative_trapezoid(temp_acc_filtered, dx=self.record_interval, axis=0, initial=0)
            temp_vel_filtered = ss.sosfilt(sos, temp_vel, axis=0)
            
            temp_disp = si.cumulative_trapezoid(temp_vel_filtered, dx=self.record_interval, axis=0, initial=0)
            temp_disp_filtered = ss.sosfilt(sos, temp_disp, axis=0)
           
            self.record_data["NS_vel"] = temp_vel_filtered[:, 0]
            self.record_data["EW_vel"] = temp_vel_filtered[:, 1]
            self.record_data["UD_vel"] = temp_vel_filtered[:, 2]
            
            self.record_data["NS_disp"] = temp_disp_filtered[:, 0]
            self.record_data["EW_disp"] = temp_disp_filtered[:, 1]
            self.record_data["UD_disp"] = temp_disp_filtered[:, 2]
            
        else:
            raise ValueError("Invalid integration method!")
        
        self.col_names_all = self.record_data.columns.values
        print("finish calcurate Velocity and Displacement!")
    
    
    # get record data
    def get_abs_max(self):

        return (self.col_names_in_use, self.record_data_abs_max)


    # get start time
    def get_start_time(self):
        
        return self.start_time
    
    
    # get time-series data
    def get_time_series_data(self):
        
        return self.record_data
    
    
    # get fft data
    def get_fft_data(self):
        
        return self.fft_record_data


    # get response spectrum data
    def get_response_spectrum_data(self):
        
        return self.response_spectrum_data
    
        
    # export time-series record     
    def export_time_series_record(self, xlim=[], x_axis_type="absolute", ylim=[], major_tick_location=[0], export_params = "all", 
                                  minor_tick_location = [10, 20, 30, 40, 50], force_update=False) -> None:
        """
        Export time-series record as a figure.
        
        Parameters
        ----------
        xlim : list, optional
            The x-axis limits. The default is [].
        x_axis_type : str
            The x-axis type. The default is "absolute"
            available x_axis_type: "absolute", "relative"
        ylim : list, optional
            The y-axis limits. The default is [].
        major_tick_location : list, optional
            The major tick locations. The default is [0].
        minor_tick_location : list, optional
            The minor tick locations. The default is [10, 20, 30, 40, 50].
        export_params : str or list, optional
            The parameters to export. The default is "all".
            available export_params: "all", "acc", "vel", "disp", list containing any combination of "acc", "vel", "disp"
        force_update : bool, optional
            Force update the figure. The default is False.

        Returns
        -------
        None.
        """
        # check x_axis_type
        if x_axis_type not in ["absolute", "relative"]:
            raise ValueError("Invalid x_axis_type!")

        # check export_params
        if type(export_params) == str:
            if export_params == "all":
                export_params = ["acc", "vel", "disp"]
            elif export_params == "acc":
                export_params = ["acc"]
            elif export_params == "vel":
                export_params = ["vel"]
            elif export_params == "disp":
                export_params = ["disp"]
            else:
                raise ValueError("Invalid export_params!")
        elif type(export_params) == list:
            for i in export_params:
                if i not in ["acc", "vel", "disp"]:
                    raise ValueError("Invalid export_params!")
        
        # flatten export_params to string
        fig_name_suffix = "-".join(export_params)
        fig_name =  self.result_folder / (self.record_path.stem + "_timeseries_" + fig_name_suffix + ".png")

        # check if the file already exists
        if not force_update:
            if fig_name.exists():
                print("The file already exists!")
                return
        
        flag_set_xlim = False
        
        # format xlim and ylim 
        if type(xlim) == list:
            if len(xlim) == 0:
                flag_set_xlim = False
            elif len(xlim) == 2:
                if x_axis_type == "relative":
                    if type(xlim[0]) in [int, float] and type(xlim[1]) in [int, float]:
                        xlim = [float(xlim[0]), float(xlim[1])]
                        flag_set_xlim = True
                    else:
                        raise ValueError("Invalid xlim!")
                elif x_axis_type == "absolute":
                    try:
                        xlim = [datetime.datetime.strptime(xlim[0], "%Y-%m-%d %H:%M:%S"), datetime.datetime.strptime(xlim[1], "%Y-%m-%d %H:%M:%S")]
                        flag_set_xlim = True
                    except:
                        raise ValueError("Invalid xlim!")
        else: 
            raise ValueError("Invalid xlim!")
        
        if type(ylim) == list:
            if len(ylim) == 0:
                temp_acc_max = self.record_data_abs_max[["NS_acc", "EW_acc", "UD_acc"]].abs().max()
                temp_vel_max = self.record_data_abs_max[["NS_vel", "EW_vel", "UD_vel"]].abs().max()
                temp_disp_max = self.record_data_abs_max[["NS_disp", "EW_disp", "UD_disp"]].abs().max()
                temp_abs_max = self.record_data_abs_max.drop(["H_acc"])
                ylim = [temp_acc_max, temp_vel_max, temp_disp_max]
                ylim = [temp_max * 1.1 for temp_max in ylim]
            elif len(ylim) == 3:
                pass
            else:
                raise ValueError("Invalid ylim!")
        
        # setup figure
        fig, axes = setup_figure(num_row=len(export_params)*3, hspace=.125, width=8, height=len(export_params)*4)

        color_list = ["r", "g", "b"]
        temp_comp_name = ["NS", "EW", "UD"]

        for i in range(len(export_params)*3):

            temp_param_index = i // 3
            temp_export_param= export_params[temp_param_index]
            
            temp_comp_index = i % 3
            temp_color = color_list[temp_comp_index]

            temp_col_name =  temp_comp_name[temp_comp_index] + "_"  + temp_export_param

            # plot time-series record
            if x_axis_type == "absolute":
                axes[i, 0].plot(self.record_data["Time"], self.record_data[temp_col_name], color=temp_color, linewidth=0.5)
            elif x_axis_type == "relative":
                temp_time = np.arange(len(self.record_data)) * self.record_interval
                axes[i, 0].plot(temp_time, self.record_data[temp_col_name], color=temp_color, linewidth=0.5)        
            
            # set xlim and ylim
            if flag_set_xlim:
                axes[i, 0].set_xlim(xlim)
            
            if temp_export_param == "acc":
                temp_param_label_name = "Acc."
                temp_param_unit = "cm/s$^2$"
                temp_ylim_index = 0

            elif temp_export_param == "vel":
                temp_param_label_name = "Vel."
                temp_param_unit = "cm/s"
                temp_ylim_index = 1
            
            elif temp_export_param == "disp":
                temp_param_label_name = "Disp."
                temp_param_unit = "cm"
                temp_ylim_index = 2
            
            axes[i, 0].set_ylim(-ylim[temp_ylim_index], ylim[temp_ylim_index])

            # set y label and annotation
            y_label = temp_comp_name[temp_comp_index] + " " + temp_param_label_name + " (" + temp_param_unit + ")"
            ann_text = "Abs. Max: {:.2f}".format(self.record_data_abs_max[temp_col_name]) + " " + temp_param_unit    
            axes[i, 0].set_ylabel(y_label, fontsize=8)    
            axes[i, 0].text(0.95, 0.05, ann_text, transform=axes[i, 0].transAxes, verticalalignment="bottom",
                            horizontalalignment="right", fontsize=8, color="k")    
            
            axes[i, 0].spines["top"].set_visible(False)
            axes[i, 0].spines["bottom"].set_linewidth(0.5)
            axes[i, 0].spines["right"].set_visible(False)
            axes[i, 0].spines["left"].set_linewidth(0.5)
            axes[i, 0].xaxis.set_tick_params(width=0.5)
            axes[i, 0].yaxis.set_tick_params(width=0.5)
            if x_axis_type == "absolute":
                axes[i, 0].xaxis.set_major_locator(mdates.SecondLocator(bysecond=major_tick_location))
                axes[i, 0].xaxis.set_minor_locator(mdates.SecondLocator(bysecond=minor_tick_location))

            if i == len(export_params)*3 - 1:
                axes[i, 0].tick_params(axis="x", which="major", labelsize=8)
                axes[i, 0].set_xlabel("Time")
            
            else:
                axes[i, 0].xaxis.set_ticklabels([])
            
            # annotate max value of holizontal component
            if temp_export_param == "acc" and temp_comp_index == 0:

                max_value_holizontal = self.record_data_abs_max["H_acc"]
                axes[i, 0].text(0.95, 0.95, f"Max of Hol. Comp.: {max_value_holizontal:.2f} cm/s$^2$", 
                                transform=axes[i, 0].transAxes, verticalalignment="top", 
                                horizontalalignment="right", fontsize=8, color="k")
            
            elif temp_export_param == "vel" and temp_comp_index == 0:
                
                max_value_holizontal = self.record_data_abs_max["H_vel"]
                axes[i, 0].text(0.95, 0.95, f"Max of Hol. Comp.: {max_value_holizontal:.2f} cm/s", 
                                transform=axes[i, 0].transAxes, verticalalignment="top", 
                                horizontalalignment="right", fontsize=8, color="k")
                
            elif temp_export_param == "disp" and temp_comp_index == 0:

                max_value_holizontal = self.record_data_abs_max["H_disp"]
                axes[i, 0].text(0.95, 0.95, f"Max of Hol. Comp.: {max_value_holizontal:.2f} cm", 
                                transform=axes[i, 0].transAxes, verticalalignment="top", 
                                horizontalalignment="right", fontsize=8, color="k")
            
            # set title
            title_str = self.start_time.strftime("%Y/%m/%d %H:%M:%S")
            axes[0, 0].set_title(title_str, fontsize=10)

        # export figure        
        fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
        print("Exported time-series record!")
        
        # clear figure
        plt.clf()
        plt.close()
        gc.collect()


    def export_fourier_spectrum(self, xlim=[0.05, 20], ylim=[0.1, 1000], force_update=False) -> None:
        
        fig_name =  self.result_folder / (self.record_path.stem + "_fourierspectrum.png")
        
        if not force_update:
            # check if the file already exists
            if fig_name.exists():
                return

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
        
        fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
        print("Exported Fourier spectrum record!")
        
        plt.clf()
        plt.close()
        gc.collect()
        
        
<<<<<<< HEAD
    def export_response_spectrum(self, xlim=[0.05, 20], ylim=[2, 2000], 
                                 export_type=["abs_acc", "rel_acc", "vel", "disp"], 
                                 x_label="period",
                                 force_update=False) -> None:
=======
    def export_response_spectrum(self, xlim=[0.1, 10], ylim=[2, 2000], x_axis = "period",
                                 export_type=["abs_acc", "rel_acc", "vel", "disp"], force_update=False) -> None:
        # check x_axis
        if x_axis not in ["period", "frequency", "p", "f"]:
            raise ValueError("Invalid x_axis! It should be 'period' or 'frequency'!")
>>>>>>> bee892ebc1d7e56a3b3d7013e5a5699c87401ca1
        
        # check type
        if type(export_type) != list:
            raise ValueError("Invalid export_type! It should be list containing 'abs_acc', 'rel_acc', 'vel', 'disp'!")
        
        for temp_export_type in export_type:
            
            fig_name =  self.result_folder / (self.record_path.stem + "_" + temp_export_type + "_response-spectrum.png")
            
            if not force_update:
                # check if the file already exists
                if fig_name.exists():
                    continue                
                
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
            
<<<<<<< HEAD
            if x_label == "period":
                temp_x = 1 / self.response_spectrum_data["nFreq"]
                axes[0, 0].set_xlabel("Period (s)")
            elif x_label == "freq":
                temp_x = self.response_spectrum_data["nFreq"]
                axes[0, 0].set_xlabel("Frequency (Hz)")
            else:
                raise ValueError("Invalid x_label!")
            
            axes[0, 0].plot(temp_x, self.response_spectrum_data["NS_" + temp_base_col_name], "r", linewidth=0.5, label="NS")
            axes[0, 0].plot(temp_x, self.response_spectrum_data["EW_" + temp_base_col_name], "g", linewidth=0.5, label="EW")
            axes[0, 0].plot(temp_x, self.response_spectrum_data["UD_" + temp_base_col_name], "b", linewidth=0.5, label="UD")
        
=======
            if x_axis == "period" or x_axis == "p":
                axes[0, 0].plot(self.response_spectrum_data["nPeriod"], self.response_spectrum_data["NS_" + temp_base_col_name], "r", linewidth=0.5, label="NS")
                axes[0, 0].plot(self.response_spectrum_data["nPeriod"], self.response_spectrum_data["EW_" + temp_base_col_name], "g", linewidth=0.5, label="EW")
                axes[0, 0].plot(self.response_spectrum_data["nPeriod"], self.response_spectrum_data["UD_" + temp_base_col_name], "b", linewidth=0.5, label="UD")
                axes[0, 0].set_xlabel("Period (s)")
                
            elif x_axis == "frequency" or x_axis == "f":
                
                axes[0, 0].plot(self.response_spectrum_data["nFreq"], self.response_spectrum_data["NS_" + temp_base_col_name], "r", linewidth=0.5, label="NS")
                axes[0, 0].plot(self.response_spectrum_data["nFreq"], self.response_spectrum_data["EW_" + temp_base_col_name], "g", linewidth=0.5, label="EW")
                axes[0, 0].plot(self.response_spectrum_data["nFreq"], self.response_spectrum_data["UD_" + temp_base_col_name], "b", linewidth=0.5, label="UD")
                axes[0, 0].set_xlabel("Frequency (Hz)")
            
>>>>>>> bee892ebc1d7e56a3b3d7013e5a5699c87401ca1
            axes[0, 0].set_xscale("log")
            axes[0, 0].set_yscale("log")
            axes[0, 0].set_xlim(xlim)
            axes[0, 0].set_ylim(ylim)
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
            
            fig.savefig(fig_name, format="png", dpi=600, pad_inches=0.05, bbox_inches="tight")
            print("Exported", temp_export_type, "response spectrum record!")
            
            plt.clf()
            plt.close()
            gc.collect()


class DesignCodeSpectrum:
    
    def __init__(self, xlabel="period", predefined_x=None):
        
        if predefined_x is None:
            
            predefined_x = np.logspace(-1, 1, 100)
        
        if xlabel == "period":
            self.freq = 1 / predefined_x
            self.period = predefined_x
        
        else:
            self.freq = predefined_x
            self.period = 1 / predefined_x
                
    def JRA(self, year=2017, level=1, type=1, ground_type=1, c=1):

        """
        JRA Design Code Spectrum
        
        Parameters
        ----------
        year : int
            Year of the JRA Design Code
        level : int
            Earthquake level (1 or 2)
        type : int
            Earthquake type (1 or 2)
        ground_type : int
            Ground type (1, 2 or 3)
        c : float
            Coefficient for Reginal Correction
        
        Returns
        -------
        Sa : array
            Standard Acceleration Response Spectrum (m/s/s)

        Notes
        -----
        - Standard Acceleration Response Spectrum (Sa) does not include the effect of seismic epicenter distribution and probability of exceedance.
        - Guidelines for seismic performance verification of river structures is also based on the JRA Design Code.
        - 標準加速度応答スペクトルは、地域別補正係数を考慮していません。
        - 令和6年において河川構造物の耐震性能照査指針も道路橋司法書に準じています。       
      
        """
        
        self.design_code = "JRA"
        self.year = year
        self.level = level
        self.type = type
        self.ground_type = ground_type
        self.c = c

        if year == 2017:
            if level == 1:
                if ground_type == 1:
                    self.Sa0 = self._JRA_2017_1_1_1(self.period)
                elif ground_type == 2:
                    self.Sa0 = self._JRA_2017_1_1_2(self.period)
                elif ground_type == 3:
                    self.Sa0 = self._JRA_2017_1_1_3(self.period)
                else:
                    raise ValueError("Invalid ground type!")
            elif level == 2:
                if type == 1:
                    if ground_type == 1:
                        self.Sa0 = self._JRA_2017_2_1_1(self.period)
                    elif ground_type == 2:
                        self.Sa0 = self._JRA_2017_2_1_2(self.period)
                    elif ground_type == 3:
                        self.Sa0 = self._JRA_2017_2_1_3(self.period)
                    else:
                        raise ValueError("Invalid ground type!")
                elif type == 2:
                    if ground_type == 1:
                        self.Sa0 = self._JRA_2017_2_2_1(self.period)
                    elif ground_type == 2:
                        self.Sa0 = self._JRA_2017_2_2_2(self.period)
                    elif ground_type == 3:
                        self.Sa0 = self._JRA_2017_2_2_3(self.period)
                    else:
                        raise ValueError("Invalid ground type!")
                else:
                    raise ValueError("Invalid earthquake type!")
            else:
                raise ValueError("Invalid earthquake level!")
        else:
            raise ValueError("Invalid JRA Design code year!")
        
        self.Sa = self.Sa0 * self.c

        return self.Sa

    def get_Sa(self):
        
        temp_df = pd.DataFrame({"Frequency": self.freq, "Period": self.period, "Sa": self.Sa})
        
        return temp_df
    
    def _debug_plot(self, design_code = "JRA", level=1, type=1):
        
        fig, ax = setup_figure()

        Sa0 = []

        if design_code == "JRA":
            if level == 1:
                Sa0.append(self._JRA_2017_1_1_1(self.period))
                Sa0.append(self._JRA_2017_1_1_2(self.period))
                Sa0.append(self._JRA_2017_1_1_3(self.period))

            elif level == 2:
                if type == 1:
                    Sa0.append(self._JRA_2017_2_1_1(self.period))
                    Sa0.append(self._JRA_2017_2_1_2(self.period))
                    Sa0.append(self._JRA_2017_2_1_3(self.period))

                elif type == 2:
                    Sa0.append(self._JRA_2017_2_2_1(self.period))
                    Sa0.append(self._JRA_2017_2_2_2(self.period))
                    Sa0.append(self._JRA_2017_2_2_3(self.period))
                                                
                else:
                    raise ValueError("Invalid earthquake type!")
            else:
                raise ValueError("Invalid earthquake level!")
        else:
            raise ValueError("Invalid Design code!")

        for i in range(len(Sa0)):
            ax[0, 0].plot(self.period, Sa0[i], label=f"Ground Type {i+1}")
        
        ax[0, 0].legend()
        ax[0, 0].set_xscale("log")
        ax[0, 0].set_yscale("log")
        ax[0, 0].set_xlim([0.1, 5])
        if level == 1:
            ax[0, 0].set_ylim([0.3, 5])
        elif level == 2:
            ax[0, 0].set_ylim([0.3, 30])
        ax[0, 0].set_xlabel("Period (s)")
        ax[0, 0].set_ylabel("Sa0 (m/s/s)")

        ax[0, 0].xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax[0, 0].yaxis.set_major_formatter(mticker.ScalarFormatter())

        ax[0, 0].xaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax[0, 0].yaxis.set_minor_formatter(mticker.ScalarFormatter())

        # change the fontsize of minor ticks label
        ax[0, 0].tick_params(axis="both", which="major", labelsize=8)
        ax[0, 0].tick_params(axis="both", which="minor", labelsize=8)

        ax[0, 0].grid(visible=True, which="both", axis="both", color="k", linewidth=0.25, linestyle="--")
        ax[0, 0].spines["top"].set_linewidth(0.5)
        ax[0, 0].spines["bottom"].set_linewidth(0.5)
        ax[0, 0].spines["right"].set_linewidth(0.5)
        ax[0, 0].spines["left"].set_linewidth(0.5)
        ax[0, 0].xaxis.set_tick_params(width=0.5)
        ax[0, 0].yaxis.set_tick_params(width=0.5)

        plt.show()

    def _JRA_2017_1_1_1(self, T):

        Sa0 = np.zeros(len(T))

        indices_1 = np.where(T < 0.1)
        indices_2 = np.where((T >= 0.1) & (T <= 1.1))
        indices_3 = np.where(T > 1.1)

        Sa0[indices_1] = 4.31 * T[indices_1] ** (1/3)
        Sa0[indices_1 * (Sa0[indices_1] < 1.6)] =  1.6
        Sa0[indices_2] = 2.0
        Sa0[indices_3] = 2.2 / T[indices_3]

        return Sa0
                
    def _JRA_2017_1_1_2(self, T):

        Sa0 = np.zeros(len(T))

        indices_1 = np.where(T < 0.2)
        indices_2 = np.where((T >= 0.2) & (T <= 1.3))
        indices_3 = np.where(T > 1.3)

        Sa0[indices_1] = 4.27 * T[indices_1] ** (1/3)
        Sa0[indices_1 * (Sa0[indices_1] < 2.0)] = 2.0
        Sa0[indices_2] = 2.5
        Sa0[indices_3] = 3.25 / T[indices_3]

        return Sa0
    
    def _JRA_2017_1_1_3(self, T):

        Sa0 = np.zeros(len(T))

        indices_1 = np.where(T < 0.34)
        indices_2 = np.where((T >= 0.34) & (T <= 1.5))
        indices_3 = np.where(T > 1.5)

        Sa0[indices_1] = 4.3 * T[indices_1] ** (1/3)
        print(Sa0[indices_1])
        Sa0[indices_1 * (Sa0[indices_1] < 2.4)] = 2.4   
        Sa0[indices_2] = 3.0
        Sa0[indices_3] = 4.50 / T[indices_3]

        return Sa0
                
    def _JRA_2017_2_1_1(self, T):

        Sa0 = np.zeros(len(T))

        indices_1 = np.where(T < 0.16)
        indices_2 = np.where((T >= 0.16) & (T <= 0.6))
        indices_3 = np.where(T > 0.6)

        Sa0[indices_1] = 25.79 * T[indices_1] ** (1/3)
        Sa0[indices_2] = 14.0
        Sa0[indices_3] = 8.4 / T[indices_3]

        return Sa0
    
    def _JRA_2017_2_1_2(self, T):

        Sa0 = np.zeros(len(T))

        indices_1 = np.where(T < 0.22)
        indices_2 = np.where((T >= 0.22) & (T <= 0.9))
        indices_3 = np.where(T > 0.9)

        Sa0[indices_1] = 21.53 * T[indices_1] ** (1/3)
        Sa0[indices_2] = 13.0
        Sa0[indices_3] = 11.7 / T[indices_3]

        return Sa0
    
    def _JRA_2017_2_1_3(self, T):

        Sa0 = np.zeros(len(T))

        indices_1 = np.where(T < 0.34)
        indices_2 = np.where((T >= 0.34) & (T <= 1.4))
        indices_3 = np.where(T > 1.4)

        Sa0[indices_1] = 17.19 * T[indices_1] ** (1/3)
        Sa0[indices_2] = 12.0
        Sa0[indices_3] = 16.8 / T[indices_3]

        return Sa0

    def _JRA_2017_2_2_1(self, T):

        Sa0 = np.zeros(len(T))

        indices_1 = np.where(T < 0.3)
        indices_2 = np.where((T >= 0.3) & (T <= 0.7))
        indices_3 = np.where(T > 0.7)

        Sa0[indices_1] = 44.63 * T[indices_1] ** (2/3)
        Sa0[indices_2] = 20
        Sa0[indices_3] = 11.04 / T[indices_3] ** (5/3)

        return Sa0

    def _JRA_2017_2_2_2(self, T):
        
        Sa0 = np.zeros(len(T))

        indices_1 = np.where(T < 0.4)
        indices_2 = np.where((T >= 0.4) & (T <= 1.2))
        indices_3 = np.where(T > 1.2)
        Sa0[indices_1] = 32.24 * T[indices_1] ** (2/3)
        Sa0[indices_2] = 17.5
        Sa0[indices_3] = 23.71/ T[indices_3] ** (5/3)

        return Sa0
    
    def _JRA_2017_2_2_3(self, T):

        Sa0 = np.zeros(len(T))

        indices_1 = np.where(T < 0.5)
        indices_2 = np.where((T >= 0.5) & (T <= 1.5))
        indices_3 = np.where(T > 1.5)

        Sa0[indices_1] = 23.81 * T[indices_1] ** (2/3)
        Sa0[indices_2] = 15.0
        Sa0[indices_3] = 29.48 / T[indices_3] ** (5/3)

        return Sa0

class GroundType:
    def __init__(self, ground_model = None, physical_param = "vs"):
        
        """
        Ground Type Evaluation
        
        Parameters
        ----------
        ground_model : array
            Ground model data (N x 3)
        physical_param : str
            Physical parameter (vs or SPT-N)

        Returns
        -------
        Tg : float
            Dominant period of subsurface ground (s)
        ground_type : int
            Ground type (1, 2 or 3)
        
        Notes
        -----
        - Ground model should be N x 3 array.
            - 1st column: Thickness of layer (m)
            - 2nd column: Vs (m/s) or SPT-N
            - 3rd column: Soil Type (S: Sandy Soil or C: Clayey Soil)
        
        """
        self.physical_param = physical_param
        
        # check ground_model which should be N x 3 array
        try:
            temp_ground_model = pd.DataFrame(ground_model, columns=["Thickness", "Physical Parameter", "Soil Type"])
        except:
            raise ValueError("Invalid ground model!")
        else:
            # convert 0 and 1 column to float
            temp_ground_model[["Thickness", "Physical Parameter"]] = temp_ground_model[["Thickness", "Physical Parameter"]].astype(float)
            self.ground_model = temp_ground_model
        
        if not temp_ground_model.shape[1] == 3:
            raise ValueError("Invalid ground model!")
        
        # check physical_param
        if not physical_param in ["vs", "SPT-N"]:
            raise ValueError("Invalid physical parameter!")
        
        elif physical_param == "SPT-N":
            temp_vs = self._convert_N_to_vs(self.ground_model)
            self.ground_model["Physical Parameter"] = temp_vs
            
        self._calc_soil_type()
    
    def get_ground_type(self):
        
        return (self.Tg, self.ground_type)
    
    def _convert_N_to_vs(self, ground_model):
        
        temp_vs = np.zeros(len(ground_model))
        
        for i in range(len(ground_model)):
            temp_N = ground_model.loc[i, "Physical Parameter"]
            temp_soil_type = ground_model.loc[i, "Soil Type"]
            
            if temp_soil_type == "S":
                if temp_N <= 50:
                    temp_vs[i] = 80 * temp_N ** (1/3)
                else:
                    temp_vs[i] = 80 * 50 ** (1/3)
            elif temp_soil_type == "C":
                if temp_N <= 25:
                    temp_vs[i] = 100 * temp_N ** (1/3)
                else:
                    temp_vs[i] = 100 * 25 ** (1/3)
            else:
                raise ValueError("Invalid soil type found in ground model!")
                    
        return temp_vs

    def _calc_soil_type(self):
        
        # calculate subsurface ground dominant period
        self.Tg = 4 * (self.ground_model["Thickness"] / self.ground_model["Physical Parameter"]).sum()
        
        if self.Tg < 0.2:
            self.ground_type = 1
        
        elif self.Tg < 0.6:
            self.ground_type = 2
        
        else:
            self.ground_type = 3
        
        