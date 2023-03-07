#import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import electrocardiogram
from scipy.signal import find_peaks
from scipy.integrate import simpson
import pandas as pd
import os


#get the data of the required experiment
os.chdir('D:\\old mouse')
preaccl_file=pd.read_excel("26_10_rt_inferred.xlsx")
accl_file=pd.read_excel("02_12_rt_inferred.xlsx")


"""Shifting the data trace, so that the zero of the graph
corresponds to the minimum value"""

def file_process(file):
    cols=list(file.columns)
    f1_new=pd.DataFrame(columns=cols)
    for i in cols:
        m=file1[i].min()
        temp=file1[i]-m
        f1_new[i]=temp
    return f1_new

preaccl_processed=file_process(preaccl_file)

"""Plotting the neuron traces as subplots"""
def plot_fig(file):
    n=len(file.columns)
    neurons = list(map(str, file.columns))
    l=[ i for i in range(1, n+1)]
    nrows=int(n/4)
    fig, axs = plt.subplots(nrows=nrows+1, ncols=4, figsize=(n,n))
    for q, ax in zip(neurons,axs.ravel()):
        data=file[str(q)]
        data.plot(ax=ax)
        peaks,i=find_peaks(data,width=5)
        ax.plot(peaks, data[peaks], "rx")

plot_fig(preaccl_processed)


"""Get the peak and area metrics for each neuron"""

def get_metrics(file):
    metrics={}
    N_peaks_list=[]
    N_peaks_norm_list=[]
    areas_list=[]
    peak_height_list=[]
    areas_norm_list=[]
    neurons = list(map(str, file.columns))
    total_frames=len(file)
    for neuron in neurons:
        data=file[neuron]
        peaks,amps=find_peaks(data,height=0,width=5)
        no_peaks=len(peaks)
        no_peaks_norm=no_peaks/total_frames
        area_trapz=np.trapz(data,dx=5)
        area_norm=area_trapz/total_frames
        peak_height=amps["peak_heights"]
        N_peaks_list.append(no_peaks)
        N_peaks_norm_list.append(no_peaks_norm)
        peak_height_list.append(peak_height)
        areas_list.append(area_trapz)
        areas_norm_list.append(area_norm)
    metrics["Neuron_number"]=   neurons
    metrics["No_peaks"]=N_peaks_list
    metrics["No_peaks_normalized"]=N_peaks_norm_list
    metrics["peak_height"]=peak_height_list
    metrics["area"]=areas_list
    metrics["area_normalized"]=areas_norm_list
    return metrics



""""get the preacclimated and acclimated metrics dataframes""""
pre_met=get_metrics(preaccl_processed)
accl_met=get_metrics(accl_processed)

pre_met_df=pd.DataFrame.from_dict(pre_met)
accl_met_df=pd.DataFrame.from_dict(accl_met)


"""organize the data to get the peaks of the neurons in different columns"""
def organize_data(file):
    split_list = lambda x: pd.Series(x)
    # apply lambda function to column with lists
    file_tmp= file['peak_height'].apply(split_list)
    file_tmp.columns = ['peak_height_' + str(i+1) for i in range(len(file_tmp.columns))]
    new_f=file.drop("peak_height",axis=1)
    df = pd.concat([new_f,file_tmp], axis=1)
return df

df_pre_met=organize_data(pre_met_df)
df_pre_met.to_excel("preaccl_{}_mouse_full_peak.xlsx".format(f_name),index=False)


df_accl_met=organize_data(accl_met_df)
df.to_excel("accl_{}_mouse_full_peak.xlsx".format(f_name),index=False)
