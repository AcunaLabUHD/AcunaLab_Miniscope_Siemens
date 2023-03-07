
# Import necessary packages
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
import os


#change to the file directory
os.chdir("1615_rt_rt/")

"""Load the required file and give the
total time during the varying temperature conditions"""

data=loadmat("26_10//26_10..mat")
f_name='26_10' #file name to save
total_rt_frame=2459
total_heat_frame=2464


#Getting the raw fluorescence, deconvolved and inferred values as a dataframe
data_f=pd.DataFrame.from_dict(data["F_raw"])
data_inf=pd.DataFrame.from_dict(data["F_inferred"])
data_deconv=pd.DataFrame.from_dict(data["S_deconv"])
total_frames=len(data_f.columns)

def file_transform(file):

    """Changing the dataframe row and column numbers so that it
    starts from 1 and not 0"""

    tmp=file
    frames=len(tmp.columns)
    print("Total number of frames: {}".format(frames))
    frame_cols=[str(i) for i in range(1,frames+1)]
    tmp.columns=frame_cols
    file_transposed=tmp.T
    n=len(file_transposed.columns)
    print("Number of Neurons: {}".format(n))
    cols=[str(i) for i in range(1,n+1)]
    file_transposed.columns=cols
    return file_transposed

def plot_fig(file,rt_frame_end):

    """Plotting the neuron traces as subplots"""

    n=len(file.columns)
    neurons = list(map(str, file.columns))
    l=[ i for i in range(1, n+1)]
    nrows=int(n/4)
    fig, axs = plt.subplots(nrows=nrows+1, ncols=4, figsize=(n,n))
    for q, ax in zip(neurons,axs.ravel()):
        file[str(q)].plot(ax=ax)
        ax.axvline(x = rt_frame_end,color='green')
#     plt.show()

def zscore_baseline(file,rt_frame_end,total_frames):

    """calculating the zscore to incorporate the baseline"""

    mean_base=file[:rt_frame_end].mean()
    std_base=file[:rt_frame_end].std()
    df=pd.DataFrame(columns=file.columns)
    for i in file.columns:
        col_values=[]
        for j in range(total_frames):
            val=(file[i][j]-mean_base[int(i)-1])/std_base[int(i)-1]
            col_values.append(val)
        df[i]=col_values
    tmp=df.T
    frames=len(tmp.columns)
    print("Total number of frames: {}".format(frames))
    frame_cols=[str(i) for i in range(1,frames+1)]
    tmp.columns=frame_cols
    df=tmp.T
    return df

def plt_heat(z_data,rt_frame_end,total_frames):

    """Plotting the heatmap of the neuron"""

    n=len(z_data.columns)
    final_req_data=z_data.T
    fig, ax = plt.subplots()
    sns.heatmap(final_req_data,ax=ax,cmap="jet",vmin=-8,vmax=10)
    ax.tick_params(axis='x', which='both', length=0)
    plt.xticks([rt_frame_end/2, (rt_frame_end+(total_frames-rt_frame_end)*0.5)], ["Room","Heat"],rotation='horizontal')
    plt.yticks([0,n],[1,n])
    ax.set_xlabel("Time (s)",
                  fontweight ='bold')
    ax.set_ylabel("Cell #",
                  fontweight ='bold')
    plt.axvline(x = rt_frame_end)

def plt_heat_sort(z_data,rt_frame_end,total_frames):

    """Plotting the heatmap of the neuron based on their zscore sorted after the heated condition"""


    df2=z_data.reindex(z_data[rt_frame_end:].mean().sort_values(ascending=False).index, axis=1)
    n=len(df2.columns)
    final_req_data=df2.T
    fig, ax = plt.subplots()
    sns.heatmap(final_req_data,ax=ax,cmap="jet",vmin=-8,vmax=10)
    ax.tick_params(axis='x', which='both', length=0)
    plt.xticks([rt_frame_end/2, (rt_frame_end+(total_frames-rt_frame_end)*0.5)], ["Room","Heat"],rotation='horizontal')
    plt.yticks([0,n],[1,n])
    ax.set_xlabel("Time (s)",
                  fontweight ='bold')
    ax.set_ylabel("Cell #",
                  fontweight ='bold')
    plt.axvline(x = rt_frame_end)


data_f_processed=file_transform(data_f)
data_f_processed.to_excel("{0}_dff.xlsx".format(f_name),index=False) #saving the delta f over f


data_inf_processed=file_transform(data_inf)
data_inf_processed.to_excel("{0}_inferred.xlsx".format(f_name),index=False) #saving the inferred file


data_deconv_processed=file_transform(data_deconv)
data_deconv_processed.to_excel("{0}_deconv.xlsx".format(f_name),index=False) #saving the deconvolved


rt_frame_start=1
rt_frame_end=total_rt_frame+1
heat_frame_start=total_frames-rt_frame_end
heat_frame_end=total_frames



plot_fig(data_f_processed,rt_frame_end)
plt.savefig("{}_dff_traces.pdf".format(f_name),dpi=500)

z_score=zscore_baseline(data_f_processed,rt_frame_end,total_frames)


plt_heat(z_score,rt_frame_end,total_frames)
plt.savefig("{}_unsort.pdf".format(f_name),dpi=500)


plt_heat_sort(z_score,rt_frame_end,total_frames)
plt.savefig("{}.pdf".format(f_name),dpi=500)


# Creating a dataframe of the activity of matched neurons
file_matched=pd.read_excel("26_and_02_matched_new.xlsx")
req_cols=list(file['02_12']) #selecting the experiment number
req_col_string = list(map(str, req_cols)

req_data_f_inf=data_f_inf_processed[req_col_string]# getting the data of the required Neurons
req_data_f_inf.to_excel("02_12_rt_inferred_new.xlsx",index=False)
