__author__ = '@Tssp'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rc('text',usetex=True)
plt.rc('font',family='serif')
plt.rcParams['xtick.labelsize']=17
plt.rcParams['ytick.labelsize']=17
plt.rcParams['axes.labelsize']=20
plt.rcParams['axes.titlesize']=20

# Loading:
##########
CsvPath = '~/CIEMAT/Rn_Time_Series/AEMET/logs/'
l = []
for i in range(1, 10):
    try:
        l.append(pd.read_csv(CsvPath + f'ErrorsANN{i}Fw.csv'))
    except:
        continue

lbl = ['Rn', 'Rn+T BCN', 'Rn+T PMP', 'Rn+T ZGZ', 'Rn+T HSC']
# 1Fw:
######

def MAE_Boxplot(datalistnFw, n, label):
    # datalistnFw: List with the DF of the nFw cases
    # n: number of forward values
    # label: names
    XnFw = [ datalistnFw[n-1]['Rn'], datalistnFw[n-1]['BCN'], datalistnFw[n-1]['PMP'], datalistnFw[n-1]['ZGZ'], datalistnFw[n-1]['HSC']]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    bplot = ax.boxplot(XnFw, sym='+', labels=label, notch=True, patch_artist=True,
                       medianprops=dict(linestyle='-', linewidth=1.4, color='k'))
    plt.grid()
    plt.ylabel('$MAE\ (Bq \cdot m^{-3})$')
    colors = ['dimgray']*5
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    fig.savefig(f'../../Figures/Boxplots/ANN/ANN_MAE_Boxplot_wcolor_{n}Fw_v3.eps', dpi=300)

for nFw in range(1,9):
    MAE_Boxplot(l, nFw, lbl)