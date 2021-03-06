__author__ = '@Tssp'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.stats.stats import pearsonr
from global_functions import read_datatxt
plt.rc('text',usetex=True)
plt.rc('font',family='serif')
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
plt.rcParams['axes.labelsize']=18
plt.rcParams['axes.titlesize']=18

# Rn + T

mdnRnA = np.loadtxt('../../../mdnRnA.txt', delimiter=',')
neuron = [256, 128]

# Plot:
newValuesReal = [102., 94., 103., 113.5, 73.,  81., 109., 109.]
predval = [91.77031 , 86.84759 , 70.75325 , 61.546257, 76.41941 , 88.68657, 91.441536, 86.76352]
startdayahead = pd.to_datetime('2019-07-22 00:00:00')

def plot_validation(data_val, pred_val, startday_ahead):
    import matplotlib.pyplot as plt
    import pandas as pd
    # ahead plot
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,4))#
    #plt.figure(2)
    xaxis = ax.get_xaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

    ax.plot(pd.date_range(startday_ahead, periods=len(data_val), freq='W'), data_val, color='k', linewidth=1)
    ax.plot(pd.date_range(startday_ahead, periods=len(pred_val), freq='W'), pred_val, linestyle='-.', color='dimgray', linewidth=1)
    ax.set_ylabel(r'$^{222}$Rn ($Bq\cdot m^{-3}$)')
    plt.xlabel('Dates')
    plt.ylim([30, 140])
    plt.grid()
    ax.legend(['Data', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig('../Paper/validationwcolor8Fw.eps', dpi=200)
    plt.show()

plot_validation(newValuesReal, predval, startdayahead)