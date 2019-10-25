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
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13
plt.rcParams['axes.labelsize']=16
plt.rcParams['axes.titlesize']=16

# Rn + T

mdnRnA = np.loadtxt('../../../mdnRnA.txt', delimiter=',')
neuron = [256, 128]

# Plot:
CNN_loss = np.loadtxt('./CNN_Loss_RnT_PMP_{}_{}_v2.txt'.format(neuron[0], neuron[1]), delimiter=',')
train_loss = CNN_loss[0]
test_loss = CNN_loss[1]
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,6), dpi=100)
plt.plot(train_loss, color='k')
plt.plot(test_loss, color='dimgray', marker='x', linestyle='--')
plt.ylabel('loss', fontsize=14)
plt.xlabel('epoch', fontsize=14)
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
fig.savefig('./CNN_Loss_RnT_PMP_{}_{}_wcolor.eps'.format(neuron[0], neuron[1]))

## Plot 2:
preds = np.loadtxt('./CNN_RnT_PMP_{}_{}_v2.txt'.format(neuron[0], neuron[1]), delimiter=',')
predmins = preds[0]
predmaxs = preds[1]
predavgs = preds[2]
startday = pd.datetime(2013, 7, 1)
startdaypred = pd.datetime(2013, 7, 1) + 7*pd.Timedelta( len(mdnRnA)-94, unit='D') - 7*pd.Timedelta(8, unit='D')
startdayahead = pd.datetime(2013, 7, 1) + 7*pd.Timedelta( len(mdnRnA), unit='D')
dist = mdnRnA[214:] - predavgs
def plot_fill_errors(data, predmins, predmaxs, predavgs, errors, startday, startdaypred):
    import pandas as pd
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,4))
    plt.figure(1)
    xaxis = ax.get_xaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.fill_between(pd.date_range(startdaypred, periods=len(predmins), freq='W')[:-5], predmins[:-5], predmaxs[:-5],
                    facecolor='g')
    ax.plot(pd.date_range(startday, periods=len(data), freq='W')[:-5], data[:-5], 'k') 
    ax.set_xlabel('Dates')
    ax.set_ylabel(r'$^{222}$Rn ($Bq\cdot m^{-3}$)', fontsize=16)
    ax.legend(['Data', 'CNN (Rn + T)'], loc='upper left')
    ax.set_ylim([30, 140])
    ax.grid()

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    axins = zoomed_inset_axes(ax, 1.7, loc='lower left', bbox_to_anchor=(673,223))
    axins.xaxis.set_major_locator(mdates.YearLocator())
    axins.xaxis.set_minor_locator(mdates.MonthLocator())
    axins.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axins.fill_between(pd.date_range(startdaypred, periods=len(predmins), freq='W')[:-5], predmins[:-5], predmaxs[:-5],
                       facecolor='g')
    axins.plot(pd.date_range(startday, periods=len(data), freq='W')[:-5], data[:-5], 'k') 
    axins.set_xlim('2017-08-15', '2019-06-24')
    axins.set_ylim(50, 110)
    axins.set_yticks([60, 80, 100])
    axins.set_title('a)', loc='right', x=0.98, y=0.85)
    axins.grid()

    axins2 = zoomed_inset_axes(ax, 1.7, loc='lower left', bbox_to_anchor=(673,97))
    axins2.xaxis.set_major_locator(mdates.YearLocator())
    axins2.xaxis.set_minor_locator(mdates.MonthLocator())
    axins2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axins2.vlines(pd.date_range(startdaypred, periods=len(predavgs), freq='W')[:-5], 0, errors)
    axins2.set_xlim('2017-08-15', '2019-06-24')
    axins2.set_yticks([int(np.min(errors)+10), 0, int(np.max(errors)+1)])
    axins2.set_ylim([-20, 20])
    axins2.grid()
    axins2.yaxis.set_label_position('right')
    axins2.yaxis.labelpad = 20
    axins2.set_ylabel('Y - pred', fontsize=13, rotation=270)
    axins2.set_title('b)', loc='right', x=0.98, y=0.85)

    plt.xticks(visible=True)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.suptitle('Weekly Fitting at LSC - Hall A')
    fig.savefig('./CNN_RnT_PMP_{}_{}_wcolor.eps'.format(neuron[0], neuron[1]), bbox_inches='tight', dpi=100)
plot_fill_errors(mdnRnA, predmins, predmaxs, predavgs, dist, startday, startdaypred)
plt.show()
