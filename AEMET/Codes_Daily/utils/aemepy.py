def DF_subplots2x2(df_list, xcol, ycol, titles, ylabel='', xlabel='', c='#1f77b4', save=False, v=''):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    for i in range(len(df_list)):
        ax = plt.subplot(2, 2, i+1)
        df_list[i].plot(ax=ax, x=xcol, y=ycol, style='-', color=c)
        plt.title('{}'.format(titles[i]), fontsize=16)
        plt.grid(True)
        if i == 0:
            plt.ylabel(r'${}$'.format(ylabel), fontsize=16)
            plt.xlabel('')
        elif i == 1:
            plt.xlabel('')
            plt.ylabel('')
        elif i == 2:
            plt.xlabel(r'${}$'.format(xlabel), fontsize=16)
            plt.ylabel(r'${}$'.format(ylabel), fontsize=16)
        elif i == 3:
            plt.xlabel(r'${}$'.format(xlabel), fontsize=16)
        ax.get_legend().remove()
        plt.tight_layout()
        if save:
            plt.savefig('../Figures/{}.png'.format('_'.join(titles) + '-' + (str(v))), bbox_inches='tight')

def concat_dictionary(DIC):
    import pandas as pd
    import numpy as np
    keys_list = [key for key in DIC.keys()]
    for j in range(len(keys_list)):
        if j == 0:
            Final = DIC[keys_list[0]]
        else:
            Final = pd.concat([Final, DIC[keys_list[j]]])
    return Final

def filter_duplicates(DF):
    import pandas as pd
    import numpy as np
    duplicated = DF.duplicated()
    filter_duplicates = DF.duplicated(keep='first')
    return Data[~filter_duplicates]
    
def fill_avg_per_month(DF):
    import pandas as pd
    import numpy as np
    DF['fecha'] = pd.to_datetime(DF['fecha'])
    output = DF.groupby([DF['fecha'].dt.year, DF['fecha'].dt.month]).transform(lambda x: x.fillna(x.mean()))
    output['fecha'] = DF['fecha']
    return output

def fill_arima_per_month(DF):
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.arima_model import ARIMA
    from sklearn.metrics import mean_squared_error
    DFavg = fill_avg_per_month(DF)
    DFavg['fecha'] = pd.to_datetime(DFavg['fecha'])
    cols = ['tmed', 'presmed', 'velmedia']
    for col in cols:
        model = ARIMA(endog=DFavg[col], order=(3, 0, 1), dates=DFavg['fecha'], freq='D')
        results = model.fit()
        print('MSE ARIMA {}: '.format(col), mean_squared_error(DFavg[col].values, results.fittedvalues))
        DF['ARIMA_' + col] = results.fittedvalues
        DF[col] = DFavg[col].fillna(DF[DF[col].isnull()]['ARIMA_' + col])
        DF = DF.drop(['ARIMA_' + col], axis=1)
    return DF
        
def avg_per_weeks(DF):
    import pandas as pd
    DF_weeks = DF.set_index(pd.DatetimeIndex(DF['fecha']))
    return DF_weeks.resample('W').mean()

def Rn_Clima_plot(DF_list, mdnRnA, dates, ycol, titles, xcol='fecha', ylabel='', legend='', xlabel='Dates', c='#1f77b4', save=False, v=''):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    for i in range(len(DF_list)):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 6), dpi=300)
        xaxis = ax.get_xaxis()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax.plot(dates, mdnRnA, 'k')
        ax.set_ylabel('$^{222}$Rn($Bq \cdot m^{-3}$)', fontsize=16)

        ax2 = ax.twinx()
        ax2.plot(DF_list[i][xcol].values, DF_list[i][ycol].values, alpha=0.7, color=c)
        #ax2.set_yscale('log')
        ax2.set_ylabel(r'${}$'.format(legend), fontsize=16, rotation=-90, labelpad=30)

        plt.xlim([dates[0], dates[-1]])
        ax.set_xlabel('{}'.format(xlabel), fontsize=16)
        ax.legend(['$^{222}$Rn'], fontsize=14, loc='upper left')
        ax2.legend(['${}$'.format(legend)], fontsize=14, loc='upper right')
        if save:
                plt.savefig('../Figures/{}'.format(titles[i]) + '-Rn-'+ str(v) + '.png', bbox_inches='tight')

def Rn_Clima_subplots(DF_list, mdnRnA, dates, ycol, titles, xcol='fecha', ylabel='', legend='', xlabel='Dates', c='#1f77b4', save=False, v=''):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    for i in range(len(DF_list)):
        ax = plt.subplot(2, 2, i+1)
        xaxis = ax.get_xaxis()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        # Rn
        ax.plot(dates, mdnRnA, 'k')
        ax.set_ylabel('$^{222}$Rn($Bq \cdot m^{-3}$)', fontsize=16)
        # Climatology
        ax2 = ax.twinx()
        ax2.plot(DF_list[i][xcol].values, DF_list[i][ycol].values, alpha=0.7, color=c)
        plt.xlim([dates[0], dates[-1]])
        plt.title('{}'.format(titles[i]), fontsize=16)

        if i == 0:
            ax.set_ylabel('$^{222}$Rn($Bq \cdot m^{-3}$)', fontsize=16)
            ax.set_xlabel('')
            ax2.set_xlabel('')
        if i == 1:
            ax2.set_ylabel(r'${}$'.format(ylabel), fontsize=16, rotation=-90, labelpad=30)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax2.set_xlabel('')
        elif i == 2:
            ax.set_ylabel('$^{222}$Rn($Bq \cdot m^{-3}$)', fontsize=16)
            ax.set_xlabel(xlabel, fontsize=16)
        elif i == 3:
            ax2.set_ylabel(r'${}$'.format(ylabel), fontsize=16, rotation=-90, labelpad=30)
            ax.set_ylabel('')
            ax.set_xlabel(xlabel, fontsize=16)
        ax.legend(['$^{222}$Rn'], fontsize=14, loc='upper left')
        ax2.legend([r'${}$'.format(legend)], fontsize=14, loc='upper right')
        plt.tight_layout()
        if save:
                plt.savefig('../Figures/{}.png'.format('_'.join(titles) + '-Rn-' + (str(v))), bbox_inches='tight')