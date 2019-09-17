def DF_subplots2x2(df_list, xcol, ycol, titles, ylabel='', xlabel='', c='purple', save=False, v=''):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    for i in range(len(df_list)):
        ax = plt.subplot(2, 2, i+1)
        df_list[i].plot(ax=ax, x=xcol, y=ycol, style='-', color=c)
        plt.title('{}'.format(titles[i]), fontsize=16)
        plt.grid(True)
        if i == 0 or i==2:
            plt.ylabel(r'${}$'.format(ylabel), fontsize=16)
            plt.xlabel('')
        elif i == 1 or i == 3:
            plt.xlabel(r'${}$'.format(xlabel), fontsize=16)
        else:
            plt.xlabel('')
        ax.get_legend().remove()
        plt.xticks(rotation=30, fontsize=12)
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