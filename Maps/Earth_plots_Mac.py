
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pgf')
# create new figure, axes instances.
fig=plt.figure(figsize=(12, 8))
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.\n
m = Basemap(llcrnrlon=-5.,llcrnrlat=40,urcrnrlon=3.,urcrnrlat=43.7,
            rsphere=(6378137.00,6356752.3142),
            resolution='l',projection='merc',
            lat_0=40.,lon_0=-20.,lat_ts=20.)
m.drawcoastlines()
m.fillcontinents()
m.drawcountries(linewidth=1.1)
# draw stations
## BCN\n",
x, y = m(2+0.4/60+42/60**2, 41+17/60+49/60**2)
m.plot(x, y, marker='^', markeredgecolor='k', color='r', markersize=11)
plt.annotate('BCN', xycoords='data', xy=(x-1e4, y-3e4), fontsize=12)
## PMP
x, y = m(-(1 + 38/60 + 49/60**2), 42 + 48/60 + 21/60**2)
m.plot(x, y, marker='^', markeredgecolor='k', color='r', markersize=11)
plt.annotate('PMP', xycoords='data', xy=(x-1e4, y-3e4), fontsize=12)
## HSC
x, y = m(-(19/60 + 24/60**2), 42 + 4/60 + 51/60**2)
m.plot(x, y, marker='^', markeredgecolor='k', color='r', markersize=11)
plt.annotate('HSC', xycoords='data', xy=(x-1e4, y-3e4), fontsize=12)
## ZGZ
x, y = m(-(1 + 2/60 + 30/60**2), 41 + 39/60 + 58/60**2)
m.plot(x, y, marker='^', markeredgecolor='k', color='r', markersize=11)
plt.annotate('ZGZ', xycoords='data', xy=(x-1e4, y-3e4), fontsize=12)
## LSC
x, y = m(-(33/60 + 28/60**2), 42 + 48/60 + 21/60**2)
m.plot(x, y, marker='*', markeredgecolor='k', color='b', markersize=16)
plt.annotate('LSC', xycoords='data', xy=(x-1e4, y+3e4), fontsize=12)
plt.show()
