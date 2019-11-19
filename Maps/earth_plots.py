import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-90, 75, 10, 60])
ax.stock_img()
plt.show()