import pandas as pd
import calendar
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from ...datasets.subseasonal.dataloader import get_data_matrix


def show_measurement_on_map(data_matrix, title, vmax):
    """Show sequential measurements on the U.S. map in an matplotlib.animation plot

    Parameters
    ----------
    data_matrix: array of formatted data matrices (see get_data_matrix)

    title: array of titles to accompany the data matrices

    vmax: Maximum value on colorbar. Minimum is 0.
    """
    # Set figure
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())

    # Draw coastlines, US states
    ax.coastlines(linewidth=0.2, color='black')  # add coast lines
    ax.add_feature(cfeature.STATES)  # add US states
    ax.set_yticks(np.arange(25, 50 + 1, 5), crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(-125, -67 + 1, 8), crs=ccrs.PlateCarree())
    lats = np.linspace(26, 50, data_matrix[0].shape[0] + 1)
    lons = np.linspace(-125, -68, data_matrix[0].shape[1] + 1)
    color_map = 'RdBu_r'
    plot = ax.pcolormesh(lons + 0.5, lats - 0.5, data_matrix[0],
                         vmin=0, vmax=vmax,
                         cmap=color_map, snap=True)
    cb = plt.colorbar(plot, fraction=0.02, pad=0.04)

    def animate(i):
        plot.set_array(data_matrix[i].ravel())
        plt.title(title[i])
        return plot

    ani = animation.FuncAnimation(
        fig, animate, frames=len(data_matrix), interval=700, blit=False, repeat=False)
    return ani
