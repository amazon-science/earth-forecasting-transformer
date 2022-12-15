"""Code is adapted from https://github.com/earthnet2021/earthnet-toolkit."""
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
import matplotlib.colors as clr
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import pandas as pd
from pathlib import Path


LANDCOVER_CLASSES = {
    0: "Clouds",
    62: "Artificial surfaces and constructions",
    73: "Cultivated areas",
    75: "Vineyards",
    82: "Broadleaf tree cover",
    83: "Coniferous tree cover",
    102: "Herbaceous vegetation",
    103: "Moors and Heathland",
    104: "Sclerophyllous vegetation",
    105: "Marshes",
    106: "Peatbogs",
    121: "Natural material surfaces", 
    123: "Permanent snow covered surfaces",
    162: "Water bodies",
    255: "No data",
}

LC_CONVERTED = {k: (i,LANDCOVER_CLASSES[k]) for i, k in enumerate(sorted(list(LANDCOVER_CLASSES.keys())))}
LC_CONVERTED_CLASSES = {v[0]: v[1] for k, v in LC_CONVERTED.items()}
lc_convert = np.vectorize(lambda x: LC_CONVERTED[x][0])

COLORS = np.array(
    [[255,255,255],
    [210,0,0],
    [253,211,39],
    [176,91,16],
    [35,152,0],
    [8,98,0],
    [249,150,39],
    [141,139,0],
    [95,53,6],
    [149,107,196],
    [77,37,106],
    [154,154,154],
    [106,255,255],
    [20,69,249],
    [255,255,255]]
)

def colorize(data, colormap = "ndvi", mask_red = None, mask_blue = None):
    t,h,w = data.shape
    in_data = data.reshape(-1)
    if mask_red is not None:
        in_data = np.ma.array(in_data, mask = mask_red.reshape(-1))  

    cmap = clr.LinearSegmentedColormap.from_list('ndvi', ["#cbbe9a","#fffde4","#bccea5","#66985b","#2e6a32","#123f1e","#0e371a","#01140f","#000d0a"], N=256) if colormap == "ndvi" else copy.copy(plt.get_cmap(colormap))
    cmap.set_bad(color='red')

    if mask_blue is None:
        return cmap(in_data)[:,:3].reshape((t,h,w,3))
    else:
        out = cmap(in_data)[:,:3].reshape((t,h,w,3))
        return np.stack([np.where(mask_blue, out[:,:,:,0],np.zeros_like(out[:,:,:,0])), 
                            np.where(mask_blue, out[:,:,:,1],np.zeros_like(out[:,:,:,1])), 
                            np.where(mask_blue, out[:,:,:,2],0.1*np.ones_like(out[:,:,:,2]))], axis = -1)

def gallery(array, ncols=10):
    nindex, height, width, intensity = array.shape
    nrows = ceil(nindex / ncols)
    nindex_pad = nrows * ncols
    height_pad = height + 2
    width_pad = width + 2
    padded = np.zeros((nindex_pad, height_pad, width_pad, intensity))
    padded[:nindex,1:-1,1:-1,:] = array
    result = (padded.reshape(nrows, ncols, height_pad, width_pad, intensity)
              .swapaxes(1,2)
              .reshape(height_pad*nrows, width_pad*ncols, intensity))
    return result

def cube_gallery(cube, variable = "rgb", vegetation_mask = None, cloud_mask = True, save_path = None):
    """

    Plots a gallery view from a given Cube.

    Args:
        cube (np.ndarray): Numpy Array or loaded NPZ of Cube or path to Cube.
        variable (str, optional):  One of "rgb", "ndvi", "rr","pp","tg","tn","tx". Defaults to "rgb".
        vegetation_mask (np.ndarray, optional): If given uses this as red mask over non-vegetation. S2GLC data. Defaults to None.
        cloud_mask (bool, optional): If True tries to use the last channel from the cubes sat imgs as blue cloud mask, 1 where no clouds, 0 where there are clouds. Defaults to True.
        save_path (str, optional): If given, saves PNG to this path. Defaults to None.

    Returns:
        plt.Figure: Matplotlib Figure
    """    

    assert(variable in ["rgb", "ndvi", "rr","pp","tg","tn","tx"])

    if isinstance(cube, str) or isinstance(cube, Path):
        cube = np.load(cube)

    if isinstance(cube, np.lib.npyio.NpzFile):
        if variable in ["rgb","ndvi"]:
            if "highresdynamic" in cube:
                data = cube["highresdynamic"]
            else:
                for k in cube:
                    if 128 in cube[k].shape:
                        data = cube[k]
                        break
                raise ValueError("data does not contain satellite imagery.")
        elif variable in ["rr","pp","tg","tn","tx"]:
            if "mesodynamic" in cube:
                data = cube["mesodynamic"]
            else:
                raise ValueError("data does not contain E-OBS.")
    elif isinstance(cube, np.ndarray):
        data = cube

    hw = 128 if variable in ["rgb","ndvi"] else 80
    hw_idxs = [i for i,j in enumerate(data.shape) if j == hw]
    assert(len(hw_idxs) > 1)
    if len(hw_idxs) == 2 and hw_idxs != [1,2]:
        c_idx = [i for i,j in enumerate(data.shape) if j == min([j for j in data.shape if j != hw])][0]
        t_idx = [i for i,j in enumerate(data.shape) if j == max([j for j in data.shape if j != hw])][0]
        data = np.transpose(data,(t_idx,hw_idxs[0],hw_idxs[1],c_idx))

    if variable == "rgb":
        targ = np.stack([data[:,:,:,2],data[:,:,:,1],data[:,:,:,0]], axis = -1)
        targ[targ<0] = 0
        targ[targ>0.5] = 0.5
        targ = 2*targ
        if data.shape[-1] > 4 and cloud_mask:
            mask = data[:,:,:,-1]
            zeros = np.zeros_like(targ)
            zeros[:,:,:,2] = 0.1
            targ = np.where(np.stack([mask]*3,-1).astype(np.uint8) | np.isnan(targ).astype(np.uint8), zeros, targ)
        else:
            targ[np.isnan(targ)] = 0

    elif variable == "ndvi":
        if data.shape[-1] == 1:
            targ = data[:,:,:,0]
        else:
            targ = (data[:,:,:,3] - data[:,:,:,2]) / (data[:,:,:,2] + data[:,:,:,3] + 1e-6)
        if data.shape[-1] > 4 and cloud_mask:
            cld_mask = 1 - data[:,:,:,-1]
        else:
            cld_mask = None
        
        if vegetation_mask is not None:
            if isinstance(vegetation_mask, str) or isinstance(vegetation_mask, Path):
                vegetation_mask = np.load(vegetation_mask)
            if isinstance(vegetation_mask, np.lib.npyio.NpzFile):
                vegetation_mask = vegetation_mask["landcover"]
            vegetation_mask = vegetation_mask.reshape(hw, hw)
            lc_mask = 1 - (vegetation_mask > 63) & (vegetation_mask < 105)
            lc_mask = np.repeat(lc_mask[np.newaxis,:,:],targ.shape[0], axis = 0)
        else:
            lc_mask = None
        targ = colorize(targ, colormap = "ndvi", mask_red = lc_mask, mask_blue = cld_mask)
    
    elif variable == "rr":
        targ = data[:,:,:,0]
        targ = colorize(targ, colormap = 'Blues', mask_red = np.isnan(targ))
    elif variable == "pp":
        targ = data[:,:,:,1]
        targ = colorize(targ, colormap = 'rainbow', mask_red = np.isnan(targ))
    elif variable in ["tg","tn","tx"]:
        targ = data[:,:,:, 2 if variable == "tg" else 3 if variable == "tn" else 4]
        targ = colorize(targ, colormap = 'coolwarm', mask_red = np.isnan(targ))

    grid = gallery(targ)

    fig = plt.figure(dpi = 300)
    plt.imshow(grid)
    plt.axis('off')
    if variable != "rgb":
        colormap = {"ndvi": "ndvi", "rr": "Blues", "pp": "rainbow", "tg": "coolwarm", "tn": "coolwarm", "tx": "coolwarm"}[variable]
        cmap = clr.LinearSegmentedColormap.from_list('ndvi', ["#cbbe9a","#fffde4","#bccea5","#66985b","#2e6a32","#123f1e","#0e371a","#01140f","#000d0a"], N=256) if colormap == "ndvi" else copy.copy(plt.get_cmap(colormap))
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.1)
        vmin, vmax = {"ndvi": (0,1), "rr": (0,50), "pp": (900,1100), "tg": (-50,50), "tn": (-50,50), "tx": (-50,50)}[variable]
        label = {"ndvi": "NDVI", "rr": "Precipitation in mm/d", "pp": "Sea-level pressure in hPa", "tg": "Mean temperature in °C", "tn": "Minimum Temperature in °C", "tx": "Maximum Temperature in °C"}[variable]
        plt.colorbar(cm.ScalarMappable(norm = clr.Normalize(vmin = vmin, vmax = vmax), cmap = cmap), cax = cax, label = label)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parents[0].mkdir(parents = True, exist_ok = True)
        plt.savefig(save_path, dpi = 300, bbox_inches='tight', transparent=True)

    return fig

def cube_ndvi_timeseries(pred, targ, vegetation_mask = None, save_path = None):
    """

    Plots a timeseries view of a predicted cube vs its respective target.

    Args:
        pred (str, Path, np.lib.npyio.NpzFile, np.ndarray): Cube with prediction
        targ (str, Path, np.lib.npyio.NpzFile, np.ndarray): Cube with target
        vegetation_mask (str, Path, np.lib.npyio.NpzFile, np.ndarray, optional): Cube with S2GLC Landcover mask. Defaults to None.
        save_path (str, optional): If given, saves PNG to this path. Defaults to None.

    Returns:
        plt.Figure: Matplotlib Figure
    """    
    if isinstance(pred, str) or isinstance(pred, Path):
        pred_cube = np.load(pred)
        pred_ndvi = pred_cube["highresdynamic"].astype(np.float32)
    elif isinstance(pred, np.lib.npyio.NpzFile):
        pred_cube = pred
        pred_ndvi = pred_cube["highresdynamic"].astype(np.float32)
    else:
        assert(isinstance(pred, np.ndarray))
        pred_ndvi = pred_ndvi
    
    if pred_ndvi.shape[-2] > 1:
        pred_ndvi = (pred_ndvi[:,:,3,:] - pred_ndvi[:,:,2,:]) / (pred_ndvi[:,:,2,:] + pred_ndvi[:,:,3,:] + 1e-6)
    
    if isinstance(targ, str) or isinstance(targ, Path):
        targ_cube = np.load(targ)
        targ_data = targ_cube["highresdynamic"].astype(np.float32)
    elif isinstance(targ, np.lib.npyio.NpzFile):
        targ_cube = targ
        targ_data = targ_cube["highresdynamic"].astype(np.float32)
    else:
        assert(isinstance(targ, np.ndarray))
        targ_data = targ

    if targ_data.shape[-2] == 1:
        targ_ndvi = targ_data
    else:
        targ_ndvi = (targ_data[:,:,3,:] - targ_data[:,:,2,:]) / (targ_data[:,:,2,:] + targ_data[:,:,3,:] + 1e-6)
    if targ_data.shape[-2] > 4:
        targ_ndvi = np.ma.array(targ_ndvi, mask = targ_data[:,:,-1,:].astype(bool))

    if vegetation_mask is not None:
        if isinstance(vegetation_mask, str) or isinstance(vegetation_mask, Path):
            vegetation_mask = np.load(vegetation_mask)
        if isinstance(vegetation_mask, np.lib.npyio.NpzFile):
            landcover = vegetation_mask["landcover"]
        else:
            landcover = vegetation_mask
        temp = np.concatenate([landcover.reshape((1,128,128)), np.indices((128,128))], axis = 0)
        df = pd.DataFrame(temp.reshape(3,-1).T, columns = ["lc", "x", "y"])
        coords = df[(df.lc > 63) & (df.lc < 105)].groupby('lc').agg(np.random.choice).to_numpy()
        if coords.shape[0] < 8:
            coords = np.concatenate([coords, np.indices((128,128)).reshape(2,-1).T[np.random.choice(128*128, 8-coords.shape[0])]], axis = 0)
    else:
        coords = np.indices((128,128)).reshape(2,-1).T[np.random.choice(128*128, 8)]

    fig, axs = plt.subplots(4,3, dpi = 450)
    for idx, ax in enumerate(axs.reshape(-1)): 
        if idx == 0:
            cmap = clr.LinearSegmentedColormap.from_list('ndvi', ["#cbbe9a","#fffde4","#bccea5","#66985b","#2e6a32","#123f1e","#0e371a","#01140f","#000d0a"], N=256)
            cmap.set_bad(color='red')
            ndvi = ax.imshow(targ_ndvi.mean(-1), cmap = cmap, vmin = 0, vmax = 1)
            ncbar = fig.colorbar(ndvi, ax=axs[0,0])
            ncbar.ax.tick_params(labelsize=6)
            ax.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            ax.scatter(coords[:,0], coords[:,1], c = "grey", s = 1)
            annotations=[f"{i}" for i in range(1,9)]
            for i, label in enumerate(annotations):
                ax.annotate(label, (coords[:,0][i], coords[:,1][i]), fontsize = 4, color = "orange")
            ax.set_title("Mean Target NDVI", fontsize = 6, loc = "left")
        elif idx == 1:
            dem = ax.imshow(2000 * (2*targ_cube["highresstatic"].astype(np.float32).reshape((128,128))-1), cmap = "terrain")
            dcbar = fig.colorbar(dem, ax=axs[0,1])
            dcbar.ax.tick_params(labelsize=6)
            ax.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            ax.scatter(coords[:,0], coords[:,1], c = "grey", s = 1)
            annotations=[f"{i}" for i in range(1,9)]
            for i, label in enumerate(annotations):
                ax.annotate(label, (coords[:,0][i], coords[:,1][i]), fontsize = 4, color = "black")
            ax.set_title("EU-DEM", fontsize = 6, loc = "left")
            #ax.tick_params(labelsize = 6)
        elif idx == 2:
            if vegetation_mask is not None:
                cmap = clr.ListedColormap(COLORS/255.)
                bounds = [i-0.5 for i in range(16)]
                norm = clr.BoundaryNorm(bounds, cmap.N)
                lac = ax.imshow(lc_convert(landcover.reshape((128,128))), cmap = cmap, norm = norm)
                lcbar = fig.colorbar(lac, ax=axs[0,2], ticks=sorted(list(LC_CONVERTED_CLASSES.keys())))
                lcbar.ax.set_yticklabels([LC_CONVERTED_CLASSES[i] for i in sorted(list(LC_CONVERTED_CLASSES.keys()))], fontsize = 3)
                lcbar.ax.tick_params(labelsize=4)
                ax.scatter(coords[:,0], coords[:,1], c = "black", s = 1)
                annotations=[f"{i}" for i in range(1,9)]
                for i, label in enumerate(annotations):
                    ax.annotate(label, (coords[:,0][i], coords[:,1][i]), fontsize = 4, color = "black")
                ax.set_title("S2GLC Landcover", fontsize = 6, loc = "left")
                ax.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            else:
                continue
        elif idx == 3:
            ax.scatter(np.linspace(0,targ_ndvi.shape[-1],targ_ndvi.shape[-1], endpoint = False),targ_ndvi.mean((0,1)).flatten(), c = "orange", s = 0.5)
            ax.plot(np.linspace(targ_ndvi.shape[-1]-pred_ndvi.shape[-1],targ_ndvi.shape[-1],pred_ndvi.shape[-1], endpoint = False),pred_ndvi.mean((0,1)).flatten(), label = f"Pred Mean", lw = 0.5)
            ax.set_ylim(0.,1.)
            ax.set_title("NDVI mean value", fontsize = 6, loc = "left")
            ax.tick_params(labelsize = 6)
        else:
            i = idx - 4
            x, y = coords[i]
            ax.scatter(np.linspace(0,targ_ndvi.shape[-1],targ_ndvi.shape[-1], endpoint = False),targ_ndvi[x,y,:].flatten(), c = "orange", s = 0.5)
            ax.plot(np.linspace(targ_ndvi.shape[-1]-pred_ndvi.shape[-1],targ_ndvi.shape[-1],pred_ndvi.shape[-1], endpoint = False),pred_ndvi[x,y,:].flatten(), lw = 0.5)
            ax.set_ylim(0.,1.)
            if vegetation_mask is None:
                ax.set_title(f"NDVI Point {i+1}", fontsize = 6, loc = "left")
            else:
                lc = LANDCOVER_CLASSES[int(vegetation_mask["landcover"].reshape((128,128))[x,y])]
                ax.set_title(f"NDVI Point {i+1},\n{lc}", fontsize = 6, loc = "left")
            ax.tick_params(labelsize = 6)

    plt.subplots_adjust(wspace=0.4, hspace=0.8)

    p00 = axs[0, 0].get_position()
    p01 = axs[0, 1].get_position()
    
    p10 = axs[1, 0].get_position()
    p11 = axs[1, 1].get_position()
    p12 = axs[1, 2].get_position()
    p00c = ncbar.ax.get_position()
    p00c = [(p10.x0 + (p00c.x0 - p00.x0)), p00c.y0, p00c.width, p00c.height]
    ncbar.ax.set_position(p00c)
    p01c = dcbar.ax.get_position()
    p01c = [(p11.x0 + (p01c.x0 - p01.x0)), p01c.y0, p01c.width, p01c.height]
    dcbar.ax.set_position(p01c)
    if vegetation_mask is not None:
        p02 = axs[0, 2].get_position()
        p02c = lcbar.ax.get_position()
        p02c = [(p12.x0 + (p02c.x0 - p02.x0)), p02c.y0, p02c.width, p02c.height]
        lcbar.ax.set_position(p02c)
        p02 = [p12.x0, p02.y0, p02.width, p02.height]
        axs[0, 2].set_position(p02)

    p00 = [p10.x0, p00.y0, p00.width, p00.height]
    axs[0, 0].set_position(p00)
    p01 = [p11.x0, p01.y0, p01.width, p01.height]
    axs[0, 1].set_position(p01)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parents[0].mkdir(parents = True, exist_ok = True)
        plt.savefig(save_path, dpi = 450, bbox_inches='tight', transparent=True)

    return fig

if __name__ == "__main__":
    import fire
    fire.Fire(cube_gallery)
