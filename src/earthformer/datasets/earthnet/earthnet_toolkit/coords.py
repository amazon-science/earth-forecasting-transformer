"""Code is adapted from https://github.com/earthnet2021/earthnet-toolkit."""
import warnings
import os
import json
from pyproj import Transformer
from .coords_dict import COORDS


EXTREME_TILES = ["32UMC", "32UNC", "32UPC", "32UQC"]

def get_coords_from_cube(cubename: str, return_meso: bool = False, ignore_warning = False):
    """

    Get the coordinates for a Cube in Lon-Lat-Grid.

    Args:
        cubename (str): cubename (has format tile_startyear_startmonth_startday_endyear_endmonth_endday_hrxmin_hrxmax_hrymin_hrymax_mesoxmin_mesoxmax_mesoymin_mesoymax.npz)
        return_meso (bool, optional): If True returns also the coordinates for the Meso-scale variables in the cube. Defaults to False.

    Returns:
        tuple: Min-Lon, Min-Lat, Max-Lon, Max-Lat or Min-Lon-HR, Min-Lat-HR, Max-Lon-HR, Max-Lat-HR, Min-Lon-Meso, Min-Lat-Meso, Max-Lon-Meso, Max-Lat-Meso
    """    
    
    if not ignore_warning:
        warnings.warn('Getting coordinates to a cube is experimental. The resulting coordinates on Lon-Lat-Grid will never be pixel perfect. Under certain circumstances, the whole bounding box might shifted by up to 0.02° in either direction. Use with caution. EarthNet2021 does not provide geo-referenced data.')
    
    cubetile,_, _,hr_x_min, hr_x_max, hr_y_min, hr_y_max, meso_x_min, meso_x_max, meso_y_min, meso_y_max = os.path.splitext(cubename)[0].split("_")

    tile = COORDS[cubetile]

    transformer = Transformer.from_crs(tile["EPSG"], 4326, always_xy = True)

    tile_x_min, tile_y_max = transformer.transform(tile["MinLon"],tile["MaxLat"], direction = "INVERSE")

    if cubetile in EXTREME_TILES:
        hr_x_min = int(hr_x_min) + 57
        hr_x_max = int(hr_x_max) + 57

    cube_x_min = tile_x_min + 20 * float(hr_y_min)
    cube_x_max = tile_x_min + 20 * float(hr_y_max)
    cube_y_min = tile_y_max - 20 * float(hr_x_min)
    cube_y_max = tile_y_max - 20 * float(hr_x_max)
    
    cube_lon_min, cube_lat_min = transformer.transform(cube_x_min, cube_y_max)
    cube_lon_max, cube_lat_max = transformer.transform(cube_x_max, cube_y_min)
    
    if return_meso:
        meso_x_min = tile_x_min + 20 * float(meso_x_min)
        meso_x_max = tile_x_min + 20 * float(meso_x_max)
        meso_y_min = tile_y_max - 20 * float(meso_y_min)
        meso_y_max = tile_y_max - 20 * float(meso_y_max)

        meso_lon_min, meso_lat_min = transformer.transform(meso_x_min, meso_y_max)
        meso_lon_max, meso_lat_max = transformer.transform(meso_x_max, meso_y_min)

        return cube_lon_min, cube_lat_min, cube_lon_max, cube_lat_max, meso_lon_min, meso_lat_min, meso_lon_max, meso_lat_max

    else:
        return cube_lon_min, cube_lat_min, cube_lon_max, cube_lat_max

def get_coords_from_tile(tilename: str):
    """
    Get the Coordinates for a Tile in Lon-lat-grid.

    Args:
        tilename (str): 5 Letter MGRS tile

    Returns:
        tuple: Min-Lon, Min-Lat, Max-Lon, Max-Lat
    """    
    tile = COORDS[tilename]
    
    return tile["MinLon"], tile["MinLat"], tile["MaxLon"], tile["MaxLat"]

if __name__ == "__main__":
    import fire
    fire.Fire()
