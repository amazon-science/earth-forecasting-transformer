"""Code is adapted from https://github.com/earthnet2021/earthnet-toolkit."""
from typing import Tuple, Optional, Sequence, Union
import argparse
import re
import json
import time
import numpy as np
from skimage import metrics
import scipy.stats
from pathlib import Path
import multiprocessing
from tqdm import tqdm
import warnings


class CubeCalculator:
    """Loads single cube and calculates subscores for EarthNetScore

    Example:

        >>> scores = CubeCalculator.get_scores({"pred_filepath": Path/to/pred.npz, "targ_filepath": Path/to/targ.npz})
    """    

    @staticmethod
    def MAD(preds: np.ndarray, targs: np.ndarray, masks: np.ndarray) -> Tuple[float, dict]:
        """Median absolute deviation score

        Median absolute deviation between non-masked target and predicted pixels. Scaled by a scaling factor such that a distance the size of a 99.7% confidence interval of the variance of the pixelwise centered timeseries is scaled to 0.9 (such that the mad-score becomes 0.1). The mad-score is 1-MAD, it is scaled from 0 (worst) to 1 (best).

        Args:
            preds (np.ndarray): Predictions, shape h,w,c,t
            targs (np.ndarray): Targets, shape h,w,c,t
            masks (np.ndarray): Masks, shape h,w,c,t, 1 if non-masked, else 0
        Returns:
            Tuple[float, dict]: mad-score, debugging information
        """        
        dists = np.abs(preds-targs)
        dists[masks == 0] = np.nan

        scaling_factor = 0.06649346971087526 # Computed via the expected distance from pixelwise timeseries variance
        dists = dists.astype(np.float64)

        scaled_dists = dists ** scaling_factor

        distmedian = np.nanmedian(scaled_dists)
        if distmedian is None:
            mad = None
        else:
            mad = max(0,min(1,1-distmedian))
        
        MAE_frames = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for t in range(dists.shape[-1]):
                mean = np.nanmean(dists[:,:,:,t])
                if mean is np.nan:
                    mean = 1000
                MAE_frames.append(mean)
        
        debug_info = {
                        "minimum distance": float(np.nanmin(dists)), 
                        "maximum distance": float(np.nanmax(dists)), 
                        "mean distance": float(np.nanmean(dists)), 
                        "median distance": float(np.nanmedian(dists)), 
                        "number nan": float(np.isnan(dists).sum()), 
                        "MAD score": float(mad),
                        "frames": MAE_frames 
                    }
        return mad, debug_info

    @staticmethod
    def OLS(preds: np.ndarray, targs: np.ndarray, masks: np.ndarray) -> Tuple[float, dict]:
        """Ordinary least squares slope deviation score

        Mean absolute difference between ordinary least squares slopes of target and predicted pixelwise NDVI timeseries. Target slopes are calculated over non-masked values. Predicted slopes are calculated for all values between the first and last non-masked value of a given timeseries. Scaled by a scaling factor such that a distance the size of a 99.7% confidence interval of the variance of the pixelwise centered NDVI timeseries is scaled to 0.9 (such that the ols-score becomes 0.1). If the timeseries is longer than 40 steps, it is split up into parts of length 20. The ols-score is 1-mean(abs(b_targ - b_pred)), it is scaled from 0 (worst) to 1 (best).

        Args:
            preds (np.ndarray): NDVI Predictions, shape h,w,1,t
            targs (np.ndarray): NDVI Targets, shape h,w,1,t
            masks (np.ndarray): NDVI Masks, shape h,w,1,t, 1 if non-masked, else 0

        Returns:
            Tuple[float, dict]: ols-score, debugging information
        """        

        h, w, c, t = preds.shape


        if t > 40: # Checking if pixelwise timeseries is too long
            assert(t%20 == 0)
            preds = np.reshape(preds, (h, w, -1, 20))
            targs = np.reshape(targs, (h, w, -1, 20))
            masks = np.reshape(masks, (h, w, -1, 20))
            h, w, c, t = preds.shape
        
        A = np.vstack([np.linspace(1,2,t),np.ones(t)]).T[np.newaxis,:,:].repeat(c*h*w, 0)
        targs = np.reshape(targs, (-1, t))[:,:,np.newaxis]
        preds = np.reshape(preds, (-1, t))[:,:,np.newaxis]
        masks = np.reshape(masks, (-1, t))[:,:,np.newaxis]
        masks[(masks.sum(1, keepdims = True) < 2).repeat(t,1)] = 0
        targsmasked = targs * masks

        Atarg = A * masks
        Atargmin = np.ma.masked_equal(Atarg, 0.0, copy=True)
        Atarg[:,:,0] = np.where(Atarg[:,:,0] > 0, Atarg[:,:,0] - Atargmin[:,:,0].min(1,keepdims = True), Atarg[:,:,0]) 
        Atarg[:,:,0] = 2 * np.where(Atarg[:,:,1] > 0, Atarg[:,:,0] / (Atarg[:,:,0].max(1, keepdims = True) + 1e-8) + 1, Atarg[:,:,0])
        AtargT = Atarg.transpose(0,2,1)

        predmask = np.where((A[:,:,0] >= Atargmin[:,:,0].min(1,keepdims = True)) & (A[:,:,0] <= Atargmin[:,:,0].max(1,keepdims = True)), np.ones_like(A[:,:,0]), np.zeros_like(A[:,:,0]))[:,:,np.newaxis]
        A = A * predmask
        A[:,:,0] = np.where(A[:,:,0] > 0, A[:,:,0] - Atargmin[:,:,0].min(1,keepdims = True), A[:,:,0]) 
        A[:,:,0] = 2 * np.where(A[:,:,1] > 0, A[:,:,0] / (A[:,:,0].max(1, keepdims = True) + 1e-8) + 1, A[:,:,0])

        predsmasked = preds * predmask

        AT = A.transpose(0,2,1)

        noise = np.random.rand(c*h*w,2,2)/10000

        btarg = np.matmul(np.linalg.inv(np.matmul(AtargT,Atarg)+noise),np.matmul(AtargT,targsmasked))
        
        bpred = np.matmul(np.linalg.inv(np.matmul(AT ,A) + noise),np.matmul(AT,predsmasked))      

        dists = np.abs(btarg[:,0,0] - bpred[:,0,0])/2

        scaling_factor = 0.10082047548620601 # Computed via the expected distance from pixelwise timeseries variance
        
        dists = dists.astype(np.float64)

        scaled_dists = dists ** scaling_factor
        
        distmean = np.nanmean(scaled_dists)
        if distmean is None:
            ols = None
        else:
            ols = max(0,min(1, 1-distmean))

        debug_info = {
                        #"target slopes": btarg[:,0,0].tolist(), 
                        #"predicted slopes": bpred[:,0,0].tolist(), 
                        "min target slope": float(np.nanmin(btarg[:,0,0])), 
                        "max target slope": float(np.nanmax(btarg[:,0,0])), 
                        "mean target slope": float(np.nanmean(btarg[:,0,0])), 
                        "number nan target slope": float(np.isnan(btarg[:,0,0]).sum()),
                        "min pred slope": float(np.nanmin(bpred[:,0,0])), 
                        "max pred slope": float(np.nanmax(bpred[:,0,0])), 
                        "mean pred slope": float(np.nanmean(bpred[:,0,0])), 
                        "number nan pred slope": float(np.isnan(bpred[:,0,0]).sum()),
                        "minimum distance": float(np.nanmin(dists)), 
                        "maximum distance": float(np.nanmax(dists)), 
                        "mean distance": float(np.nanmean(dists)), 
                        "median distance": float(np.nanmedian(dists)), 
                        "number nan": float(np.isnan(dists).sum()), 
                        "ols score": float(ols)}

        return ols, debug_info

    @classmethod
    def EMD(cls, preds: np.ndarray, targs: np.ndarray, masks: np.ndarray) -> Tuple[float, dict]:
        """Earth mover distance score

        The earth mover distance (w1 metric) is computed between target and predicted pixelwise NDVI timeseries value distributions. For the target distributions, only non-masked values are considered. Scaled by a scaling factor such that a distance the size of a 99.7% confidence interval of the variance of the pixelwise centered NDVI timeseries is scaled to 0.9 (such that the ols-score becomes 0.1). The emd-score is 1-mean(emd), it is scaled from 0 (worst) to 1 (best).

        Args:
            preds (np.ndarray): NDVI Predictions, shape h,w,1,t
            targs (np.ndarray): NDVI Targets, shape h,w,1,t
            masks (np.ndarray): NDVI Masks, shape h,w,1,t, 1 if non-masked, else 0

        Returns:
            Tuple[float, dict]: emd-score, debugging information
        """        
        data = np.concatenate([preds, targs, masks], axis = -1)
        dists = np.apply_along_axis(cls.compute_w1, axis = -1, arr = data)
        
        scaling_factor = 0.10082047548620601 # Computed via the expected distance from pixelwise timeseries variance

        dists = np.abs(dists).astype(np.float64)

        scaled_dists = dists ** scaling_factor

        distmean = np.nanmean(scaled_dists)

        if distmean is None:
            emd = None
        else:
            emd = max(0,min(1, 1-distmean))

        debug_info = {
                        "minimum distance": float(np.nanmin(dists)), 
                        "maximum distance": float(np.nanmax(dists)), 
                        "mean distance": float(np.nanmean(dists)), 
                        "median distance": float(np.nanmedian(dists)), 
                        "number nan": float(np.isnan(dists).sum()), 
                        "w1 score": float(emd)
                    }

        return emd, debug_info

    @staticmethod
    def compute_w1(datarow: np.ndarray) -> Union[np.ndarray, None]:
        """Computing w1 distance for np.apply_along_axis

        Args:
            datarow (np.ndarray): 1-dimensional array that can be split into three parts of equal size, these are in order: predictions, targets and masks for a single pixel and channel through time.

        Returns:
            Union[np.ndarray, None]: w1 distance between prediction and target, if not completely masked, else None.
        """        
        preds, targs, masks = np.split(datarow, 3)
        targs = targs[masks == 1]
        if len(targs) > 1:
            return scipy.stats.wasserstein_distance(preds, targs)
        else:
            return np.nan

    @staticmethod
    def SSIM(preds: np.ndarray, targs: np.ndarray, masks: np.ndarray) -> Tuple[float, dict]:
        """Structural similarity index score

        Structural similarity between predicted and target cube computed for all channels and frames individually if the given target is less than 30% masked. Scaled by a scaling factor such that a mean SSIM of 0.8 is scaled to a ssim-score of 0.1. The ssim-score is mean(ssim), it is scaled from 0 (worst) to 1 (best).

        Args:
            preds (np.ndarray): Predictions, shape h,w,c,t
            targs (np.ndarray): Targets, shape h,w,c,t
            masks (np.ndarray): Masks, shape h,w,c,t, 1 if non-masked, else 0

        Returns:
            Tuple[float, dict]: ssim-score, debugging information
        """        

        ssim_targs = np.where(masks, targs, preds)
        new_shape = (-1, preds.shape[0], preds.shape[1])
        ssim_targs = np.transpose(np.reshape(np.transpose(ssim_targs, (3,2,0,1)), new_shape),(1,2,0))
        ssim_preds = np.transpose(np.reshape(np.transpose(preds, (3,2,0,1)), new_shape),(1,2,0))
        ssim_masks = np.transpose(np.reshape(np.transpose(masks, (3,2,0,1)), new_shape),(1,2,0))
        running_ssim = 0
        counts = 0
        ssim_frames = []
        for i in range(ssim_targs.shape[-1]):
            if ssim_masks[:,:,i].sum() > 0.7*ssim_masks[:,:,i].size:
                curr_ssim = metrics.structural_similarity(ssim_targs[:,:,i], ssim_preds[:,:,i])
                running_ssim += curr_ssim
                counts += 1
            else:
                curr_ssim = 1000
            ssim_frames.append(curr_ssim)
        
        if counts == 0:
            ssim = None
        else:
            ssim = max(0,(running_ssim/max(counts,1)))

            scaling_factor = 10.31885115 # Scales SSIM=0.8 down to 0.1

            ssim = float(ssim ** scaling_factor)
        
        debug_info = {
                        #"framewise SSIM, 1000 if frame was too much masked": ssim_frames, 
                        "Min SSIM": str(np.ma.filled(np.ma.masked_equal(np.array(ssim_frames), 1000.0).min(), np.nan)),
                        "Max SSIM": str(np.ma.filled(np.ma.masked_equal(np.array(ssim_frames), 1000.0).max(), np.nan)),
                        "Mean SSIM": str(np.ma.filled(np.ma.masked_equal(np.array(ssim_frames), 1000.0).mean(), np.nan)),
                        "Standard deviation SSIM": str(np.ma.filled(np.ma.masked_equal(np.array(ssim_frames), 1000.0).std(), np.nan)),
                        "Valid SSIM frames": counts,
                        "SSIM score": ssim,
                        "frames": ssim_frames
                    }

        return ssim, debug_info

    @staticmethod
    def load_file(pred_filepath: Path, targ_filepath: Path) -> Sequence[np.ndarray]:
        """Load a single target cube and a matching prediction

        Args:
            pred_filepath (Path): Path to predicted cube
            targ_filepath (Path): Path to target cube

        Returns:
            Sequence[np.ndarray]: preds, targs, masks, ndvi_preds, ndvi_targs, ndvi_masks
        """        
        
        pred_npz = np.load(pred_filepath)
        targ_npz = np.load(targ_filepath)

        pred_key = "highresdynamic" if "highresdynamic" in pred_npz.keys() else list(pred_npz.keys())[0]

        preds = pred_npz[pred_key][:,:,:4,:]
        targs = targ_npz["highresdynamic"][:,:,:4,:]
        masks = ((1 - targ_npz["highresdynamic"][:,:,-1,:])[:,:,np.newaxis,:]).repeat(4,2)

        if preds.shape[-1] < targs.shape[-1]:
            targs = targs[:,:,:,-preds.shape[-1]:]
            masks = masks[:,:,:,-preds.shape[-1]:]
        
        assert(preds.shape == targs.shape)

        preds[preds < 0] = 0
        preds[preds > 1] = 1

        targs[np.isnan(targs)] = 0
        targs[targs > 1] = 1
        targs[targs < 0] = 0

        ndvi_preds = ((preds[:,:,3,:] - preds[:,:,2,:])/(preds[:,:,3,:] + preds[:,:,2,:] + 1e-6))[:,:,np.newaxis,:]
        ndvi_targs = ((targs[:,:,3,:] - targs[:,:,2,:])/(targs[:,:,3,:] + targs[:,:,2,:] + 1e-6))[:,:,np.newaxis,:]
        ndvi_masks = masks[:,:,0,:][:,:,np.newaxis,:]

        return preds, targs, masks, ndvi_preds, ndvi_targs, ndvi_masks

    @classmethod
    def get_scores(cls, filepaths: dict) -> dict:
        """Get all subscores for a given cube

        Args:
            filepaths (dict): Has keys "pred_filepath", "targ_filepath" with respective paths.

        Returns:
            dict: subscores and debugging info for the input cube
        """        
        assert({"pred_filepath", "targ_filepath"}.issubset(set(filepaths.keys())))
        
        preds, targs, masks, ndvi_preds, ndvi_targs, ndvi_masks = cls.load_file(filepaths["pred_filepath"], filepaths["targ_filepath"])

        debug_info = {}

        mad, debug_info["MAD"] = cls.MAD(preds, targs, masks)

        ols, debug_info["OLS"] = cls.OLS(ndvi_preds, ndvi_targs, ndvi_masks)

        emd, debug_info["EMD"] = cls.EMD(ndvi_preds, ndvi_targs, ndvi_masks)

        ssim, debug_info["SSIM"] = cls.SSIM(preds, targs, masks)

        return {
            "pred_filepath": str(filepaths["pred_filepath"]),
            "targ_filepath": str(filepaths["targ_filepath"]),
            "MAD": mad,
            "OLS": ols,
            "EMD": emd,
            "SSIM": ssim,
            "debug_info": debug_info
        }

class EarthNetScore:
    """EarthNetScore class, fast computation using multiprocessing

    Example:

        Direct computation
        >>> EarthNetScore.get_ENS(Path/to/predictions, Path/to/targets, data_output_file = Path/to/data.json, ens_output_file = Path/to/ens.json)

        More control (for further plotting)
        >>> ENS = EarthNetScore(Path/to/predictions, Path/to/targets)
        >>> data = ENS.compute_scores()
        >>> ens = ENS.summarize()

    """    
    def __init__(self, pred_dir: str, targ_dir: str):
        """Initialize EarthNetScore

        Args:
            pred_dir (str): Directory with predictions, format is one of {pred_dir/tile/cubename.npz, pred_dir/tile/experiment_cubename.npz}
            targ_dir (str): Directory with targets, format is one of {targ_dir/target/tile/target_cubename.npz, targ_dir/target/tile/cubename.npz, targ_dir/tile/target_cubename.npz, targ_dir/tile/cubename.npz}
        """        
        self.get_paths(pred_dir, targ_dir)

    def get_paths(self, pred_dir: str, targ_dir: str):
        """Match paths of target cubes with predicted cubes

        Each target cube gets 1 or more predicted cubes.

        Args:
            pred_dir (str): Directory with predictions, format is one of {pred_dir/tile/cubename.npz, pred_dir/tile/experiment_cubename.npz}
            targ_dir (str): Directory with targets, format is one of {targ_dir/target/tile/target_cubename.npz, targ_dir/target/tile/cubename.npz, targ_dir/tile/target_cubename.npz, targ_dir/tile/cubename.npz}
        """        
        print("Initializing filepaths...")

        pred_dir, targ_dir = Path(pred_dir), Path(targ_dir)

        if "target" in [d.name for d in targ_dir.glob("*") if d.is_dir()]:
            targ_dir = targ_dir/"target"

        assert({d.name for d in pred_dir.glob("*") if d.is_dir()}.issubset({d.name for d in targ_dir.glob("*") if d.is_dir()}))

        targ_paths = sorted(list(targ_dir.glob("**/*.npz")))

        filepaths = []
        for targ_path in tqdm(targ_paths):

            pred_paths = sorted(list(pred_dir.glob(f"**/*{self.__name_getter(targ_path)}")))
            assert (len(pred_paths) <= 10),"EarthNetScore is calculated with up to 10 predictions for a target, but more than 10 predictions were found."
            for pred_path in pred_paths:
                filepaths.append({"pred_filepath": pred_path, "targ_filepath": targ_path})
        
        self.filepaths = filepaths

        print("Filepaths initialized.")
        
    def __name_getter(self, path: Path) -> str:
        """Helper function gets Cubename from a Path

        Args:
            path (Path): One of Path/to/cubename.npz and Path/to/experiment_cubename.npz

        Returns:
            [str]: cubename (has format tile_startyear_startmonth_startday_endyear_endmonth_endday_hrxmin_hrxmax_hrymin_hrymax_mesoxmin_mesoxmax_mesoymin_mesoymax.npz)
        """        
        components = path.name.split("_")
        regex = re.compile('\d{2}[A-Z]{3}')
        if bool(regex.match(components[0])):
            return path.name
        else:
            assert(bool(regex.match(components[1])))
            return "_".join(components[1:]) 

    def compute_scores(self, n_workers: Optional[int] = -1) -> dict:
        """Compute subscores for all cubepaths

        Args:
            n_workers (Optional[int], optional): Number of workers, if -1 uses all CPUs, if 0 uses no multiprocessing. Defaults to -1.

        Returns:
            dict: data of format {cubename: score_dict}
        """        
        if n_workers == 0:
            all_scores = []
            print("Iteratively computing components for EarthNetScore")
            for filepaths in tqdm(self.filepaths):
                all_scores.append(CubeCalculator.get_scores(filepaths))
        else:
            if n_workers == -1:
                n_workers = multiprocessing.cpu_count()
            print(f"Computing components for EarthNetScore using {n_workers} processes")
            with multiprocessing.Pool(n_workers) as p:
                all_scores = list(tqdm(p.imap(CubeCalculator.get_scores, self.filepaths), total = len(self.filepaths)))

        data = {}
        for scores in all_scores:
            name = self.__name_getter(Path(scores["targ_filepath"])) 
            if name not in data:
                data[name] = [scores]
            else:
                data[name].append(scores)

        self.data = data

        print("Done computing scores.")

        return data

    def save_scores(self, output_file: str):
        """Save all subscores and debugging info as JSON

        Args:
            output_file (str): Output filepath, recommended to end with .json
        """        
        print("Saving data...")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as fp:
            json.dump(self.data, fp)   
        print(f"Saved data to {output_file}.")
    
    def summarize(self, output_file: Optional[str] = None) -> Tuple[float, float, float, float, float]:
        """Calculate EarthNetScore from subscores and optionally save to file as JSON

        Args:
            output_file (Optional[str], optional): If not None, saves EarthNetScore to this path, recommended to end with .json. Defaults to None.

        Returns:
            Tuple[float, float, float, float, float]: ens, mad, ols, emd, ssim
        """        
        print("Calculating Earth Net Score...")
        scores = []
        for cube in tqdm(self.data):
            best_sample = self.__get_best_sample(self.data[cube])
            scores.append([best_sample["MAD"],best_sample["OLS"],best_sample["EMD"],best_sample["SSIM"]])
        scores = np.array(scores, dtype = np.float64)
        mean_scores = np.nanmean(scores, axis = 0).tolist()
        ens = self.__harmonic_mean(mean_scores)

        if output_file is not None:
            output_dict = {
                "EarthNetScore": ens,
                "Value (MAD)": mean_scores[0],
                "Trend (OLS)": mean_scores[1],
                "Distribution (EMD)": mean_scores[2],
                "Perceptual (SSIM)": mean_scores[3]
            }
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as fp:
                json.dump(output_dict, fp)

        print(f"EarthNetScore: {ens}\t Value (MAD): {mean_scores[0]}\t Trend (OLS): {mean_scores[1]}\t Distribution (EMD): {mean_scores[2]}\t Perceptual (SSIM): {mean_scores[3]}")
        return [ens]+mean_scores

    def __harmonic_mean(self, vals: Sequence[float]) -> Union[float, None]:
        """

        Calculates the harmonic mean of a list of values, safe for NaNs

        Args:
            vals (list): List of Floats
        Returns:
            float: harmonic mean
        """        
        vals = list(filter(None, vals))
        if len(vals) == 0:
            return None
        else:
            return min(1,len(vals)/sum([1/(v+1e-8) for v in vals]))
                
    def __get_best_sample(self, samples: Sequence[dict]) -> dict:
        """Gets best prediction out of 1 to n predictions. Safe to NaNs

        Args:
            samples (Sequence[dict]): List of dicts with subscores per sample

        Returns:
            dict: dict with subscores of best sample
        """        
        ens = np.array([self.__harmonic_mean([sample["MAD"],sample["OLS"],sample["EMD"],sample["SSIM"]]) for sample in samples], dtype = np.float64)

        try:
            min_idx = np.nanargmax(ens)
        except ValueError:
            min_idx = 0
        
        return samples[min_idx]
    
    @classmethod
    def get_ENS(cls, pred_dir: str, targ_dir: str, n_workers: Optional[int] = -1, data_output_file: Optional[str] = None, ens_output_file: Optional[str] = None):
        """Method to directly compute EarthNetScore

        Args:
            pred_dir (str): Directory with predictions, format is one of {pred_dir/tile/cubename.npz, pred_dir/tile/experiment_cubename.npz}
            targ_dir (str): Directory with targets, format is one of {targ_dir/target/tile/target_cubename.npz, targ_dir/target/tile/cubename.npz, targ_dir/tile/target_cubename.npz, targ_dir/tile/cubename.npz}
            n_workers (Optional[int], optional): Number of workers, if -1 uses all CPUs, if 0 uses no multiprocessing. Defaults to -1.
            data_output_file (Optional[str], optional): Output filepath for subscores and debugging information, recommended to end with .json. Defaults to None.
            ens_output_file (Optional[str], optional): Output filepath for EarthNetScore, recommended to end with .json. Defaults to None.
        """        

        self = cls(pred_dir, targ_dir)
        
        self.compute_scores(n_workers = n_workers)

        if data_output_file is not None:
            self.save_scores(output_file = data_output_file)
        
        self.summarize(output_file = ens_output_file)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Parallel evaluation using EarthNetScore")
    parser.add_argument('--pred_dir', type = str, help='Path where Predictions are saved')
    parser.add_argument('--targ_dir', type = str, help ='Path where targets are saved')
    parser.add_argument('--data_output_file', type = str, help ='Filepath where output data will be saved')
    parser.add_argument('--ens_output_file', type = str, help ='Filepath where resulting EarthNetScore will be saved')

    args = parser.parse_args()
    start = time.time()
    EarthNetScore.get_ENS(args.pred_dir, args.targ_dir, data_output_file = args.data_output_file, ens_output_file = args.ens_output_file)
    end = time.time()

    print(f"Calculation done in {end - start} seconds.")
