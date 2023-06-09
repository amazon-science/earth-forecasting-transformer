import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import h5py, png, PIL
import numpy as np
import os, json

H5_FILE_NAME_FORMAT = "IMS_{img_type}_{year}_{start_day}_{end_day}.h5"
H5_FILES_DIRECTORY = "/ims_projects/Research/Oren/Lia_Ofir/ims_data/data/{img_type}/{year}"
IMG_TYPES = ["MIDDLE_EAST_VIS", "MIDDLE_EAST_DAY_CLOUDS", "MIDDLE_EAST_COLORED", "MIDDLE_EAST_IR"]
IMG_FORMATS = ["png", "jpeg"]
IMG_SHAPES = {'png': (600, 600)}
CATALOG_HEADERS = ['id', 'file_name', 'file_index', 'img_type', 'time_utc', 'min_delta']
EUMETSAT_DATE_PATH = "/ims_archive/Operational/MSG/images/HRIT_RSS/{img_type}/{year}/{month}/{day}/{year}{month}{day}{hour}{minute}.{img_format}"
EUMETSAT_FRAME_NAME = "{year}{month}{day}{hour}{minute}.{img_format}"
CFG_FILE_PATH = "/ims_projects/Research/Oren/Lia_Ofir/ims_data/extract_cfg.json"


class IMSH5():
    """
    represents an h5 file containing IMS data.
    """

    def __init__(self, img_type,
                 time_delta,
                 start_date, end_date,
                 sunrise=timedelta(hours=2, minutes=40),
                 sunset=timedelta(hours=16, minutes=40),
                 event_length=None,
                 sample_mode='sequent',
                 img_format='png',
                 shape=IMG_SHAPES['png'],
                 slice_x=None,
                 slice_y=None,
                 channels='grayscale',
                 verbose=True,
                 catalog_headers=None,
                 h5_files_directory=None,
                 h5_file_name_format=None,
                 eumetsat_date_path=None,
                 eumetsat_frame_name=None):

        # image type
        assert img_type in IMG_TYPES
        self.img_type = img_type

        # events sampling
        assert start_date.year == end_date.year
        assert start_date <= end_date
        self.start_date = start_date
        self.end_date = end_date
        self.year = start_date.year
        assert sample_mode in ['sequent', 'random']
        self.sample_mode = sample_mode

        # frames sampling
        assert (time_delta.seconds / 60) % 5 == 0
        self.time_delta = time_delta
        if sunrise and sunset:
            assert sunrise <= sunset
            assert sunset < timedelta(hours=24)
            self.sunrise = sunrise
            self.sunset = sunset
        else:
            self.sunrise = timedelta(hours=2, minutes=40)
            self.sunset = timedelta(hours=16, minutes=40)

        if event_length is None:
            self.event_length = int((self.sunset.seconds - self.sunrise.seconds) / self.time_delta.seconds + 1)
        else:
            assert event_length > 0
            self.event_length = event_length

        # frame parameters
        assert img_format in IMG_FORMATS
        self.img_format = img_format

        if img_format == 'png':
            assert min(np.array(shape) <= np.array(IMG_SHAPES['png']))
        self.shape = shape

        if not slice_x:
            self.slice_x = slice(0, self.shape[0])
        else:
            assert not slice_x.step
            assert (slice_x.stop - slice_x.start) + 1 <= shape[0]
            self.slice_x = slice_x

        if not slice_y:
            self.slice_y = slice(0, self.shape[1])
        else:
            assert not slice_y.step
            assert (slice_y.stop - slice_y.start) + 1 <= shape[1]
            self.slice_y = slice_y

        assert channels in ['grayscale', 'rgba']
        self.channels = channels

        # path parameters
        self.h5_files_directory = (H5_FILES_DIRECTORY if h5_files_directory is None else h5_files_directory).format(
            img_type=self.img_type, year=self.year)
        self._h5_file_name(h5_file_name_format)
        if not catalog_headers:
            self.catalog_headers = CATALOG_HEADERS
        else:
            self.catalog_headers = catalog_headers
        if not eumetsat_date_path:
            self.eumetsat_date_path = EUMETSAT_DATE_PATH
        else:
            self.eumetsat_date_path = eumetsat_date_path
        if not eumetsat_frame_name:
            self.eumetsat_frame_name = EUMETSAT_FRAME_NAME
        else:
            self.eumetsat_frame_name = eumetsat_frame_name

        # setup
        self.verbose = verbose
        self.catalog = pd.DataFrame(columns=self.catalog_headers)
        self._open_file()

        self._index = 0
        self._events = []
        self._ids = []

    def _h5_file_name(self, h5_file_name_format):
        h5_start_day = self.start_date.strftime("%m%d")
        h5_end_day = self.end_date.strftime("%m%d")
        self.h5_file_name = (h5_file_name_format if h5_file_name_format is not None else H5_FILE_NAME_FORMAT).format(
            img_type=self.img_type, year=self.year,
            start_day=h5_start_day, end_day=h5_end_day)

    def _open_file(self):
        # create directory for h5 file
        out_dir = Path(self.h5_files_directory)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)
        # open h5 file for writing
        self.file = h5py.File(os.path.join(self.h5_files_directory, self.h5_file_name), "w")

    def extract_all(self):
        self._load_data()
        self._save_data()
        if self.verbose:
            print(f"saved {self.h5_file_name}")

    def _save_data(self):
        self.file.close()

    def _load_data(self):
        if self.sample_mode == 'sequent':
            # go over all days on between start_date and end_date, insert them to catalog and to the h5 file
            curr_date = self.start_date
            day_time_delta = timedelta(days=1)
            while curr_date <= self.end_date:
                self._discover_events(curr_date)
                curr_date += day_time_delta

        if self.sample_mode == 'random':
            # TODO: write this
            pass

        # insert data to h5 file
        self.file.create_dataset('id', data=self._ids)
        self.file.create_dataset(self.img_type, data=self._events)

    def _discover_events(self, date):
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        frames_path = self.eumetsat_date_path.format(img_type=self.img_type, year=year, month=month, day=day)

        if not os.path.isdir(frames_path):
            if self.verbose:
                print(f"Did not find directory {frames_path}")
                return

        frame_names = os.listdir(frames_path)
        frame_df = pd.DataFrame(frame_names, columns=["frame_name"])
        frame_df["frame_time"] = pd.to_datetime(frame_df["frame_name"], format=f'%Y%m%d%H%M.{self.img_format}')
        frame_df["frame_path"] = frames_path + "/" + frame_df["frame_name"]
        # filter
        start_time = date + self.sunrise
        end_time = date + self.sunset
        frame_df = frame_df[(start_time <= frame_df["frame_time"]) & (frame_df["frame_time"] <= end_time)]
        # sort
        frame_df.sort_values(by='frame_time', inplace=True, ascending=True)
        # dummy frame for the last sequence
        dummy = {'frame_time': None, 'frame_name': None, 'frame_path': None}
        frame_df = frame_df._append(dummy, ignore_index=True)

        seq_length = 0
        for i, frame_time in enumerate(frame_df['frame_time']):
            # the sequence will be broken if the previous frame is not timedelta before the current
            # or when we are at the last frame
            if i > 0 and frame_df.iloc[i - 1]["frame_time"] != (frame_time - self.time_delta):
                j = i - seq_length

                while j <= (i - self.event_length):
                    event_frames = frame_df.iloc[j: j + self.event_length]
                    self._load_event(event_frames)
                    j += self.event_length

                if (i - self.event_length - j) < 0 and self.verbose:
                    print(
                        f"frames from {frame_df.iloc[j]['frame_time']} to {frame_df.iloc[i - 1]['frame_time']} was not appended.")

                seq_length = 1 # because the frame that broke the sequence will be getting into the next sequence

            else:
                seq_length += 1

        return

    def _load_event(self, frames: pd.DataFrame):
        event_frames = []
        start_event_time = frames.iloc[0]["frame_time"]
        event_id = start_event_time.strftime("%Y%m%d%H%M")

        for frame in frames.iterrows():
            frame_data = self._load_frame(frame[1].loc["frame_path"])
            event_frames.append(frame_data)

        # add event
        self._events.append(event_frames)
        self._ids.append(event_id)

        # append event to CATALOG
        self.catalog = self.catalog._append(
            {'id': event_id, 'file_name': self.h5_file_name, 'file_index': self._index,
             'time_utc': start_event_time,
             'img_type': self.img_type, 'min_delta': self.time_delta.seconds / 60}, ignore_index=True)

        self._index += 1

        if self.verbose:
            print(f"V appended event {event_id} to catalog of {self.h5_file_name}")

    def _load_frame(self, frame_path):
        pixels = None
        if Path(frame_path).exists():
            if self.img_format == 'png':
                raw_img = png.Reader(file=open(frame_path, "rb")).asRGBA8()
                raw_pixels = raw_img[2]
                pixels = np.array([list(row) for row in raw_pixels], dtype="uint8").reshape((*self.shape, 4))

            elif self.img_format == 'jpeg':
                raw_img = PIL.Image.open(frame_path)
                pixels = np.array(raw_img)

            pixels = pixels[self.slice_x, self.slice_y, :]
            if self.channels == 'grayscale':
                pixels = 0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :,
                                                                                     2]  # https://spec.oneapi.io/oneipl/0.6/convert/rgb-gray-conversion.html, https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py
                pixels = pixels.reshape(*pixels.shape, 1)  # HWC

        return pixels


def main():
    # extract config
    cfg = json.load(open(CFG_FILE_PATH, "r"))

    catalog_file_path = Path(cfg["catalog_file_path"])
    catalog_headers = cfg["catalog_headers"]
    eumetsat_date_path = cfg["eumetsat_date_path"]
    eumetsat_frame_name = cfg["eumetsat_frame_name"]
    h5_file_name_format = cfg["h5_file_name_format"]
    h5_files_directory = cfg["h5_files_directory"]
    h5_files = cfg["h5_files"]
    event_length = int(cfg["event_length"])

    catalog = pd.DataFrame(columns=catalog_headers)

    # check if there is existing CATALOG file in path
    if catalog_file_path.is_file():
        print(f"WARNING: CATALOG file already exists at {catalog_file_path}."
              f" The script is going to add to the existing file.")
    else:
        # create an empty CATALOG file
        catalog.to_csv(str(catalog_file_path), index=False)

    for f in h5_files:
        img_type = f["img_type"]
        time_delta = timedelta(minutes=f["time_delta"])
        start_date = datetime(*tuple(f["start_date"]))
        end_date = datetime(*tuple(f["end_date"]))
        sunrise = None if len(f["sunrise"]) == 0 else timedelta(hours=f["sunrise"][0], minutes=f["sunrise"][1])
        sunset = None if len(f["sunset"]) == 0 else timedelta(hours=f["sunset"][0], minutes=f["sunset"][1])
        sample_mode = f["sample_mode"]
        img_format = f["img_format"]
        shape = None if len(f["shape"]) == 0 else tuple(f["shape"])
        slice_x = None if len(f["slice_x"]) == 0 else slice(*tuple(f["slice_x"]))
        slice_y = None if len(f["slice_y"]) == 0 else slice(*tuple(f["slice_y"]))
        channels = f["channels"]

        h5_file = IMSH5(img_type=img_type,
                        time_delta=time_delta,
                        start_date=start_date,
                        end_date=end_date,
                        event_length=event_length,
                        sunrise=sunrise,
                        sunset=sunset,
                        sample_mode=sample_mode,
                        img_format=img_format,
                        shape=shape,
                        slice_x=slice_x,
                        slice_y=slice_y,
                        channels=channels,
                        catalog_headers=catalog_headers,
                        h5_files_directory=h5_files_directory,
                        h5_file_name_format=h5_file_name_format,
                        eumetsat_date_path=eumetsat_date_path,
                        eumetsat_frame_name=eumetsat_frame_name)

        h5_file.extract_all()
        catalog.to_csv(str(catalog_file_path), mode='a', index=False, header=False)  # adds to existing CATALOG file


if __name__ == '__main__':
    main()
