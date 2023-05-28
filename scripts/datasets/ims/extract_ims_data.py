import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import h5py, png, PIL
import numpy as np
import os, json

H5_FILE_NAME = "IMS_{img_type}_{year}_{start_day}_{end_day}.h5"
H5_FILES_DIRECTORY = "/ims_projects/Research/Oren/Lia_Ofir/ims_data/data/{img_type}/{year}"
IMG_TYPES = ["MIDDLE_EAST_VIS", "MIDDLE_EAST_DAY_CLOUDS", "MIDDLE_EAST_COLORED", "MIDDLE_EAST_IR"]
IMG_FORMATS = ["png", "jpeg"]
IMG_SHAPES = {'png': (600, 600)}
CATALOG_HEADERS = ['id', 'file_name', 'file_index', 'img_type', 'time_utc', 'min_delta']

CATALOG_PATH = "/ims_projects/Research/Oren/Lia_Ofir/ims_data/CATALOG.csv"
CFG_FILE_PATH = "/ims_projects/Research/Oren/Lia_Ofir/ims_data/extract_cfg.json"
EUMETSAT_FRAME_PATH = "/ims_archive/Operational/MSG/images/HRIT_RSS/{img_type}/{year}/{month}/{day}/{year}{month}{day}{hour}{minute}.{img_format}"

class IMSH5():
    """
    represents an h5 file containing IMS data.
    """

    def __init__(self, img_type,
                 time_delta,
                 start_date, end_date,
                 sunrise=timedelta(hours=2, minutes=40),
                 sunset=timedelta(hours=16, minutes=40),
                 sample_mode='sequent',
                 img_format='png',
                 shape=IMG_SHAPES['png'],
                 slice_x=None,
                 slice_y=None,
                 channels='grayscale',
                 file_directory=None, file_name=None,
                 verbose=True):

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

        self.samples_per_event = int((self.sunset.seconds - self.sunrise.seconds) / self.time_delta.seconds + 1)

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
        if not file_directory:
            self.file_directory = H5_FILES_DIRECTORY.format(img_type=self.img_type, year=self.year)
        if not file_name:
            self._h5_file_name()

        # setup
        self.verbose = verbose
        self.catalog = pd.DataFrame(columns=CATALOG_HEADERS)
        self._open_file()

        self._index = 0
        self._events = []
        self._ids = []

    def _h5_file_name(self):
        h5_start_day = self.start_date.strftime("%m%d")
        h5_end_day = self.end_date.strftime("%m%d")
        self.file_name = H5_FILE_NAME.format(img_type=self.img_type, year=self.year,
                                             start_day=h5_start_day, end_day=h5_end_day)

    def _open_file(self):
        # create directory for h5 file
        out_dir = Path(self.file_directory)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)
        # open h5 file for writing
        self.file = h5py.File(os.path.join(self.file_directory, self.file_name), "w")

    def extract_all(self):
        self._load_data()
        self._save_data()
        if self.verbose:
            print(f"saved {self.file_name}")

    def _save_data(self):
        self.file.close()

    def _load_data(self):
        if self.sample_mode == 'sequent':
            # go over all days on between start_date and end_date, insert them to catalog and to the h5 file
            event_date = self.start_date
            day_time_delta = timedelta(days=1)
            while event_date <= self.end_date:
                self._load_event(event_date)
                event_date += day_time_delta

        if self.sample_mode == 'random':
            # TODO: write this
            pass

        # insert data to h5 file
        self.file.create_dataset('id', data=self._ids)
        self.file.create_dataset(self.img_type, data=self._events)

    def _load_event(self, event_date):
        event_frames = []

        start_event_time = event_date + self.sunrise
        end_event_time = event_date + self.sunset
        event_id = event_date.strftime("%Y%m%d")
        full_event = True

        frame_time = start_event_time
        while frame_time <= end_event_time and full_event:
            frame = self._load_frame(frame_time)  # TODO: editing the frame_path
            if frame is None:
                full_event = False
            event_frames.append(frame)
            frame_time += self.time_delta

        if full_event:
            self._index += 1
            self._events.append(event_frames)
            self._ids.append(event_id)

            # append event to CATALOG
            self.catalog = self.catalog._append(
                {'id': event_id, 'file_name': self.file_name, 'file_index': self._index,
                 'time_utc': event_date + self.sunrise,
                 'img_type': self.img_type, 'min_delta': self.time_delta.seconds / 60}, ignore_index=True)

            if self.verbose:
                print(f"V appended event {event_id} to catalog of {self.file_name}")
        else:
            if self.verbose:
                print(f"X event {event_id} was not appended to catalog of {self.file_name}")

    def _load_frame(self, frame_time, frame_path=None):
        if not frame_path:
            img_year = frame_time.strftime("%Y")
            img_month = frame_time.strftime("%m")
            img_day = frame_time.strftime("%d")
            img_hour = frame_time.strftime("%H")
            img_minute = frame_time.strftime("%M")
            frame_path = EUMETSAT_FRAME_PATH.format(year=img_year, month=img_month, day=img_day, hour=img_hour,
                                                    minute=img_minute, img_format=self.img_format,
                                                    img_type=self.img_type)

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
                pixels = 0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2] # https://spec.oneapi.io/oneipl/0.6/convert/rgb-gray-conversion.html, https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py
                pixels = pixels.reshape(*pixels.shape, 1) # HWC

        return pixels


def main():
    # extract config
    catalog = pd.DataFrame(columns=CATALOG_HEADERS)
    cfg = json.load(open(CFG_FILE_PATH, "r"))
    h5_files = cfg["h5_files"]
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

        h5_file = IMSH5(img_type=img_type, time_delta=time_delta, start_date=start_date, end_date=end_date, sunrise=sunrise, sunset=sunset, \
              sample_mode=sample_mode, img_format=img_format, shape=shape, slice_x=slice_x, slice_y=slice_y, \
              channels=channels)

        h5_file.extract_all()
        catalog = pd.concat([catalog, h5_file.catalog], axis=0)
        catalog.to_csv(CATALOG_PATH, index=False)


if __name__ == '__main__':
    main()
