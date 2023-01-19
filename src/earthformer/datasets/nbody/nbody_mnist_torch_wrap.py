import os
import torch
from pytorch_lightning import LightningDataModule
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from .nbody_mnist import NBodyMovingMNISTIterator


class NBodyMovingMNISTTorchDataset(Dataset):

    def __init__(self,
                 data_path=None,
                 force_regenerate=False,
                 num_samples=10000,
                 digit_num=None,
                 img_size=64,
                 raw_img_size=128,
                 seq_len=20,
                 raw_seq_len_multiplier=10,
                 distractor_num=None,
                 distractor_size=5,
                 max_velocity_scale=3.6,
                 initial_velocity_range=(0.0, 3.6),
                 random_acceleration_range=(0.0, 0.0),
                 scale_variation_range=(1.0, 1.0),
                 rotation_angle_range=(-0, 0),
                 illumination_factor_range=(1.0, 1.0),
                 period=5,
                 global_rotation_prob=0.5,
                 index_range=(0, 40000),
                 mnist_data_path=None,
                 rescale_01=True,
                 # N-Body params
                 nbody_acc_mode=None,
                 nbody_G=1.0,
                 nbody_softening_distance=1.0,
                 nbody_mass=None,
                 # for debugging
                 return_raw_seq=False,
                 ):
        r"""
        Parameters
        ----------
        data_path: str
            Path of stored data. If not exists, generated to this path.
        force_regenerate: bool
            If True, force to regenerate data even if data stored in data_path already exists.
        num_samples: int
            Number of sequences
        digit_num: int
            Number of digits
        img_size: int
            Sampled sequences have shape (batch_size, seq_len, img_size, img_size, 1)
        raw_img_size: int
            The original image size for generating sequences
            before downsampling (or upsampling) to `img_size`,
            since the raw digits are rather hard to scale.
        seq_len: int
            Sampled sequences have shape (batch_size, seq_len, img_size, img_size, 1)
        raw_seq_len_multiplier: int
            The original generated sequences have higher temporal resolution for
            more precise calculation of acceleration effect. `raw_seq_len == seq_len * raw_seq_len_multiplier'.
        distractor_num: int
            Number of distractors
        distractor_size: int
            Size of the distractors
        max_velocity_scale: float
            Maximum scale of the velocity
        initial_velocity_range: tuple
        random_acceleration_range: tuple
        scale_variation_range: tuple
        rotation_angle_range: tuple
        period: int
            period of the
        index_range: Tuple[int]
            The range of indices of used mnist images
        mnist_data_path: str
            The path storing raw mnist digits images
        rescale_01:
            pixel value range from [0, 255] or [0, 1]
        nbody_acc_mode
            Should be one of [None, "r0", "r1", "r2"]
        nbody_G
            Newton's Gravitational constant
        nbody_softening_distance
            softening length
        nbody_mass
        """
        super(NBodyMovingMNISTTorchDataset, self).__init__()
        # for debugging
        self.return_raw_seq = return_raw_seq
        self.rescale_01 = rescale_01

        self.img_size = img_size
        self.raw_img_size = raw_img_size
        self.seq_len = seq_len
        raw_seq_len = seq_len * raw_seq_len_multiplier
        self.raw_seq_len_multiplier = raw_seq_len_multiplier
        self.raw_seq_len = raw_seq_len
        self.raw_period = period * raw_seq_len_multiplier

        self.num_samples = num_samples

        self.nbody_iterator = NBodyMovingMNISTIterator(
            digit_num=digit_num,
            distractor_num=distractor_num,
            img_size=raw_img_size,
            distractor_size=distractor_size,
            max_velocity_scale=max_velocity_scale,
            initial_velocity_range=initial_velocity_range,
            random_acceleration_range=random_acceleration_range,
            scale_variation_range=scale_variation_range,
            rotation_angle_range=rotation_angle_range,
            illumination_factor_range=illumination_factor_range,
            period=self.raw_period,
            global_rotation_prob=global_rotation_prob,
            index_range=index_range,
            mnist_data_path=mnist_data_path,
            # N-Body params
            nbody_acc_mode=nbody_acc_mode,
            nbody_G=nbody_G,
            nbody_softening_distance=nbody_softening_distance,
            nbody_mass=nbody_mass,)

        if data_path is None:
            data_path = f"generated_nbody.npz"
        self.data_path = data_path
        if force_regenerate or not os.path.exists(data_path):
            print(f"Generating new dataset to {data_path}")
            self.save(file=data_path)
        else:
            print(f"Loading existing dataset from {data_path}")
        num_samples, raw_seq_len = self.load(file=data_path)
        assert num_samples == self.num_samples, f"`num_samples` mismatch! " \
                                                f"loaded num_samples = {num_samples}, " \
                                                f"self.num_samples = {self.num_samples}"
        assert raw_seq_len == self.raw_seq_len, f"`raw_seq_len` mismatch! " \
                                                f"loaded raw_seq_len = {raw_seq_len}, " \
                                                f"self.raw_seq_len = {self.raw_seq_len}"

    def save(self, file=None):
        self.nbody_iterator.save(seqlen=self.raw_seq_len,
                                 num_samples=self.num_samples,
                                 file=file)

    def load(self, file=None):
        num_samples, raw_seq_len = self.nbody_iterator.load(file=file)
        return num_samples, raw_seq_len

    def __getitem__(self, index):
        r"""
        The layout of self.nbody_iterator.sample() is "NTHW"
        The layout of returned seq is "NTHWC" with C = 1
        """
        seq = self.nbody_iterator.sample(batch_size=1,
                                         seqlen=self.raw_seq_len,
                                         replay_index=index)
        ret = torch.from_numpy(seq[0, ...])
        if self.return_raw_seq:
            ret = ret.unsqueeze(-1)
        else:
            ret = ret[::self.raw_seq_len_multiplier, ...]
            ret = F.interpolate(input=ret.unsqueeze(1),
                                size=(self.img_size, self.img_size),
                                mode="bicubic",
                                align_corners=False)
            ret = ret.squeeze(1).unsqueeze(-1)
        ret.clamp_(0.0, 255.0)
        if self.rescale_01:
            ret /= 255.0
        return ret

    def __len__(self):
        return self.nbody_iterator.replay_numsamples


class NBodyMovingMNISTLightningDataModule(LightningDataModule):

    def __init__(self,
                 data_dir=None,
                 force_regenerate=False,
                 num_train_samples=8100,
                 num_val_samples=900,
                 num_test_samples=1000,
                 digit_num=None,
                 img_size=64,
                 raw_img_size=128,
                 seq_len=20,
                 raw_seq_len_multiplier=5,
                 distractor_num=None,
                 distractor_size=5,
                 max_velocity_scale=2.0,
                 initial_velocity_range=(0.0, 2.0),
                 random_acceleration_range=(0.0, 0.0),
                 scale_variation_range=(1.0, 1.0),
                 rotation_angle_range=(-0, 0),
                 illumination_factor_range=(1.0, 1.0),
                 period=5,
                 global_rotation_prob=0.5,
                 index_range=(0, 40000),
                 mnist_data_path=None,
                 rescale_01=True,
                 # N-Body params
                 nbody_acc_mode="r0",
                 nbody_G=0.05,
                 nbody_softening_distance=10.0,
                 nbody_mass=None,
                 # datamodule_only
                 batch_size=1,
                 num_workers=8,
                 ):
        super(NBodyMovingMNISTLightningDataModule, self).__init__()
        if data_dir is None:
            data_dir = "nbody_datamodule"
        self.data_dir = data_dir
        self.force_regenerate = force_regenerate
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.digit_num = digit_num
        self.seq_len = seq_len
        self.raw_seq_len_multiplier = raw_seq_len_multiplier
        self.img_size = img_size
        self.raw_img_size = raw_img_size
        self.max_velocity_scale = max_velocity_scale
        self.initial_velocity_range = initial_velocity_range
        self.random_acceleration_range = random_acceleration_range
        self.distractor_num = distractor_num
        self.distractor_size = distractor_size
        self.rotation_angle_range = rotation_angle_range
        self.scale_variation_range = scale_variation_range
        self.illumination_factor_range = illumination_factor_range
        self.period = period
        self.global_rotation_prob = global_rotation_prob
        self.index_range = index_range
        self.mnist_data_path = mnist_data_path
        self.rescale_01 = rescale_01
        # N-Body
        self.nbody_acc_mode = nbody_acc_mode
        self.nbody_G = nbody_G
        self.nbody_softening_distance = nbody_softening_distance
        self.nbody_mass = nbody_mass
        # datamodule_only
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_data_path = os.path.join(self.data_dir, "nbody_train.npz")
        self.val_data_path = os.path.join(self.data_dir, "nbody_val.npz")
        self.test_data_path = os.path.join(self.data_dir, "nbody_test.npz")

    def prepare_data(self):
        NBodyMovingMNISTTorchDataset(
            data_path=self.train_data_path,
            force_regenerate=self.force_regenerate,
            num_samples=self.num_train_samples,
            digit_num=self.digit_num,
            seq_len=self.seq_len,
            raw_seq_len_multiplier=self.raw_seq_len_multiplier,
            img_size=self.img_size,
            raw_img_size=self.raw_img_size,
            max_velocity_scale=self.max_velocity_scale,
            initial_velocity_range=self.initial_velocity_range,
            random_acceleration_range=self.random_acceleration_range,
            distractor_num=self.distractor_num,
            distractor_size=self.distractor_size,
            rotation_angle_range=self.rotation_angle_range,
            scale_variation_range=self.scale_variation_range,
            illumination_factor_range=self.illumination_factor_range,
            period=self.period,
            global_rotation_prob=self.global_rotation_prob,
            index_range=self.index_range,
            mnist_data_path=self.mnist_data_path,
            rescale_01=self.rescale_01,
            # N-Body
            nbody_acc_mode=self.nbody_acc_mode,
            nbody_G=self.nbody_G,
            nbody_softening_distance=self.nbody_softening_distance,
            nbody_mass=self.nbody_mass,)
        NBodyMovingMNISTTorchDataset(
            data_path=self.val_data_path,
            force_regenerate=self.force_regenerate,
            num_samples=self.num_val_samples,
            digit_num=self.digit_num,
            seq_len=self.seq_len,
            raw_seq_len_multiplier=self.raw_seq_len_multiplier,
            img_size=self.img_size,
            raw_img_size=self.raw_img_size,
            max_velocity_scale=self.max_velocity_scale,
            initial_velocity_range=self.initial_velocity_range,
            random_acceleration_range=self.random_acceleration_range,
            distractor_num=self.distractor_num,
            distractor_size=self.distractor_size,
            rotation_angle_range=self.rotation_angle_range,
            scale_variation_range=self.scale_variation_range,
            illumination_factor_range=self.illumination_factor_range,
            period=self.period,
            global_rotation_prob=self.global_rotation_prob,
            index_range=self.index_range,
            mnist_data_path=self.mnist_data_path,
            rescale_01=self.rescale_01,
            # N-Body
            nbody_acc_mode=self.nbody_acc_mode,
            nbody_G=self.nbody_G,
            nbody_softening_distance=self.nbody_softening_distance,
            nbody_mass=self.nbody_mass,)
        NBodyMovingMNISTTorchDataset(
            data_path=self.test_data_path,
            force_regenerate=self.force_regenerate,
            num_samples=self.num_test_samples,
            digit_num=self.digit_num,
            seq_len=self.seq_len,
            raw_seq_len_multiplier=self.raw_seq_len_multiplier,
            img_size=self.img_size,
            raw_img_size=self.raw_img_size,
            max_velocity_scale=self.max_velocity_scale,
            initial_velocity_range=self.initial_velocity_range,
            random_acceleration_range=self.random_acceleration_range,
            distractor_num=self.distractor_num,
            distractor_size=self.distractor_size,
            rotation_angle_range=self.rotation_angle_range,
            scale_variation_range=self.scale_variation_range,
            illumination_factor_range=self.illumination_factor_range,
            period=self.period,
            global_rotation_prob=self.global_rotation_prob,
            index_range=self.index_range,
            mnist_data_path=self.mnist_data_path,
            rescale_01=self.rescale_01,
            # N-Body
            nbody_acc_mode=self.nbody_acc_mode,
            nbody_G=self.nbody_G,
            nbody_softening_distance=self.nbody_softening_distance,
            nbody_mass=self.nbody_mass,)

    def setup(self, stage = None):
        if stage in (None, "fit"):
            self.nbody_train = NBodyMovingMNISTTorchDataset(
                data_path=self.train_data_path,
                force_regenerate=False,
                num_samples=self.num_train_samples,
                digit_num=self.digit_num,
                seq_len=self.seq_len,
                raw_seq_len_multiplier=self.raw_seq_len_multiplier,
                img_size=self.img_size,
                raw_img_size=self.raw_img_size,
                max_velocity_scale=self.max_velocity_scale,
                initial_velocity_range=self.initial_velocity_range,
                random_acceleration_range=self.random_acceleration_range,
                distractor_num=self.distractor_num,
                distractor_size=self.distractor_size,
                rotation_angle_range=self.rotation_angle_range,
                scale_variation_range=self.scale_variation_range,
                illumination_factor_range=self.illumination_factor_range,
                period=self.period,
                global_rotation_prob=self.global_rotation_prob,
                index_range=self.index_range,
                mnist_data_path=self.mnist_data_path,
                rescale_01=self.rescale_01,
                # N-Body
                nbody_acc_mode=self.nbody_acc_mode,
                nbody_G=self.nbody_G,
                nbody_softening_distance=self.nbody_softening_distance,
                nbody_mass=self.nbody_mass,)
            self.nbody_val = NBodyMovingMNISTTorchDataset(
                data_path=self.val_data_path,
                force_regenerate=False,
                num_samples=self.num_val_samples,
                digit_num=self.digit_num,
                seq_len=self.seq_len,
                raw_seq_len_multiplier=self.raw_seq_len_multiplier,
                img_size=self.img_size,
                raw_img_size=self.raw_img_size,
                max_velocity_scale=self.max_velocity_scale,
                initial_velocity_range=self.initial_velocity_range,
                random_acceleration_range=self.random_acceleration_range,
                distractor_num=self.distractor_num,
                distractor_size=self.distractor_size,
                rotation_angle_range=self.rotation_angle_range,
                scale_variation_range=self.scale_variation_range,
                illumination_factor_range=self.illumination_factor_range,
                period=self.period,
                global_rotation_prob=self.global_rotation_prob,
                index_range=self.index_range,
                mnist_data_path=self.mnist_data_path,
                rescale_01=self.rescale_01,
                # N-Body
                nbody_acc_mode=self.nbody_acc_mode,
                nbody_G=self.nbody_G,
                nbody_softening_distance=self.nbody_softening_distance,
                nbody_mass=self.nbody_mass,)
        if stage in (None, "test"):
            self.nbody_test = NBodyMovingMNISTTorchDataset(
                data_path=self.test_data_path,
                force_regenerate=False,
                num_samples=self.num_test_samples,
                digit_num=self.digit_num,
                seq_len=self.seq_len,
                raw_seq_len_multiplier=self.raw_seq_len_multiplier,
                img_size=self.img_size,
                raw_img_size=self.raw_img_size,
                max_velocity_scale=self.max_velocity_scale,
                initial_velocity_range=self.initial_velocity_range,
                random_acceleration_range=self.random_acceleration_range,
                distractor_num=self.distractor_num,
                distractor_size=self.distractor_size,
                rotation_angle_range=self.rotation_angle_range,
                scale_variation_range=self.scale_variation_range,
                illumination_factor_range=self.illumination_factor_range,
                period=self.period,
                global_rotation_prob=self.global_rotation_prob,
                index_range=self.index_range,
                mnist_data_path=self.mnist_data_path,
                rescale_01=self.rescale_01,
                # N-Body
                nbody_acc_mode=self.nbody_acc_mode,
                nbody_G=self.nbody_G,
                nbody_softening_distance=self.nbody_softening_distance,
                nbody_mass=self.nbody_mass,)
        if stage in (None, "predict"):
            self.nbody_predict = NBodyMovingMNISTTorchDataset(
                data_path=self.test_data_path,
                force_regenerate=False,
                num_samples=self.num_test_samples,
                digit_num=self.digit_num,
                seq_len=self.seq_len,
                raw_seq_len_multiplier=self.raw_seq_len_multiplier,
                img_size=self.img_size,
                raw_img_size=self.raw_img_size,
                max_velocity_scale=self.max_velocity_scale,
                initial_velocity_range=self.initial_velocity_range,
                random_acceleration_range=self.random_acceleration_range,
                distractor_num=self.distractor_num,
                distractor_size=self.distractor_size,
                rotation_angle_range=self.rotation_angle_range,
                scale_variation_range=self.scale_variation_range,
                illumination_factor_range=self.illumination_factor_range,
                period=self.period,
                global_rotation_prob=self.global_rotation_prob,
                index_range=self.index_range,
                mnist_data_path=self.mnist_data_path,
                rescale_01=self.rescale_01,
                # N-Body
                nbody_acc_mode=self.nbody_acc_mode,
                nbody_G=self.nbody_G,
                nbody_softening_distance=self.nbody_softening_distance,
                nbody_mass=self.nbody_mass,)

    def train_dataloader(self):
        return DataLoader(self.nbody_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.nbody_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.nbody_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.nbody_predict,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
