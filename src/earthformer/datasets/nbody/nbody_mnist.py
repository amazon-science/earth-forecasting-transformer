"""Code is adapted from https://github.com/sxjscience/HKO-7. Their license is MIT License"""

import cv2
import numpy as np
import math
import os


def load_mnist(training_num=50000, mnist_data_path=None):
    r"""Load the mnist dataset
    """
    if mnist_data_path is None:
        from ...config import cfg
        mnist_data_path = os.path.join(cfg.datasets_dir, "nbody", "mnist.npz")
    dat = np.load(mnist_data_path)
    X = dat['X'][:training_num]
    Y = dat['Y'][:training_num]
    X_test = dat['X_test']
    Y_test = dat['Y_test']
    Y = Y.reshape((Y.shape[0],))
    Y_test = Y_test.reshape((Y_test.shape[0],))
    return X, Y, X_test, Y_test

def crop_mnist_digit(digit_img, tol=5):
    """Return the cropped version of the mnist digit
    Parameters
    ----------
    digit_img : np.ndarray
        Shape: ()
    Returns
    -------
    """
    tol = float(tol) / float(255)
    mask = digit_img > tol
    return digit_img[np.ix_(mask.any(1), mask.any(0))]

class NBodyMovingMNISTIterator():

    def __init__(self,
                 digit_num=None,
                 distractor_num=None,
                 img_size=None,
                 distractor_size=5,
                 max_velocity_scale=3.6,
                 initial_velocity_range=(0.0, 3.6),
                 random_acceleration_range=(0.0, 0.0),
                 scale_variation_range=(1 / 1.1, 1.1),
                 rotation_angle_range=(-30, 30),
                 illumination_factor_range=(0.6, 1.0),
                 period=5,
                 global_rotation_prob=0.5,
                 index_range=(0, 40000),
                 mnist_data_path=None,
                 # N-Body params
                 nbody_acc_mode=None,
                 nbody_G=1.0,
                 nbody_softening_distance=1.0,
                 nbody_mass=None,
                 ):
        """
        Parameters
        ----------
        digit_num : int
            Number of digits
        distractor_num : int
            Number of distractors
        img_size : int
            Size of the image
        distractor_size : int
            Size of the distractors
        max_velocity_scale : float
            Maximum scale of the velocity
        initial_velocity_range : tuple
        random_acceleration_range
        scale_variation_range
        rotation_angle_range
        period : period of the
        index_range
        """
        self.mnist_train_img, self.mnist_train_label, \
        self.mnist_test_img, self.mnist_test_label = load_mnist(mnist_data_path=mnist_data_path)
        self._digit_num = digit_num if digit_num is not None else 3
        self._img_size = img_size if img_size is not None else 64
        self._distractor_size = distractor_size
        self._distractor_num = distractor_num if distractor_num is not None else 0
        self._max_velocity_scale = max_velocity_scale
        self._initial_velocity_range = initial_velocity_range
        self._random_acceleration_range = random_acceleration_range
        self._scale_variation_range = scale_variation_range
        self._rotation_angle_range = rotation_angle_range
        self._illumination_factor_range = illumination_factor_range
        self._period = period
        self._global_rotation_prob = global_rotation_prob
        self._index_range = index_range
        self._h5py_f = None
        self._seq = None
        self._motion_vectors = None
        self.replay = None
        self.replay_index = 0
        self.replay_numsamples = -1
        # N-Body
        self.nbody_acc_mode = nbody_acc_mode
        if nbody_acc_mode is not None:
            self.use_nbody_acc = True
            self.nbody_G = nbody_G
            self.nbody_softening_distance = nbody_softening_distance
            if nbody_mass is None:
                nbody_mass = np.ones((self._digit_num, 1))
            else:
                nbody_mass = np.array(nbody_mass).reshape(shape=(self._digit_num, 1))
            self.nbody_mass = nbody_mass
        else:
            self.use_nbody_acc = False

    def _choose_distractors(self, distractor_seeds):
        """Choose the distractors
        We use the similar approach as
         https://github.com/deepmind/mnist-cluttered/blob/master/mnist_cluttered.lua
        Returns
        -------
        ret : list
            list of distractor images
        """
        ret = []
        for i in range(self._distractor_num):
            ind = math.floor(distractor_seeds[i, 2] * self._index_range[1])
            distractor_img = self.mnist_train_img[ind].reshape((28, 28))
            distractor_h_begin = math.floor(distractor_seeds[i, 3] * (28 - self._distractor_size))
            distractor_w_begin = math.floor(distractor_seeds[i, 4] * (28 - self._distractor_size))
            distractor_img = distractor_img[
                             distractor_h_begin:distractor_h_begin + self._distractor_size,
                             distractor_w_begin:distractor_w_begin + self._distractor_size]
            ret.append(distractor_img)
        return ret

    def draw_distractors(self, canvas_img, distractor_seeds):
        """
        Parameters
        ----------
        canvas_img
        Returns
        -------
        """
        distractor_imgs = self._choose_distractors(distractor_seeds)
        for i, img in enumerate(distractor_imgs):
            r_begin = math.floor(distractor_seeds[i][0] * (self._img_size - img.shape[0]))
            c_begin = math.floor(distractor_seeds[i][1] * (self._img_size - img.shape[1]))
            canvas_img[r_begin:r_begin + img.shape[0], c_begin:c_begin +
                                                               img.shape[1]] = img
        return canvas_img

    def draw_imgs(self,
                  base_img,
                  affine_transforms,
                  prev_affine_transforms=None):
        """
        Parameters
        ----------
        base_img : list
            Inner Shape: (H, W)
        affine_transforms : np.ndarray
            Shape: (digit_num, 2, 3)
        prev_affine_transforms : np.ndarray
            Shape: (digit_num, 2, 3)
        Returns
        -------
        """
        canvas_img = np.zeros(
            (self._img_size, self._img_size), dtype=np.float32)
        for i in range(self._digit_num):
            tmp_img = cv2.warpAffine(base_img[i], affine_transforms[i],
                                     (self._img_size, self._img_size))
            canvas_img = np.maximum(canvas_img, tmp_img)
        return canvas_img

    def _find_center(self, img):
        x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
        raise NotImplementedError

    def _bounce_border(self, inner_boundary, affine_transform, digit_shift,
                       velocity, img_h, img_w):
        # top-left, top-right, down-left, down-right
        center = affine_transform.dot(
            np.array([img_w / 2.0, img_h / 2.0, 1], dtype=np.float32))
        new_velocity = velocity.copy()
        new_center = center.copy()
        if center[0] < inner_boundary[0]:
            new_velocity[0] = -new_velocity[0]
            new_center[0] = inner_boundary[0]
        if center[0] > inner_boundary[2]:
            new_velocity[0] = -new_velocity[0]
            new_center[0] = inner_boundary[2]
        if center[1] < inner_boundary[1]:
            new_velocity[1] = -new_velocity[1]
            new_center[1] = inner_boundary[1]
        if center[1] > inner_boundary[3]:
            new_velocity[1] = -new_velocity[1]
            new_center[1] = inner_boundary[3]
        affine_transform[:, 2] += new_center - center
        digit_shift += new_center - center
        return affine_transform, digit_shift, new_velocity

    def get_nbody_acceleration(self, pos):
        r"""
        Calculate the acceleration on each particle due to Newton's Law
        We only consider motion within 2D plane
        find original impl at https://github.com/pmocz/nbody-python

        pos  is an N x 2 matrix of positions
        mass is an N x 1 vector of masses
        G is Newton's Gravitational constant
        softening is the softening length
        a is N x 2 matrix of accelerations
        """

        # positions r = [x, y] for all particles
        x = pos[:, 0:1]
        y = pos[:, 1:2]

        # matrix that stores all pairwise particle separations: r_j - r_i
        dx = x.T - x
        dy = y.T - y

        # matrix that stores 1/r^3 for all particle pairwise particle separations
        r2 = (dx ** 2 + dy ** 2 + self.nbody_softening_distance ** 2)
        if self.nbody_acc_mode == "r2":
            inv_r3 = r2
            inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)
            ax = self.nbody_G * (dx * inv_r3) @ self.nbody_mass
            ay = self.nbody_G * (dy * inv_r3) @ self.nbody_mass
        elif self.nbody_acc_mode == "r1":
            inv_r2 = 1 / r2
            ax = self.nbody_G * (dx * inv_r2) @ self.nbody_mass
            ay = self.nbody_G * (dy * inv_r2) @ self.nbody_mass
        elif self.nbody_acc_mode == "r0":
            inv_r1 = r2
            inv_r1[inv_r1 > 0] = inv_r1[inv_r1 > 0] ** (-0.5)
            ax = self.nbody_G * (dx * inv_r1) @ self.nbody_mass
            ay = self.nbody_G * (dy * inv_r1) @ self.nbody_mass
        else:
            raise NotImplementedError

        # pack together the acceleration components
        a = np.hstack((ax, ay))

        return a

    def sample(self, batch_size, seqlen, replay_index=None):
        """
        Parameters
        ----------
        batch_size : int
        seqlen : int
        random: take random samples from loaded parameters. Ignored if no parameters are loaded.
        Returns
        -------
        seq : np.ndarray
            Shape = (batch_size, seqlen, img_size, img_size)
        """

        if self.replay is not None:
            sequential_sample_flag = False
            if replay_index is None:
                replay_index = self.replay_index
                sequential_sample_flag = True
            elif replay_index == "random":
                replay_index = np.random.randint(self.replay_numsamples - batch_size)
            elif isinstance(replay_index, int):
                if replay_index + batch_size > self.replay_numsamples:
                    raise IndexError("Not enough pre-generated parameters to create new sample.")
            else:
                raise ValueError(f"replay_index should be an int or one of [None, 'random']")

        seq = np.zeros(
            (batch_size, seqlen, self._img_size, self._img_size),
            dtype=np.float32)
        inner_boundary = np.array(
            [10, 10, self._img_size - 10, self._img_size - 10],
            dtype=np.float32)
        for b in range(batch_size):
            affine_transforms = np.zeros(
                (seqlen, self._digit_num, 2, 3), dtype=np.float32)
            appearance_variants = np.ones(
                (seqlen, self._digit_num), dtype=np.float32)
            scale = np.ones((seqlen, self._digit_num), dtype=np.float32)
            rotation_angle = np.zeros(
                (seqlen, self._digit_num), dtype=np.float32)
            init_velocity = np.zeros(
                shape=(self._digit_num, 2), dtype=np.float32)
            velocity = np.zeros((seqlen, self._digit_num, 2), dtype=np.float32)
            digit_shift = np.zeros(
                (seqlen, self._digit_num, 2), dtype=np.float32)

            if self.replay is not None:
                digit_indices = self.replay["digit_indices"][replay_index + b]
                appearance_mult = self.replay["appearance_mult"][replay_index + b]
                scale_variation = self.replay["scale_variation"][replay_index + b]
                base_rotation_angle = self.replay["base_rotation_angle"][replay_index + b]
                affine_transforms_multipliers = self.replay["affine_transforms_multipliers"][replay_index + b]
                init_velocity_angle = self.replay["init_velocity_angle"][replay_index + b]
                init_velocity_magnitude = self.replay["init_velocity_magnitude"][replay_index + b]
                random_acceleration_angle = self.replay["random_acceleration_angle"][replay_index + b]
                random_acceleration_magnitude = self.replay["random_acceleration_magnitude"][replay_index + b]
                distractor_seeds = self.replay["distractor_seeds"][replay_index + b]
                assert (distractor_seeds.shape[0] == seqlen)

            else:
                digit_indices = np.random.randint(
                    low=self._index_range[0],
                    high=self._index_range[1],
                    size=self._digit_num)
                appearance_mult = np.random.uniform(
                    low=self._illumination_factor_range[0],
                    high=self._illumination_factor_range[1])
                scale_variation = np.random.uniform(
                    low=self._scale_variation_range[0],
                    high=self._scale_variation_range[1],
                    size=(self._digit_num,))
                base_rotation_angle = np.random.uniform(
                    low=self._rotation_angle_range[0],
                    high=self._rotation_angle_range[1],
                    size=(self._digit_num,))
                affine_transforms_multipliers = np.random.uniform(
                    size=(self._digit_num, 2))
                init_velocity_angle = np.random.uniform(
                    size=(self._digit_num,)) * (2 * np.pi)
                init_velocity_magnitude = np.random.uniform(
                    low=self._initial_velocity_range[0],
                    high=self._initial_velocity_range[1],
                    size=self._digit_num)
                distractor_seeds = np.random.uniform(
                    size=(seqlen, self._distractor_num, 5))
                random_acceleration_angle = np.random.random() * 2 * np.pi
                random_acceleration_magnitude = np.random.uniform(
                    low=self._random_acceleration_range[0],
                    high=self._random_acceleration_range[1],
                    size=self._digit_num)

            random_acceleration = np.zeros(shape=(self._digit_num, 2), dtype=np.float32)
            random_acceleration[:, 0] = random_acceleration_magnitude * np.cos(random_acceleration_angle)
            random_acceleration[:, 1] = random_acceleration_magnitude * np.sin(random_acceleration_angle)

            base_digit_img = [
                crop_mnist_digit(self.mnist_train_img[i].reshape((28, 28)))
                for i in digit_indices
            ]

            for i in range(1, seqlen):
                appearance_variants[i, :] = appearance_variants[i - 1, :] * \
                                            (appearance_mult ** -(2 * ((i // self._period) % 2) - 1))

            for i in range(1, seqlen):
                base_factor = (2 * ((i // self._period) % 2) - 1)
                scale[i, :] = scale[i - 1, :] * (scale_variation ** base_factor)
                rotation_angle[i, :] = rotation_angle[i - 1, :] + base_rotation_angle

            affine_transforms[0, :, 0, 0] = 1.0
            affine_transforms[0, :, 1, 1] = 1.0
            for i in range(self._digit_num):
                affine_transforms[0, i, 0, 2] = affine_transforms_multipliers[i, 0] * \
                                                (self._img_size - base_digit_img[i].shape[1])
                affine_transforms[0, i, 1, 2] = affine_transforms_multipliers[i, 1] * \
                                                (self._img_size - base_digit_img[i].shape[0])

            init_velocity[:, 0] = init_velocity_magnitude * np.cos(init_velocity_angle)
            init_velocity[:, 1] = init_velocity_magnitude * np.sin(init_velocity_angle)
            curr_velocity = init_velocity

            for i in range(self._digit_num):
                digit_shift[0, i, 0] = affine_transforms[0, i, 0, 2]  # + (base_digit_img[i].shape[1] / 2.0)
                digit_shift[0, i, 1] = affine_transforms[0, i, 1, 2]  # + (base_digit_img[i].shape[0] / 2.0)

            for i in range(seqlen - 1):
                velocity[i, :, :] = curr_velocity
                curr_velocity += random_acceleration
                # curr_velocity += random_acceleration * (2 * ((i / self._period) % 2) - 1)
                if self.use_nbody_acc:
                    nbody_acceleration = self.get_nbody_acceleration(pos=digit_shift[i, ...])
                    curr_velocity += nbody_acceleration
                curr_velocity = np.clip(
                    curr_velocity,
                    a_min=-self._max_velocity_scale,
                    a_max=self._max_velocity_scale)
                for j in range(self._digit_num):
                    digit_shift[i + 1, j, :] = digit_shift[i, j, :] + curr_velocity[j]
                    rotation_mat = cv2.getRotationMatrix2D(
                        center=(base_digit_img[j].shape[1] / 2.0,
                                base_digit_img[j].shape[0] / 2.0),
                        angle=rotation_angle[i + 1, j],
                        scale=scale[i + 1, j])
                    affine_transforms[i + 1, j, :, :2] = rotation_mat[:, :2]
                    affine_transforms[i + 1, j, :, 2] = digit_shift[i + 1, j, :] + rotation_mat[:, 2]
                    affine_transforms[i + 1, j, :, :], digit_shift[i + 1, j, :], curr_velocity[j] = \
                        self._bounce_border(inner_boundary=inner_boundary,
                                            affine_transform=affine_transforms[i + 1, j, :, :],
                                            digit_shift=digit_shift[i + 1, j, :],
                                            velocity=curr_velocity[j],
                                            img_h=base_digit_img[j].shape[0],
                                            img_w=base_digit_img[j].shape[1])
            for i in range(seqlen):
                seq[b, i, :, :] = self.draw_imgs(
                    base_img=[
                        base_digit_img[j] * appearance_variants[i, j]
                        for j in range(self._digit_num)
                    ],
                    affine_transforms=affine_transforms[i])
                self.draw_distractors(seq[b, i, :, :], distractor_seeds[i])
        if self.replay is not None:
            if sequential_sample_flag:
                self.replay_index += batch_size
        return seq

    def load(self, file=None):
        """Initialize to draw samples from pre-computed parameters.
        Args:
            file: Either the file name (string) or an open file (file-like
                object) from which the data will be loaded.
        """
        self.replay_index = 0
        if file is None:
            file = f"generated_nbody.npz"
        with np.load(file) as f:
            self.replay = dict(f)

        assert (self.replay["distractor_seeds"].shape[2] == self._distractor_num)

        num_samples, seqlen = self.replay["distractor_seeds"].shape[0:2]
        self.replay_numsamples = num_samples
        return num_samples, seqlen

    def save(self, seqlen, num_samples=10000, file=None, seed=None):
        """Draw random numbers for num_samples sequences and save them.
        This initializes the state of MovingMNISTAdvancedIterator to generate
        sequences based on the hereby drawn parameters.
        Note that each call to sample(batch_size, seqlen) will use batch_size
        of the num_samples parameters.
        Args:
            num_samples: Number of unique MovingMNISTAdvanced sequences to draw
                parameters for
            file: Either the file name (string) or an open file (file-like
                object) where the data will be saved. If file is a string or a
                Path, the .npz extension will be appended to the file name if
                it is not already there.
        """
        if file is None:
            file = f"generated_nbody.npz"
        if isinstance(file, str):
            data_dir = os.path.dirname(file)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

        self.replay = dict()
        rng = np.random.RandomState(seed=seed)
        self.replay["digit_indices"] = rng.randint(
            low=self._index_range[0],
            high=self._index_range[1],
            size=(num_samples, self._digit_num))
        self.replay["appearance_mult"] = rng.uniform(
            low=self._illumination_factor_range[0],
            high=self._illumination_factor_range[1],
            size=(num_samples, ))
        self.replay["scale_variation"] = rng.uniform(
            low=self._scale_variation_range[0],
            high=self._scale_variation_range[1],
            size=(num_samples, self._digit_num))
        self.replay["base_rotation_angle"] = rng.uniform(
            low=self._rotation_angle_range[0],
            high=self._rotation_angle_range[1],
            size=(num_samples, self._digit_num))
        self.replay["affine_transforms_multipliers"] = rng.uniform(
            size=(num_samples, self._digit_num, 2))
        self.replay["init_velocity_angle"] = rng.uniform(
            size=(num_samples, self._digit_num)) * 2 * np.pi
        self.replay["init_velocity_magnitude"] = rng.uniform(
            low=self._initial_velocity_range[0],
            high=self._initial_velocity_range[1],
            size=(num_samples, self._digit_num))
        self.replay["random_acceleration_angle"] = rng.random(
            size = (num_samples, )) * 2 * np.pi
        self.replay["random_acceleration_magnitude"] = rng.uniform(
            low=self._random_acceleration_range[0],
            high=self._random_acceleration_range[1],
            size=(num_samples, self._digit_num))
        self.replay["distractor_seeds"] = rng.uniform(
            size=(num_samples, seqlen, self._distractor_num, 5))

        self.replay_numsamples = num_samples

        np.savez_compressed(file=file, **self.replay)
