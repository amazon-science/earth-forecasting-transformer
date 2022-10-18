from typing import Sequence
import random
import torchvision.transforms.functional as TF


class TransformsFixRotation:
    r"""
    Rotate by one of the given angles.

    Example: `rotation_transform = MyRotationTransform(angles=[-30, -15, 0, 15, 30])`
    """

    def __init__(self, angles):
        if not isinstance(angles, Sequence):
            angles = [angles, ]
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
