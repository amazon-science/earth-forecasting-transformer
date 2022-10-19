import numpy as np
from PIL import Image


def save_gif(single_seq, fname):
    """Save a single gif consisting of image sequence in single_seq to fname."""
    img_seq = [Image.fromarray(img.astype(np.float32) * 255, 'F').convert("L") for img in single_seq]
    img = img_seq[0]
    img.save(fname, save_all=True, append_images=img_seq[1:])
