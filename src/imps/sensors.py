import numpy as np

from .consts import yup2zup
from .math_utils import rotX

# OV works in y-up. Convert the camera poses to z-up.
# OV cameras faces down when the rotation is 0 degrees. Make it face upwards

class OVCamera:
    def __init__(self, params):
        pose = np.copy(params['pose']).T

        # Convert to meters
        pose[:-1, -1] /= 100

        self.c2w = pose

        # Convert from y-up to z-up
        self.c2w = yup2zup @ self.c2w
        self.w2c = np.linalg.inv(self.c2w)
        
        # Convert to face-up (+z)
        self.w2c = rotX(np.pi, True) @ self.w2c
        
        self.c2w = np.linalg.inv(self.w2c)

        self.aspect_ratio = params['resolution']['height'] / params['resolution']['width']
        self.horizontal_aperture = params['horizontal_aperture']
        self.focal_mm = params['focal_length']
        
    def focal_px(self, image_width, y=False):
        return (self.focal_mm * image_width) / self.horizontal_aperture
        