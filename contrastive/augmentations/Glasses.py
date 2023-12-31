from utils.utils import crop_mask, rotate_image, angle_and_distance
from Landmarks import Landmarks
import numpy as np
import cv2
import os
import random
import imageio
import albumentations as A


class GlassAugmentations:
    """
    Augments images by adding glasses
    """

    def __init__(self):
        self.landmarks_detector = Landmarks()
        self.glasses = []
        self.init_glasses()
        self.mask_augmentation = A.Compose([
            A.RGBShift(r_shift_limit=100,
                       g_shift_limit=100,
                       b_shift_limit=100, ),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.2)),
        ])

    def init_glasses(self):
        """
        Loads all the glasses
        """
        for filename in os.listdir("filtered-glasses"):
            if filename.endswith("png"):
                mask = cv2.imread(os.path.join("filtered-glasses", filename), cv2.IMREAD_UNCHANGED)
                self.glasses.append(crop_mask(mask))

    def augment(self, original_image, bbox=None, n=1):
        """
        :param original_image: BGR image
        :return: list of images with added glasses
        """
        if bbox is None:
            bbox = self.landmarks_detector.retina_landmarks(original_image.copy())[0]

        x1, y1 = [int(t) for t in bbox[2]]
        x2, y2 = [int(t) for t in bbox[3]]
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        angle, d = angle_and_distance(x1, x2, y1, y2)

        res = []
        for i in range(n):
            image = original_image.copy()
            mask = random.choice(self.glasses)

            new_w = int(random.uniform(2.3, 2.7) * d)
            new_h = new_w * mask.shape[0] // mask.shape[1]

            mask = cv2.resize(mask, (new_w, new_h))
            mask = rotate_image(mask, -random.uniform(1, 1.2) * angle)

            mh, mw = mask.shape[:2]
            center_x = (x2 + x1) // 2
            center_y = (y2 + y1) // 2

            region = image[center_y - mh // 2:center_y + mh // 2 + mh % 2,
                     center_x - mw // 2:center_x + mw // 2 + mw % 2]

            overlay_mask = np.squeeze(mask[:, :, 3:])
            overlay_mask_3D = np.repeat(overlay_mask[:, :, np.newaxis], 3, axis=2)

            region = np.where(overlay_mask_3D, self.mask_augmentation(image=mask[:, :, :3])['image'], region)
            image[center_y - mh // 2:center_y + mh // 2 + mh % 2,
            center_x - mw // 2:center_x + mw // 2 + mw % 2] = region

            res.append(image)
        return res


if __name__ == '__main__':
    augmentations = GlassAugmentations()
    image = cv2.imread('sample.png')
    image = cv2.resize(image, (178, 178))

    augmented_images = augmentations.augment(image, n=200)
    # imageio.mimsave('rotated.gif', [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in augmented_images],
    #                 duration=1000 // 10)

    for im in augmented_images:
        cv2.imshow('augmented image', im)
        cv2.waitKey()
