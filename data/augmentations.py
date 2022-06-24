import cv2
import torch
import numpy as np
from numpy import random


def intersect(box_a,
              box_b):

    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    intersection = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)

    return intersection[:, 0] * intersection[: 1]


def jaccard_numpy(box_a,
                  box_b):

    intersection = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    union = area_a + area_b - intersection

    return intersection / union


class Compose(object):
    """This class applies a list of transformation to an image."""

    def __init__(self,
                 transforms):
        """Class constructor of Compose

        Arguments:
            transforms {list} -- list of transformation to be applied to the
            image
        """

        super(Compose, self).__init__()
        self.transforms = transforms

    def __call__(self,
                 image,
                 label=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            label {str} -- corresponding class of the image (default: {None})

        Returns:
            np.ndarray, str -- transformed image pixels, corresponding class
            of the image
        """

        for transform in self.transforms:
            image, label = transform(image, label)

        return image, label


class ConvertToFloat(object):
    """This class casts an np.ndarray to floating-point data type."""

    def __init__(self):
        """Class constructor for ConvertToFloat"""

        super(ConvertToFloat, self).__init__()

    def __call__(self,
                 image,
                 label=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            label {str} -- corresponding class of the image (default: {None})

        Returns:
            np.ndarray, str -- image pixels casted to floating-point data type,
            corresponding class of the image
        """

        return image.astype(np.float32), label.astype(np.float32)


class SubtractMeans(object):
    """This class subtracts the mean from the pixel values of the image"""

    def __init__(self,
                 mean):
        """Class constructor for SubtractMeans

        Arguments:
            mean {tuple} -- mean
        """

        super(SubtractMeans, self).__init__()
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self,
                 image,
                 label=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            label {str} -- corresponding class of the image (default: {None})

        Returns:
            np.ndarray, str -- image pixels subtracted by the mean,
            corresponding class of the image
        """

        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), label


class Resize(object):
    """This class resizes the image to output size"""

    def __init__(self,
                 size=300):
        """Class constructor for Resize

        Keyword Arguments:
            size {int} -- output size (default: {300})
        """

        super(Resize, self).__init__()
        self.size = size

    def __call__(self,
                 image,
                 label=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            label {str} -- corresponding class of the image (default: {None})

        Returns:
            np.ndarray, str -- pixels of the resized image, corresponding class
            of the image
        """

        image = cv2.resize(image, (self.size, self.size))
        label = cv2.resize(label, (self.size, self.size))
        return image, label

class ConvertColor(object):
    """This class converts the image to another color space. This class
    supports the conversion from BGT to HSV color space and vice-versa."""

    def __init__(self,
                 current='BGR',
                 transform='HSV'):
        """Class constructor for ConvertColor

        Keyword Arguments:
            current {str} -- the input color space of the image
            (default: {'BGR'})
            transform {str} -- the output color space of the image
            (default: {'HSV'})
        """

        super(ConvertColor, self).__init__()
        self.current = current
        self.transform = transform

    def __call__(self,
                 image,
                 label=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray

        Keyword Arguments:
            label {str} -- corresponding class of the image (default: {None})

        Returns:
            np.ndarray, str -- pixels of the image converted to the output
            color space, corresponding class of the image
        """

        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError

        return image, label

class ChangeBrightness(object):
    """This class adjusts the brightness of the image. This adds a random
    constant to the pixel values of the image."""

    def __init__(self,
                 delta=32.0):
        """Class constructor for RandomBrightness

        Keyword Arguments:
            delta {int} -- lower bound and upper bound of the interval used
            in generating a random number from a uniform distribution to adjust
            brightness (default: {32.0})
        """

        super(ChangeBrightness, self).__init__()
        assert delta >= -255.0 and delta <= 255.0
        self.delta = delta

    def __call__(self,
                 image,
                 label=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            label {str} -- corresponding class of the image (default: {None})

        Returns:
            np.ndarray, str -- image pixels with adjusted brightness,
            corresponding class of the image
        """

        if self.delta > 0:
            image = self.increase_brightness(image, value=self.delta)
        elif self.delta < 0:
            image = self.decrease_brightness(image, value=self.delta)

        return image, label

    def increase_brightness(self, img, value=50):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        value = int(value)
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def decrease_brightness(self, img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        value = int(abs(value))
        lim = value
        v[v < lim] = 0
        v[v >= lim] -= value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

class LowerResolution(object):
    """This class adjusts the brightness of the image. This adds a random
    constant to the pixel values of the image."""

    def __init__(self,
                 scale=3):
        """Class constructor for RandomBrightness

        Keyword Arguments:
            delta {int} -- lower bound and upper bound of the interval used
            in generating a random number from a uniform distribution to adjust
            brightness (default: {32.0})
        """

        super(LowerResolution, self).__init__()
        assert scale >= 1.0
        self.scale = scale

    def __call__(self,
                 image,
                 label=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            label {str} -- corresponding class of the image (default: {None})

        Returns:
            np.ndarray, str -- image pixels with adjusted brightness,
            corresponding class of the image
        """

        if self.scale > 1.0:
            h, w, _ = image.shape
            h2, w2 = (int(w/self.scale), int(h/self.scale))
            img_smaller = cv2.resize(image, (w2, h2), interpolation = cv2.INTER_NEAREST)
            lbl_smaller = cv2.resize(label, (w2, h2), interpolation = cv2.INTER_NEAREST)

            image = cv2.resize(img_smaller, (w, h), interpolation = cv2.INTER_NEAREST)
            label = cv2.resize(lbl_smaller, (w, h), interpolation = cv2.INTER_NEAREST)

        return image, label

class ToCV2Image(object):
    """This class converts the torch.Tensor representation of an image to
    np.ndarray. The channels of the image are also converted from RGB to
    BGR"""

    def __init__(self):
        """Class constructor for ToCV2Image"""

        super(ToCV2Image, self).__init__()

    def __call__(self,
                 image,
                 label=None):
        """Executed when the class is called as a function

        Arguments:
            image {torch.Tensor} -- image pixels represented as torch.Tensor.

        Keyword Arguments:
            label {str} -- corresponding class of the image (default: {None})

        Returns:
            np.ndarray, str -- image pixels represented as np.ndarray,
            corresponding class of the image
        """

        # permute() is used to switch the channels from RGB to BGR
        image = image.cpu().numpy().astype(np.float32).transpose((2, 1, 0))
        return image, label


class ToTensor(object):
    """This class converts the np.ndarray representation of an image to
    torch.Tensor. The channels of the image are also converted from BGR to
    RGB"""

    def __init__(self):
        """Class constructor for ToTensor"""

        super(ToTensor, self).__init__()

    def __call__(self,
                 image,
                 label=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            label {str} -- corresponding class of the image (default: {None})

        Returns:
            torch.Tensor, str -- image pixels represented as torch.Tensor,
            corresponding class of the image
        """

        # permute() is used to switch the channels from BGR to RGB
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 1, 0)
        return image, label

class Augmentations(object):
    """This class applies different augmentation techniques to the image.
    This is used for training the model."""

    def __init__(self,
                 # size,
                 # mean,
                 brightness=0,
                 scale=1):
        """Class constructor for Augmentations

        Arguments:
            size {int} -- output size
            mean {tuple} -- mean
        """

        super(Augmentations, self).__init__()
        # self.size = size
        # self.mean = mean
        # self.augment = Compose([ConvertToFloat(),
        #                         PhotometricDistort(),
        #                         RandomSampleCrop(),
        #                         RandomMirror(),
        #                         # Resize(self.size),
        #                         SubtractMeans(self.mean)])

        # self.augment = Compose([RandomSampleCrop(),
        #                         RandomMirror()])

        self.augment = Compose([ChangeBrightness(delta=brightness),
                                LowerResolution(scale=scale)])

    def __call__(self,
                 image,
                 label):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            label {str} -- corresponding class of the image

        Returns:
            np.ndarray, str -- image pixels applied with different augmentation
            techniques, corresponding class of the image
        """

        return self.augment(image, label)


class BaseTransform(object):
    """This class applies different base transformation techniques to the
    image. This includes resizing the image and subtracting the mean from the
    image pixels. This is used for testing the model."""

    def __init__(self,
                 # size,
                 mean):
        """Class constructor for BaseTransform

        Arguments:
            size {int} -- output size
            mean {tuple} -- mean
        """

        super(BaseTransform, self).__init__()
        # self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self,
                 image,
                 label=None):
        """Executed when the class is called as a function

        Arguments:
            image {np.ndarray} -- image pixels represented as np.ndarray.

        Keyword Arguments:
            label {str} -- corresponding class of the image (default: {None})

        Returns:
            np.ndarray, str -- image pixels applied with different augmentation
            techniques, corresponding class of the image
        """

        # dimensions = (self.size, self.size)
        # image = cv2.resize(image, dimensions).astype(np.float32)
        image = image.astype(np.float32)
        image -= self.mean
        image = image.astype(np.float32)

        # label = cv2.resize(image, dimensions).astype(np.float32)
        return image, label
