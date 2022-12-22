import numpy as np
import h5py
import os
import nibabel as nib
import json
import random
import cv2
import copy


class Params:
    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """"
        Save dict to json file

        Parameters
        ----------
        json_path : string
            Path to save location
        """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """
        Load parameters from json file

        Parameters
        ----------
        json_path : string
            Path to json file
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """"
        Give dict-like access to Params instance by: 'params.dict['learning_rate']'
        """
        return self.__dict__


#def load_dataset(fname):
#    with h5py.File(fname, "r") as f:
#
#        a_group_key2 = list(f.keys())
#        # TODO: Changed images dtype
#        images = np.zeros([512, 512, len(a_group_key2)], dtype=np.float32)
#        labels = np.zeros([512, 512, len(a_group_key2)]).astype(np.int16)
#        for i, name in enumerate(a_group_key2):
#            image = f[str(name)]['image'][()]
#            label = f[str(name)]['labels'][()]
#            # TODO: Added 2 lines for labels + dtype image
#            label[label == 5] = 2
#            label[label == 7] = 3
#            images[:, :, i] = image[:, :].astype(np.float32)
#            labels[:, :, i] = label
#    return images, labels, a_group_key2


def load_dataset(fname, params, norm=True):
    with h5py.File(fname, "r") as f:

        a_group_key2 = list(f.keys())
        # TODO: Changed images dtype
        images = np.zeros([512, 512, len(a_group_key2)], dtype=np.float32)
        labels = np.zeros([512, 512, len(a_group_key2)]).astype(np.int16)
        for i, name in enumerate(a_group_key2):
            image = f[str(name)]['images'][()]
            label = f[str(name)]['labels'][()]
            # TODO: Added 2 lines for labels + dtype image
            # Ralph: removed these lines because this should be handled in createh5.py
            # label[label == 5] = 2
            # label[label == 7] = 3
            # TODO: Added normalization here
            if norm:
                image = normalize(image[:, :], 'True', params.dict['min_bound'], params.dict['max_bound'])
            images[:, :, i] = image.astype(np.float32)
            labels[:, :, i] = label
    print('>>> Unique labels in {}: {}'.format(fname, np.unique(labels)))
    return images, labels, a_group_key2


def load_samples(sample_txt_file, seed=42):
    """
    Load samples from a .txt file, extracts relevant information, splits between CT and GT, and shuffles samples
    according to a specified seed.
    Parameters
    ----------
    sample_txt_file : str
        Text file containing samples.
    seed : int - default = 42
        Seed for shuffling

    Returns
    -------
    samples_dict : dict
        Dictionary containing strings with locations to CT and GT patches.
    """

    with open(sample_txt_file, 'r') as infile:
        data = infile.readlines()

        ct_patches = []
        gt_patches = []
        for i in data:
            line = i.strip(',')
            line = line.split(',')

            ct_patches.append(line[0])
            gt_patches.append(line[1])
    array = list(zip(ct_patches, gt_patches))
    random.seed(seed)
    random.shuffle(array)
    ct_patches, gt_patches = zip(*array)
    samples_dict = {'ct_patches': list(ct_patches),
                    'gt_patches': list(gt_patches)
                    }

    return samples_dict


def shuffle_samples(samples, seed=42):
    """

    Parameters
    ----------
    samples : list
        List containing paths to patches
    seed : int
        Seed for shuffling

    Returns
    -------
    Shuffled Samples
    """
    random.seed(seed)
    return random.shuffle(list(samples))


def load_batch(samples_dict, patch_path, iteration, batch_size):

    def _load_batch(sample_list, patch_path_dir):
        batch = []
        for sample_path in sample_list:
            patch = nib.load(os.path.join(patch_path_dir, sample_path)).get_fdata()
            # patch = np.expand_dims(patch, 0)
            patch = np.expand_dims(patch, -1)
            batch.append(patch)
        # batch = np.expand_dims(batch, 0)
        return np.array(batch)

    min_index = (iteration * batch_size) - batch_size
    max_index = iteration * batch_size
    ct_samples = samples_dict['ct_patches'][min_index:max_index]
    gt_samples = samples_dict['gt_patches'][min_index:max_index]

    ct_batch = _load_batch(ct_samples, patch_path)
    gt_batch = _load_batch(gt_samples, patch_path)

    return ct_batch, gt_batch


def normalize(img, bound, min_bound, max_bound):
    """
    Normalize an image between "min_bound" and "max_bound", and scale between 0 and 1. If "bound" = 'True', scale
    between 2.5th and 97.5th percentile.
    Parameters
    ----------
    img : np.ndarray
        Image to normalize.
    bound : str - True or False.
        Whether to scale between percentiles.
    min_bound : int
        Lower bound for normalization.
    max_bound : int
        Upper bound for normalization.

    Returns
    -------
    img : np.ndarray
        Normalized and scaled image.
    """
    norm = 2.5
    img = (img - min_bound) / (max_bound - min_bound)
    img[img > 1] = 0
    img[img < 0] = 0
    #if bound == 'True':
     #   mn = np.percentile(img, norm)
      #  mx = np.percentile(img, 100 - norm)
       # a = (img - mn)
        #b = (mx - mn)
        #img = np.divide(a, b, np.zeros_like(a), where=b != 0)   
    #print(np.min(img))
    #print(np.max(img))
    c = (img - np.min(img))
    d = (np.max(img) - np.min(img))
    img = np.divide(c, d, np.zeros_like(c), where=d != 0)

   
    # img += np.abs(img.min())
    # img *= 1/img.max()    
    return img


def largest_component_mask(inputs):
    """Finds the largest component in a binary image and returns the component as a mask."""
    img = copy.deepcopy(inputs)
    # img = np.expand_dims(img, -1)
    img *= 255
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = np.invert(thresh)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    # contours = np.squeeze(contours)
    # should be [1] if OpenCV 3+

    max_area = 0
    max_contour_index = 0
    for i, contour in enumerate(contours):
        contour_area = cv2.moments(contour)['m00']
        if contour_area > max_area:
            max_area = contour_area
            max_contour_index = i

    labeled_img = np.zeros(img.shape, dtype=np.uint8)
    labeled_img = cv2.drawContours(labeled_img, contours, max_contour_index, color=255, thickness=-1)

    return np.uint8(labeled_img[:, :, 0])
