import os
import h5py
import pydicom
import numpy as np

from caipirinha_cmdtools.lib.dicom import Dcm2Numpy, Tag2NumPy

HOME = os.environ['HOME']

ROOT_DIR = '/Volumes/USB_SECURE1/data/radiomics/projects/deepseg/data/mega/final'

H5_MEGA = '{}/Desktop/data_mega.h5'.format(HOME)


def count_nr_patients():
    count = 0
    for root, dirs, files in os.walk(ROOT_DIR):
        for f in files:
            f = os.path.join(ROOT_DIR, f)
            if f.endswith('.dcm') and not f.startswith('._'):
                count += 1
    print('found total of {} patients'.format(count))


def get_dicom_pixels(f, normalize):
    dcm2numpy = Dcm2Numpy()
    dcm2numpy.set_input_dicom_file_path(f)
    dcm2numpy.set_normalize_enabled(normalize)
    dcm2numpy.execute()
    pixels = dcm2numpy.get_output_numpy_array()
    return pixels


def get_tag_pixels(f, shape):
    tag2numpy = Tag2NumPy(shape)
    tag2numpy.set_input_tag_file_path(f)
    tag2numpy.execute()
    return tag2numpy.get_output_numpy_array()


def has_dimension(f, rows, columns):
    p = pydicom.read_file(f)
    if p.Rows == rows and p.Columns == columns:
        return True
    return False


def has_correct_labels(labels):
    if len(labels) == 5 and labels[0] == 0 and labels[1] == 1 and labels[2] == 5 and labels[3] == 7 and labels[4] == 12:
        return True
    return False


def update_labels(pixels):
    # http://www.tomovision.com/Sarcopenia_Help/index.htm
    labels_to_keep = [0, 1, 5, 7]
    labels_to_remove = [2, 12, 14]
    for label in np.unique(pixels):
        if label in labels_to_remove:
            pixels[pixels == label] = 0
    for label in np.unique(pixels):
        if label not in labels_to_keep:
            return None
    if len(np.unique(pixels)) != 4:
        print('Incorrect nr. of labels: {}'.format(len(np.unique(pixels))))
        return None
    return pixels


def create_h5(target_file, root_dir, rows, columns, normalize):
    with h5py.File(target_file, 'w') as h5f:
        count = 1
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.endswith('.dcm') and not f.startswith('._'):
                    f = os.path.join(root, f)
                    if has_dimension(f, rows=rows, columns=columns):
                        dicom_pixels = get_dicom_pixels(f, normalize=normalize)
                        tag_pixels = get_tag_pixels(f[:-4] + '.tag', dicom_pixels.shape)
                        tag_pixels = update_labels(tag_pixels)
                        if tag_pixels is not None:
                            group = h5f.create_group('{:04d}'.format(count))
                            group.create_dataset('image', data=dicom_pixels)
                            group.create_dataset('labels', data=tag_pixels)
                            print('{:04d} added {}'.format(count, f))
                            count += 1
                        else:
                            print(' >> Missing labels in {}'.format(f))
        print('Created HDF5 based on {} patients'.format(count))


def run():
    create_h5(H5_MEGA, ROOT_DIR, 512, 512, True)


if __name__ == '__main__':
    run()
