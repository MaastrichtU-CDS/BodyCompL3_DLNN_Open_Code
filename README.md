# BodyCompL3_DLNN_Open_Code
Pre-processing and deep learning network open code for training an L3 CT automatic segmentation of fat and muscle

To train the network, simply run

python run_offline.py

This script will load the params.json config file and take all settings from there. Important are the following settings:

- data_path_train/val/test: These are the file paths to the training, validation and test sets in HDF5 format (see below for details about how to structure your images and ground truth segmentations inside the .h5 file)
- patch_path/val_patch_path: These are not used but you might create these folders if you run into errors about missings
- log_path: This folder should exist because that where the trained model will be stored in

All others settings are probably best left as is. 

## Creating HDF5 files

The run_offline.py script expects training, validation and test sets in HDF5 format. These files contain both the orginal L3 images as well as the ground truth segmentations. Both the image and the ground truth should be NumPy arrays. Since the original images are DICOM, we need to convert the pixel data inside the DICOM file to NumPy. The same must be done for the ground truth segmentations. Our segmentations were created with Tomovision Slice-o-matic which produces a proprietary TAG format for these segmentations. You need to convert the pixel data to NumPy before you run the training procedure. 

The script create_h5.py describes how we did this. Below some explanation about parts of this procedure.

To convert DICOM and TAG files to Numpy you can use the Python package barbell2 which contains components for the above-mentioned conversions. After pip installing barbell2, run the following code:

    from barbell2.imaging import dcm2npy
    d2n = dcm2npy.Dicom2Numpy('/path/to/l3.dcm')
    d2n.set_normalize_enabeld(True) # Applies DICOM rescale intercept and slope to the pixel values 
                                    # to move them into the positive range
    dcm_npy_array = d2n.execute()

To convert the TAG file to NumPy do the following:

    from barbell2.imaging import tag2npy
    t2n = tag2npy.Tag2Numpy('/path/to/l3.tag', shape=(512, 512))  # Assuming your image size is (512, 512)
    tag_npy_array = t2n.execute()

Once you have the DICOM and TAG files converted to NumPy arrays, you can start building the HDF5 file as follows:

# License
This project is licensed under Creative Commons by 4.0 Public International (see LICENSE)

# Final notes
This repo does not contain the *trained* model, only the untrained network architecture. You need to train the network yourself using your own data. If you want to use the trained network to segment muscle and fat in new images, you can contact Ralph Brecheisen (r.brecheisen@maastrichtuniversity.nl) or submit a processing request by filling in the form [here](https://mosamatic.rbeesoft.nl).