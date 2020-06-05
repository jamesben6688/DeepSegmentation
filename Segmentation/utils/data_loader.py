import h5py
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import math
from functools import partial
import tensorflow as tf
from glob import glob

from Segmentation.utils.augmentation import flip_randomly_left_right_image_pair_2d, rotate_randomly_image_pair_2d, \
    translate_randomly_image_pair_2d

from Segmentation.plotting.voxels import plot_slice

def get_multiclass(label):

    # label shape
    # (batch_size, height, width, channels)

    batch_size = label.shape[0]
    height = label.shape[1]
    width = label.shape[2]
    channels = label.shape[3]

    background = np.zeros((batch_size, height, width, 1))
    label_sum = np.sum(label, axis=3)
    background[label_sum == 0] = 1

    label = np.concatenate((label, background), axis=3)

    return label

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float /p double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_OAI_dataset(data_folder, tfrecord_directory, get_train=True, use_2d=True, crop_size=None):

    if not os.path.exists(tfrecord_directory):
        os.mkdir(tfrecord_directory)

    train_val = 'train' if get_train else 'valid'
    files = glob(os.path.join(data_folder, f'*.im'))

    for idx, f in enumerate(files):
        f_name = f.split("/")[-1]
        f_name = f_name.split(".")[0]

        fname_img = f'{f_name}.im'
        fname_seg = f'{f_name}.seg'

        img_filepath = os.path.join(data_folder, fname_img)
        seg_filepath = os.path.join(data_folder, fname_seg)

        assert os.path.exists(seg_filepath), f"Seg file does not exist: {seg_filepath}"

        with h5py.File(img_filepath, 'r') as hf:
            img = np.array(hf['data'])
        with h5py.File(seg_filepath, 'r') as hf:
            seg = np.array(hf['data'])

        if crop_size is not None:

            img_mid = (int(img.shape[0]/2), int(img.shape[1]/2))
            seg_mid = (int(seg.shape[0]/2), int(seg.shape[1]/2))

            assert img_mid == seg_mid, "We expect the mid shapes to be the same size"

            seg_total = np.sum(seg)

            img = img[img_mid[0] - crop_size:img_mid[0] + crop_size,
                    img_mid[1] - crop_size:img_mid[1] + crop_size, :]
            seg = seg[seg_mid[0] - crop_size:seg_mid[0] + crop_size,
                    seg_mid[1] - crop_size:seg_mid[1] + crop_size, :, :]

            #assert np.sum(seg) == seg_total, "We are losing information in the initial cropping."
            assert img.shape == (crop_size * 2, crop_size * 2, 160)
            assert seg.shape == (crop_size * 2, crop_size * 2, 160, 6)

        img = np.rollaxis(img, 2, 0)
        seg = np.rollaxis(seg, 2, 0)
        seg_temp = np.zeros((*seg.shape[0:3], 1), dtype=np.int8)

        assert seg.shape[0:3] == seg_temp.shape[0:3]

        seg_sum = np.sum(seg, axis=-1)
        seg_temp[seg_sum == 0] = 1
        seg = np.concatenate([seg_temp, seg], axis=-1)  # adds additional channel for no class
        img = np.expand_dims(img, axis=-1)
        assert img.shape[-1] == 1
        assert seg.shape[-1] == 7

        shard_dir = f'{idx:03d}-of-{len(files) - 1:03d}.tfrecords'
        tfrecord_filename = os.path.join(tfrecord_directory, shard_dir)

        with tf.io.TFRecordWriter(tfrecord_filename) as writer:
            if use_2d:
                for k in range(len(img)):
                    img_slice = img[k, :, :, :]
                    seg_slice = seg[k, :, :, :]

                    img_raw = img_slice.tostring()
                    seg_raw = seg_slice.tostring()

                    height = img_slice.shape[0]
                    width = img_slice.shape[1]
                    num_channels = seg_slice.shape[-1]

                    feature = {
                        'height': _int64_feature(height),
                        'width': _int64_feature(width),
                        'num_channels': _int64_feature(num_channels),
                        'image_raw': _bytes_feature(img_raw),
                        'label_raw': _bytes_feature(seg_raw)
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
            else:
                height = img.shape[0]
                width = img.shape[1]
                depth = img.shape[2]
                num_channels = seg.shape[-1]

                print(img.shape)
                print(seg.shape)

                img_raw = img.tostring()
                seg_raw = seg.tostring()

                feature = {
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'depth': _int64_feature(depth),
                    'num_channels': _int64_feature(num_channels),
                    'image_raw': _bytes_feature(img_raw),
                    'label_raw': _bytes_feature(seg_raw)
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        print(f'{idx} out of {len(files) - 1} datasets have been processed')

def parse_fn_2d(example_proto, training, multi_class=True):

    features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'num_channels': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string)
    }

    # Parse the input tf.Example proto using the dictionary above.
    image_features = tf.io.parse_single_example(example_proto, features)
    image_raw = tf.io.decode_raw(image_features['image_raw'], tf.float32)
    image = tf.reshape(image_raw, [image_features['height'], image_features['width'], 1])

    seg_raw = tf.io.decode_raw(image_features['label_raw'], tf.int16)
    seg = tf.reshape(seg_raw, [image_features['height'], image_features['width'], image_features['num_channels']])
    seg = tf.cast(seg, tf.float32)

    #if training:
    #    image, seg = flip_randomly_left_right_image_pair_2d(image, seg)
    #    image, seg = translate_randomly_image_pair_2d(image, seg, 24, 12)
    #    image, seg = rotate_randomly_image_pair_2d(image, seg, tf.constant(-math.pi / 12), tf.constant(math.pi / 12))

    # need to add binary option - see below how to

    return (image, seg)

def parse_fn_3d(example_proto, training, multi_class=True):

    features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'num_channels': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string)
    }

    # Parse the input tf.Example proto using the dictionary above.
    image_features = tf.io.parse_single_example(example_proto, features)
    image_raw = tf.io.decode_raw(image_features['image_raw'], tf.float32)
    image = tf.reshape(image_raw, [image_features['height'], image_features['width'], image_features['depth'], 1])

    seg_raw = tf.io.decode_raw(image_features['label_raw'], tf.int16)
    seg = tf.reshape(seg_raw, [image_features['height'], image_features['width'],
                               image_features['depth'], image_features['num_channels']])
    seg = tf.cast(seg, tf.float32)

    if not multi_class:
        seg_cartilage = tf.slice(seg, [0, 0, 0, 1], [-1, -1, -1, 6])
        seg_cartilage = tf.math.reduce_sum(seg_cartilage, axis=-1)
        seg_cartilage = tf.expand_dims(seg_cartilage, axis=-1)
        seg = tf.clip_by_value(seg_cartilage, 0, 1)

    # print(image_features['height'])
    # print(seg.shape)

    # print("========================== p")
    # if crop_size is not None:
    #     h_centre = tf.math.divide(image_features['height'], 2)
    #     w_centre = image_features['width']/2

    #     h_centre = tf.random.normal([1], mean=tf.cast(h_centre, tf.float32), stddev=tf.cast(h_centre/2, tf.float32))
        
    #     w_centre = tf.random.normal([1], mean=tf.cast(w_centre, tf.float32), stddev=tf.cast(w_centre/2, tf.float32))
        
    #     h_centre = tf.clip_by_value(h_centre, h_centre - crop_size, h_centre + crop_size)
    #     h_centre = tf.clip_by_value(h_centre, tf.cast(0, tf.float32), image_features['height'])
    #     w_centre = tf.clip_by_value(w_centre, w_centre - crop_size, w_centre + crop_size)
    #     w_centre = tf.clip_by_value(w_centre, tf.cast(0, tf.float32), image_features['width'])
    #     h_centre = tf.cast(h_centre, tf.int32)
    #     w_centre = tf.cast(w_centre, tf.int32)
        

    #     tmp_x = [0, 0, tf.squeeze(h_centre) - crop_size, tf.squeeze(w_centre) - crop_size, 0]
    #     tmp_y = [-1, -1, crop_size * 2, crop_size * 2, -1]

    #     tf.print(tmp_x)
    #     tf.print(tmp_y)

    #     tf.print(image.shape)
    #     tf.print(seg.shape)

    #     image = tf.slice(image, [0, 192 - 144 - 1, 192 - 144 - 1, 0], [-1, 144 * 2, 144 * 2, -1])

    #     seg = tf.slice(seg, [0, 192 - 144 - 1, 192 - 144 - 1, 0], [-1, 144 * 2, 144 * 2, -1])


    #     # if training:        
    #     #     h_centre = tf.random.normal([1], mean=tf.cast(h_centre, tf.float32), stddev=tf.cast(h_centre/2, tf.float32))
    #     #     w_centre = tf.random.normal([1], mean=tf.cast(w_centre, tf.float32), stddev=tf.cast(w_centre/2, tf.float32))
    #     #     h_centre = tf.clip_by_value(h_centre, h_centre - crop_size, h_centre + crop_size)
    #     #     w_centre = tf.clip_by_value(w_centre, w_centre - crop_size, w_centre + crop_size)
    #     #     h_centre = tf.cast(h_centre, tf.int32)
    #     #     w_centre = tf.cast(w_centre, tf.int32)

    #     #     tmp_x = [0, 0, tf.squeeze(h_centre), tf.squeeze(w_centre), 0]
    #     #     tmp_y = [-1, -1, crop_size * 2, crop_size * 2, -1]

    #     #     image = tf.expand_dims(image, axis=0)
    #     #     image = tf.slice(image, tmp_x, tmp_y)

    #     #     seg = tf.expand_dims(seg,axis=0)
    #     #     seg = tf.slice(seg, tmp_x, tmp_y)
    #     #     #image = tf.slice(image, [tf.constant([0]), tf.constant([0]), tf.squeeze(h_centre), tf.squeeze(w_centre), tf.constant([0])], [-1, -1, crop_size * 2, crop_size * 2, -1])
    #     #     #seg = tf.slice(seg, [0, 0, h_centre, w_centre, 0], [-1, -1, crop_size * 2, crop_size * 2, -1])

    #     #     print(image.shape)
    #     #     print("########################## d")
    #     # else:
    #     #     print("not training =========================================")      
    #     #     h_centre = tf.random.normal([1], mean=tf.cast(h_centre, tf.float32), stddev=tf.cast(h_centre/2, tf.float32))
    #     #     w_centre = tf.random.normal([1], mean=tf.cast(w_centre, tf.float32), stddev=tf.cast(w_centre/2, tf.float32))
    #     #     print(h_centre, w_centre)
    #     #     print(h_centre - crop_size, h_centre + crop_size)
    #     #     h_centre = tf.clip_by_value(h_centre, h_centre - crop_size, h_centre + crop_size)
    #     #     w_centre = tf.clip_by_value(w_centre, w_centre - crop_size, w_centre + crop_size)
    #     #     print(h_centre, w_centre)
    #     #     print("------------------------------------- q ")

    #     #     print(crop_size * 2)
    #     #     print("------------------------------------- r")
    #     #     print(h_centre.shape)
    #     #     print(tf.squeeze(h_centre).shape)
    #     #     print("------------------------------------- s")
    #     #     h_centre = tf.cast(h_centre, tf.int32)
    #     #     w_centre = tf.cast(w_centre, tf.int32)

    #     #     tmp_x = [0, 0, tf.squeeze(h_centre), tf.squeeze(w_centre), 0]
    #     #     tmp_y = [-1, -1, crop_size * 2, crop_size * 2, -1]

    #     #     print(tmp_x)
    #     #     print(tmp_y)

    #     #     print("########################## a ")

    #     #     print(image.shape)
    #     #     print("========================= b")

    #     #     image = tf.expand_dims(image, axis=0)
            
    #     #     print(image.shape)
    #     #     print("========================= c")

    #     #     image = tf.slice(image, tmp_x, tmp_y)
    #     #     #image = tf.slice(image, [tf.constant([0]), tf.constant([0]), tf.squeeze(h_centre), tf.squeeze(w_centre), tf.constant([0])], [-1, -1, crop_size * 2, crop_size * 2, -1])
    #     #     #seg = tf.slice(seg, [0, 0, h_centre, w_centre, 0], [-1, -1, crop_size * 2, crop_size * 2, -1])

    #     #     print(image.shape)
    #     #     print("########################## d")
        # pass
    
    return (image, seg)

def read_tfrecord(tfrecords_dir, batch_size, buffer_size, parse_fn=parse_fn_2d,
                  multi_class=True, is_training=False, use_keras_fit=True, crop_size=None):

    file_list = tf.io.matching_files(os.path.join(tfrecords_dir, '*-*'))
    shards = tf.data.Dataset.from_tensor_slices(file_list)
    if is_training:
        shards = shards.shuffle(tf.cast(tf.shape(file_list)[0], tf.int64))
    if use_keras_fit:
        shards = shards.repeat()
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    parser = partial(parse_fn, training=is_training, multi_class=multi_class)
    dataset = dataset.map(map_func=parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    # optimise dataset performance
    options = tf.data.Options()
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_vectorization.enabled = True
    options.experimental_optimization.map_parallelization = True
    dataset = dataset.with_options(options)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def read_tfrecord_3d(tfrecords_dir, batch_size, buffer_size, is_training, crop_size=None, **kwargs):
    dataset = read_tfrecord(tfrecords_dir, batch_size, buffer_size, parse_fn_3d, is_training=is_training, crop_size=crop_size, **kwargs)
    if crop_size is not None:
        if is_training:
            parse_rnd_crop = partial(apply_random_crop, crop_size=crop_size)
            dataset = dataset.map(map_func=parse_rnd_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            parse_rnd_crop = partial(apply_centre_crop, crop_size=crop_size)
            dataset = dataset.map(map_func=parse_rnd_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def apply_random_crop(image_tensor, label_tensor, crop_size):
    print("==============")
    print("RANDOM")
    centre = (tf.cast(tf.math.divide(tf.shape(image_tensor)[2], 2), tf.int32), 
              tf.cast(tf.math.divide(tf.shape(image_tensor)[3], 2), tf.int32))
    hrc = tf.random.normal([], mean=tf.cast(centre[0], tf.float32), stddev=tf.cast(centre[0]/2, tf.float32))
    wrc = tf.random.normal([], mean=tf.cast(centre[1], tf.float32), stddev=tf.cast(centre[1]/2, tf.float32))
    hh = tf.shape(image_tensor)[2] - crop_size
    hrc = tf.clip_by_value(hrc, tf.cast(crop_size, tf.float32), tf.cast(tf.shape(image_tensor)[2] - crop_size, tf.float32))
    wrc = tf.clip_by_value(wrc, tf.cast(crop_size, tf.float32), tf.cast(tf.shape(image_tensor)[3] - crop_size, tf.float32))
    hrc = tf.cast(tf.math.round(hrc), tf.int32)
    wrc = tf.cast(tf.math.round(wrc), tf.int32)
    centre = (hrc, wrc)
    image_tensor, label_tensor = crop(image_tensor, label_tensor, crop_size, centre)
    return image_tensor, label_tensor

def apply_centre_crop(image_tensor, label_tensor, crop_size):
    print("==============")
    print("CENTRE")
    centre = (tf.cast(tf.math.divide(tf.shape(image_tensor)[2], 2), tf.int32), 
              tf.cast(tf.math.divide(tf.shape(image_tensor)[3], 2), tf.int32))
    image_tensor, label_tensor = crop(image_tensor, label_tensor, crop_size, centre)
    return image_tensor, label_tensor

def crop(image_tensor, label_tensor, crop_size, centre):
    print("============= crop ================")
    print(image_tensor.shape)
    print(label_tensor.shape)
    print(centre)
    hc, wc = centre
    image_tensor = tf.slice(image_tensor, [0, 0, hc - crop_size, wc - crop_size, 0], [-1, -1, crop_size * 2, crop_size * 2, -1])
    label_tensor = tf.slice(label_tensor, [0, 0, hc - crop_size, wc - crop_size, 0], [-1, -1, crop_size * 2, crop_size * 2, -1])
    return image_tensor, label_tensor
