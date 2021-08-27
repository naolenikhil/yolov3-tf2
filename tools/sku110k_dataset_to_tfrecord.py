import time
import os
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
#import lxml.etree
import tqdm
import pandas as pd
from struct import unpack


flags.DEFINE_string('data_dir',
                    './datasets/sku110k_cvpr19/SKU110K_fixed/annotations',
                    'path to raw SKU110k train/test/val directory')
flags.DEFINE_string('image_dir',
                    './datasets/sku110k_cvpr19/SKU110K_fixed/images',
                    'path to all images'
                    )
flags.DEFINE_string('split_name', 'annotations_train.csv', 'specify annotations_{train|test|val}.csv')
flags.DEFINE_string('classes', 'classes/sku110k.names', 'classes file')


# we need to check Jpeg images as some are corrupt causing model training to fail.
# Lets remove such images
# source https://github.com/tensorflow/tpu/issues/455

marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}

class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2 + lenchunk:]
            if len(data) == 0:
                break


def build_example(row, class_map):
    img_raw = open(row['image_fpath'], 'rb').read()

    # check if image can be encoded and decoded correctly. If not, remove/skip it
    try:
        # check of corrupt images which are corrupt
        _ = tf.image.decode_jpeg(img_raw)
        _image = JPEG(row['image_fpath'])
        _image.decode()
    except Exception as e:
        logging.info(f'Skipping image: {row["image_fpath"].split("/")[-1]} because it cant be decoded')
        return None

    key = hashlib.sha256(img_raw).hexdigest()

    width = int(row['image_width'])
    height = int(row['image_height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    for bbox in row['bboxes']:
        xmin.append(float(bbox['x1']) / width)
        ymin.append(float(bbox['y1']) / height)
        xmax.append(float(bbox['x2']) / width)
        ymax.append(float(bbox['y2']) / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_map[row['class']])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            row['image_fpath'].encode('utf8')])),
        # 'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
        #     row['image_fpath'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        # 'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        # 'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        # 'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example

def main(_argv):
    classes_fpath = os.path.join(FLAGS.data_dir, FLAGS.classes)
    class_map = {name: idx for idx, name in enumerate(
        open(classes_fpath).read().splitlines())}
    logging.info("Class mapping loaded: %s", class_map)
    df = pd.read_csv(os.path.join(FLAGS.data_dir, FLAGS.split_name), header=None)
    col = {
        0: 'image_fpath',
        1: 'x1',
        2: 'y1',
        3: 'x2',
        4: 'y2',
        5: 'class',
        6: 'image_width',
        7: 'image_height'
    }
    df.rename(columns=col, inplace=True)
    df['image_fpath'] = FLAGS.image_dir.rstrip('/') + '/' + df['image_fpath'].astype('str')
    logging.info(f"raw dataset loaded: {len(df)}")
    output_fpath = os.path.join(FLAGS.data_dir, FLAGS.split_name.split('.')[0] + '.tfrecord')
    logging.info(f'output_fpath: {output_fpath}')
    writer = tf.io.TFRecordWriter(output_fpath)
    data = {}
    for ind, row in df.iterrows():
        if row['image_fpath'] not in data:
            data[row['image_fpath']] = {'bboxes': [{'x1': row['x1'],
                                                    'y1': row['y1'],
                                                    'x2': row['x2'],
                                                    'y2': row['y2'],
                                                    }],
                                        'image_width': row['image_width'],
                                        'image_height': row['image_height'],
                                        'class': 'object',
                                        'image_fpath': row['image_fpath']
                                        }
        else:
            data[row['image_fpath']]['bboxes'].append(
                {'x1': row['x1'],
                 'y1': row['y1'],
                 'x2': row['x2'],
                 'y2': row['y2'],
                 }
            )
    for image_fpath in data:
        tf_example = build_example(data[image_fpath], class_map)
        if tf_example is None:
            # skip images which are corrupt
            continue
        writer.write(tf_example.SerializeToString())
    writer.close()
    logging.info("Done")



if __name__ == '__main__':
    app.run(main)
