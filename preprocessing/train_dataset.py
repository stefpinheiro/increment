# from google.colab import auth
from google.api_core import retry
from IPython.display import Image
from matplotlib import pyplot as plt
from numpy.lib import recfunctions as rfn

import concurrent
import ee
import google
import io
import multiprocessing
import numpy as np
import requests
import tensorflow as tf

PROJECT = 'sad-deep-learning-274812'
credentials, _ = google.auth.default()
ee.Initialize(credentials, project=PROJECT, opt_url='https://earthengine-highvolume.googleapis.com')

assetMosaic = 'projects/nexgenmap/MapBiomas2/SENTINEL/mosaics-3'
asset_tiles = 'users/stefanypinheiro/roads/increment/PA/PA-tiles-1k-samples'
assetRoads22 = 'users/stefanypinheiro/roads/increment/inference/2022/inference_v2'

tiles = ee.FeatureCollection(asset_tiles)
inference = ee.Image(assetRoads22).rename(['reference']).clip(tiles)

mosaicT0 = ee.ImageCollection(assetMosaic)\
    .filter(ee.Filter.eq('version', '3'))\
    .filter(ee.Filter.eq('year', 2020))\
    .select(['swir1_median', 'nir_median', 'red_median'], ['swir1_median_t0', 'nir_median_t0', 'red_median_t0'])\
    .mosaic()

mosaicT1 = ee.ImageCollection(assetMosaic)\
    .filter(ee.Filter.eq('version', '3'))\
    .filter(ee.Filter.eq('year', 2022))\
    .select(['swir1_median', 'nir_median', 'red_median'], ['swir1_median_t1', 'nir_median_t1', 'red_median_t1'])\
    .mosaic()

mosaic = ee.ImageCollection(assetMosaic)\
    .filter(ee.Filter.eq('version', '3'))\
    .filter(ee.Filter.eq('year', 2022))\
    .select(['swir1_median', 'nir_median', 'red_median'])\
    .mosaic()

stacked_mosaic_t0t1 = mosaicT0.addBands(mosaicT1).addBands(inference)
stacked_mosaic = mosaic.addBands(inference)

# REPLACE WITH YOUR BUCKET!
OUTPUT_FILE = 'gs://imazon/roads/training_dataset.tfrecord.gz'

# Output resolution in meters.
SCALE = 10

# Pre-compute a geographic coordinate system.
proj = ee.Projection('EPSG:4326').atScale(SCALE).getInfo()

# Get scales in degrees out of the transform.
SCALE_X = proj['transform'][0]
SCALE_Y = -proj['transform'][4]

# Patch size in pixels.
PATCH_SIZE = 128

# Offset to the upper left corner.
OFFSET_X = -SCALE_X * PATCH_SIZE / 2
OFFSET_Y = -SCALE_Y * PATCH_SIZE / 2

# Request template.
REQUEST = {
      'fileFormat': 'NPY',
      'grid': {
          'dimensions': {
              'width': PATCH_SIZE,
              'height': PATCH_SIZE
          },
          'affineTransform': {
              'scaleX': SCALE_X,
              'shearX': 0,
              'shearY': 0,
              'scaleY': SCALE_Y,
          },
          'crsCode': proj['crs']
      }
  }

# SWIR NIR Red.
FEATURES_T0T1 = ['swir1_median_t0', 'nir_median_t0', 'red_median_t0',
            'swir1_median_t1', 'nir_median_t1', 'red_median_t1',
            'reference'
            ]

FEATURES = [
   'swir1_median', 'nir_median', 'red_median',
   'reference'
]

# Number of samples per ROI, and per TFRecord file.
N = 64

# Specify the size and shape of patches expected by the model.
KERNEL_SHAPE = [PATCH_SIZE, PATCH_SIZE]
COLUMNS = [
  tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in FEATURES
]
FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

@retry.Retry()
def get_patch(coords, image):
  """Get a patch centered on the coordinates, as a numpy array."""
  request = dict(REQUEST)
  request['expression'] = image
  request['grid']['affineTransform']['translateX'] = coords[0] + OFFSET_X
  request['grid']['affineTransform']['translateY'] = coords[1] + OFFSET_Y
  return np.load(io.BytesIO(ee.data.computePixels(request)))

@retry.Retry()
def get_display_image(coords, image):
  """Helper to display a patch using notebook widgets."""
  point = ee.Geometry.Point(coords)
  region = point.buffer(64 * 10).bounds()
  url = image.getThumbURL({
      'region': region,
      'dimensions': '128x128',
      'format': 'jpg',
      'min': 0, 'max': 5000,
      'bands': ['B4_median', 'B3_median', 'B2_median']
  })

  r = requests.get(url, stream=True)
  if r.status_code != 200:
    raise google.api_core.exceptions.from_http_response(r)

  return r.content

def get_centroids(fc):
   return fc.centroid()

def get_sample_coords(roi, n):
  """"Get a random sample of N points in the ROI."""
  points = ee.FeatureCollection.randomPoints(region=roi, points=n, maxError=1)
  return points.aggregate_array('.geo').getInfo()

def array_to_example(structured_array):
  """"Serialize a structured numpy array into a tf.Example proto."""
  feature = {}
  for f in FEATURES:
    feature[f] = tf.train.Feature(
        float_list = tf.train.FloatList(
            value = structured_array[f].flatten()))
  return tf.train.Example(
      features = tf.train.Features(feature = feature))

def write_dataset(image, sample_points, file_name):
  """"Write patches at the sample points into a TFRecord file."""
  future_to_point = {
    EXECUTOR.submit(get_patch, point['coordinates'], image): point for point in sample_points
  }

  # Optionally compress files.
  writer = tf.io.TFRecordWriter(file_name)

  for future in concurrent.futures.as_completed(future_to_point):
      point = future_to_point[future]
      try:
          np_array = future.result()
          example_proto = array_to_example(np_array)
          writer.write(example_proto.SerializeToString())
          writer.flush()
      except Exception as e:
          print(e)
          pass

  writer.close()

EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=N)

# IMAGE = ''

# sample_points = get_sample_coords(TEST_ROI, N)

sample_points = tiles.map(get_centroids)
print('sample_points size', sample_points.size().getInfo())


sample_points = sample_points.randomColumn()
training_points = sample_points.filter('random <= 0.756')
test_points = sample_points.filter('random > 0.756 and random <=0.887')
validation_points = sample_points.filter('random > 0.887')
print('training_points size', training_points.size().getInfo())
print('test_points size', test_points.size().getInfo())
print('validation_points size', validation_points.size().getInfo())

coords = training_points.aggregate_array('.geo').getInfo()
# print(coords)

# Sample patches from the image at each point.  Each sample is
# fetched in parallel using the ThreadPoolExecutor.
path = '/home/stefany/imazon/increment/data/train.tfrecord.gz'
write_dataset(stacked_mosaic, coords, path)

