import keras
import keras.models as kmodels
import keras.backend as K

import numpy as np
from matplotlib import pyplot as plt

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_activation, get_num_filters

from PIL import Image


def get_class(y_pred):
    return K.round(y_pred)


def count_true(x):
    return K.cast(K.sum(K.cast(x, dtype='int64')), dtype='float64')


def purity_metric(class_index):
    def purity(y_true, y_pred):
        idx_true = K.equal(y_true, class_index)
        idx_pred = K.equal(get_class(y_pred), class_index)
        #print('# predicted to be class {}: {}'.format(class_index, count_true(idx_pred)))
        return count_true(idx_true & idx_pred) / (count_true(idx_pred) + K.epsilon())
    return purity


def completeness_metric(class_index):
    def completeness(y_true, y_pred):
        idx_true = K.equal(y_true, class_index)
        idx_pred = K.equal(get_class(y_pred), class_index)
        return count_true(idx_true & idx_pred) / (count_true(idx_true) + K.epsilon())
    return completeness


# Build the VGG16 network with ImageNet weights
with open('weights/sixth_try.json', 'r') as f:
    model_json = f.read()

model = kmodels.model_from_json(model_json)
model.summary()
model.load_weights('weights/sixth_try.h5')
#model = VGG16(weights='imagenet', include_top=True)
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'conv2d_4'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Visualize all filters in this layer.
filters = np.arange(get_num_filters(model.layers[layer_idx]))
#filters = filters[:2]

# Generate input image for each filter. Here `text` field is used to overlay `filter_value` on top of the image.
vis_images = []
for idx in filters:
    img = visualize_activation(model, layer_idx, filter_indices=idx) 
    img.shape = img.shape[:2]
    img = Image.fromarray(img)
    img.save('plots/{}_filter_{}.png'.format(layer_name, idx))
    #print(type(img))
    #print(img.shape)
    #print(img.dtype)
    #img = np.repeat(img, 3, axis=2)
    #img = utils.draw_text(img, str(idx))
    #vis_images.append(img)

# Generate stitched image palette with 8 cols.
#stitched = utils.stitch_images(vis_images, cols=8)    
#plt.axis('off')
#plt.imshow(stitched)
#plt.title(layer_name)
#plt.savefig('filters.png', dpi=100)
#plt.show()
