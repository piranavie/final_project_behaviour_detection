import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer


class RoiPoolingConv(Layer):
    """
    Define ROI Pooling Convolutional Layer for 2D inputs.
    """

    def __init__(self, pool_size, num_rois, **kwargs):

        self.image_data_format = K.image_data_format()
        assert self.image_data_format in {'channels_last',
                                          'channels_first'}, 'image_data_format must be in {channels_last, channels_first}'

        self.pool_size = pool_size
        self.num_rois = num_rois
        self.nb_channels = None

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.image_data_format == 'channels_first':
            # input_shape = (num_rois,512,7,7)
            self.nb_channels = input_shape[0][1]
        elif self.image_data_format == 'channels_last':
            # input_shape = (num_rois,7,7,512)
            self.nb_channels = input_shape[0][3]

        super(RoiPoolingConv, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.image_data_format == 'channels_first':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert (len(x) == 2)

        # x[0] is image with shape (rows, cols, channels)
        img = x[0]
        # x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        input_shape = tf.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            row_length = w / float(self.pool_size)
            col_length = h / float(self.pool_size)

            num_pool_regions = self.pool_size

            if self.image_data_format == 'channels_first':
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = x + ix * row_length
                        x2 = x1 + row_length
                        y1 = y + jy * col_length
                        y2 = y1 + col_length

                        x1 = tf.cast(x1, tf.int32)
                        x2 = tf.cast(x2, tf.int32)
                        y1 = tf.cast(y1, tf.int32)
                        y2 = tf.cast(y2, tf.int32)

                        x2 = x1 + tf.maximum(1, x2 - x1)
                        y2 = y1 + tf.maximum(1, y2 - y1)

                        new_shape = [input_shape[0], input_shape[1], y2 - y1, x2 - x1]

                        x_crop = img[:, :, y1:y2, x1:x2]
                        xm = tf.reshape(x_crop, new_shape)
                        pooled_val = tf.math.maximum(xm, axis=(2, 3))
                        outputs.append(pooled_val)

            elif self.image_data_format == 'channels_last':
                x = tf.cast(x, tf.int32)
                y = tf.cast(y, tf.int32)
                w = tf.cast(w, tf.int32)
                h = tf.cast(h, tf.int32)

                rs = tf.image.resize(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
                outputs.append(rs)

        final_output = tf.concat(outputs, axis=0)
        final_output = tf.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        if self.image_data_format == 'channels_first':
            final_output = K.permute_dimensions(final_output, (0, 1, 4, 2, 3))
        else:
            final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
