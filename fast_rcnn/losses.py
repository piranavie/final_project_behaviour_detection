import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy


# Defining the loss functions
lambda_cls_regr = 1.0
lambda_cls_class = 1.0

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

epsilon = 1e-4


def rpn_loss_regression(num_anchors):
    """
    Calculating the loss function for rpn regression.

    :param num_anchors: get the number of anchors = 9.
    :return: depends on the L1 loss function, where it can return x_abx - 0.5  or 0.5*x*x (if x_abs < 1).
    """
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        y_true = tf.dtypes.cast(y_true, tf.float32)
        if K.image_data_format() == 'channels_first':
            x = y_true[:, 4 * num_anchors:, :, :] - y_pred
            x_abs = tf.abs(x)
            x_bool = tf.cast(tf.less_equal(x_abs, 1.0), tf.float32)
            x_sum = tf.reduce_sum(epsilon + y_true[:, :4 * num_anchors, :, :])
            return lambda_rpn_regr * tf.reduce_sum(
                y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / x_sum
        else:
            x = y_true[:, :, :, 4 * num_anchors:] - y_pred
            x_abs = tf.abs(x)
            x_bool = tf.cast(tf.less_equal(x_abs, 1.0), tf.float32)
            x_sum = tf.reduce_sum(epsilon + y_true[:, :, :, :4 * num_anchors])
            return lambda_rpn_regr * tf.reduce_sum(
                y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / x_sum

    return rpn_loss_regr_fixed_num


def rpn_loss_classifier(num_anchors):
    """
    Calculating loss function for rpn classification.

    :param num_anchors: get the number of anchors= 9.
    :return: fixed number of the rpn loss classifier.
    """
    def fixed_num_rpn_loss_classifier(y_true, y_pred):
        if K.image_data_format() == 'channels_last':
            x = y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :],
                                                                      y_true[:, :, :,
                                                                      num_anchors:])
            y = epsilon + y_true[:, :, :, :num_anchors]
            return lambda_rpn_class * tf.reduce_sum(x) / tf.reduce_sum(y)
        else:
            x_sum = tf.reduce_sum(epsilon + y_true[:, :num_anchors, :, :])
            return lambda_rpn_class * tf.reduce_sum(
                y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :],
                                                                      y_true[:,
                                                                      num_anchors:, :, :])) / x_sum

    return fixed_num_rpn_loss_classifier


def class_loss_regression(num_classes):
    """
    Calculating the loss function for rpn regression.

    :param num_classes: get the number of anchors= 9.
    :return: fixed number of the rpn loss regression.
    """
    def fixed_num_class_loss_regression(y_true, y_pred):
        y_true = tf.dtypes.cast(y_true, tf.float32)
        x = y_true[:, :, 4 * num_classes:] - y_pred
        x_abs = tf.abs(x)
        x_bool = tf.cast(tf.less_equal(x_abs, 1.0), tf.float32)
        x_sum = tf.reduce_sum(epsilon + y_true[:, :, :4 * num_classes])
        return lambda_cls_regr * tf.reduce_sum(
            y_true[:, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / x_sum

    return fixed_num_class_loss_regression


def class_loss_classifier(y_true, y_pred):
    return lambda_cls_class * tf.reduce_mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
