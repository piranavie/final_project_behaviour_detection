"""
Train videos are convert into the image frames according to what UCF annotation and readMe.
Training models is created if no training has been done before, weights can be loaded from a pretrained model.
Training process is done using Faster R-CNN with VGG16 network.
The length of each epoch used to do training is 1000 and the total number of epochs trained is 5.
Generating some graph results for loss rpn classifier, loss rpn regression, loss class classifier, loss class regression,
total_loss and class_accuracy by calling plot_graph class and using the csv file that is generated.
"""
from __future__ import division

import random
import pprint
import sys
import time
import pickle
import pandas as pd
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from fast_rcnn import config, data_generators
from fast_rcnn import losses as rcnn_losses
import fast_rcnn.roi_helpers as roi_helpers
from tensorflow.python.keras.utils import generic_utils
from fast_rcnn.simple_parser import get_data
from fast_rcnn import vgg as nn
from format_dataset import remove_file
from plot_graph import train_process_graph

# maximum depth of the Python interpreter stack to the required limit.
# This limit prevents any program from getting into infinite recursion,
# Otherwise infinite recursion will lead to overflow of the C stack and crash the Python.
sys.setrecursionlimit(40000)


def update_execute_config():
    """
    Loading the config and updating the config class.
    Removing results data in the folder when running this class.
    Setting the path to weights based on Faster R-CNN and VGG16 model.
    Parsing the data from annotation file for training process.

    :return classes_count: number of abnormal behaviour labels.
    :return class_mapping: abnormal behaviour classes(labels).
    :return all_imgs: all the training dataset.
    :return curr_config: updated config class.
    """
    curr_config = config.Config()

    print("Clear the graph data in the folder {}".format(curr_config.result_graphs_path))
    remove_file(curr_config.train_process_data_file)

    # check if weight path was passed via command line
    # Input path for weights. try to load default weights provided by keras.
    # set the path to weights based on backend and model
    curr_config.base_net_weights = nn.get_weight_path()

    # Path to training data.
    all_imgs, classes_count, class_mapping = get_data(curr_config.train_path_file, header=True)

    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    curr_config.class_mapping = class_mapping

    print('Training images per class: ')
    pprint.pprint(classes_count)
    print('Num classes (including bg) = {}'.format(len(classes_count)))

    with open(curr_config.train_config_output_filename, 'wb') as config_f:
        pickle.dump(curr_config, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
            curr_config.train_config_output_filename))

    num_imgs = len(all_imgs)
    print('Num all images {}'.format(num_imgs))

    return classes_count, class_mapping, all_imgs, curr_config


def image_generate_optimizer_classifier(classes_count, train_imgs, curr_config):
    """
    Checking the pre-trained model weights is loaded or not.
    Generate the ground_truth anchors.
    Generating image classification.
    This is done by defining the RPN and classifier to built on the vgg layers and using the Model import package to
    to holds the RPN and the classifier, used to load/save weights for the models.
    Then these model will be compiled to get the optimal rpn and classifier model.

    :param classes_count: number of abnormal behaviour labels.
    :param all_imgs: all the training dataset.
    :param curr_config: updated config class.

    :return model_rpn: optimal rpn model.
    :return model_all: optimal models for faster r-cnn and vgg.
    :return data_gen_train: the ground_truth anchors data.
    :return model_classifier: optimal classifier model.
    """
    data_gen_train = data_generators.get_anchor_ground_truth(train_imgs, classes_count, curr_config, nn.get_img_output_length,
                                                             K.image_data_format())

    if K.image_data_format() == 'channels_first':
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # define the base network (VGG)
    shared_layers = nn.vgg_network(img_input)

    # define the RPN, built on the base layers
    num_anchors = len(curr_config.anchor_box_scales) * len(curr_config.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(shared_layers, roi_input, curr_config.num_rois, nb_classes=len(classes_count))

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    try:
        # This will continue training
        print('loading weights from {}'.format(curr_config.base_net_weights))
        model_rpn.load_weights(curr_config.base_net_weights, by_name=True)
        model_classifier.load_weights(curr_config.base_net_weights, by_name=True)
    except:
        print('Could not load pre-trained model weights')

    # compile models
    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer,
                      loss=[rcnn_losses.rpn_loss_classifier(num_anchors), rcnn_losses.rpn_loss_regression(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[rcnn_losses.class_loss_classifier, rcnn_losses.class_loss_regression(len(classes_count) - 1)],
                             metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    return model_rpn, model_all, data_gen_train, model_classifier


def image_training(curr_config, class_mapping, model_all, model_rpn, model_classifier, data_gen_train):
    """
    Starting the training process.
    Generate X (resizeImg) and label Y ([y_rpn_classifier, y_rpn_regr])
    Training rpn model and get loss value for loss rpn classifier and loss rpn regression.
    Getting predicted rpn from rpn model for rpn classifier and loss rpn regression.
    Converting rpn layer to roi bounding boxes.
    Generating data such as loss rpn classifier, loss rpn regression, loss class classifier, loss class regression,
    total_loss and class_accuracy for each number of Epochs. These data will be used to generate graphs.

    :param curr_config: updated config class.
    :param class_mapping: abnormal behaviour classes(labels).
    :param model_all: optimal models for faster r-cnn and vgg16.
    :param model_rpn: optimal rpn model.
    :param model_classifier: optimal classifier model.
    :param data_gen_train: the ground_truth anchors data.
    """
    print('Starting image training')
    # start time
    start_time = time.time()
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    losses = np.zeros((curr_config.epoch_length, 5))
    best_loss = np.Inf
    iter_num = 0
    all_data = []

    # record_dataset = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls','loss_rpn_regr', 'loss_class_cls', 'loss_class_regr','curr_loss', 'elapsed_time'])
    for epoch_num in range(curr_config.num_epochs):

        progbar = generic_utils.Progbar(curr_config.epoch_length)
        print('Epoch {}/{}'.format(epoch_num + 1, curr_config.num_epochs))

        while True:
            try:

                if len(rpn_accuracy_rpn_monitor) == curr_config.epoch_length and curr_config.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print(
                        'Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                            mean_overlapping_bboxes, curr_config.epoch_length))

                    if mean_overlapping_bboxes == 0:
                        print(
                            'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                X, Y, img_data = next(data_gen_train)

                loss_rpn = model_rpn.train_on_batch(X, Y)

                P_rpn = model_rpn.predict_on_batch(X)

                R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], curr_config, K.image_data_format(), use_regr=True,
                                           overlap_thresh=0.7,
                                           max_boxes=300)

                # calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format.
                # X2: bounding boxes that IntersectionOfUnion is greater than config.classifierMinOverlap.
                # for all bounding boxes ground truth in 300 non_max_suppression bounding boxes.
                # Y1: one hot code for bounding boxes from above => x_roi (X).
                # Y2: corresponding labels and corresponding gt bounding boxes.
                X2, Y1, Y2, IouS = roi_helpers.calculate_iou(R, img_data, curr_config, class_mapping)

                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if curr_config.num_rois > 1:
                    # if number of positive anchors is larger than 4//2 --> randomly choose 2 pos samples.
                    if len(pos_samples) < curr_config.num_rois // 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, curr_config.num_rois // 2,
                                                                replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples,
                                                                curr_config.num_rois - len(selected_pos_samples),
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples,
                                                                curr_config.num_rois - len(selected_pos_samples),
                                                                replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()

                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                             [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                progbar.update(iter_num + 1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
                                              ('detector_cls', losses[iter_num, 2]),
                                              ('detector_regr', losses[iter_num, 3])])

                iter_num += 1

                if iter_num == curr_config.epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if curr_config.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                            mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))
                    elapsed_time = (time.time() - start_time)

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    # finish time
                    start_time = time.time()

                    if curr_loss < best_loss:
                        if curr_config.verbose:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                        best_loss = curr_loss
                        model_all.save_weights(curr_config.model_path)

                    # Create record csv file to store the train process
                    new_row = {'mean_overlapping_bboxes': round(mean_overlapping_bboxes, 3),
                               'class_acc': round(float(class_acc), 3),
                               'loss_rpn_cls': round(float(loss_rpn_cls), 3),
                               'loss_rpn_regr': round(float(loss_rpn_regr), 3),
                               'loss_class_cls': round(float(loss_class_cls), 3),
                               'loss_class_regr': round(float(loss_class_regr), 3),
                               'curr_loss': round(curr_loss, 3),
                               'elapsed_time': round(elapsed_time, 3)}
                    all_data.append(new_row)
                    record_dataset = pd.DataFrame(all_data)
                    record_dataset.to_csv(curr_config.train_process_data_file, index=0)
                    break

            except Exception as e:
                print('Exception while Training images: {}'.format(e))
                continue

    print('Training complete, exiting.')
rm -rf .git

def main():
    """
    Executing all the methods above.
    Generating some graph results for loss rpn classifier, loss rpn regression, loss class classifier, loss class regression,
    total_loss and class_accuracy by calling plot_graph class and using the csv file that is generated.
    """
    classes_count, class_mapping, train_imgs, curr_config = update_execute_config()
    model_rpn, model_all, data_gen_train, model_classifier = image_generate_optimizer_classifier(classes_count,
                                                                                                 train_imgs,
                                                                                                 curr_config)

    # image_training(curr_config, class_mapping, model_all, model_rpn, model_classifier, data_gen_train)

    train_process_graph(curr_config.train_process_data_file)


if __name__ == "__main__":
    main()
    quit(0)
