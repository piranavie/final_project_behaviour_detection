"""
The test video is converted into image frames.
Detection process is done using Faster R-CNN with VGG 16 network.
Display the detected results containing the bounding boxes of where the abnormal behaviour is found in the image frame,
the name of the crime occurs in the image frame, accuracy score and a warning sign to alert the user that there is a
suspicious person is found. This warning message is only displayed when the accuracy score is more than 70%. Also, if
there is no human and no abnormal behaviour is found, then there is no video output.
Displaying produces some graph results such as ground truth classes, predicted classes, mAP, Precision and Recall
calculations.
"""
from __future__ import division
import os
import pprint
import cv2
import sys
import pickle
import time
import numpy as np

import fast_rcnn.vgg as nn
from fast_rcnn import config
from tensorflow.python.keras.utils import generic_utils
from format_dataset import remove_all_files
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from fast_rcnn import roi_helpers
from plot_graph import main_graph_process
from fast_rcnn.simple_parser import get_data

sys.setrecursionlimit(40000)


def load_config():
    """
    Loading the config and updating the data arguments(e.g., turn off the use_horizontal_flips, use_vertical_flips and
    rot_90) in the config class.
    Removing results data in the folder when running this class.
    Parsing the data from annotation file for testing process.
    """
    curr_config = config.Config()

    if not os.path.exists(curr_config.results_image_path):
        os.makedirs(curr_config.results_image_path)

    if not os.path.exists(curr_config.results_video_path):
        os.makedirs(curr_config.results_video_path)

    if not os.path.exists(curr_config.result_graphs_path):
        os.makedirs(curr_config.result_graphs_path)

    if not os.path.exists(curr_config.result_graphs_path):
        os.makedirs(curr_config.result_graphs_path)

    remove_all_files(curr_config.result_graphs_path)
    remove_all_files(curr_config.results_image_path)
    remove_all_files(curr_config.results_video_path)

    with open(curr_config.train_config_output_filename, 'rb') as f_in:
        curr_config = pickle.load(f_in)

    # turn off any data augmentation at test time
    curr_config.use_horizontal_flips = False
    curr_config.use_vertical_flips = False
    curr_config.rot_90 = False

    if 'bg' not in curr_config.class_mapping:
        curr_config.class_mapping['bg'] = len(curr_config.class_mapping)

    class_mapping = {v: k for k, v in curr_config.class_mapping.items()}
    print('Class Mappings: ')
    pprint.pprint(class_mapping)

    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    # Save abnormal behaviour classes in class_mapping in config class.
    curr_config.class_mapping = class_mapping

    gt_records_count, ground_truth_data = get_data(curr_config.train_path_file, header=True, ground_truth=True)

    return curr_config, class_to_color, gt_records_count, ground_truth_data


def create_new_image_size(img, curr_config):
    """
    Formats the image size based on config class.
    """
    img_min_side = float(curr_config.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return img, ratio


def create_image_channels(img, curr_config):
    """
    Formats the image channels based on config class.
    """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= curr_config.img_channel_mean[0]
    img[:, :, 1] -= curr_config.img_channel_mean[1]
    img[:, :, 2] -= curr_config.img_channel_mean[2]
    img /= curr_config.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


def create_image(img, curr_config):
    """
    Formats an image for model prediction based on config class.
    """
    new_img, ratio = create_new_image_size(img, curr_config)
    new_new_img = create_image_channels(new_img, curr_config)
    return new_new_img, ratio


def get_original_coordinates(ratio, x1, y1, x2, y2):
    """
    Transform the the bounding box coordinates to its original size.

    :param ratio: get ratio
    :param x1: get xmin coordinate
    :param y1: get ymin coordinate
    :param x2: get xmax coordinate
    :param y2: get ymax coordinate

    :return real_x1: original xmin coordinate
    :return real_y1: original ymin coordinate
    :return real_x2: original xmax coordinate
    :return real_y2: original ymax coordinate
    """
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2


def image_classifier(curr_config):
    """
    Generating image classification.
    This is done by defining the RPN and classifier to built on the vgg 16 layers and using the Model import package to
    to holds the RPN and the classifier, used to load/save weights for the models.
    Then these model will be compiled to get the optimal rpn and classifier model.

    :param curr_config: get the variables from the config class.

    :return model_rpn: optimal rpn model.
    :return model_classifier_only: optimal classifier model.
    """
    num_features = 512

    if K.image_data_format() == 'channels_first':
        input_shape_img = (3, None, None)
        input_shape_features = (num_features, None, None)
    else:
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(curr_config.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network ( VGG network)
    shared_layers = nn.vgg_network(img_input)

    # define the RPN, built on the base layers
    num_anchors = len(curr_config.anchor_box_scales) * len(curr_config.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(feature_map_input, roi_input, curr_config.num_rois,
                               nb_classes=len(curr_config.class_mapping))

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(curr_config.model_path))

    # Load weights
    model_rpn.load_weights(curr_config.model_path, by_name=True)
    model_classifier.load_weights(curr_config.model_path, by_name=True)
    # compile models
    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    return model_rpn, model_classifier_only


def create_boxes_probs(R, F, curr_config, model_classifier_only, bbox_threshold):
    """
    Calculating the bounding boxes probability for each image frame.
    Calculating the probability of the classifier and the regression.
    Getting the maximum probability and it's class name -- this is tge result of the detection.

    :param R: contain the information of the bounding boxes, it's probabilities.
    :param F: the feature maps.
    :param curr_config: get the variables from the config class.
    :param model_classifier_only: probability of the classifier.
    :param bbox_threshold: get bounding box threshold.

    :return bboxes: bounding boxes coordinates for each image frame.
    :return probs: maximum probability that the image belong to specific class(known as label).
    """
    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // curr_config.num_rois + 1):
        ROIs = np.expand_dims(R[curr_config.num_rois * jk:curr_config.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // curr_config.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], curr_config.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):
            """
            Calculate bboxes coordinates on resized image.
            Ignore 'bg' class and Get the class name
            """
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = curr_config.class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= curr_config.classifier_regr_std[0]
                ty /= curr_config.classifier_regr_std[1]
                tw /= curr_config.classifier_regr_std[2]
                th /= curr_config.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regression(x, y, w, h, tx, ty, tw, th)
            except:
                pass

            bboxes[cls_name].append(
                [curr_config.rpn_stride * x, curr_config.rpn_stride * y, curr_config.rpn_stride * (x + w),
                 curr_config.rpn_stride * (y + h)])

            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    return bboxes, probs


def drawAlertMark(img):
    """
    Add alert (triangle with "!" sign) to mark frame with anomaly.
    """
    pts = np.array([[155, 15], [172, 45], [138, 45]])
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (255, 0, 0))
    cv2.putText(img, "!", (152, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def image_show_save(bboxes, probs, new_img, ratio, class_to_color, image_file, max_probability_score):
    """
   Generate the detected results containing the bounding boxes, class name, probability (accuracy score) and
    warning message. The warning message display when the accuracy score is more than 70%.

    :param bboxes: bounding boxes coordinates.
    :param probs: probabilities of the detect image belong to specific class(known as label).
    :param new_img: get the detected image frame.
    :param ratio:  get ratio.
    :param class_to_color: get colour for the bounding boxes as each class(label) represent different colour.
    :param image_file:  directory path for saving the related image.
    :param max_probability_score: max probabilities that are detected in the image frame.

    :return all_data: detected data in a dic format like
    {{image1: {class, probability},{class, probability}},
    ...
    {image10: {class, probability}}
    }
    :return new_img: get the detected result image containing the bounding boxes, class name, accuracy score and
    warning message.
    """
    all_data = []
    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)

        for jk in range(new_boxes.shape[0]):
            prob = int(100 * new_probs[jk])

            (x1, y1, x2, y2) = new_boxes[jk, :]
            (real_x1, real_y1, real_x2, real_y2) = get_original_coordinates(ratio, x1, y1, x2, y2)

            det = {'filepath': image_file, 'x1': real_x1, 'y1': real_y1, 'x2': real_x2, 'y2': real_y2, 'class': key,
                   'prob': prob}
            all_data.append(det)

    all_data.sort(key=lambda x: x['prob'], reverse=True)
    max_class_name = all_data[0]['class']
    max_probability = all_data[0]['prob']

    cv2.rectangle(new_img, (all_data[0]['x1'], all_data[0]['y1']), (all_data[0]['x2'], all_data[0]['y2']),
                  (int(class_to_color[max_class_name][0]), int(class_to_color[max_class_name][1]),
                   int(class_to_color[max_class_name][2])), 2)

    cv2.rectangle(new_img, (8, 5), (190, 50), (255, 255, 255), cv2.FILLED)

    text_label1 = 'Actual: {}: {}%'.format(max_class_name, max_probability)
    cv2.putText(new_img, text_label1, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    k = 0
    for det in list(all_data)[1:4]:
        text_label1 = '{}: {}%'.format(det['class'], det['prob'])
        cv2.putText(new_img, text_label1, (10, k * 15 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        k += 1

    if max_probability > max_probability_score:
        drawAlertMark(new_img)

    return all_data, new_img


def image_testing(filepath, curr_config, model_rpn, model_classifier_only, class_to_color, online=None):
    """
    Get output layer Y1, Y2 from the RPN and the feature maps.
    Y1: y_rpn_cls
    Y2: y_rpn_regr
    Get the feature maps and output from the RPN.
    Generate and save the detected results containing the bounding boxes, class name, probability (accuracy score) and
    warning message.

    :param filepath: get test image frame.
    :param curr_config: get config class.
    :param model_rpn: get optimal rpn model.
    :param model_classifier_only: get optimal classifier model.
    :param class_to_color: get colour for the bounding boxes as each class(label) represent different colour.
    :param online: control to get input test image frame.

    :return all_data: detected data in a dic format like
    {{image1: {class, probability},{class, probability}},
    ...
    {image10: {class, probability}}
    }
    :return new_img: get the detected result image containing the bounding boxes, class name, accuracy score and
    warning message.
    """
    all_data = None
    new_img = None
    tmp_file_path = 'video_frame'
    if online:
        img = filepath
    else:
        img = cv2.imread(filepath)
        tmp_file_path = filepath

    X, ratio = create_image(img, curr_config)

    if K.image_data_format() == 'channels_last':
        X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)

    R = roi_helpers.rpn_to_roi(Y1, Y2, curr_config, K.image_data_format(), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    bboxes, probs = create_boxes_probs(R, F, curr_config, model_classifier_only, curr_config.bbox_threshold)

    # Call show/save
    if bboxes:
        all_data, new_img = image_show_save(bboxes, probs, img, ratio, class_to_color, tmp_file_path, curr_config.max_probability)

    return all_data, new_img


def get_frame(vidcap, sec):
    """
    Capture a frame at every 'sec' seconds.
    :param vidcap: video image frame.
    :param sec: get sec of the frame.

    :return hasFrames and image: getting the frame at every 'sec' seconds.
    """
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()

    return hasFrames, image


def convert_video_frame(input_video_file):
    """
    Convert test videos into every image frame per second.

    :param input_video_file: get the test video.
    :return video_frames: get every image frame per second.
    """
    # Read the video from specified path
    vidcap = cv2.VideoCapture(input_video_file)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Estimated frames per second: {0}, frames: {0}".format(fps, total_frames))

    # frame
    currentframe = 0

    video_frames = []

    # //it will capture image in each second
    frame_rate = .5
    if total_frames > 50:
        frame_rate = 1.5
    sec = 0

    while (True):
        hasFrames, frame = get_frame(vidcap, sec)
        if hasFrames:
            video_frames.append(frame)
            currentframe += 1
            sec += frame_rate
        else:
            break

    # Release all space and windows once done
    vidcap.release()
    cv2.destroyAllWindows()

    print("Totals frames : {0}".format(str(currentframe)))
    return video_frames


def write_video_output(save_file, new_images):
    """
    Converting the detected result images into video. This will be the output for the uploaded video.
    The video is saved in avc1 format.

    :param save_file: directory path for the video saving
    :param new_images: get the detected result images
    """
    video = None
    for i, image in enumerate(new_images):
        if i == 0:
            height, width, _ = image.shape
            video = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*'avc1'), 1.0, (width, height))

        video.write(image)
    video.release()


def main_video_process(curr_config, class_to_color, model_rpn, model_classifier_only, video_file_name):
    """
    Extracting the test video into image frame by calling convert_video_frame method.
    Detection process is called using Faster R-CNN with VGG 16 network.
    Display the detected results as a video output.

    :param curr_config: get config class.
    :param class_to_color: get colour for the bounding boxes as each class(label) represent different colour.
    :param model_rpn: get optimal rpn model.
    :param model_classifier_only: get optimal classifier model.
    :param video_file_name: get test image frame.

    :return test_data: get the dict of detected results.
    :return new_images: get the list of detected result image.
    """
    test_data = {}
    new_images = []
    print('Test video : {}'.format(video_file_name))

    video_frames = convert_video_frame(video_file_name)
    print('Start Image Testing')
    progbar = generic_utils.Progbar(len(video_frames))

    for idx, frame in enumerate(video_frames):
        all_data, new_img = image_testing(frame, curr_config, model_rpn, model_classifier_only, class_to_color,
                                          online=True)
        progbar.update(idx + 1, [('Image Testing Continue.', idx)])
        if all_data:
            new_images.append(new_img)
            for data in all_data:
                class_name = data['class']
                if class_name not in test_data:
                    test_data[class_name] = []

                test_data[class_name].append(data)

    print('End Image Testing')
    return test_data, new_images


def main_video():
    """
    before running the test r-cnn, updating the data augmentation in the config class.
    Executing all the methods above by callin main_video_process method. -- testing process and result detection
    Generating some graph results such as ground truth classes, predicted classes, mAP, Precision and Recall
    calculations by calling plot_graph class.
    """
    print('Load configuration')
    curr_config, class_to_color, gt_records_count, ground_truth_data = load_config()

    print('Load image classification')
    model_rpn, model_classifier_only = image_classifier(curr_config)

    file_list = os.listdir(curr_config.test_file_path)
    for img_idx, video_file_name in enumerate(sorted(file_list)):
        if not video_file_name.lower().endswith(('.mp4', '.mov', '.avi')):
            continue

        st = time.time()
        filepath = os.path.join(curr_config.test_file_path, video_file_name)
        test_data, new_images = main_video_process(curr_config, class_to_color, model_rpn, model_classifier_only, filepath)

        if test_data:
            print('Plot graph')
            main_graph_process(curr_config.result_graphs_file, gt_records_count, ground_truth_data, test_data)

            print('Create video for testing images')
            write_video_output(curr_config.results_video_file.format(video_file_name), new_images)
        else:
            print('Detection has not been found while testing image')
        print('Elapsed time = {}'.format(time.time() - st))
    return


if __name__ == "__main__":
    main_video()
    quit(0)
