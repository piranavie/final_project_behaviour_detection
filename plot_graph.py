import matplotlib
import numpy as np
import pandas as pd
import operator
from collections import OrderedDict
import matplotlib.pyplot as plt
from fast_rcnn import data_generators
matplotlib.pyplot.switch_backend('Agg')

def voc_ap(rec, prec):
    """
     This part makes the precision monotonically decreasing
      (goes from the end to the beginning)
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]

    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
    """
    # matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
      (numerical integration)
    """
    # matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def get_rec_prec(true_positive, false_positive, false_negative,ground_truth_records):
    """
     Calculate precision/recall based on true_positive, false_positive, false_negative result.

    :param true_positive: get the true_positive result.
    :param false_positive: get the false_positive result.
    :param false_negative: get the false_negative result.
    :param ground_truth_records: get the ground_truth_records information.

    :return rec: result the recall.
    :return prec: result the precision.
    """
    cumsum = 0
    for idx, val in enumerate(false_positive):
        false_positive[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(true_positive):
        true_positive[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(false_negative):
        false_negative[idx] += cumsum
        cumsum += val

    rec = true_positive[:]
    for idx, val in enumerate(true_positive):
        sum_rec = false_negative[idx] + true_positive[idx]
        if sum_rec == 0:
            continue
        rec[idx] = (float(true_positive[idx]) / sum_rec)

    prec = true_positive[:]
    for idx, val in enumerate(true_positive):
        prec[idx] = float(true_positive[idx]) / (false_positive[idx] + true_positive[idx])

    return rec, prec


def match_ground_truth_box(detection_record, ground_truth_records):
    """
    Search ground_truth_records list and try to find a matching box for the detection box

    :param detection_record: get data with format ['filepath': image_file', 'x1': xmin, 'y1': ymin, 'x2': xmax,
     'y2': ymax', 'prob': score]
    :param ground_truth_records: record list with format
                     [
                      ['filepath': image_file', 'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax', 'used': True],
                      ['filepath': image_file', 'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax', 'used': False],
                      ...
                     ]

    :return max_iou: resulting the max iou
    :return ground_truth_records: resulting the ground_truth_records information
    """

    max_iou = 0.0
    # get predict box coordinate
    detection_box = (detection_record['x1'], detection_record['y1'], detection_record['x2'], detection_record['y2'])

    for i, gt_record in enumerate(ground_truth_records):
        # get ground truth box coordinate
        ground_truth_box = (gt_record['x1'], gt_record['y1'], gt_record['x2'], gt_record['y2'])
        iou = data_generators.iou(ground_truth_box, detection_box)

        # if the ground truth has been assigned to other
        # prediction, we couldn't reuse it
        if iou > max_iou and not gt_record['used']:
            max_iou = iou
            max_index = i + 1
            gt_record['used'] = True

    return max_iou, ground_truth_records


def get_mean_metric(metric_records, ground_truth_classes_records):
    """
    Calculate mean metric, but only count classes which have ground truth object.

    :param metric_records: get the metric dict like:
                metric_records = {'arson':0.77, 'assault':0.60, 'shooting':0.88,..}
    :param ground_truth_classes_records: get ground truth class dict like:
            ground_truth_classes_records = {'class': {
                ['filepath': image_file', 'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax']
                } }

    :return mean_metric: float value of mean metric.
    """
    mean_metric = 0.0
    count = 0
    for (class_name, metric) in metric_records.items():
        if (class_name in ground_truth_classes_records) and (len(ground_truth_classes_records[class_name]) != 0):
            mean_metric += metric
            count += 1
    mean_metric = (mean_metric / count) * 100 if count != 0 else 0.0
    return mean_metric


def calc_AP(rec_prec_plot_file, ground_truth_records, detection_records, class_name, iou_threshold, show_result=None):
    """
    Calculate AP value for one class records

    :param rec_prec_plot_file: recall precision plot file path.
    :param ground_truth_records: ground truth records list for one class, with format:
                     [
                      ['filepath': image_file', 'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax'],
                      ['filepath': image_file', 'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax'],
                      ...
                     ]
    :param detection_records: predict records for one class, with format (in score descending order):
                     [
                      ['filepath': image_file', 'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax', 'prob': score],
                      ['filepath': image_file', 'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax', 'prob': score],
                      ...
                     ]
    :param class_name: name of the abnormal behaviour labels.
    :param iou_threshold: get iou threshold.
    :param show_result: control the graph to show or not instead of saving into folder.

    :return ap: Ap value for the class
    ":return true_positive_count: number of true_positive found
    """

    # append usage flag in gt_records for matching gt search
    for gt_record in ground_truth_records:
        gt_record['used'] = False

    # init true_positive and false_positive list
    nd = len(detection_records)  # number of predict data
    true_positive = [0] * nd
    false_positive = [0] * nd
    false_negative = [0] * nd
    true_positive_count = 0

    # assign predictions to ground truth objects
    for idx, detection_record in enumerate(detection_records):
        '''
        matching ground_truth_records index.
        > 0 there's matching ground truth with > iou_threshold
        == 0 there's no matching ground truth
        -1 there's poor matching ground truth with < iou_threshold
        '''
        max_iou, ground_truth_records = match_ground_truth_box(detection_record, ground_truth_records)
        if max_iou >= iou_threshold:
            # find a valid gt obj to assign, set
            true_positive[idx] = 1
            true_positive_count += 1

        elif max_iou == 0:
            false_negative[idx] = 1
            # false_positive[idx] = 1

        else:
            false_positive[idx] = 1

    # compute precision/recall
    rec, prec = get_rec_prec(true_positive, false_positive, false_negative, ground_truth_records)
    ap, mrec, mprec = voc_ap(rec, prec)

    # draw_rec_prec(rec_prec_plot_file, rec, prec, mrec, mprec, class_name, ap, show_result)

    return ap, true_positive_count


# DRAW GRAPH
def draw_rec_prec(rec_prec_plot_file, rec, prec, mrec, mprec, class_name, ap, show_result=None):
    """
     Draw graph for recall and precision
    """
    plt.plot(rec, prec, '-o')
    # add a new penultimate point to the list (mrec[-2], 0.0)
    # since the last line segment (and respective area) do not affect the AP value
    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')

    # set window title
    fig = plt.gcf()  # gcf - get current figure
    fig.canvas.set_window_title('AP ' + class_name)

    # set plot title
    plt.title('class: ' + class_name + ' AP = {}%'.format(ap * 100))

    # plt.suptitle('This is a somewhat long figure title', fontsize=16)
    # set axis titles
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # optional - set axes
    axes = plt.gca()  # gca - get current axes
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])  # .05 to give some extra space

    # Alternative option -> wait for button to be pressed
    # while not plt.waitforbuttonpress(): pass # wait for key display
    # Alternative option -> normal display

    if show_result:
        plt.show()
    else:
        # save the plot
        fig.savefig(rec_prec_plot_file.format(class_name))

    plt.cla()  # clear axes for next plot
    return


def compute_mAP_draw_graph(result_graphs_file, annotation_records_len, ground_truth_classes_records,
                           detection_classes_records, iou_threshold=0.5, show_result=None):
    """
    Compute PascalVOC style mAP and getting the mAP percentage value

    :param result_graphs_file: plot result graph path.
    :param annotation_records_len: get the length of annotation data.
    :param ground_truth_classes_records: get data with format like detection_classes_records = {'class': {
                            ['filepath': image_file', 'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax', 'prob': score]
                        } }
    :param detection_classes_records: get data with format like detection_classes_records = {'class': {
                            ['filepath': image_file', 'x1': xmin, 'y1': ymin, 'x2': xmax, 'y2': ymax']
                        } }
    :param iou_threshold: get iou threshold.
    :param show_result: control the graph to show or not instead of saving into folder.
    """
    APs = {}
    count_true_positives = {class_name: 0 for class_name in list(ground_truth_classes_records.keys())}

    # get AP value for each of the ground truth classes
    for class_name in ground_truth_classes_records:

        gt_records = ground_truth_classes_records[class_name]

        # if we didn't detect any obj for a class, record 0
        if class_name not in detection_classes_records:
            APs[class_name] = 0.
            continue

        pred_records = detection_classes_records[class_name]
        pred_records.sort(key=lambda x: x['prob'], reverse=True)

        ap, true_positive_count = calc_AP(result_graphs_file.format('precision-recall-' + class_name), gt_records,
                                          pred_records, class_name, iou_threshold, show_result)
        APs[class_name] = ap
        count_true_positives[class_name] = true_positive_count

    # sort AP result by value, in descending order
    APs = OrderedDict(sorted(APs.items(), key=operator.itemgetter(1), reverse=True))

    # get mAP percentage value
    # mAP = np.mean(list(APs.values()))*100
    mAP = get_mean_metric(APs, ground_truth_classes_records)

    # get GroundTruth count per class
    gt_counter_per_class = {}
    for (class_name, info_list) in ground_truth_classes_records.items():
        gt_counter_per_class[class_name] = len(info_list)

    # get Precision count per class
    pred_counter_per_class = {class_name: 0 for class_name in list(ground_truth_classes_records.keys())}
    for (class_name, info_list) in detection_classes_records.items():
        pred_counter_per_class[class_name] = len(info_list)

    # get the precision & recall
    precision_dict = {}
    recall_dict = {}
    for (class_name, gt_count) in gt_counter_per_class.items():
        if (class_name not in pred_counter_per_class) or (class_name not in count_true_positives) or \
                pred_counter_per_class[class_name] == 0:
            precision_dict[class_name] = 0.
        else:
            precision_dict[class_name] = float(count_true_positives[class_name]) / pred_counter_per_class[class_name]

        if class_name not in count_true_positives or gt_count == 0:
            recall_dict[class_name] = 0.
        else:
            recall_dict[class_name] = float(count_true_positives[class_name]) / gt_count

    # get mPrec, mRec
    # mPrec = np.mean(list(precision_dict.values()))*100
    # mRec = np.mean(list(recall_dict.values()))*100
    mPrec = get_mean_metric(precision_dict, ground_truth_classes_records)
    mRec = get_mean_metric(recall_dict, ground_truth_classes_records)

    if show_result:

        # show result
        print('\nPascal VOC AP evaluation')
        for (class_name, AP) in APs.items():
            print('%s: AP %.4f, precision %.4f, recall %.4f' % (
                class_name, AP, precision_dict[class_name], recall_dict[class_name]))
        print('mAP@IoU=%.2f result: %f' % (iou_threshold, mAP))
        print('mPrec@IoU=%.2f result: %f' % (iou_threshold, mPrec))
        print('mRec@IoU=%.2f result: %f' % (iou_threshold, mRec))

    plot_Pascal_AP_result(result_graphs_file, annotation_records_len, count_true_positives,
                          len(ground_truth_classes_records),
                          gt_counter_per_class, pred_counter_per_class, precision_dict, recall_dict, mPrec, mRec,
                          APs, mAP, iou_threshold, show_result)

    # return mAP percentage value
    return


def plot_Pascal_AP_result(result_graphs_file, count_images, count_true_positives, num_classes,
                          gt_counter_per_class, pred_counter_per_class,
                          precision_dict, recall_dict, mPrec, mRec,
                          APs, mAP, iou_threshold, to_show):
    '''
     Plot the total number of occurrences of each class in the ground-truth
    '''
    window_title = "Ground-Truth Info"
    plot_title = "Ground-Truth\n" + "(" + str(count_images) + " files and " + str(num_classes) + " classes)"
    x_label = "Number of objects per class"
    output_path = result_graphs_file.format('Ground-Truth_Info')
    draw_plot_func(gt_counter_per_class, num_classes, window_title, plot_title, x_label, output_path,
                   plot_color='forestgreen', show_result=to_show)

    '''
     Plot the total number of occurrences of each class in the "predicted" folder
    '''
    window_title = "Predicted Objects Info"
    # Plot title
    plot_title = "Predicted Objects\n" + "(" + str(count_images) + " files and "
    count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(pred_counter_per_class.values()))
    plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
    # end Plot title
    x_label = "Number of objects per class"
    output_path = result_graphs_file.format('Predicted_Objects_Info')
    draw_plot_func(pred_counter_per_class, len(pred_counter_per_class), window_title, plot_title, x_label, output_path,
                   plot_color='forestgreen', true_p_bar=count_true_positives, show_result=to_show)

    '''
     Draw mAP plot (Show AP's of all classes in decreasing order)
    '''
    window_title = "mAP"
    plot_title = "mAP@IoU={0}: {1:.2f}%".format(iou_threshold, mAP)
    x_label = "Average Precision"
    output_path = result_graphs_file.format('mAP')
    draw_plot_func(APs, num_classes, window_title, plot_title, x_label, output_path, plot_color='royalblue',
                   show_result=to_show)

    '''
     Draw Precision plot (Show Precision of all classes in decreasing order)
    '''
    window_title = "Precision"
    plot_title = "mPrec@IoU={0}: {1:.2f}%".format(iou_threshold, mPrec)
    x_label = "Precision rate"
    output_path = result_graphs_file.format('Precision')
    draw_plot_func(precision_dict, len(precision_dict), window_title, plot_title, x_label, output_path,
                   plot_color='royalblue', show_result=to_show)

    '''
     Draw Recall plot (Show Recall of all classes in decreasing order)
    '''
    window_title = "Recall"
    plot_title = "mRec@IoU={0}: {1:.2f}%".format(iou_threshold, mRec)
    x_label = "Recall rate"
    output_path = result_graphs_file.format('Recall')
    draw_plot_func(recall_dict, len(recall_dict), window_title, plot_title, x_label, output_path,
                   plot_color='royalblue', show_result=to_show)


def adjust_axes(r, t, fig, axes):
    """
     Plot - adjust axes
    """
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, plot_color, true_p_bar=None,
                   show_result=None):
    """
     Draw plot using Matplotlib
    """
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar:
        """
         Special case to draw in (green=true predictions) & (red=false predictions)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions',
                 left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            #   first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)  # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    # fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15  # in percentage of the figure height
    bottom_margin = 0.05  # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()

    # show image
    if show_result:
        plt.show()
    else:
        # save the plot
        fig.savefig(output_path)

    # close the plot
    plt.close()


def train_process_graph(train_process_data_file):
    """
    Getting the training data and drawing a graph of loss rpn classifier, loss rpn regression, loss class classifier,
    loss class regression, total_loss and class_accuracy for each number of Epochs.

    :param train_process_data_file: get data from csv file containing information about loss rpn classifier, loss rpn regression,
    loss class classifier, loss class regression, total_loss and class_accuracy.
    """
    record_dataset = pd.read_csv(train_process_data_file)

    record_epochs = len(record_dataset)
    print(record_epochs)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, record_epochs), record_dataset['loss_rpn_cls'], 'r')
    plt.title('loss_rpn_cls')
    plt.xlabel('number of Epochs')
    plt.ylabel('loss_rpn_cls')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, record_epochs), record_dataset['loss_rpn_regr'], 'r')
    plt.title('loss_rpn_regr')
    plt.xlabel('number of Epochs')
    plt.ylabel('loss_rpn_regr')
    plt.savefig("./data/result_graphs/loss_rpn_regr_and_loss_rpn_cls")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, record_epochs), record_dataset['loss_class_cls'], 'r')
    plt.title('loss_class_cls')
    plt.xlabel('number of Epochs')
    plt.ylabel('loss_class_cls')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, record_epochs), record_dataset['loss_class_regr'], 'r')
    plt.title('loss_class_regr')
    plt.xlabel('number of Epochs')
    plt.ylabel('loss_class_regr')
    plt.savefig("./data/result_graphs/loss_class_regr_and_loss_class_cls")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, record_epochs), record_dataset['curr_loss'], 'r')
    plt.title('total_loss')
    plt.xlabel('number of Epochs')
    plt.ylabel('total_loss')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, record_epochs), record_dataset['class_acc'], 'r')
    plt.title('class_acc')
    plt.xlabel('number of Epochs')
    plt.ylabel('class_acc')
    plt.savefig("./data/result_graphs/total_loss_and_class_acc")


def main_graph_process(save_graph_file, gt_records_count, ground_truth_data, detection_data):
    """
    Executing this class to calculate and drawing graph for map, recall and precision.

    :param save_graph_file: get path directory to save result graphs.
    :param gt_records_count: get number of ground truth records.
    :param ground_truth_data: get ground truth data.
    :param detection_data: get detection data,
    """
    compute_mAP_draw_graph(save_graph_file, gt_records_count, ground_truth_data, detection_data)
    return
