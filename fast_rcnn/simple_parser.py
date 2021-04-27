import cv2


def get_data(input_path, header=None, ground_truth=None, temporal_anomaly_data=None):
    """
    Parse the data from annotation file for training and testing process.

    :param input_path: get the input path.
    :param header:  get the header of the file.
    :param ground_truth: get the ground-truth.
    :param temporal_anomaly_data:  get the temporal anomaly data file.

    :return all_data: Parse data format.
    :return classes_count: total number of classes.
    :return class_mapping: abnormal behaviour classes.
    """
    found_bg = False
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    ground_truth_detection_data = {}
    records_count = 0
    temporal_anomaly_annotation = []

    with open(input_path, 'r') as f:

        # print('Parsing annotation files as .txt: {}'.format(input_path))

        for line in f:
            line_split = line.replace('"', '').strip().split(',')
            if header:
                header = None
                continue
            records_count += 1

            if temporal_anomaly_data:
                (filename, class_name, start_frame1, end_frame1, start_frame2, end_frame2) = line_split
                if int(start_frame1) == -1 or int(end_frame1) == -1:
                    continue
                temporal_anomaly_annotation.append(
                    {'filepath': filename, 'class': class_name, 'start_frame1': int(start_frame1),
                     'end_frame1': int(end_frame1), 'start_frame2': int(start_frame2), 'end_frame2': int(end_frame2)})
                continue
            else:
                (filename, x1, y1, x2, y2, class_name) = line_split

                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if class_name not in class_mapping:
                    if class_name == 'bg' and found_bg == False:
                        print('Found class name with special name bg. Will be treated as a background region (known as hard negative mining).')
                        found_bg = True
                    class_mapping[class_name] = len(class_mapping)

                if filename not in all_imgs:
                    all_imgs[filename] = {}

                    img = cv2.imread(filename)
                    (rows, cols) = img.shape[:2]
                    all_imgs[filename]['filepath'] = filename
                    all_imgs[filename]['width'] = cols
                    all_imgs[filename]['height'] = rows
                    all_imgs[filename]['bboxes'] = []

                all_imgs[filename]['bboxes'].append(
                    {'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

                if ground_truth:
                    if class_name not in ground_truth_detection_data:
                        ground_truth_detection_data[class_name] = []

                    ground_truth_detection_data[class_name].append(
                        {'filepath': filename, 'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)})

        if temporal_anomaly_data:
            return temporal_anomaly_annotation

        if ground_truth:
            return records_count, ground_truth_detection_data

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping
