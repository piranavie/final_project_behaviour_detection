import os
import cv2
import pandas as pd

from fast_rcnn import config
from fast_rcnn.simple_parser import get_data


def create_csv_file(all_data, save_file):
    """
    Create output csv file to store image path, classname and it's probability.

    :param all_data: get the dataset.
    :param save_file: get the path directory to save.
    """
    test_result_dataset = pd.DataFrame(all_data)
    test_result_dataset.to_csv(save_file, index=0)
    return


def remove_file(filePath):
    """
    Removing files when doing a new test.

    :param filePath: get the file path that need to be deleted.
    """
    try:
        os.ulink(filePath)
    except:
        print("Error while deleting file ", filePath)


def remove_all_files(folder):
    """
    Removing data in folder when doing a new test.
    :param folder: get the folder path that need to be deleted.
    :return:
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete files in %s. Reason: %s' % (file_path, e))
    return


def convert_frame_image_create_annotation(curr_config, temporal_anomaly_annotation):
    """
    Converting videos into image frame and creating a new annotation file containing image path directory,
    class name, xmin, xmax, ymin and y max.

    :param curr_config: get the config class.
    :param temporal_anomaly_annotation:  get the temporal anomaly annotation file.
    """
    output_path = curr_config.train_images_dataset_path
    annotation_data = []

    for anomaly in temporal_anomaly_annotation:
        input_video_file = '{}/{}'.format(curr_config.train_videos_dataset_path, anomaly['filepath'])
        start_frame = anomaly['start_frame1']
        end_frame = anomaly['end_frame1']
        class_name = anomaly['class']
        start_frame_i = anomaly['start_frame2']
        end_frame_i = anomaly['end_frame2']

        basename_without_ext = os.path.basename(input_video_file).split('.', 1)[0]

        # frame
        currentframe = 1

        # Read the video from specified path
        cam = cv2.VideoCapture(input_video_file)
        fps = cam.get(cv2.CAP_PROP_FPS)
        print("Estimated frames per second : {0}".format(fps))

        width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

        # Grab a few frames
        # for i in range(0, num_frames):
        while (True):
            ret, frame = cam.read()

            if ret:
                if currentframe >= start_frame:
                    # if video is still left continue creating images
                    file_name = output_path + '/{}-{}.jpg'.format(basename_without_ext, str(currentframe))

                    # writing the extracted images
                    cv2.imwrite(file_name, frame)

                    annotation_data.append({'filepath': file_name, 'x1': int(1), 'y1': int(1),
                                            'x2': int(width), 'y2': int(height), 'class': class_name})

                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1

                if currentframe > end_frame:
                    break
            else:
                break

        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()

        if start_frame_i != -1 and end_frame_i != -1:
            # Read the video from specified path
            cam = cv2.VideoCapture(input_video_file)
            fps = cam.get(cv2.CAP_PROP_FPS)
            print("Estimated frames per second : {0}".format(fps))

            width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
            height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

            # frame
            currentframe = 1

            # Grab a few frames
            while (True):
                ret, frame = cam.read()

                if ret:
                    if currentframe >= start_frame:
                        # if video is still left continue creating images
                        file_name = output_path + '/{}-{}.jpg'.format(basename_without_ext, str(currentframe))

                        # writing the extracted images
                        cv2.imwrite(file_name, frame)

                        annotation_data.append({'filepath': file_name, 'x1': int(1), 'y1': int(1),
                                                'x2': int(width), 'y2': int(height), 'class': class_name})

                    # increasing counter so that it will
                    # show how many frames are created
                    currentframe += 1

                    if currentframe > end_frame:
                        break
                else:
                    break

            # Release all space and windows once done
            cam.release()
            cv2.destroyAllWindows()

    create_csv_file(annotation_data, curr_config.train_path_file)
    return


def load_config_train():
    """
    Loading the config class and calling the remove_all_files and remove_all_file methods.
    Calling the get_data method from fast_rcnn folder to format the temporal_anomaly_annotation.

    :return curr_config: modified config class.
    :return temporal_anomaly_annotation: modified temporal_anomaly_annotation file.
    """
    curr_config = config.Config()
    # creating a folder named data
    if not os.path.exists(curr_config.train_images_dataset_path):
        os.makedirs(curr_config.train_images_dataset_path)

    remove_all_files(curr_config.train_images_dataset_path)
    remove_file(curr_config.train_path_file)

    temporal_anomaly_annotation = get_data(curr_config.train_videos_annotation_file, temporal_anomaly_data=True)

    return curr_config, temporal_anomaly_annotation


def main():
    curr_config, temporal_anomaly_annotation = load_config_train()
    convert_frame_image_create_annotation(curr_config, temporal_anomaly_annotation)


if __name__ == "__main__":
    main()
    quit(0)
