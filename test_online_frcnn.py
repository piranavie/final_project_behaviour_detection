"""
---- video uploading detection-------
Once they upload a video, it starts the detection process by handling video upload on the local server using POST
method in the flask as a server-end point. Then, it will extract the input video into one frame per second, which will
be stored in a test image folder. These image frames will be used to process the testing using Faster R-CNN on
VGG 16 network (Faster R-CNN on VGG 16 network.
After the Detection process is done, displaying the detected results as a video output. Also, displaying graph results such as
ground truth classes, predicted classes, mAP, Precision and Recall calculations.

---- live streaming detection-------
The live detection done by getting an image frame for every 3 seconds from backend end-point services
(from flask API via call HTTP API (using AJAX in browser). These image frames will be used to process the testing
using Faster R-CNN on VGG 16 network.
After the Detection process is done, displaying the output the bounding boxes, name of the crime that occurs in the
image frame, accuracy score and alert message as a warning sign to alert the user that there is a suspicious person
is found.
"""
import time
import cv2

from test_frcnn import main_video_process, write_video_output, load_config, image_classifier, image_testing
from plot_graph import main_graph_process

video_capture = None


def main_online_output_process(video_file_name, curr_config, class_to_color, gt_records_count, ground_truth_data,
                               model_rpn, model_classifier_only):
    """
    video uploading detection --> called when the sever is up
    """
    st = time.time()
    test_data, new_images = main_video_process(curr_config, class_to_color, model_rpn, model_classifier_only,
                                               video_file_name)

    if test_data:
        write_video_output(curr_config.result_online_video_file, new_images)
        print('Elapsed time = {}'.format(time.time() - st))

        print('Plot graph')
        main_graph_process(curr_config.result_online_graphs_file, gt_records_count, ground_truth_data, test_data)

    else:
        print('Detection has not been found while testing image')
        print('Elapsed time = {}'.format(time.time() - st))
        return False
    return True


def camera_start():
    """
    Video capture is process when the webcam is on.
    """
    global video_capture
    video_capture = cv2.VideoCapture(0)


def camera_stop():
    """
    Stop Video capture when the webcam is off.
    """
    video_capture.release()


def camera_stream(curr_config, model_rpn, model_classifier_only, class_to_color):
    """
    live streaming detection --> called when the sever is up
    """
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if ret:
        _, new_img = image_testing(frame, curr_config, model_rpn, model_classifier_only, class_to_color, online=True)
        if new_img is not None:
            # Display the resulting frame in browser
            return cv2.imencode('.jpg', new_img)[1].tobytes()

    return cv2.imencode('.jpg', frame)[1].tobytes()


if __name__ == "__main__":
    # Test run using an example video
    file_path = 'data/test_online_images/Abuse028_x264.mp4'
    print('Load configuration')
    curr_config, class_to_color, gt_records_count, ground_truth_data = load_config()

    print('Load image classification')
    model_rpn, model_classifier_only = image_classifier(curr_config)

    # filename = secure_filename(file_path)
    main_online_output_process(file_path, curr_config, class_to_color, gt_records_count, ground_truth_data, model_rpn,
                               model_classifier_only)
    quit(0)
