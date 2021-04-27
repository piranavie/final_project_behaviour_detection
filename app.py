"""
This module is a program that simulates a remote controlled
user-interfaced web server.
"""
import os
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from werkzeug.utils import secure_filename
from format_dataset import remove_all_files
from test_frcnn import load_config, image_classifier
from test_online_frcnn import main_online_output_process, camera_stream, camera_start, camera_stop

# Allowed video file format
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi"}

file_uploaded_msg = {"message": ""}
tab_no = {"tab": ""}

app = Flask(__name__, static_folder='static')

# To make flask point to our current app
app.app_context().push()

# Max video file size can be inputted
app.config['MAX_CONTENT_PATH'] = 70 * 1024 * 1024
global curr_config, class_to_color, gt_records_count, ground_truth_data, model_rpn, model_classifier_only


def remove_file():
    """
    Removing results data every time uploading a new video.
    """
    remove_all_files(curr_config.result_online_graphs_path)
    remove_all_files(curr_config.result_online_videos_path)


if os.environ.get("WERKZEUG_RUN_MAIN") == 'true':
    print('Loading configuration -----wait for few seconds')
    curr_config, class_to_color, gt_records_count, ground_truth_data = load_config()
    model_rpn, model_classifier_only = image_classifier(curr_config)

    if not os.path.exists(curr_config.result_online_graphs_path):
        os.makedirs(curr_config.result_online_graphs_path)

    if not os.path.exists(curr_config.result_online_videos_path):
        os.makedirs(curr_config.result_online_videos_path)
    remove_file()


def allowed_file(filename):
    """
    :param filename: video path name
    :return: checking and allowing the video file that is "mp4", "mov" or "avi" formats.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def root():
    """
    Redirecting back to the homepage.
    """
    return redirect(url_for('home'))


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/home', methods=['GET'])
def home():
    return render_template("homepage.html", MESSAGE=file_uploaded_msg, TAB_NO=tab_no["tab"])


@app.route('/uploadFootage', methods=['POST'])
def get_upload_footage():
    """
    Get the uploaded video file from the server and start processing the detection methods.
    The detection and plotting graph is done by calling main_online_output_process.
    Allowing the video file that is "mp4", "mov" or "avi" formats.
    Allowing the video file size that is less than or equal to 70MB.
    """
    remove_file()
    tab_no.update({"tab": "tab1"})
    if request.method == 'POST':
        file = request.files['file']

        if "filesize" in request.cookies:

            if file and allowed_file(file.filename) and int(request.cookies["filesize"]) <= app.config["MAX_CONTENT_PATH"]:
                file_stream = secure_filename(file.filename)
                file.save(os.path.join(curr_config.test_file_path, file_stream))
                file_uploaded_msg.update({"message": "file uploaded successfully"})

                result = main_online_output_process(os.path.join(curr_config.test_file_path, file_stream), curr_config,
                                                    class_to_color, gt_records_count, ground_truth_data, model_rpn,
                                                    model_classifier_only)
                if not result:
                    file_uploaded_msg.update({"message": "Detection has not been found while testing image"})

            else:
                file_uploaded_msg.update({
                    "message": "ERROR: Video must be no larger than 70Mb and be in the following format: mp4,mov or avi format"})

        return redirect(url_for('home'))


@app.route('/cameraOperation', methods=['POST'])
def camera_operation():
    '''
    Setting up a live camera.
    If user click start button - allows to switch on the webcam.
    If user click stop button - allows to switch off the webcam.
    :return 'ok',200: successfully recognise the user action of controlling the webcam
    (live streaming detection).
    '''
    content = request.json
    if content is None:
        return jsonify({"message": "text not found"})

    tab_no.update({"tab": "tab0"})
    if content['operation'] == 'start':
        camera_start()
    elif content['operation'] == 'stop':
        camera_stop()
    else:
        return jsonify({"message": "operation not yet defined"})

    return 'OK', 200


def gen_frame():
    """
    Video streaming generator function.
    :return frame: get the frame from the webcam.
    """
    frame = camera_stream(curr_config, model_rpn, model_classifier_only, class_to_color)
    return frame


@app.route('/video_feed', methods=['GET'])
def video_feed():
    """
    :return: displaying the results of the detection on the live streaming detection.
    """
    tab_no.update({"tab": "tab0"})
    response = make_response(gen_frame())
    response.headers.set('Content-Type', 'image/jpeg')
    return response


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
    app.config['SESSION_TYPE'] = 'filesystem'
