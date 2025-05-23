import shutil

import qrcode
import json
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory, render_template, send_file, redirect, url_for

import re
import os
from werkzeug.utils import secure_filename
import librosa  # You might need to install this library for handling audio files
from datetime import date
import time
from sound_processing import combine_wav_files
from clear_folder import clear_folder
from extract_features import extact_features_from_file

from datetime import datetime

import random

app = Flask(__name__)

UPLOAD_FOLDER = 'records'
ALLOWED_EXTENSIONS = {'wav', 'mp3', '3gp', 'aac', 'flac'}
SETUP_TIME_ALLOWANCE = 0
extract_features_flag = True

processing_started = 0

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class Record:

    def __init__(self, id, save_folder = UPLOAD_FOLDER, model = '', brand = ''):
        self.id = id
        self.chunks = []
        self.chunk_quality = []
        self.successful = False
        self.files = []
        self.save_folder = save_folder
        self.tmp_folder = f"{self.save_folder}/TMP{self.id}"
        self.sound_folder = f"{self.save_folder}/{self.id}"
        self.model = model
        self.brand = brand
        self.active_point = 0
        self.pointRecordProcessId = "0"
        self.requestsInProcess = 0
        self.recordState = [False for i in range(10)]
        self.processing_tmp_files = False
        self.numChunksRequired = 3

    def get_recorded_point_numbers(self):
        self.recordState = [False for i in range(10)]
        for filename in os.listdir(self.sound_folder):
            match = re.search(r"(\d+)\.wav$", filename)
            if match:
                pointNo = int(match.group(1))
                recordStateIndex =  pointNo - 1
                if recordStateIndex not in list(range(len(self.recordState))):
                    raise Warning(f"Point number extracted is {pointNo }, value should ")
                self.recordState[recordStateIndex] = True # Returns the point number as an integer
        return self.recordState

    def get_pointRecordProcessId(self):
        self.pointRecordProcessId = generate_ID(4)
        return self.pointRecordProcessId

    def reset_pointRecordProcessId(self):
        self.pointRecordProcessId = "0"
    def point_data_reset(self):
        self.chunks = []
        self.chunk_quality = []
        self.active_point = 0
        clear_folder(self.tmp_folder)
        print(f"Point data cleared , active_point {self.active_point}")


    def get_filename(self, point_number, sep = 'point'):
        return f"{self.id}{sep}{point_number}.wav"

    def get_good_subseq_ind(self, subs_len):
        ones = ''.join(['1' for i in range(subs_len)])
        start = self.quality_string().find(ones)
        end = start + subs_len
        return start, end
    def num_chunks(self):
        return len(self.chunk_quality)
    def quality_string(self):
        return "".join(self.chunk_quality)
    def check_sucsess(self):
        crit = "".join(['1' for i in range(self.numChunksRequired)])
        self.successful = crit in self.quality_string()
        return self.successful

    def combine_wav_from_tmp(self, point_number, filename):
        subs_len = self.num_chunks()
        while self.get_good_subseq_ind(subs_len)[0] < 0:
            subs_len -= 1
        start, end = self.get_good_subseq_ind(subs_len)
        files_to_combine = self.chunks[start: end]
        for tmpFile in files_to_combine:
            if tmpFile.split('/')[-1] not in os.listdir(self.tmp_folder):
                self.point_data_reset()
                print(f"ERROR: file not found in the directory {tmpFile.split('/')[-1]}")
                return False
        # filename = self.get_filename(point_number)
        save_file_path = os.path.join(self.sound_folder, filename)
        self.files.append((save_file_path))
        combine_wav_files(save_file_path, files_to_combine)

        # Update using point number as the key
        self.update_labels({str(point_number): "No Label"}, mode="add")

        self.point_data_reset()
        return filename

    def save_metadata(self, string):
        file_path = f"{self.sound_folder}/meta.txt"
        with open (file_path, "w") as file:
            file.write(string)

    def update_metadata(self, dict):

        file_path = f"{self.sound_folder}/meta.json"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
        else:
            data = {}

        entries_to_pop = []
        for entry in dict:
            if entry in data:
                data[entry] = dict[entry]
                entries_to_pop.append(entry)
        for entry in entries_to_pop:
                dict.pop(entry)

        data.update(dict)
        with open(file_path, 'w') as file:
            json.dump(data, file)

    def get_labels(self):

        file_path = f"{self.sound_folder}/labels.json"
        if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    data = json.load(file)
                return data
        else:
            return False

    def update_labels(self, dict, mode="replace"):
        file_path = f"{self.sound_folder}/labels.json"

        if mode == "add":
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    data = json.load(file)
            else:
                data = {}

            entries_to_pop = []
            for entry in dict:
                if entry in data:
                    data[entry] = dict[entry]
                    entries_to_pop.append(entry)
            for entry in entries_to_pop:
                dict.pop(entry)

            data.update(dict)
        else:
            data = dict

        with open(file_path, 'w') as file:
            json.dump(data, file)


    def destroy(self):
        folders = [self.tmp_folder, self.sound_folder]
        for folder in folders:
            if os.path.exists(folder):
                print(f"Clearing folder {folder}")
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        # Check if it's a file and then remove it
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.remove(file_path)
                            print(f"Deleted {file_path}")
                        elif os.path.isdir(file_path):
                            # Optionally remove directories as well
                            # Use os.rmdir(file_path) for empty directories
                            # Or use a recursive delete function if directories contain files
                            print(f"Skipping directory {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
                os.rmdir(folder)

def get_all_records_with_meta(upload_folder = UPLOAD_FOLDER):
    records = []
    folders = [item for item in os.listdir(upload_folder)]
    for folder in folders:
        if folder[:3] == "TMP" or folder[0] == ".":
            continue
        meta = os.path.join(upload_folder, folder, "meta.json")
        with open(meta) as file:
            metadata = json.load(file)
        metadata["ID"] = folder
        records.append(metadata)
    return records

print(get_all_records_with_meta(upload_folder = UPLOAD_FOLDER))

record_dict = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/extract_features_off', methods=['GET'])  # Change to POST to accept data
def extract_features_off():
    global extract_features_flag
    extract_features_flag = False
    print("Feature extracting turned off")
    return jsonify("OK")


@app.route('/getUniqueId', methods=['POST'])
def get_unique_id():
    try:
        # Parse request data
        data = request.get_json()

        # Extract ID sent from the device
        unique_id = data.get('id')
        if not unique_id:
            return jsonify({"error": "No ID provided in request"}), 400

        records_data = get_all_records_with_meta(UPLOAD_FOLDER)
        if unique_id in [record_data['ID'] for record_data in records_data]:
            print(f"\nID is already on the server: {unique_id}")
            # Return the same ID back
            return jsonify(unique_id)

        # Extract device information
        maker = data.get('maker', 'Unknown')
        model = data.get('model', 'Unknown')
        device_id = data.get('deviceId', 'Unknown')

        # Get the number of chunks from query params
        num_chunks = int(request.args.get("numChunks", 10))

        # Log the received information
        print(f"Received ID registration: {unique_id}")
        print(f"Device info - Maker: {maker}, Model: {model}, DeviceID: {device_id}")

        # Create metadata for the record
        metadata = {
            "Date": str(date.today()),
            "Maker": maker,
            "Model": model,
            "DeviceID": device_id,
            "Comment": ""
        }

        # Create and initialize the record with the client-provided ID
        record = Record(unique_id)  # Use the ID from the client instead of generating one
        record.numChunksRequired = num_chunks
        print(f"Required length of good record set to {record.numChunksRequired}")

        # Store the record and create necessary directories
        record_dict[unique_id] = record
        os.makedirs(record.sound_folder, exist_ok=True)
        record.update_metadata(metadata)

        print(f"Registered client ID: {unique_id}")

        # Return the same ID back to confirm registration
        return jsonify(unique_id)

    except Exception as e:
        print(f"Error in get_unique_id: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/checkConnection', methods=['GET'])  # Change to POST to accept data
def connectionOK():
    return jsonify("OK")

@app.route('/start_point_recording')
def start_point_recording():
    ID = request.args.get('record_id').strip().strip('"')
    if ID not in record_dict:
        print("Error: referring to non-existant ID")
        return jsonify({"error": "No record created for ID"}), 400
    else:
        record = record_dict[ID]
        print(f"Record exists, tmp folder {record.tmp_folder}")
    pointID = record.get_pointRecordProcessId()
    print(f"ID for the point recording issued: {pointID}")
    return jsonify({"pointRecordId" : pointID})

@app.route('/upload_file', methods=['POST'])
def upload_recorded_file():

    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 401

    file = request.files.get('file')
    button_number = int(request.form.get('button_number'))
    filename = request.form.get('fileName')
    ID = request.form.get('record_id').strip().strip('"')
    pointID = request.form.get('pointRecordId').strip().strip('"')
    print(f"Save request received ID {ID} button {button_number} pointID {pointID}")
    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        # filename = f"{pointID}{button_number}"
        save_path = os.path.join(record_dict[ID].sound_folder, filename)
        file.save(save_path)
        print("File saved")
    return jsonify({"message" : "OK"}), 200

@app.route('/upload', methods=['POST'])
def upload_file():

    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 401

    file = request.files.get('file')
    button_number = int(request.form.get('button_number'))
    ID = request.form.get('record_id').strip().strip('"')
    pointID = request.form.get('pointRecordId').strip().strip('"')

    if ID not in record_dict:
        print("Error: referring to non-existant ID")
        return jsonify({"error": "No record created for ID"}), 402
    else:
        record = record_dict[ID]
    record.requestsInProcess += 1

    os.makedirs(record.tmp_folder, exist_ok=True)

    if pointID != record.pointRecordProcessId:
        print(f"ID should be {record.pointRecordProcessId} received {pointID}")
        return jsonify({"error": "Wrong point record ID"}), 409
    else:
        print(f"Continuing with button No {button_number} point record ID {pointID}")


    record_quality = '1' if record.num_chunks() >= SETUP_TIME_ALLOWANCE else '0'
    print(f"request received ID {ID} button {button_number} response {record_quality}")
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':

        print("error: No selected file")
        return jsonify({"error": "No selected file"}), 403
    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        filename = f"{ID}No{len(record.chunks)}"
        save_path = os.path.join(record_dict[ID].tmp_folder, filename + ".wav")
        file.save(save_path)
        print("Temporary file saved")

        #******* Surfboard feature extraction *******
        if extract_features_flag:
            print("Extracting features with surfboard")
            try:
                # extact_features_from_file(os.path.join(records[ID].tmp_folder, filename))
                print("Features extracted")
            except Exception as er:
                print(f"Surfboard failed, {er}")

        record.chunks.append(save_path)
        print(f"Chunk saved {save_path} , num_chunks {record.num_chunks()}")
        record.chunk_quality.append(record_quality)
        record.requestsInProcess -= 1
        response_message = "Record sucsessfull" if record.check_sucsess() else "Continue"
        print(f"Returning response {response_message}")
        return jsonify({"message": response_message, "filename": filename})
    else:
        record.requestsInProcess -= 1
        return jsonify({"error": "File type not allowed"}), 404

@app.route('/save_record', methods=['POST'])
def save_record():
    message = "Undefined"
    ID = request.form.get('record_id').strip().strip('"')
    if ID not in record_dict:
        return jsonify({"error": "Record not found"}), 400
    record = record_dict[ID]
    record.reset_pointRecordProcessId()
    result = request.form.get('result').strip().strip('"')
    filename = request.form.get('filename').strip().strip('"')
    print(f"\n****************\nSaving request result {result} filename {filename}")
    if result == "success":
        message = "Point recording completed. "
        button_number = int(request.form.get('button_number'))
        print(f"\n****************\nSaving record to the database, ID {ID} point {button_number}")
        if record.processing_tmp_files:
            print("******* DOUBLE CALL ********")
            return jsonify(message), 200
        if (button_number > 0):
            record.processing_tmp_files = True
            res = record.combine_wav_from_tmp(button_number, filename)
            if res:
                filename = filename.split('.')[0]
                record.recordState[button_number - 1] = True
                message += f"Record saved successfully file name {filename}"
            else:
                print("Failed to save record")
                message += "Failed to save"
            record.processing_tmp_files = False
        else:
            message = "Wrong button number, record not saved"
    elif result == "abort":
        message = "Record aborted"
    elif result == "timeout":
        message = "Record unsuccessfull"
    record.point_data_reset()
    print(f"Recording activity completed with the result {message}")
    return jsonify({"message": message, "filename": filename}), 200


@app.route('/get_wav_files', methods=['GET'])
def get_wav_files():
    ID = request.args.get('folderId', default='default_folder', type=str).strip().strip('"')
    record = record_dict[ID]
    sound_folder = record.sound_folder

    # Get files
    files = [f for f in os.listdir(sound_folder) if f.endswith('.wav')]
    file_names = " ".join([f[:-4] for f in files])

    # Get labels
    file_labels = record.get_labels()

    return jsonify({'files': file_names, 'labels': file_labels})

@app.route('/get_full_path_to_id/<ID>', methods=['GET'])
def get_full_path_to_id(ID):
    # ID = request.args.get('folderId', default='default_folder', type=str).strip().strip('"')
    # record = records[ID.strip().strip('"')]
    http_address =  'http://' + request.host + '/show_wav_files?folderId=' + ID.strip().strip('"')

    img = qrcode.make(http_address)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr)
    img_byte_arr.seek(0)

    # Send the bytes stream as a file response with MIME type for PNG images
    return send_file(img_byte_arr, mimetype='image/png')


@app.route('/show_wav_files')
def show_wav_files():
    folder = request.args.get('folderId', default='default_folder', type=str).strip().strip('"')
    upload_folder = request.args.get('upload_folder', UPLOAD_FOLDER)
    full_path = os.path.join(upload_folder, folder)
    wav_files = [f[:-4] for f in os.listdir(full_path) if f.endswith('.wav')]
    metafile = os.path.join(full_path, 'meta.json')
    labelfile = os.path.join(full_path, 'labels.json')
    if not os.path.exists(labelfile):
        create_label_file(full_path)

    with open(metafile) as file:
        metadata = json.load(file)
        comment = metadata.get("Comment", "")
        date = metadata.get("Date", "")

    with open(labelfile) as file:
        labels = json.load(file)

    route_to_file = 'http://' + request.host + '/file_download' + '?folderId=' + folder + '&fileName='
    print(f"Fetching files with request {route_to_file}")

    return render_template('list_wav.html', wav_files=wav_files, folderId=folder, route_to_file=route_to_file,
                           comment=comment, date=date, labels=labels)


@app.route('/show_all_records')
def show_all_records():
    upload_folder = request.args.get('upload_folder', UPLOAD_FOLDER)
    records_data = get_all_records_with_meta(upload_folder)

    # Add recording state and label information to each record
    for record_data in records_data:
        record_id = record_data['ID']
        record_folder = os.path.join(upload_folder, record_id)

        # Get recording state by examining the .wav files directly
        record_data['recordState'] = [False] * 10  # Initialize with 10 false values

        # Get labels from the label file
        label_file = os.path.join(record_folder, 'labels.json')
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                try:
                    labels = json.load(f)
                    record_data['labels'] = labels
                    print(f"Labels for {record_id}: {labels}")
                except json.JSONDecodeError:
                    print(f"Error reading labels file for {record_id}")
                    record_data['labels'] = {}
        else:
            record_data['labels'] = {}

        # Get recording state by scanning the wav files
        try:
            for filename in os.listdir(record_folder):
                if filename.endswith('.wav'):
                    match = re.search(r"(\d+)\.wav$", filename)
                    if match:
                        point_num = int(match.group(1))
                        if 1 <= point_num <= len(record_data['recordState']):
                            record_data['recordState'][point_num - 1] = True
        except Exception as e:
            print(f"Error reading wav files for {record_id}: {e}")

    print(f"Record data formed: {records_data}, upload folder {upload_folder}")
    return render_template('all_records.html', records=records_data, upload_folder=upload_folder)

@app.route('/file_download', methods=['GET'])
def download_file():
    folderId = request.args.get('folderId', default='default_folder').strip().strip('"')
    fileName = request.args.get('fileName', default='default_filename').strip().strip('"')
    folder = os.path.join(UPLOAD_FOLDER, folderId)
    print(f"Playing {fileName} from {folder}")

    try:
        return send_from_directory(folder, fileName, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404

@app.route('/file_delete', methods=['GET'])
def delete_file():
    folderId = request.args.get('folderId', default='default_folder').strip().strip('"')
    fileName = request.args.get('fileName', default='default_filename').strip().strip('"')
    record = record_dict[folderId]
    sound_folder = record.sound_folder
    print(f"Deleting {fileName} from {sound_folder}")

    try:
        os.remove(sound_folder + '/' + fileName)
        return "File deleted", 200
    except FileNotFoundError:
        return "File not found", 404

@app.route('/delete_record_folder', methods=['GET', 'POST'])
def delete_record_folder():
    record_id = request.args.get('record_id')
    upload_folder = request.args.get('upload_folder', UPLOAD_FOLDER)
    record_path = os.path.join(upload_folder, record_id)

    if os.path.exists(record_path):
        shutil.rmtree(record_path)
        return redirect(url_for('show_all_records', upload_folder=upload_folder))
    else:
        return "Record not found", 404


@app.route('/update_labels', methods=['POST'])
def update_labels():
    folderId = request.args.get('folderId', default="").strip().strip('"')
    data = request.get_json()
    print(f"Updating labels, record ID {folderId}")
    print(f"Labels: {data}")

    if folderId not in record_dict:
        return jsonify({"Error": f"Folder {folderId} not found"}), 415

    record = record_dict[folderId]

    # Convert filename-based labels to point-number-based labels
    point_labels = {}
    for filename, label in data.items():
        # First, strip extension if it exists
        base_filename = filename.rsplit('.', 1)[0] if '.' in filename else filename

        # Try to find the point number at the end of the filename
        match = re.search(r"(\d+)$", base_filename)
        if match:
            point_number = match.group(1)
            point_labels[point_number] = label
        else:
            # Try alternative pattern for offline filenames
            match = re.search(r"point_(\d+)_", filename)
            if match:
                point_number = match.group(1)
                point_labels[point_number] = label
            else:
                # Last resort - any digits in the filename
                match = re.search(r"(\d+)", filename)
                if match:
                    point_number = match.group(1)
                    point_labels[point_number] = label
                else:
                    print(f"Warning: Could not extract point number from filename: {filename}")

    record.update_labels(point_labels, "add")
    return jsonify({"ok": "Labels file saved"}), 200


@app.route('/submit_comment', methods=['POST'])
def submit_comment():
    data = request.get_json()
    ID = data.get('record_id', '').strip().strip('"')
    comment = data.get('comment', '')
    print(f"ID: {ID} comment: {comment}")
    if ID not in record_dict:
        return jsonify({"error": "Record not found"}), 400
    record = record_dict[ID]
    record.update_metadata({"Comment" : comment})
    return jsonify({"ok": "Record saved"}), 200

@app.route('/record_delete', methods=['GET'])
def record_delete():
    ID = request.args.get('record_id')
    print(ID)
    ID = ID.strip().strip('"')
    if ID not in record_dict:
        return jsonify({"error": "Record not found"}), 400
    record = record_dict[ID]
    record.destroy()
    record_dict.pop(ID)
    return jsonify({"OK": "Record deleted successfully"}), 200

@app.route('/get_button_states')
def get_button_colors():
    ID = request.args.get('record_id')
    print(ID)
    ID = ID.strip().strip('"')
    if ID not in record_dict:
        return jsonify({"error": "Record not found"}), 400
    record = record_dict[ID]
    return jsonify(record.get_recorded_point_numbers())

import re

# Register custom Jinja2 filters
@app.template_filter('regex_extract')
def regex_extract_filter(s, pattern):
    """Extract the first group from a regex match"""
    match = re.search(pattern, s)
    if match:
        # Return the first capturing group or the entire match if no groups
        return match.group(1) if len(match.groups()) > 0 else match.group(0)
    return ""

##### APK download routes
APK_FOLDER = 'android'
def get_file_info(filepath):
    """Get file information including size and modification date"""
    stat = os.stat(filepath)
    size_mb = round(stat.st_size / (1024 * 1024), 2)
    mod_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
    return {
        'size_mb': size_mb,
        'modified': mod_time
    }
@app.route('/download_application')
def download_application():
    """Main page showing all available APK files"""
    apk_files = []
    # Get all APK files in the folder
    if os.path.exists(APK_FOLDER):
        for filename in os.listdir(APK_FOLDER):
            if filename.lower().endswith('.apk'):
                filepath = os.path.join(APK_FOLDER, filename)
                file_info = get_file_info(filepath)
                apk_files.append({
                    'name': filename,
                    'size_mb': file_info['size_mb'],
                    'modified': file_info['modified']
                })

    # Sort by modification date (newest first)
    apk_files.sort(key=lambda x: x['modified'], reverse=True)

    return render_template('app_APK.html', apk_files=apk_files)
@app.route('/download_apk/<filename>')
def download_apk(filename):
    """Download APK file"""
    # Security check - only allow APK files
    if not filename.lower().endswith('.apk'):
        return jsonify({"error": "File not found"}), 404

    filepath = os.path.join(APK_FOLDER, filename)

    # Check if file exists
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    # Set proper MIME type for APK
    return send_file(
        filepath,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.android.package-archive'
    )
@app.route('/info/<filename>')
def file_info(filename):
    """Get detailed info about an APK file"""
    if not filename.lower().endswith('.apk'):
        return jsonify({"error": "File not found"}), 404

    filepath = os.path.join(APK_FOLDER, filename)

    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    info = get_file_info(filepath)
    return render_template('file_info.html', filename=filename, info=info)


def process_file(file_path):
    """Simple processing function to check if the audio is longer than 5 seconds."""
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)
    # Check if duration is greater than 5 seconds
    if duration > 5:
        return 1
    else:
        return 0


def create_label_file(path):
    wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]

    # Initialize labels dictionary with point numbers as keys
    labels = {}
    for filename in wav_files:
        match = re.search(r"(\d+)\.wav$", filename)
        if match:
            point_number = match.group(1)
            labels[point_number] = "No Label"

    labelfile = os.path.join(path, 'labels.json')
    with open(labelfile, 'w') as file:
        json.dump(labels, file)


import random
import string

def generate_ID(seq_length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=seq_length)).lower()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5057, debug=True)

