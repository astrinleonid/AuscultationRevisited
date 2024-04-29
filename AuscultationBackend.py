import qrcode
import json
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory, render_template, send_file

import os
from werkzeug.utils import secure_filename
import librosa  # You might need to install this library for handling audio files
from datetime import date
import time
from sound_processing import combine_wav_files
from clear_folder import clear_folder
from extract_features import extact_features_from_file

import random

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', '3gp', 'aac', 'flac'}

processing_started = 0

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class Record:

    def __init__(self, save_folder = UPLOAD_FOLDER, model = '', brand = ''):
        self.id = generate_ID(8)
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

    def get_pointRecordProcessId(self):
        self.pointRecordProcessId = generate_ID(4)
        return self.pointRecordProcessId

    def reset_pointRecordProcessId(self):
        self.pointRecordProcessId = "0"
    def point_data_reset(self):

        clear_folder(self.tmp_folder)
        self.chunks = []
        self.chunk_quality = []
        self.active_point = 0
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
        self.successful = '111' in self.quality_string()
        return self.successful
    def combine_wav_from_tmp(self, point_number):
        subs_len = self.num_chunks()
        while self.get_good_subseq_ind(subs_len)[0] < 0:
            subs_len -= 1
        start, end = self.get_good_subseq_ind(subs_len)
        files_to_combine = self.chunks[start : end]
        filename = self.get_filename(point_number)
        save_file_path = os.path.join(self.sound_folder, filename)
        self.files.append((save_file_path))
        combine_wav_files(save_file_path, files_to_combine)
        self.point_data_reset()

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

        for entry in dict:
            if entry in data:
                data[entry] = dict[entry]
                dict.pop(entry)

        data.update(dict)
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


records = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/getUniqueId', methods=['POST'])  # Change to POST to accept data
def get_unique_id():
    # Extract maker and model from the posted data
    data = request.get_json()  # Parse JSON payload
    maker = data.get('maker')
    model = data.get('model')
    print(data)

    metadata = {"Date": str(date.today()), "Maker" : maker, "Model": model}
    print(metadata)

    # Generate a unique ID - here, we use a UUID for simplicity

    record = Record()
    unique_id = record.id
    records[unique_id] = record
    os.makedirs(record.sound_folder, exist_ok=True)
    record.update_metadata(metadata)
    print(f"ID generated: {unique_id}")

    return jsonify(unique_id)

@app.route('/checkConnection', methods=['GET'])  # Change to POST to accept data
def connectionOK():
    return jsonify("OK")

@app.route('/start_point_recording')
def start_point_recording():
    ID = request.args.get('record_id').strip().strip('"')
    if ID not in records:
        print("Error: referring to non-existant ID")
        return jsonify({"error": "No record created for ID"}), 400
    else:
        record = records[ID]
        print(f"Record exists, tmp folder {record.tmp_folder}")
    pointID = record.get_pointRecordProcessId()
    print(f"ID for the point recording issued: {pointID}")
    return jsonify({"pointRecordId" : pointID})

@app.route('/upload', methods=['POST'])
def upload_file():

    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 401

    file = request.files.get('file')
    button_number = int(request.form.get('button_number'))
    ID = request.form.get('record_id').strip().strip('"')
    pointID = request.form.get('pointRecordId').strip().strip('"')

    if ID not in records:
        print("Error: referring to non-existant ID")
        return jsonify({"error": "No record created for ID"}), 402
    else:
        record = records[ID]
    record.requestsInProcess += 1

    os.makedirs(record.tmp_folder, exist_ok=True)

    if pointID != record.pointRecordProcessId:
        print(f"ID should be {record.pointRecordProcessId} received {pointID}")
        return jsonify({"error": "Wrong point record ID"}), 409
    else:
        print(f"Continuing with button No {button_number} point record ID {pointID}")


    record_quality = '1' if record.num_chunks() > 3 else '0'
    print(f"request received ID {ID} button {button_number} response {record_quality}")
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':

        print("error: No selected file")
        return jsonify({"error": "No selected file"}), 403
    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        filename = f"{ID}No{len(record.chunks)}"
        save_path = os.path.join(records[ID].tmp_folder, filename + ".wav")
        file.save(save_path)
        print("Temporary file saved")

        try:
            extact_features_from_file(os.path.join(records[ID].tmp_folder, filename))
            print("Features extracted")
        except Exception as er:
            print(f"Surfboard failed, {er}")

        record.chunks.append(save_path)
        print(f"Chunk saved {save_path} , num_chunks {record.num_chunks()}")
        record.chunk_quality.append(record_quality)
        record.requestsInProcess -= 1
        if record.check_sucsess():
            return jsonify({"message": "Record sucsessfull", "filename": filename})
        else:
            return jsonify({"message": "Continue", "filename": filename})
    else:
        record.requestsInProcess -= 1
        return jsonify({"error": "File type not allowed"}), 404

@app.route('/save_record', methods=['POST'])
def save_record():

    ID = request.form.get('record_id').strip().strip('"')
    if ID not in records:
        return jsonify({"error": "Record not found"}), 400
    record = records[ID]
    record.reset_pointRecordProcessId()

    message = "Point recording completed. "
    button_number = int(request.form.get('button_number'))
    print(f"\n****************\nSaving record to the database, ID {ID} point {button_number}")

    if record.requestsInProcess > 0:
        print(f"Waiting for {record.requestsInProcess} processes to conclude")
        # while processing_started > 0:
        #     time.sleep(0.1)
        # print("All processes ready, continuing")
    if (button_number > 0):
        record.combine_wav_from_tmp(button_number)
        message += "Record saved successfully"
    record.point_data_reset()
    return jsonify({"message": message}), 200


@app.route('/get_wav_files', methods=['GET'])
def get_wav_files():
    ID = request.args.get('folderId', default='default_folder', type=str).strip().strip('"')
    record = records[ID]
    sound_folder = record.sound_folder
    files = " ".join([f[:-4] for f in os.listdir(sound_folder) if f[-3:] == 'wav'])
    return(files)

@app.route('/get_full_path_to_id/<ID>', methods=['GET'])
def get_full_path_to_id(ID):
    # ID = request.args.get('folderId', default='default_folder', type=str).strip().strip('"')
    record = records[ID.strip().strip('"')]
    http_address =  'http://' + request.host + '/show_wav_files?folderId=' + record.sound_folder

    img = qrcode.make(http_address)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr)
    img_byte_arr.seek(0)

    # Send the bytes stream as a file response with MIME type for PNG images
    return send_file(img_byte_arr, mimetype='image/png')


@app.route('/show_wav_files')
def show_wav_files():
    folderId = request.args.get('folderId', default='default_folder', type=str).strip().strip('"')
    folder = os.path.join(UPLOAD_FOLDER, folderId)
    wav_files = [f[:-4] for f in os.listdir(folder) if f.endswith('.wav')]
    metafile = os.path.join(folder, 'meta.json')
    with open(metafile) as file:
        metadata = json.load(file)
    if "Comment" in metadata:
        comment = metadata["Comment"]
    date = metadata["Date"]
    route_to_file = 'http://' + request.host + '/file_download' + '?folderId=' + folderId + '&fileName='
    return render_template('list_wav.html',
                           wav_files=wav_files,
                           folderId = folderId,
                           route_to_file = route_to_file,
                           comment = comment,
                           date = date)

@app.route('/file_download', methods=['GET'])
def download_file():
    folderId = request.args.get('folderId', default='default_folder').strip().strip('"')
    fileName = request.args.get('fileName', default='default_filename').strip().strip('"') + '.wav'
    folder = os.path.join(UPLOAD_FOLDER, folderId)
    print(f"Playing {fileName} from {folder}")

    try:
        return send_from_directory(folder, fileName, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404

@app.route('/file_delete', methods=['GET'])
def delete_file():
    folderId = request.args.get('folderId', default='default_folder').strip().strip('"')
    fileName = request.args.get('fileName', default='default_filename').strip().strip('"') + '.wav'
    record = records[folderId]
    sound_folder = record.sound_folder
    print(f"Deleting {fileName} from {sound_folder}")

    try:
        os.remove(sound_folder + '/' + fileName)
        return "File deleted", 200
    except FileNotFoundError:
        return "File not found", 404

@app.route('/submit_comment', methods=['POST'])
def submit_comment():
    data = request.get_json()
    ID = data.get('record_id', '').strip().strip('"')
    comment = data.get('comment', '')
    print(f"ID: {ID} comment: {comment}")
    if ID not in records:
        return jsonify({"error": "Record not found"}), 400
    record = records[ID]
    record.update_metadata({"Comment" : comment})
    return jsonify({"ok": "Record saved"}), 200

@app.route('/record_delete', methods=['GET'])
def delete_record():
    ID = request.args.get('record_id')
    print(ID)
    ID = ID.strip().strip('"')
    if ID not in records:
        return jsonify({"error": "Record not found"}), 400
    record = records[ID]
    record.destroy()
    records.pop(ID)
    return jsonify({"OK": "Record deleted successfully"}), 200




# @app.route("/")
# def view():
#     return request.host, 200


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

import random
import string

def generate_ID(seq_length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=seq_length)).lower()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)