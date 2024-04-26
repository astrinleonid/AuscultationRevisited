from flask import Flask, request, jsonify, send_from_directory, render_template

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

    metadata_string = f"Date: {date.today()} Maker: {maker} Model: {model}"
    print(metadata_string)  # Print maker and model

    # Generate a unique ID - here, we use a UUID for simplicity
    unique_id = generate_ID(6)
    record = Record(unique_id)
    records[unique_id] = record
    os.makedirs(record.sound_folder, exist_ok=True)
    record.save_metadata(metadata_string)
    print(f"ID generated: {unique_id}")

    return jsonify(unique_id)

@app.route('/checkConnection', methods=['GET'])  # Change to POST to accept data
def connectionOK():
    return jsonify("OK")

@app.route('/upload', methods=['POST'])
def upload_file():
    global processing_started
    processing_started += 1
    # Check if the post request has the file part
    if 'file' not in request.files:
        processing_started -= 1
        return jsonify({"error": "No file part"}), 400
    file = request.files.get('file')
    button_number = int(request.form.get('button_number'))
    ID = request.form.get('record_id').strip().strip('"')

    if ID not in records:
        print("Error: referring to non-existant ID")
        return jsonify({"error": "No record created for ID"}), 400
    else:
        record = records[ID]
        print(f"Record exists, tmp folder {record.tmp_folder}")

    os.makedirs(record.tmp_folder, exist_ok=True)

    if button_number != record.active_point:
        print(f"Button number mismatch, waiting for the previous cycle to conclude {button_number}, {record.active_point}")
        while record.active_point > 0:
            time.sleep(0.01)
        record.active_point = button_number
    else:
        print(f"Continuing with button No {button_number}")

    record_quality = '1' if record.num_chunks() > 3 else '0'
    print(f"request received ID {ID} button {button_number} response {record_quality}")
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        processing_started -= 1
        print("error: No selected file")
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        filename = f"{ID}No{len(record.chunks)}"
        save_path = os.path.join(records[ID].tmp_folder, filename + ".wav")
        file.save(save_path)
        print("Temporary file saved")
        extact_features_from_file(os.path.join(records[ID].tmp_folder, filename))
        print("Features extracted")
        record.chunks.append(save_path)
        print(f"Chunk saved {save_path} , num_chunks {record.num_chunks()}")
        record.chunk_quality.append(record_quality)
        if record.check_sucsess():
            processing_started -= 1
            return jsonify({"message": "Record sucsessfull", "filename": filename})
        else:
            processing_started -= 1
            return jsonify({"message": "Continue", "filename": filename})
    else:
        processing_started -= 1
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/save_record', methods=['POST'])
def save_record():
    message = "Point recording completed. "
    button_number = int(request.form.get('button_number'))
    ID = request.form.get('record_id').strip().strip('"')
    print(f"\n*******************************\nSaving record to the database, ID {ID} point {button_number}")
    if ID not in records:
        return jsonify({"error": "Record not found"}), 400
    record = records[ID]
    while processing_started > 0:
        time.sleep(0.1)
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
    record = records[ID]
    return '/show_wav_files/' + record.sound_folder


@app.route('/show_wav_files/<folder>')
def show_wav_files(folder):
    wav_files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    return render_template('list_wav.html', wav_files=wav_files)

@app.route('/file_download', methods=['GET'])
def download_file():
    folderId = request.args.get('folderId', default='default_folder').strip().strip('"')
    fileName = request.args.get('fileName', default='default_filename').strip().strip('"') + '.wav'
    record = records[folderId]
    sound_folder = record.sound_folder
    print(f"Playing {fileName} from {sound_folder}")

    try:
        return send_from_directory(sound_folder, fileName, as_attachment=True)
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