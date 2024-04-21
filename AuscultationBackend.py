from flask import Flask, request, jsonify
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
        self.model = model
        self.brand = brand

    def point_data_reset(self):
        self.chunks = []
        self.chunk_quality = []

    def get_filename(self, point_number, sep = 'point'):
        return f"record{date.today()}ID{self.id}{sep}{point_number}.wav"

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
        save_file_path = os.path.join(self.save_folder, filename)
        self.files.append((save_file_path))
        combine_wav_files(save_file_path, files_to_combine)
        clear_folder(self.tmp_folder)
        self.point_data_reset()

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

    print(f"Maker: {maker}, Model: {model}")  # Print maker and model

    # Generate a unique ID - here, we use a UUID for simplicity
    unique_id = generate_ID(6)
    record = Record(unique_id)
    records[unique_id] = record
    os.makedirs(record.tmp_folder, exist_ok=True)
    print(f"ID generated: {unique_id}")

    return jsonify(unique_id)

@app.route('/upload', methods=['POST'])
def upload_file():
    global processing_started
    processing_started += 1
    # Check if the post request has the file part
    if 'file' not in request.files:
        processing_started -= 1
        return jsonify({"error": "No file part"}), 400
    file = request.files.get('file')
    button_number = request.form.get('button_number')
    ID = request.form.get('record_id').strip().strip('"')

    if ID not in records:
        record = Record(ID)
        records[ID] = record
        tmp_dir_path = record.tmp_folder
        print(f"New record created, tmp folder {tmp_dir_path}")
        os.makedirs(tmp_dir_path, exist_ok=True)
    else:
        record = records[ID]
        print(f"Record exists, tmp folder {record.tmp_folder}")

    record_quality = '1' if record.num_chunks() > 3 else '0'
    print(f"request received ID {ID} button {button_number} response {record_quality}")
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        processing_started -= 1
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        filename = f"record{date.today()}ID{ID}no{len(record.chunks)}"
        save_path = os.path.join(records[ID].tmp_folder, filename + ".wav")
        file.save(save_path)
        extact_features_from_file(os.path.join(records[ID].tmp_folder, filename))
        record.chunks.append(save_path)

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
    button_number = request.form.get('button_number')
    ID = request.form.get('record_id').strip().strip('"')
    print(f"Saving record to the database, ID {ID} point {button_number}")
    if ID not in records:
        return jsonify({"error": "Record not found"}), 400
    record = records[ID]
    while processing_started > 0:
        time.sleep(0.1)
    record.combine_wav_from_tmp(button_number)
    return jsonify({"message": "Record saved successfully"}), 200

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