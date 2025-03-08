import uuid
import os

from flask import Flask, request

app = Flask(__name__)

@app.route('/api/v1/sample/upload', methods=['POST'])
def upload_sample():
    if 'file' not in request.files:
        return 'No file part in the request', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file', 400
    
    if file and file.filename.endswith('.wav'):
        filename = "sample.wav"
        file.save('./sample/' + filename)
        
        return 'File uploaded successfully', 200
    
    return 'Invalid file type. Only .wav files are accepted', 400

@app.route('/api/v1/sample/create', methods=['POST'])
def create_sample():
    data = request.get_json()
    
    if data and 'sample' in data:
        sample_text = data['sample']
        
        with open('./sample/sample.txt', 'w') as f:
            f.write(sample_text)
        return 'Sample text saved successfully', 200
    
    return 'Invalid request data', 400

@app.route('/api/v1/transform', methods=['POST'])
def transform():
    data = request.get_json()
    
    if data and 'target' in data:
        target_text = data['target']
        unique_id = uuid.uuid4()
        directory = f'./target/{unique_id}'
        os.makedirs(directory, exist_ok=True)

        with open(f'{directory}/target.txt', 'w') as f:
            f.write(target_text)
            
        return jsonify({'receipt_id': str(unique_id)}), 200
    
    return 'Invalid request data', 400

if __name__ == '__main__':
    app.run(debug=True)
