import os
import requests
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import json

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = "exoplanet_secret_key"

UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HF_API_URL = os.environ.get('HF_API_URL', 'https://sawabedarain-nasa-space-api.hf.space/predict')

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', models_loaded=True)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                df = pd.read_csv(filepath)
                payload = {"type": "batch", "data": df.to_json(orient='records')}
                response = requests.post(HF_API_URL, json=payload, timeout=30)
                if response.status_code != 200:
                    flash(f"AI Engine Error: {response.text}")
                    return redirect(request.url)
                api_results = response.json()
                return render_template('results.html',
                                      df_html=api_results['df_html'],
                                      visualizations=api_results['visualizations'],
                                      metrics=api_results.get('metrics'),
                                      filename=filename,
                                      shape=api_results['shape'],
                                      model_name=api_results['model_name'])
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)
    return render_template('upload.html')

@app.route('/manual_input', methods=['GET', 'POST'])
def manual_input():
    if request.method == 'POST':
        try:
            form_data = {k: v for k, v in request.form.items()}
            payload = {"type": "manual", "data": form_data}
            response = requests.post(HF_API_URL, json=payload, timeout=30)
            if response.status_code != 200:
                flash(f"AI Engine Error: {response.text}")
                return redirect(request.url)
            results = response.json()
            results['data'] = form_data
            return render_template('manual_result.html', results=results)
        except Exception as e:
            flash(f'Error processing input: {str(e)}')
            return redirect(request.url)
    return render_template('manual_input.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
