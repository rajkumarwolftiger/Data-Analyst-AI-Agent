# app.py
import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
from dotenv import load_dotenv

load_dotenv()
import agent_logic 

app = Flask(__name__)

# ========================================================================= #
# WEB INTERFACE ROUTES                                                      #
# ========================================================================= #

@app.route('/')
def index():
    """Renders the homepage."""
    return render_template('index.html')

@app.route('/upload')
def upload_form():
    """Renders the file upload form page."""
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze_from_web():
    """
    Handles the file upload from the web form.
    It calls the same agent logic but renders an HTML result page.
    """
    if 'questions_txt' not in request.files:
        return "No file part in the request", 400
    
    file = request.files['questions_txt']
    if file.filename == '':
        return "No selected file", 400

    if file:
        try:
            # We get the task description from the uploaded file
            task_description = file.read().decode('utf-8')
            
            # We call the EXACT same agent logic as our API
            # NOTE: Our web form only handles the main task file, not extra attachments.
            result = agent_logic.run_analysis(task_description, attached_files={})
            
            # Instead of returning JSON, we render the result.html template
            return render_template('result.html', result=result)

        except Exception as e:
            # If an error occurs, we render the same result.html template but pass the error
            print(f"An error occurred: {e}") # Log the error for debugging
            return render_template('result.html', error=str(e))

    return redirect(url_for('upload_form'))


# ========================================================================= #
# SCRIPTABLE API ROUTE (FOR CURL) - UNCHANGED!                              #
# ========================================================================= #

@app.route('/api/', methods=['POST'])
def handle_analysis_request_from_api():
    """
    This is the original API endpoint. It works exactly as before and is
    intended for use with curl or other scripts. It returns raw JSON.
    """
    if 'questions.txt' not in request.files:
        return jsonify({"error": "questions.txt is missing"}), 400

    task_description = request.files['questions.txt'].read().decode('utf-8')
    
    attached_files = request.files.to_dict()
    if 'questions.txt' in attached_files:
        del attached_files['questions.txt']

    try:
        result = agent_logic.run_analysis(task_description, attached_files)
        # Return the raw JSON result for scripts
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
