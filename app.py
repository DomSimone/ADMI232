from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_moment import Moment
import json
from datetime import datetime
import os
from werkzeug.utils import secure_filename

# For data processing terminal
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)
moment = Moment(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///admi.db' # SQLite database file
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads' # Folder to temporarily store uploaded files
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max upload size

db = SQLAlchemy(app)

# --- Database Models ---

class Survey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    # Relationship to questions - ordered by position
    questions = db.relationship('Question', backref='survey', lazy=True, order_by='Question.position')
    responses = db.relationship('SurveyResponse', backref='survey', lazy=True)

    def __repr__(self):
        return f'<Survey {self.id}: {self.title}>'

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    survey_id = db.Column(db.Integer, db.ForeignKey('survey.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(50), nullable=False) # e.g., 'text', 'number', 'multiple_choice'
    options = db.Column(db.Text, nullable=True) # Stored as JSON string: ["Option A", "Option B"]
    validation = db.Column(db.Text, nullable=True) # Stored as JSON string: {"required": true, "min": 0}
    branching_logic = db.Column(db.Text, nullable=True) # Stored as JSON string: {"if_answer": "Yes", "show_question_id": 5}
    position = db.Column(db.Integer, nullable=False) # For ordering questions

    def __repr__(self):
        return f'<Question {self.id} (Survey {self.survey_id}): {self.text[:30]}>'

    def get_options_list(self):
        return json.loads(self.options) if self.options else []

    def get_validation_dict(self):
        return json.loads(self.validation) if self.validation else {}

    def get_branching_logic_dict(self):
        return json.loads(self.branching_logic) if self.branching_logic else {}

class SurveyResponse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    survey_id = db.Column(db.Integer, db.ForeignKey('survey.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    # Relationship to answers
    answers = db.relationship('Answer', backref='survey_response', lazy=True)

    def __repr__(self):
        return f'<SurveyResponse {self.id} for Survey {self.survey_id}>'

class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    response_id = db.Column(db.Integer, db.ForeignKey('survey_response.id'), nullable=False)
    question_id = db.Column(db.Integer, nullable=False) # Store question ID for easy lookup
    value = db.Column(db.Text, nullable=True) # Store answer value (can be JSON for checkboxes)

    def __repr__(self):
        return f'<Answer {self.id} (Q:{self.question_id}): {self.value[:30]}>'


# --- Dummy data for demonstration (will be replaced by DB queries) ---
# research_data will still be in-memory for now, but its 'responses' count will be updated from DB
research_data = [
    {"id": 1, "title": "Study on Agricultural Practices", "status": "Active", "responses": 0, "last_activity": "N/A"},
    {"id": 2, "title": "Education Access Survey", "status": "Completed", "responses": 0, "last_activity": "N/A"},
    {"id": 3, "title": "Healthcare Infrastructure Analysis", "status": "Active", "responses": 0, "last_activity": "N/A"},
]

# Function to update research_data based on actual survey responses
def update_research_data_from_db():
    for r_item in research_data:
        survey = Survey.query.filter_by(title=r_item['title']).first()
        if survey:
            r_item['responses'] = SurveyResponse.query.filter_by(survey_id=survey.id).count()
            latest_response = SurveyResponse.query.filter_by(survey_id=survey.id).order_by(SurveyResponse.timestamp.desc()).first()
            if latest_response:
                r_item['last_activity'] = latest_response.timestamp.strftime('%Y-%m-%d')
        else:
            r_item['responses'] = 0
            r_item['last_activity'] = "N/A"


# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html') # Render a simple index page

@app.route('/research_management')
def research_management():
    update_research_data_from_db() # Update counts before rendering
    return render_template('research_management.html', research_data=research_data)

@app.route('/api/dashboard_data')
def get_dashboard_data():
    update_research_data_from_db() # Update counts before sending data
    total_research = len(research_data)
    active_research = len([r for r in research_data if r['status'] == 'Active'])
    recent_activity = sorted(research_data, key=lambda x: x['last_activity'] if x['last_activity'] != 'N/A' else '1970-01-01', reverse=True)[:5]

    responses_summary = {r['title']: r['responses'] for r in research_data}

    return jsonify({
        'total_research': total_research,
        'active_research': active_research,
        'recent_activity': recent_activity,
        'responses_summary': responses_summary
    })

# --- Survey Routes ---

@app.route('/surveys')
def survey_list():
    surveys = Survey.query.order_by(Survey.id).all()
    return render_template('survey_management.html', surveys=surveys)

@app.route('/surveys/new', methods=['GET', 'POST'])
def create_survey():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        new_survey = Survey(title=title, description=description)
        db.session.add(new_survey)
        db.session.commit()
        return redirect(url_for('edit_survey', survey_id=new_survey.id))
    return render_template('create_survey.html')

@app.route('/surveys/<int:survey_id>')
def edit_survey(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    return render_template('edit_survey.html', survey=survey)

# API endpoint to get a single question's data
@app.route('/api/surveys/<int:survey_id>/questions/<int:question_id>', methods=['GET'])
def get_question(survey_id, question_id):
    question = Question.query.filter_by(survey_id=survey_id, id=question_id).first_or_404()
    return jsonify({
        'id': question.id,
        'text': question.text,
        'type': question.type,
        'options': question.get_options_list(),
        'validation': question.get_validation_dict(),
        'branching_logic': question.get_branching_logic_dict()
    }), 200

@app.route('/api/surveys/<int:survey_id>/questions', methods=['POST'])
def add_question(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    data = request.get_json()

    question_text = data.get('question_text')
    question_type = data.get('question_type')
    options = json.dumps(data.get('options', []))
    validation = json.dumps(data.get('validation', {}))
    branching_logic = json.dumps(data.get('branching_logic', {}))

    # Determine the position for the new question
    last_question = Question.query.filter_by(survey_id=survey_id).order_by(Question.position.desc()).first()
    new_position = (last_question.position + 1) if last_question else 0

    new_question = Question(
        survey_id=survey_id,
        text=question_text,
        type=question_type,
        options=options,
        validation=validation,
        branching_logic=branching_logic,
        position=new_position
    )
    db.session.add(new_question)
    db.session.commit()

    return jsonify({
        'id': new_question.id,
        'text': new_question.text,
        'type': new_question.type,
        'options': new_question.get_options_list(),
        'validation': new_question.get_validation_dict(),
        'branching_logic': new_question.get_branching_logic_dict()
    }), 201

@app.route('/api/surveys/<int:survey_id>/questions/<int:question_id>', methods=['PUT'])
def update_question(survey_id, question_id):
    question = Question.query.filter_by(survey_id=survey_id, id=question_id).first_or_404()
    data = request.get_json()

    question.text = data.get('question_text', question.text)
    question.type = data.get('question_type', question.type)
    question.options = json.dumps(data.get('options', json.loads(question.options))) if 'options' in data else question.options
    question.validation = json.dumps(data.get('validation', json.loads(question.validation))) if 'validation' in data else question.validation
    question.branching_logic = json.dumps(data.get('branching_logic', json.loads(question.branching_logic))) if 'branching_logic' in data else question.branching_logic

    db.session.commit()

    return jsonify({
        'id': question.id,
        'text': question.text,
        'type': question.type,
        'options': question.get_options_list(),
        'validation': question.get_validation_dict(),
        'branching_logic': question.get_branching_logic_dict()
    }), 200

@app.route('/api/surveys/<int:survey_id>/questions/<int:question_id>', methods=['DELETE'])
def delete_question(survey_id, question_id):
    question = Question.query.filter_by(survey_id=survey_id, id=question_id).first_or_404()
    db.session.delete(question)
    db.session.commit()
    return jsonify({"message": "Question deleted successfully"}), 200

@app.route('/api/surveys/<int:survey_id>/reorder_questions', methods=['POST'])
def reorder_questions(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    new_order_ids = request.get_json().get('question_ids')

    if not isinstance(new_order_ids, list) or not all(isinstance(x, int) for x in new_order_ids):
        return jsonify({"error": "Invalid format for question_ids. Expected a list of integers."}), 400

    questions_in_survey = {q.id: q for q in survey.questions}

    if len(new_order_ids) != len(questions_in_survey):
        return jsonify({"error": "Mismatch in question count. Not all questions were provided in the new order."}), 400

    for index, q_id in enumerate(new_order_ids):
        if q_id not in questions_in_survey:
            return jsonify({"error": f"Question with ID {q_id} not found in survey."}), 404
        questions_in_survey[q_id].position = index # Update position

    db.session.commit()
    return jsonify({"message": "Questions reordered successfully"}), 200


# --- Survey Taking Routes ---
@app.route('/take_survey/<int:survey_id>', methods=['GET'])
def take_survey(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    # Pass questions with their parsed JSON fields
    questions_data = []
    for q in survey.questions:
        questions_data.append({
            'id': q.id,
            'text': q.text,
            'type': q.type,
            'options': q.get_options_list(),
            'validation': q.get_validation_dict(),
            'branching_logic': q.get_branching_logic_dict()
        })
    return render_template('take_survey.html', survey=survey, questions_data=questions_data)

@app.route('/api/surveys/<int:survey_id>/submit_response', methods=['POST'])
def submit_survey_response(survey_id):
    survey = Survey.query.get_or_404(survey_id)
    response_data = request.get_json()

    new_survey_response = SurveyResponse(
        survey_id=survey.id,
        timestamp=datetime.fromisoformat(response_data.get('timestamp').replace('Z', '+00:00')) # Handle Z for UTC
    )
    db.session.add(new_survey_response)
    db.session.flush() # Assigns an ID to new_survey_response before commit

    for q_id, answer_value in response_data.get('answers', {}).items():
        # Ensure q_id is int for lookup
        q_id_int = int(q_id)
        # Convert list answers (e.g., checkboxes) to JSON string
        if isinstance(answer_value, list):
            answer_value = json.dumps(answer_value)
        elif answer_value is not None:
            answer_value = str(answer_value) # Ensure it's a string for Text column

        new_answer = Answer(
            response_id=new_survey_response.id,
            question_id=q_id_int,
            value=answer_value
        )
        db.session.add(new_answer)

    db.session.commit()

    return jsonify({"message": "Survey response submitted successfully", "response_id": new_survey_response.id}), 201


# --- AI Native Vision & OCR Routes ---

ALLOWED_EXTENSIONS = {'pdf', 'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/document_processing')
def document_processing_page():
    return render_template('document_processing.html')

@app.route('/api/upload_documents', methods=['POST'])
def upload_documents():
    if 'documents' not in request.files:
        return jsonify({"error": "No document part in the request"}), 400
    
    files = request.files.getlist('documents')
    if not files:
        return jsonify({"error": "No selected file"}), 400

    uploaded_filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            uploaded_filenames.append(filename)
        else:
            return jsonify({"error": f"File type not allowed or file missing: {file.filename}"}), 400
    
    if len(uploaded_filenames) > 30:
        return jsonify({"error": "Maximum 30 documents allowed at a time"}), 400

    return jsonify({"message": "Documents uploaded successfully", "filenames": uploaded_filenames}), 200

@app.route('/api/process_document', methods=['POST'])
def process_document():
    data = request.get_json()
    filenames = data.get('filenames')
    prompt = data.get('prompt')
    output_format = data.get('output_format', 'csv') # default to csv

    if not filenames or not prompt:
        return jsonify({"error": "Missing filenames or prompt"}), 400

    # --- MOCK AI NATIVE VISION & OCR AND GENERATIVE EXTRACTION ---
    # In a real application, you would:
    # 1. Load files from app.config['UPLOAD_FOLDER']
    # 2. Call an external AI Vision/OCR service (e.g., Google Cloud Vision, AWS Textract)
    # 3. Use a Generative AI model (e.g., LLM) to extract data based on the 'prompt'
    # 4. Format the extracted data as requested

    mock_extracted_data = []
    for filename in filenames:
        # Simulate extraction based on filename and prompt
        if "invoice" in prompt.lower() and "invoice number" in prompt.lower():
            mock_extracted_data.append({
                "filename": filename,
                "Invoice Number": f"INV-{hash(filename) % 10000}",
                "Date": "2023-10-27",
                "Vendor Name": f"Vendor {filename.split('.')[0]}",
                "Total Amount": f"{ (hash(filename) % 1000) + 100 }.00"
            })
        elif "metadata" in prompt.lower() and "author" in prompt.lower():
             mock_extracted_data.append({
                "filename": filename,
                "Title": f"Document Title {filename.split('.')[0]}",
                "Author": f"Author {hash(filename) % 50}",
                "Creation Date": "2023-01-01"
            })
        else:
            mock_extracted_data.append({
                "filename": filename,
                "ExtractedField1": f"Value A from {filename}",
                "ExtractedField2": f"Value B from {filename}",
                "PromptUsed": prompt
            })

    if output_format == 'csv':
        # Convert list of dicts to CSV string
        if not mock_extracted_data:
            return jsonify({"result": "No data extracted."}), 200
        
        # Get all unique keys for header
        all_keys = sorted(list(set(k for d in mock_extracted_data for k in d.keys())))
        csv_lines = [",".join(all_keys)] # Header row
        for item in mock_extracted_data:
            row = [str(item.get(key, "")) for key in all_keys]
            csv_lines.append(",".join(row))
        return jsonify({"result": "\n".join(csv_lines), "format": "csv"}), 200
    else: # json
        return jsonify({"result": json.dumps(mock_extracted_data, indent=2), "format": "json"}), 200

@app.route('/api/upload_analog_metadata', methods=['POST'])
def upload_analog_metadata():
    if 'metadata_files' not in request.files:
        return jsonify({"error": "No metadata file part in the request"}), 400
    
    files = request.files.getlist('metadata_files')
    if not files:
        return jsonify({"error": "No selected metadata file"}), 400

    uploaded_filenames = []
    for file in files:
        if file and allowed_file(file.filename): # Re-using allowed_file for PDF/CSV
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], "analog_metadata_" + filename))
            uploaded_filenames.append(filename)
        else:
            return jsonify({"error": f"File type not allowed or file missing: {file.filename}"}), 400
    
    return jsonify({"message": "Analog metadata uploaded successfully", "filenames": uploaded_filenames}), 200


# --- Data Processing Terminal Routes ---

@app.route('/data_processing_terminal')
def data_processing_terminal_page():
    return render_template('data_processing_terminal.html')

@app.route('/api/process_data_command', methods=['POST'])
def process_data_command():
    data = request.get_json()
    csv_data = data.get('csv_data')
    command = data.get('command')

    if not csv_data:
        return jsonify({"error": "No CSV data provided."}), 400
    if not command:
        return jsonify({"error": "No command provided."}), 400

    try:
        df = pd.read_csv(BytesIO(csv_data.encode('utf-8')))
        result_text = ""
        graph_base64 = None

        command_lower = command.lower().strip()

        if "linear regression" in command_lower:
            parts = command_lower.split(" on ")
            if len(parts) > 1:
                variables_part = parts[1].strip()
                if " vs " in variables_part:
                    x_var, y_var = variables_part.split(" vs ")
                    x_var = x_var.strip()
                    y_var = y_var.strip()

                    if x_var in df.columns and y_var in df.columns:
                        X = df[[x_var]].values
                        y = df[y_var].values
                        
                        # Remove rows with NaN values in X or y
                        valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
                        X = X[valid_indices]
                        y = y[valid_indices]

                        if len(X) == 0:
                            result_text = f"Error: No valid data points for regression after handling NaNs for {x_var} and {y_var}."
                        elif len(np.unique(X)) == 1:
                            result_text = f"Error: Cannot perform regression with a single unique value in '{x_var}'."
                        else:
                            model = LinearRegression()
                            model.fit(X, y)
                            result_text = f"Linear Regression ({y_var} vs {x_var}):\n"
                            result_text += f"  Coefficient ({x_var}): {model.coef_[0]:.4f}\n"
                            result_text += f"  Intercept: {model.intercept_:.4f}\n"
                            result_text += f"  R-squared: {model.score(X, y):.4f}"

                            # Generate plot
                            plt.figure(figsize=(8, 6))
                            plt.scatter(X, y, color='blue', label='Actual Data')
                            plt.plot(X, model.predict(X), color='red', label='Regression Line')
                            plt.title(f'Linear Regression: {y_var} vs {x_var}')
                            plt.xlabel(x_var)
                            plt.ylabel(y_var)
                            plt.legend()
                            plt.grid(True)
                            
                            buf = BytesIO()
                            plt.savefig(buf, format='png')
                            buf.seek(0)
                            graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                            plt.close() # Close the plot to free memory
                    else:
                        result_text = f"Error: One or both columns '{x_var}', '{y_var}' not found in data."
                else:
                    result_text = "Error: Invalid linear regression command format. Use 'linear regression on X vs Y'."
            else:
                result_text = "Error: Invalid linear regression command format. Use 'linear regression on X vs Y'."
        
        elif "plot histogram of" in command_lower:
            column_name = command_lower.replace("plot histogram of", "").strip()
            if column_name in df.columns:
                plt.figure(figsize=(8, 6))
                df[column_name].hist()
                plt.title(f'Histogram of {column_name}')
                plt.xlabel(column_name)
                plt.ylabel('Frequency')
                plt.grid(True)

                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
                result_text = f"Generated histogram for '{column_name}'."
            else:
                result_text = f"Error: Column '{column_name}' not found in data."

        elif "describe" in command_lower:
            result_text = df.describe().to_string()
        
        elif "head" in command_lower:
            result_text = df.head().to_string()

        else:
            result_text = "Unknown command. Supported commands: 'linear regression on X vs Y', 'plot histogram of Z', 'describe', 'head'."

        return jsonify({"result_text": result_text, "graph_base64": graph_base64}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred during data processing: {str(e)}"}), 400

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/terms')
def terms_page():
    return render_template('terms.html')

@app.route('/contact')
def contact_page():
    return render_template('contact.html')


if __name__ == '__main__':
    # Create database tables if they don't exist
    # Ensure the upload folder exists
    with app.app_context():
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

    with app.app_context():
        db.create_all()
    app.run(debug=True)
