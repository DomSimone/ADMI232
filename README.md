# African Development Models Initiative (ADMI) Web Application

## Project Overview

The African Development Models Initiative (ADMI) is a web-based application designed to facilitate data management, survey creation, AI-powered document processing, and data analysis for sustainable economic and social development in Africa and beyond. This platform addresses critical systemic issues such as the lack of a unified platform for statistical agencies, quality control, and rural isolation hindering accurate statistical work in the region.

## Features

The ADMI application is built with the following key functionalities:

### 1. Research Management Dashboard
*   **Customizable Dashboard:** Users can add, remove, and arrange widgets to display key metrics.
*   **Key Metrics:** Widgets include total research count, active research projects, recent activity, and a summary of responses per research.
*   **Personalized Overview:** Provides a tailored view of research data for each user.

### 2. Survey Forms & Question Design
*   **Survey Creation:** Users can create new surveys with titles and descriptions.
*   **Question Management (CRUD):** Full Create, Read, Update, and Delete (CRUD) functionality for survey questions.
*   **Question Types:** Supports various question types (Text, Number, Multiple Choice, Checkbox, Date).
*   **Options & Validation:** Ability to define options for multiple-choice/checkbox questions and set validation rules (e.g., required, min/max length/value).
*   **Branching Logic:** Allows defining conditional logic to show/hide questions based on previous answers.
*   **Question Reordering:** Drag-and-drop interface to easily change the order of questions within a survey.

### 3. Survey-Taking Interface
*   **Dynamic Survey Display:** Renders surveys dynamically based on their design.
*   **Interactive Questions:** Users can fill out surveys, with validation and branching logic applied in real-time.
*   **Response Submission:** Submits survey responses to the backend for storage.

### 4. AI Native Vision & OCR Document Processing
*   **Document Ingestion:** Upload PDF or CSV files (up to 30 items at a time) for processing.
*   **Structured Output Prompting:** Users provide natural language prompts to instruct the AI on what data to extract (e.g., "Invoice Number," "Date," "Vendor Name").
*   **Generative AI Extraction (Mocked):** Simulates AI/ML models to "see" and interpret document context, extracting specified information. (Note: Actual AI integration would require external API services).
*   **Output Formats:** Extracted data can be generated as CSV or JSON, ready for spreadsheet conversion.
*   **Analog Study Metadata Upload:** Dedicated section for uploading CSV/PDF files containing metadata from analog studies.

### 5. Data Processing Terminal
*   **CSV Data Input:** Users can paste raw CSV data directly into a textarea or upload a CSV file.
*   **Command-Line Interface:** Execute data analysis commands on the provided CSV data.
*   **Supported Commands:**
    *   `linear regression on X vs Y`: Performs linear regression between two specified columns.
    *   `plot histogram of Z`: Generates a histogram for a specified column.
    *   `describe`: Provides descriptive statistics for the dataset.
    *   `head`: Displays the first few rows of the dataset.
*   **Textual & Graphical Output:** Displays results as text and generates illustrative graphs (e.g., regression plots, histograms) as images.

### 6. Informational Pages
*   **About Us:** A dedicated page detailing the mission, vision, and solutions offered by the African Development Models Initiative.
*   **Terms of Use and User Agreement:** Comprehensive legal document outlining the terms governing the use of the Service, intellectual property, user obligations, data privacy, disclaimers, and dispute resolution.
*   **Contact Us:** A form for users to submit general inquiries, including fields for Name, Email, Message, and an optional Phone Number.

### 7. Consistent User Interface
*   **Global Header:** A consistent header across all pages featuring the application logo and navigation links to all main sections.
*   **Responsive Design:** Basic styling for a clean and functional user experience.

## Technologies and Packages Employed

The application is built using Python with the Flask web framework and leverages several libraries for various functionalities:

*   **Backend:**
    *   **Flask:** The core web framework for routing, request handling, and templating.
    *   **Flask-SQLAlchemy:** ORM (Object Relational Mapper) for interacting with the database.
    *   **SQLAlchemy:** Python SQL toolkit and ORM (underlying Flask-SQLAlchemy).
    *   **SQLite:** Default database for local development and persistence (`admi.db`).
    *   **`json`:** For handling JSON data serialization/deserialization (e.g., for question options, validation, branching logic).
    *   **`os`:** For operating system interactions, such as managing file paths and creating directories.
    *   **`werkzeug.utils.secure_filename`:** For securely handling uploaded filenames.
    *   **`datetime`:** For handling timestamps in survey responses.

*   **Data Processing Terminal:**
    *   **`pandas`:** For data manipulation and analysis (reading CSV, DataFrame operations).
    *   **`numpy`:** For numerical operations, especially with arrays (used in regression).
    *   **`scikit-learn` (`sklearn.linear_model.LinearRegression`):** For performing linear regression.
    *   **`matplotlib.pyplot`:** For generating static, interactive, and animated visualizations (graphs).
    *   **`base64`:** For encoding generated graph images to be displayed in HTML.
    *   **`io.BytesIO`:** For handling in-memory binary streams, used for saving matplotlib figures.

*   **Frontend:**
    *   **HTML5:** Structure of web pages.
    *   **CSS3:** Styling of web pages.
    *   **JavaScript (ES6+):** For dynamic client-side interactions, form handling, API calls, drag-and-drop functionality, and real-time validation/branching logic in surveys.

## Setup and Installation

To set up and run the ADMI application locally, follow these steps:

1.  **Clone the Repository (if applicable) or ensure you have all project files.**

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Required Python Packages:**
    ```bash
    pip install Flask Flask-SQLAlchemy pandas numpy scikit-learn matplotlib
    ```

4.  **Place the Logo File:**
    Ensure your application logo (`logo.png`) is located in the `static/images/` directory:
    `C:\Users\pmali\PycharmProjects\ADMI\static\images\logo.png`
    (Create the `static` and `images` folders if they don't exist).

5.  **Run the Flask Application:**
    Navigate to your project's root directory in the terminal and run:
    ```bash
    python app.py
    ```

6.  **Access the Application:**
    Open your web browser and go to `http://127.0.0.1:5000/`.

## Usage Workflow

### General Navigation
*   The application features a consistent header with navigation links to all main modules: Research Dashboard, Survey Management, AI Document Processing, Data Terminal, About Us, Terms of Use, and Contact Us.
*   The logo in the header links back to the home page.

### Research Management Dashboard
*   Access via "Research Dashboard" link.
*   Use the "Add Widget" dropdown to select and add various data visualization widgets.
*   Widgets can be removed or reordered by dragging and dropping.

### Survey Forms & Questions
*   Access via "Survey Management" link.
*   **Create Survey:** Click "Create New Survey", provide a title and description.
*   **Edit Survey Design:** Click "Edit Design" next to a survey to add, edit, delete, or reorder questions.
    *   Fill in question text, select type, add options (for multiple-choice/checkbox), and define validation/branching logic in JSON format.
    *   Drag and drop questions to reorder them.
*   **Take Survey:** Click "Take Survey" to simulate a user filling out the survey. Observe real-time validation and branching logic.

### AI Document Processing
*   Access via "AI Document Processing" link.
*   **Upload Documents:** Select PDF/CSV files (max 30) and click "Upload Documents".
*   **Structured Extraction:** Enter a prompt (e.g., "Extract Invoice Number, Date, Total Amount") and choose output format (CSV/JSON). Click "Process Documents".
*   **Analog Metadata:** Upload CSV/PDF files for analog study metadata separately.

### Data Processing Terminal
*   Access via "Data Terminal" link.
*   **Input Data:** Paste CSV data into the textarea or upload a CSV file.
*   **Execute Commands:** Type commands like `linear regression on X vs Y`, `plot histogram of Z`, `describe`, or `head` into the command input and click "Execute Command".
*   Results will appear as text and/or graphs.

## Future Enhancements (Conceptual)

*   **Real AI Integration:** Replace mocked AI extraction with actual calls to external AI Vision/OCR services (e.g., Google Cloud Vision, AWS Textract) and Generative AI models.
*   **User Authentication & Authorization:** Implement user login, roles, and permissions.
*   **Advanced Data Visualization:** Integrate more sophisticated charting libraries for the dashboard and terminal.
*   **Database Migrations:** Use Flask-Migrate for managing database schema changes.
*   **Deployment:** Prepare the application for production deployment (e.g., Gunicorn, Nginx, Docker).
*   **Comprehensive Error Handling:** More robust error handling and user feedback.
*   **Frontend Framework:** Consider a frontend framework (e.g., React, Vue) for more complex UI interactions.
*   **API Documentation:** Generate API documentation for external integrations.
*   **Unit/Integration Tests:** Add automated tests for backend and frontend logic.
