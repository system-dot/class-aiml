import os
import torch
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from torchvision import models, transforms
from PIL import Image
from collections import Counter
import pdfplumber
from groq import Groq
import tensorflow as tf
import tempfile
import json
from dotenv import load_dotenv  # Load environment variables

# Load variables from .env file
load_dotenv()

# Initialize the Groq client with the API key from environment variables
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize Flask app
app = Flask(__name__)

# Access the second API key if needed (e.g., for Google services)
google_api_key = os.getenv("API_KEY")
if not google_api_key:
    raise ValueError("API key for Google services is not set in the environment variables.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the System!"

@app.route('/favicon.ico')
def favicon():
    return '', 204

IMG_SIZE = (224, 224)
CLASSES = ["classroom_images", "classroom_training", "extra_activities", "teacher_teaching"]

transform_classroom = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

classroom_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

for param in classroom_model.parameters():
    param.requires_grad = False

classroom_model.fc = torch.nn.Sequential(
    torch.nn.Linear(classroom_model.fc.in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256, len(CLASSES)),
    torch.nn.Softmax(dim=1)
)

classroom_model.load_state_dict(torch.load("classroom_activity_model.pth"))
classroom_model = classroom_model.to(device)
classroom_model.eval()

# Parameters for second model
model_path = "impairment_detection_model.pth"
impairment_model = models.resnet18()
impairment_model.fc = torch.nn.Linear(impairment_model.fc.in_features, 4)  # Adjust for 4 classes
impairment_model.load_state_dict(torch.load(model_path, map_location=device))
impairment_model = impairment_model.to(device)
impairment_model.eval()

class_names = ["Visual Impairment", "Hearing Impairment", "Physical Disability", "Normal Students"]

transform_impairment = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

infrastructure_suggestions = {
    "Visual Impairment": "screen magnifier.",
    "Hearing Impairment": "captioning tools.",
    "Physical Disability": "ramp access.",
    "Normal Students": "interactive learning tools (Smart board)."
}

save_dir = "uploads"
os.makedirs(save_dir, exist_ok=True)


def predict_classroom(model, image):
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)
    img = transform_classroom(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()

    return probabilities


def predict_impairment(model, image):
    img = image.convert("RGB")
    img = transform_impairment(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return predicted.item()


@app.route('/score', methods=['POST'])
def analyze():
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "No images found in the request"}), 400

    classroom_scores = [0] * len(CLASSES)
    impairment_counts = Counter({class_name: 0 for class_name in class_names})
    total_images = 0
    filepaths = []

    for file in files:
        try:
            # Save the image
            filename = secure_filename(file.filename)
            filepath = os.path.join(save_dir, filename)
            file.save(filepath)
            filepaths.append(filepath)

            # Open image
            image = Image.open(filepath)

            # Predictions for classroom model
            classroom_probabilities = predict_classroom(classroom_model, image)
            classroom_scores = [classroom_scores[i] + classroom_probabilities[i] for i in range(len(CLASSES))]

            # Predictions for impairment model
            predicted_class_idx = predict_impairment(impairment_model, image)
            predicted_class = class_names[predicted_class_idx]
            impairment_counts[predicted_class] += 1

            total_images += 1
        except Exception as e:
            return jsonify({"error": f"Error processing file {file.filename}: {str(e)}"}), 500

    # Compute classroom results
    if total_images > 0:
        average_classroom_scores = [classroom_scores[i] / total_images for i in range(len(CLASSES))]
        best_classroom_index = average_classroom_scores.index(max(average_classroom_scores))
        best_classroom_activity = CLASSES[best_classroom_index]
    else:
        best_classroom_activity = "No valid images processed"

    # Compute impairment results
    class_percentages = {class_name: (count / total_images) * 100 for class_name, count in impairment_counts.items()}

    # Prepare final response
    response = {
        "best_classroom_activity": best_classroom_activity,
        "impairment_analysis": class_percentages,
        "infrastructure_suggestions": infrastructure_suggestions
    }

    # Cleanup
    # for filepath in filepaths:
    #     if os.path.exists(filepath):
    #         os.remove(filepath)

    return jsonify(response)

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    syllabus_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                syllabus_text += page.extract_text()
        return syllabus_text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

def analyze_text_with_llama(user_input):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Analyze the following syllabus and provide the following details:\n"
                           "1. Key Topics: List the main topics covered in the syllabus (2-3 lines).\n"
                           "2. Difficulty Level: Provide an overview of the difficulty level with 2-3 sentences.\n"
                           "3. Recommended Prerequisites: Suggest the necessary prerequisites (2-3 sentences)."
                           f"\n\nSyllabus Content:\n{user_input}"
            }],
            model="llama3-8b-8192"
        )
        analysis = {
            "Key Topics": chat_completion.choices[0].message.content.split("Key Topics:")[1].split("Difficulty Level:")[0].strip(),
            "Difficulty Level": chat_completion.choices[0].message.content.split("Difficulty Level:")[1].split("Recommended Prerequisites:")[0].strip(),
            "Recommended Prerequisites": chat_completion.choices[0].message.content.split("Recommended Prerequisites:")[1].strip(),
        }
        return analysis
    except Exception as e:
        return {"error": f"Error: {e}"}

def analyze_curriculum(curriculum_text):
    return "Analyzing the curriculum to align with teacher's qualifications."

def get_alignment_percentage(gemini_response):
    if "aligned" in gemini_response.lower():
        return 80
    else:
        return 40

@app.route('/teacher', methods=['POST'])
def process_request():
    data = request.get_json()

    response = {}

    # Check if file-based syllabus processing or teacher data is provided
    course_curriculum = data.get('course_curriculum')
    if course_curriculum:
        # PDF syllabus analysis
        if allowed_file(course_curriculum):
            syllabus_text = extract_text_from_pdf(course_curriculum)
            if syllabus_text.startswith("Error"):
                return jsonify({"error": syllabus_text}), 400
            analysis_result = analyze_text_with_llama(syllabus_text)
            response["analysis_result"] = {
                "message": "Syllabus processed successfully",
                "analysis_result": analysis_result
            }

    # Teacher points calculation
    avg_experience = data.get('avg_experience')
    live_in_out_ratio = data.get('live_in_out_ratio')
    educational_qualifications = data.get('educational_qualifications')
    total_teachers = data.get('total_teachers')

    if all([avg_experience, live_in_out_ratio, educational_qualifications, total_teachers]):
        # Calculate experience points
        if avg_experience > 5:
            experience_points = 20
        elif avg_experience > 4:
            experience_points = 17
        elif avg_experience > 3:
            experience_points = 15
        elif avg_experience > 2:
            experience_points = 12
        elif avg_experience > 1:
            experience_points = 9
        else:
            experience_points = 0

        # Calculate live-in/out ratio points
        if live_in_out_ratio > 8:
            live_in_out_points = 20
        elif live_in_out_ratio > 6:
            live_in_out_points = 16
        elif live_in_out_ratio > 4:
            live_in_out_points = 14
        else:
            live_in_out_points = 0

        # Calculate qualification points
        phd_percentage = (educational_qualifications['phd'] / total_teachers) * 100
        post_graduate_percentage = (educational_qualifications['post_graduate'] / total_teachers) * 100
        graduate_percentage = (educational_qualifications['graduate'] / total_teachers) * 100

        # PhD points calculation
        if phd_percentage > 10:
            phd_points = 20
        elif phd_percentage > 7:
            phd_points = 15
        elif phd_percentage > 5:
            phd_points = 10
        else:
            phd_points = 0

        # Postgraduate points calculation
        if post_graduate_percentage > 70:
            post_graduate_points = 20
        elif post_graduate_percentage > 50:
            post_graduate_points = 16
        elif post_graduate_percentage > 40:
            post_graduate_points = 14
        elif post_graduate_percentage > 30:
            post_graduate_points = 12
        elif post_graduate_percentage > 20:
            post_graduate_points = 10
        else:
            post_graduate_points = 0

        # Graduate points calculation
        if graduate_percentage > 70:
            graduate_points = 20
        elif graduate_percentage > 50:
            graduate_points = 16
        elif graduate_percentage > 40:
            graduate_points = 14
        elif graduate_percentage > 30:
            graduate_points = 12
        elif graduate_percentage > 20:
            graduate_points = 10
        else:
            graduate_points = 0

        alignment_score = 0
        alignment_percentage = 0

        if course_curriculum:
            syllabus_text = extract_text_from_pdf(course_curriculum)
            curriculum_analysis = analyze_curriculum(syllabus_text)

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Does the teacher's qualifications align with the following curriculum?\n{curriculum_analysis}",
                    }
                ],
                model="llama3-8b-8192",
            )

            gemini_response = chat_completion.choices[0].message.content
            alignment_percentage = get_alignment_percentage(gemini_response)

            if alignment_percentage >= 80:
                alignment_score = 10
            elif alignment_percentage >= 70:
                alignment_score = 9
            elif alignment_percentage >= 60:
                alignment_score = 8
            elif alignment_percentage >= 50:
                alignment_score = 7
            elif alignment_percentage >= 40:
                alignment_score = 5
            elif alignment_percentage >= 30:
                alignment_score = 3
            else:
                alignment_score = 0

        average_points = (experience_points + live_in_out_points + phd_points + post_graduate_points +
                          graduate_points + alignment_score) / 6

        response["teacher_analysis"] = {
            "average_points": round(average_points, 2),
            "alignment_score": alignment_score
        }

    return jsonify(response)

def calculate_scores(total_lectures_taken, avg_attendance, avg_marks, curriculum_lectures):
    """
    Calculate the scores for students' performance and faculty's consistency.
    """
    # Faculty Consistency Score
    faculty_score = min((total_lectures_taken / curriculum_lectures) * 20, 20)
    
    # Student Attendance Score based on attendance percentage
    attendance_percentage = (avg_attendance / total_lectures_taken) * 100
    if attendance_percentage >= 75:
        attendance_score = 20
    elif attendance_percentage >= 65:
        attendance_score = 18
    elif attendance_percentage >= 55:
        attendance_score = 16
    elif attendance_percentage >= 45:
        attendance_score = 14
    else:
        attendance_score = 10

    # Student Marks Score based on average marks
    if avg_marks >= 80:
        marks_score = 20
    elif avg_marks >= 70:
        marks_score = 18
    elif avg_marks >= 60:
        marks_score = 16
    elif avg_marks >= 50:
        marks_score = 14
    else:
        marks_score = 10
    student_score = (attendance_score + marks_score) / 2
    student_score = min(student_score, 20) 
    
    return round(faculty_score, 2), round(attendance_score, 2), round(marks_score, 2), round(student_score, 2)

# For students
@app.route('/process_scores', methods=['POST'])
def process_scores():
    """
    Process the scores based on the POST request data from the frontend.
    """
    data = request.get_json()

    total_lectures_taken = data.get('total_lectures_taken')
    avg_attendance = data.get('avg_attendance')
    avg_marks = data.get('avg_marks')
    total_lectures_assigned = data.get('total_lectures_assigned')  # Manual input for total lectures assigned

    if total_lectures_assigned is None:
        return jsonify({"error": "Total lectures assigned must be provided."}), 400

    # Calculate scores
    faculty_score, attendance_score, marks_score, student_score = calculate_scores(total_lectures_taken, avg_attendance, avg_marks, total_lectures_assigned)

    results = {
        "faculty_score": faculty_score,
        "attendance_score": attendance_score,
        "marks_score": marks_score,
        "student_score": student_score,
        # "total_lectures_assigned": total_lectures_assigned,
    }

    return jsonify(results)

def calculate_scores(hackathons, sports, olympiads, seminars, recreational_sessions, cultural_activity):
    # Hackathons score
    if hackathons > 5:
        hackathons_score = 20
    elif hackathons == 4:
        hackathons_score = 16
    elif hackathons == 3:
        hackathons_score = 12
    elif hackathons == 2:
        hackathons_score = 8
    elif hackathons == 1:
        hackathons_score = 4
    else:
        hackathons_score = 0

    # Sports score
    if sports > 5:
        sports_score = 20
    elif sports == 4:
        sports_score = 16
    elif sports == 3:
        sports_score = 12
    elif sports == 2:
        sports_score = 8
    elif sports == 1:
        sports_score = 4
    else:
        sports_score = 0

    # Olympiads score
    if olympiads > 5:
        olympiads_score = 20
    elif olympiads == 4:
        olympiads_score = 16
    elif olympiads == 3:
        olympiads_score = 12
    elif olympiads == 2:
        olympiads_score = 8
    elif olympiads == 1:
        olympiads_score = 4
    else:
        olympiads_score = 0

    # Seminars score
    if seminars > 3:
        seminars_score = 20
    elif seminars == 2:
        seminars_score = 15
    elif seminars == 1:
        seminars_score = 10
    else:
        seminars_score = 0

    # Recreational sessions score
    recreational_sessions_score = 10 if recreational_sessions else 0

    # Cultural activity score
    cultural_activity_score = 20 if cultural_activity else 0

    # Calculate average score
    total_score = (hackathons_score + sports_score + olympiads_score + seminars_score + recreational_sessions_score + cultural_activity_score)
    average_score = total_score / 100 * 100  # out of 100
    return average_score

# Function to analyze and compare with other institutions using Groq API
def compare_with_other_institutes(data, tier):
    prompt = f"Compare the following data of a {tier}-tier institute with other similar institutions.\n{data}"
    
    # Use Groq API to generate content for comparison
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"
    )
    return response.choices[0].message.content

@app.route('/extra_curricular', methods=['POST'])
def evaluate_institute():
    # Get input data from the frontend (e.g., API request)
    data = request.get_json()

    hackathons = data.get('hackathons', 0)
    sports = data.get('sports', 0)
    olympiads = data.get('olympiads', 0)
    seminars = data.get('seminars', 0)
    recreational_sessions = data.get('recreational_sessions', False)
    cultural_activity = data.get('cultural_activity', False)
    institute_tier = data.get('institute_tier', '3')  # Default to 3-tier if not provided

    average_score = calculate_scores(hackathons, sports, olympiads, seminars, recreational_sessions, cultural_activity)

    # Prepare the data for Groq API comparison
    institute_data = {
        "hackathons": hackathons,
        "sports": sports,
        "olympiads": olympiads,
        "seminars": seminars,
        "recreational_sessions": recreational_sessions,
        "cultural_activity": cultural_activity
    }

    comparison_result = compare_with_other_institutes(institute_data, institute_tier)

    # Prepare the response
    response = {
        "extra_curricular_avg": average_score,
        # "comparison_with_other_institutes": comparison_result
    }

    return jsonify(response)

@app.route('/overall', methods=['POST'])
def calculate_institute_score():
    data = request.get_json()

    # Extracting the required data from input JSON
    impairment_analysis = data.get("impairment_analysis", {})
    infrastructure_suggestions = data.get("infrastructure_suggestions", {})
    analysis_result = data.get("analysis_result", {})
    attendance_score = data.get("attendance_score", 0)
    faculty_score = data.get("faculty_score", 0)
    marks_score = data.get("marks_score", 0)
    student_score = data.get("student_score", 0)
    extra_curricular_score = data.get("average_score of extra curricular activities", 0)
    alignment_score = data.get("alignment_score", 0)
    average_points = data.get("average_points", 0)

    # Real-time analysis using Groq
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze the following syllabus: \n\nDifficulty Level: {analysis_result.get('Difficulty Level')}\nKey Topics: {', '.join(analysis_result.get('Key Topics', []))}\nRecommended Prerequisites: {', '.join(analysis_result.get('Recommended Prerequisites', []))}"
                }
            ],
            model="llama3-8b-8192"
        )
        real_time_analysis = chat_completion.choices[0].message.content
    except Exception as e:
        real_time_analysis = f"Error in real-time analysis: {e}"

    # Calculate suitability percentages
    suitability_percentages = {
        category: max(0, 100 - value) for category, value in impairment_analysis.items()
    }

    # Calculate overall score
    scores = [attendance_score, faculty_score, marks_score, student_score, extra_curricular_score, alignment_score, average_points]
    overall_score = sum(scores) / len(scores)

    # Prepare result
    result = {
        "institute_overall_score": round(overall_score, 2),
        "suitability_percentages": suitability_percentages,
        "real_time_analysis": real_time_analysis,
        "infrastructure_suggestions": infrastructure_suggestions
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
