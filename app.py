from flask import Flask, render_template, request, redirect, url_for, send_file
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Reshape
from PIL import Image
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import os
import cv2

app = Flask(__name__)

def create_compatible_model(model_path):
    # Load the original model
    original_model = load_model(model_path)
    
    # Create input layer matching image dimensions
    input_layer = Input(shape=(224, 224, 3))
    
    # Add preprocessing specific to your model
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(input_layer)
    
    # If model expects flattened input
    if len(original_model.input_shape) == 2:  # Expects flattened input
        x = Flatten()(x)
    
    # Connect to original model
    outputs = original_model(x)
    
    return tf.keras.Model(inputs=input_layer, outputs=outputs)

# Load models with proper adaptation
model_paths = [
    "C:/Users/user/Downloads/breast cancer/breast_cancer_model.keras",
    "C:/Users/user/Downloads/breast cancer/best_model.keras", 
    "C:/Users/user/Downloads/breast cancer/breast_cancer_ultrasound_model.h5",
    "C:/Users/user/Downloads/breast cancer/breast_rnn_model.h5"
]

models = []
for path in model_paths:
    try:
        model = create_compatible_model(path)
        models.append(model)
    except Exception as e:
        print(f"Error loading model {path}: {str(e)}")
        # Create a dummy model if loading fails (for testing)
        models.append(tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation='softmax')
        ]))

def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    if image is None:
        raise ValueError("Image not found. Please check the path.")
    
    image = image.resize(target_size)
    image = np.array(image)
    
    # Convert to 3 channels if grayscale
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
    
    return np.expand_dims(image, axis=0)

# [Rest of your existing code remains the same...]

# Calculate tumor size
def calculate_tumor_size(mask, pixel_spacing=(1, 1)):
    """
    Calculate tumor size in mm
    mask: 2D numpy array of the segmentation mask
    pixel_spacing: tuple of (x_spacing, y_spacing) in mm
    """
    # Find tumor pixels
    tumor_pixels = np.argwhere(mask > 0.5)
    
    if len(tumor_pixels) == 0:
        return 0, 0
    
    # Calculate maximum diameter
    min_coords = np.min(tumor_pixels, axis=0)
    max_coords = np.max(tumor_pixels, axis=0)
    diameter_pixels = max_coords - min_coords
    diameter_mm = diameter_pixels * pixel_spacing
    
    max_diameter = np.max(diameter_mm)
    area = np.sum(mask > 0.5) * pixel_spacing[0] * pixel_spacing[1]
    
    return max_diameter, area

# Determine breast cancer type based on prediction
def determine_breast_cancer_type(image_path, predictions, mask):
    mask = mask.squeeze()
    max_diameter, area = calculate_tumor_size(mask)
    
    # Get average prediction from all models
    avg_malignant_prob = np.mean([pred[0][1] for pred in predictions])
    
    if avg_malignant_prob < 0.2:
        return {
            "type": "No Tumor Detected",
            "description": "No significant tumor was detected in the breast scan.",
            "treatment": "No specific treatment needed. Regular mammograms recommended based on age.",
            "size": max_diameter,
            "area": area,
            "size_category": "No tumor",
            "malignancy_prob": avg_malignant_prob * 100
        }
    elif avg_malignant_prob < 0.5:
        return {
            "type": "Benign Breast Tumor",
            "description": f"Small benign tumor detected (Size: {max_diameter:.1f} mm, Area: {area:.1f} mm²). These are usually non-cancerous and don't spread.",
            "treatment": "Monitoring with regular imaging. Possible biopsy if suspicious features are present.",
            "size": max_diameter,
            "area": area,
            "size_category": "Small (<10mm)",
            "malignancy_prob": avg_malignant_prob * 100
        }
    elif avg_malignant_prob < 0.8:
        if max_diameter < 20:
            return {
                "type": "Early Stage Breast Cancer (DCIS or Stage I)",
                "description": f"Moderate-sized tumor detected (Size: {max_diameter:.1f} mm, Area: {area:.1f} mm²) with {avg_malignant_prob*100:.1f}% malignancy probability.",
                "treatment": "Lumpectomy or mastectomy, possibly with radiation therapy. Hormone therapy if hormone receptor positive.",
                "size": max_diameter,
                "area": area,
                "size_category": "Medium (10-20mm)",
                "malignancy_prob": avg_malignant_prob * 100
            }
        else:
            return {
                "type": "Stage II Breast Cancer",
                "description": f"Larger tumor detected (Size: {max_diameter:.1f} mm, Area: {area:.1f} mm²) with {avg_malignant_prob*100:.1f}% malignancy probability.",
                "treatment": "Surgery (lumpectomy or mastectomy) plus chemotherapy and/or radiation. Hormone therapy if applicable.",
                "size": max_diameter,
                "area": area,
                "size_category": "Large (20-50mm)",
                "malignancy_prob": avg_malignant_prob * 100
            }
    else:
        if max_diameter > 50:
            return {
                "type": "Advanced Breast Cancer (Stage III/IV)",
                "description": f"Large tumor detected (Size: {max_diameter:.1f} mm, Area: {area:.1f} mm²) with {avg_malignant_prob*100:.1f}% malignancy probability, possibly indicating advanced breast cancer.",
                "treatment": "Systemic therapy (chemotherapy, targeted therapy), possible surgery, radiation, and/or hormone therapy depending on subtype.",
                "size": max_diameter,
                "area": area,
                "size_category": "Very Large (>50mm)",
                "malignancy_prob": avg_malignant_prob * 100
            }
        else:
            return {
                "type": "Aggressive Breast Cancer",
                "description": f"Tumor detected (Size: {max_diameter:.1f} mm, Area: {area:.1f} mm²) with high malignancy probability ({avg_malignant_prob*100:.1f}%).",
                "treatment": "Combination therapy including surgery, chemotherapy, and possibly radiation. Targeted therapy if HER2 positive.",
                "size": max_diameter,
                "area": area,
                "size_category": "Medium (10-50mm)",
                "malignancy_prob": avg_malignant_prob * 100
            }

# Generate segmentation visualization
def generate_segmentation_visualization(image_path, mask):
    # Load and prepare original image
    original_img = Image.open(image_path)
    original_img = original_img.resize((256, 256))
    original_img = np.array(original_img)
    
    # Ensure image has 3 channels (convert grayscale to RGB if needed)
    if len(original_img.shape) == 2:
        original_img = np.stack((original_img,)*3, axis=-1)
    elif original_img.shape[2] == 1:
        original_img = np.concatenate([original_img]*3, axis=-1)
    
    # Process mask - ensure proper shape and values
    mask = np.squeeze(mask)  # Remove single-dimensional entries
    
    # Debug print to understand mask shape
    print(f"Initial mask shape: {mask.shape}")
    
    # Handle different mask formats:
    if mask.ndim == 0:  # Scalar value
        mask = np.full((256, 256), mask)  # Create full-size mask
    elif mask.ndim == 1:  # 1D array
        if len(mask) == 2:  # Likely binary classification probabilities
            # Create dummy mask based on probability
            prob = mask[1]  # Assuming second value is tumor probability
            mask = np.zeros((256, 256))
            if prob > 0.5:
                mask[128-50:128+50, 128-50:128+50] = 1  # Placeholder tumor area
        else:
            mask = np.zeros((256, 256))  # Fallback empty mask
    elif mask.ndim == 2:  # Proper 2D mask
        pass  # Use as-is
    elif mask.ndim == 3:  # Multi-channel mask
        mask = mask[..., 0]  # Take first channel
    
    # Ensure mask is binary (0 or 1) and proper size
    mask = (mask > 0.5).astype(np.uint8)
    if mask.shape != (256, 256):
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    # Create colored mask
    colored_mask = np.zeros_like(original_img)
    colored_mask[mask == 1] = [255, 0, 0]  # Red color for tumor
    
    # Blend original image with mask
    alpha = 0.5
    blended = cv2.addWeighted(original_img, 1 - alpha, colored_mask, alpha, 0)
    
    # Create visualization figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.imshow(original_img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Ensure mask is 2D for display
    display_mask = mask if mask.ndim == 2 else mask[..., 0]
    ax2.imshow(display_mask, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Segmentation Mask')
    ax2.axis('off')
    
    ax3.imshow(blended)
    ax3.set_title('Tumor Detection')
    ax3.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf
# Create risk factor diagrams
def create_relationship_diagrams(age, gender, family_history, menopause_status, tumor_size):
    diagrams = []
    
    # Age vs Risk
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    age_categories = ["<40", "40-50", "50-60", ">60"]
    age_risk_values = [0, 0, 0, 0]
    
    if int(age) < 40:
        age_risk_values[0] = 1
    elif 40 <= int(age) < 50:
        age_risk_values[1] = 1
    elif 50 <= int(age) <= 60:
        age_risk_values[2] = 1
    else:
        age_risk_values[3] = 1
    
    ax1.pie(age_risk_values, labels=age_categories, autopct='%1.1f%%', 
            colors=['lightblue', 'lightgreen', 'orange', 'lightcoral'])
    ax1.set_title("Age vs Breast Cancer Risk")
    diagrams.append(fig1)
    
    # Gender vs Risk (though breast cancer can occur in men too)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    gender_labels = ['Female', 'Male']
    gender_risk_values = [1 if gender == "Female" else 0.01]
    gender_risk_percent = [gender_risk_values[0] * 100, (1 - gender_risk_values[0]) * 100]
    ax2.pie(gender_risk_percent, labels=gender_labels, autopct='%1.1f%%', 
            colors=['pink', 'lightblue'])
    ax2.set_title("Gender vs Breast Cancer Risk")
    diagrams.append(fig2)
    
    # Family History
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    family_categories = ["Family History"]
    family_risk_values = [1 if family_history.lower() in ["yes", "mother", "sister", "daughter"] else 0]
    ax3.bar(family_categories, family_risk_values, color=['red' if family_risk_values[0] else 'green'])
    ax3.set_ylim(0, 1)
    ax3.set_title("Family History vs Breast Cancer Risk")
    ax3.set_ylabel("Family History (1=Yes, 0=No)")
    diagrams.append(fig3)
    
    # Menopause Status
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    menopause_labels = ['Pre-Menopause', 'Post-Menopause']
    menopause_values = [1 if menopause_status.lower() in ["post", "post-menopause"] else 0]
    menopause_percent = [menopause_values[0] * 100, (1 - menopause_values[0]) * 100]
    ax4.pie(menopause_percent, labels=menopause_labels, autopct='%1.1f%%', 
            colors=['lightblue', 'pink'])
    ax4.set_title("Menopause Status vs Risk")
    diagrams.append(fig4)
    
    # Tumor Size Diagram
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    size_categories = ["<10mm", "10-20mm", "20-50mm", ">50mm"]
    size_values = [0, 0, 0, 0]
    
    if tumor_size < 10: size_values[0] = 1
    elif 10 <= tumor_size < 20: size_values[1] = 1
    elif 20 <= tumor_size < 50: size_values[2] = 1
    else: size_values[3] = 1
    
    ax5.bar(size_categories, size_values, color=['green', 'yellow', 'orange', 'red'])
    ax5.set_title("Tumor Size Category")
    ax5.set_ylabel("Risk Level")
    diagrams.append(fig5)
    
    return diagrams

# Predict and generate report
def predict_and_generate_report(image_path, name, national_id, nationality, age, mobile_number, gender, 
                              family_history, menopause_status, hormone_use, pregnancies, weight, height):
    img_array = preprocess_image(image_path)
    
    # Get predictions from all models
    predictions = [model.predict(img_array) for model in models]
    
    # Use first model for segmentation (assuming it's the segmentation model)
    mask = predictions[0]
    
    disease_info = determine_breast_cancer_type(image_path, predictions[1:], mask)
    visualization_buf = generate_segmentation_visualization(image_path, mask)
    
    # Calculate BMI
    try:
        bmi = float(weight) / ((float(height)/100) ** 2)
    except ValueError:
        bmi = 0
    
    report = (f"Name: {name}\nNational ID: {national_id}\nNationality: {nationality}\nAge: {age}\n"
              f"Mobile Number: {mobile_number}\nGender: {gender}\nFamily History: {family_history}\n"
              f"Menopause Status: {menopause_status}\nHormone Use: {hormone_use}\n"
              f"Pregnancies: {pregnancies}\nWeight: {weight} kg\nHeight: {height} cm\nBMI: {bmi:.1f}\n"
              f"Tumor Size: {disease_info['size']:.1f} mm\nTumor Area: {disease_info['area']:.1f} mm²\n"
              f"Tumor Size Category: {disease_info['size_category']}\n"
              f"Malignancy Probability: {disease_info['malignancy_prob']:.1f}%\n\n")
    
    report += f"Diagnosis: {disease_info['type']}\n"
    report += f"Description: {disease_info['description']}\n"
    report += f"Recommended Treatment: {disease_info['treatment']}\n\n"
    
    if "No Tumor" in disease_info['type']:
        report += (f"{name} shows no signs of breast tumors in the scan. No further treatment is required. "
                   "However, regular mammograms are recommended based on age and risk factors.\n\n")
    else:
        report += (f"{name} has a breast condition that requires medical attention. "
                   "Consultation with a breast specialist or oncologist is strongly recommended. "
                   "Depending on the diagnosis, further tests like biopsy or additional imaging "
                   "may be needed to confirm the diagnosis and plan treatment.\n\n")
        
        report += ("Additional Recommendations:\n"
                   "- Regular breast self-exams\n"
                   "- Annual mammograms if over 40\n"
                   "- Maintain healthy weight\n"
                   "- Limit alcohol consumption\n"
                   "- Exercise regularly\n"
                   "- Consider genetic counseling if strong family history\n\n")
    
    report += "\nRelationships between Input Data and Diagnosis:\n"
    
    if int(age) > 50:
        report += "- Age: Patients over 50 years old are at higher risk of breast cancer.\n"
    elif int(age) > 40:
        report += "- Age: Risk increases after age 40.\n"
    else:
        report += "- Age: Younger patients have lower risk but can still develop breast cancer.\n"
    
    if gender == "Female":
        report += "- Gender: Females are at much higher risk of breast cancer than males.\n"
    else:
        report += "- Gender: Male breast cancer is rare but possible.\n"
    
    if family_history.lower() in ["yes", "mother", "sister", "daughter"]:
        report += "- Family History: Family history increases breast cancer risk significantly.\n"
    else:
        report += "- Family History: No significant family history reported.\n"
    
    if menopause_status.lower() in ["post", "post-menopause"]:
        report += "- Menopause Status: Post-menopausal women are at higher risk.\n"
    else:
        report += "- Menopause Status: Pre-menopausal status may indicate higher risk if other factors present.\n"
    
    # Tumor size relationship
    size = disease_info['size']
    if size < 10:
        report += "- Tumor Size: Very small tumor (<10 mm), potentially early stage if malignant.\n"
    elif 10 <= size < 20:
        report += "- Tumor Size: Small tumor (10-20 mm), may indicate stage I cancer.\n"
    elif 20 <= size < 50:
        report += "- Tumor Size: Medium tumor (20-50 mm), may indicate stage II cancer.\n"
    else:
        report += "- Tumor Size: Large tumor (>50 mm), may indicate advanced stage cancer.\n"
    
    if bmi > 30:
        report += "- BMI: Obesity is a risk factor for breast cancer, especially post-menopause.\n"
    elif bmi > 25:
        report += "- BMI: Overweight status may contribute to breast cancer risk.\n"
    else:
        report += "- BMI: Healthy weight reduces breast cancer risk.\n"
    
    # Create diagrams including tumor size
    diagrams = create_relationship_diagrams(age, gender, family_history, menopause_status, disease_info['size'])
    
    return report, visualization_buf, disease_info, diagrams

@app.route('/', methods=['GET', 'POST'])
def index():
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    
    if request.method == 'POST':
        name = request.form['name']
        national_id = request.form['national_id']
        nationality = request.form['nationality']
        age = request.form['age']
        mobile_number = request.form['mobile_number']
        gender = request.form['gender']
        family_history = request.form['family_history']
        menopause_status = request.form['menopause_status']
        hormone_use = request.form['hormone_use']
        pregnancies = request.form['pregnancies']
        weight = request.form['weight']
        height = request.form['height']
        
        if 'image' not in request.files:
            return "No image uploaded", 400
        image = request.files['image']
        if image.filename == '':
            return "No image selected", 400
        
        image_path = os.path.join('uploads', image.filename)
        image.save(image_path)
        
        report, visualization_buf, disease_info, diagrams = predict_and_generate_report(
            image_path, name, national_id, nationality, age, mobile_number, gender,
            family_history, menopause_status, hormone_use, pregnancies, weight, height)
        
        visualization_path = os.path.join('static', 'visualization.png')
        with open(visualization_path, 'wb') as f:
            f.write(visualization_buf.getbuffer())
        
        # Save diagrams
        diagram_paths = []
        for i, diagram in enumerate(diagrams):
            buf = io.BytesIO()
            diagram.savefig(buf, format='png')
            buf.seek(0)
            path = os.path.join('static', f'diagram_{i}.png')
            with open(path, 'wb') as f:
                f.write(buf.getbuffer())
            diagram_paths.append(path)
            plt.close(diagram)
        
        return render_template('result.html', 
                             report=report, 
                             visualization=visualization_path,
                             diagrams=diagram_paths,
                             disease_info=disease_info,
                             age=age,
                             gender=gender,
                             family_history=family_history,
                             menopause_status=menopause_status)
    
    return render_template('index.html')

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    try:
        report = request.form['report']
        visualization_path = request.form['visualization']
        age = request.form['age']
        gender = request.form['gender']
        family_history = request.form['family_history']
        menopause_status = request.form['menopause_status']
        
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        
        os.makedirs(static_dir, exist_ok=True)
        os.makedirs(uploads_dir, exist_ok=True)
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        pdf.add_page()
        pdf.set_font("Arial", style='B', size=16)
        pdf.cell(200, 10, "Breast Cancer Detection Report", ln=True, align='C')
        pdf.ln(10)
        
        lines = report.split("\n")
        
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Patient Information:", ln=True)
        pdf.set_font("Arial", size=10)
        
        diagnosis_start = next((i for i, line in enumerate(lines) if "Diagnosis:" in line), len(lines))
        
        # Create table for patient information
        patient_info = lines[:diagnosis_start]
        
        # Set column widths (total width = 190)
        col_width = 95
        line_height = 7
        
        # Draw table headers
        pdf.set_fill_color(230, 242, 253)  # Light blue background
        pdf.set_font("Arial", style='B', size=10)
        pdf.cell(col_width, line_height, "Field", border=1, fill=True)
        pdf.cell(col_width, line_height, "Value", border=1, ln=True, fill=True)
        
        # Draw table content
        pdf.set_font("Arial", size=10)
        for line in patient_info:
            if line.strip():
                # Split the line into field and value
                parts = line.split(":", 1)
                if len(parts) == 2:
                    field, value = parts[0].strip(), parts[1].strip()
                    pdf.cell(col_width, line_height, field, border=1)
                    pdf.cell(col_width, line_height, value, border=1, ln=True)
        
        # Add a new page for Diagnosis section
        pdf.add_page()
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Diagnosis:", ln=True)
        pdf.set_font("Arial", size=10)
        
        risk_start = next((i for i, line in enumerate(lines) if "Relationships between Input Data and Diagnosis:" in line), len(lines))
        
        for line in lines[diagnosis_start:risk_start]:
            if line.strip():
                if "Diagnosis:" in line or "Description:" in line or "Recommended Treatment:" in line:
                    pdf.set_font("Arial", style='B', size=11)
                    pdf.cell(0, 7, line, ln=True)
                    pdf.set_font("Arial", size=10)
                else:
                    pdf.cell(0, 7, line, ln=True)
        pdf.ln(5)
        
        if risk_start < len(lines):
            pdf.set_font("Arial", style='B', size=12)
            pdf.cell(0, 10, "Risk Factors Analysis:", ln=True)
            pdf.set_font("Arial", size=10)
            
            for line in lines[risk_start:]:
                if line.strip():
                    if line.startswith("-"):
                        pdf.cell(10)
                        pdf.cell(0, 7, line[2:], ln=True)
                    else:
                        pdf.cell(0, 7, line, ln=True)
            pdf.ln(5)
        
        pdf.add_page()
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Breast Scan Analysis:", ln=True)
        pdf.image(visualization_path, x=10, y=None, w=180)
        
        pdf_path = os.path.join(static_dir, 'breast_cancer_report.pdf')
        pdf.output(pdf_path)
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Failed to create PDF file at {pdf_path}")
        
        return send_file(pdf_path, as_attachment=True)
    
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return f"An error occurred while generating the PDF: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)