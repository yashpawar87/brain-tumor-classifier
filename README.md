# 🧠 Brain Tumor Classification & XAI

App that classifies brain tumor MRI scans into four categories (Glioma, Meningioma, Pituitary, or No Tumor). The web-app also provides model explanations using **Grad-CAM** and **LIME** to visualize *why* the model made a specific prediction.

---

## 💡 Features

* **MRI Upload:** Upload your own `.jpg`, `.jpeg`, or `.png` brain MRI scans.
* **4-Class Classification:** The app uses a PyTorch CNN model to classify the image as:

  * Glioma
  * Meningioma
  * Pituitary Tumor
  * No Tumor
* **Explainable AI (XAI):**

  * **Grad-CAM:** Generates a heatmap overlay to show which parts of the image were most important for the model's prediction.
  * **LIME:** Segments the image and highlights the "superpixels" that contributed most to the classification.

---

## 🛠 Tech Stack

* **Framework:** Streamlit
* **Deep Learning:** PyTorch
* **XAI Libraries:** `pytorch-grad-cam`, `lime`
* **Image Processing:** `Pillow`, `opencv-python`
* **Data Handling:** `numpy`

---

## 🚀 Setup and Run Locally

Follow these steps to get the application running on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/yashpawar87/brain-tumor-classifier.git
cd brain-tumor-classifier
```

### 2. Get the Model

This project requires the pre-trained model file `custom_cnn_tumor_model.pth`.

> **Note:** If you have not already, you must add this file to the root of the project directory. 

### 3. Create a Virtual Environment

It is highly recommended to use a virtual environment.

```bash
# Create the environment
python3 -m venv my_env

# Activate the environment
# On macOS/Linux:
source my_env/bin/activate

# On Windows:
# my_env\Scripts\activate
```

### 4. Install Dependencies

Install all required Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Run the App

Use the following command to run the Streamlit app. We use `python3 -m streamlit` to ensure it runs using the venv's Python.

```bash
python3 -m streamlit run app.py
```

Your app should now be open and running in your default web browser!

---

## 📂 Project Structure

```
brain-tumor-classifier/
│
├── .streamlit/
│   └── config.toml     # Custom UI theme (light/dark modes)
│
├── .gitignore          
├── app.py              # The main Streamlit application
├── training.py         # Script used to train the CNN model
├── requirements.txt    # Python dependencies
├── custom_cnn_tumor_model.pth # The trained model (Required)
└── README.md           # This file
```

---

## 🧠 Explanation Methods

| Method       | Description                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------------- |
| **Grad-CAM** | Highlights image regions that most influenced the model’s decision using gradient-based activation maps. |
| **LIME**     | Explains model predictions by perturbing image segments (“superpixels”) and observing output changes.    |

---

##
