Satellite Imagery–Based Property Valuation
Overview

This project builds a multimodal regression pipeline to predict property prices by combining:

Tabular housing attributes (size, location, condition, etc.)

Satellite imagery extracted using latitude and longitude

The motivation is to capture environmental and neighborhood context (green cover, density, road structure, proximity effects) that traditional tabular models miss.

The system integrates computer vision, geospatial analysis, and machine learning to produce more accurate real-estate valuations.

Environment Setup
1️⃣ Create Virtual Environment
python -m venv open_venv


Activate the environment:

open_venv\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt

Satellite Image Download   

* I have downloaded the images from google earth engine using their API so you need to make your project and id then the code will run *

First, put the training dataset CSV inside the Data folder and change its name inside data_fetcher.py.

Run:

python data_fetcher.py


This creates:

sat_images/ for training

test_sat_images/ for test data

Feature Extraction / Preprocessing

Run:

preprocessing.ipynb


Cell by cell.

There is also a component of geospatial analysis where figures get saved in the plots folder.

Satellite images are passed through ResNet50 (ImageNet pretrained).
Final classification layer is removed → 2048-D embeddings.

Model Training

Open and run:

model_training.ipynb


Models trained:

Tabular-only XGBoost

Image-only XGBoost

Multimodal early-fusion XGBoost

Late-fusion ensemble (optional)

Final model saved as:

models/xgb_multimodal.json

Explainability (Grad-CAM)

Grad-CAM highlights image regions influencing predictions.

Run:

python gradcam/gradcam_visualization.py

Results
Model	                              RMSE	              R²
Tabular Only	                     ~130k	             ~0.86
Image Only	                         ~128k	             ~0.87
Multimodal (Early Fusion)	         ~119k	             ~0.89