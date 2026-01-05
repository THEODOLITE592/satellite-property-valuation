Satellite Imagery–Based Property Valuation
#Overview

This project builds a multimodal regression pipeline to predict property prices by combining:

Tabular housing attributes (size, location, condition, etc.)

Satellite imagery extracted using latitude and longitude

The motivation is to capture environmental and neighborhood context (green cover, density, road structure, proximity effects) that traditional tabular models miss.

The system integrates computer vision, geospatial analysis, and machine learning to produce more accurate real-estate valuations.


Environment Setup
1 Create Virtual Environment

python -m venv open_venv

and activate using open_venv\Scripts\activate

2️ Install Dependencies

pip install -r requirements.txt

3 Satellite Image Download

first put the training data set csv in Data folder and chnage the name of its in data_fetcher.py file

python data_fetcher.py

This creates:

sat_images/ for training

test_sat_images/ for test data

Feature Extraction/ Preprocessing
 
run preprocessing.ipynb cell by cell 

there is also component of geospatial analysis where fig gets saved in plots.

Satellite images are passed through ResNet50 (ImageNet pretrained).
Final classification layer is removed → 2048-D embeddings.

Model Training

Open and run:

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