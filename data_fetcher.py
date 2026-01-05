import os
import ee
import requests
import pandas as pd
from google.oauth2 import service_account
from tqdm import tqdm
from PIL import Image
from io import BytesIO


# INITIALIZE EARTH ENGINE

SERVICE_ACCOUNT_JSON = r"PATH OF JSON"
PROJECT_ID = "XXXX"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_JSON,
    scopes=[
        "https://www.googleapis.com/auth/earthengine",
        "https://www.googleapis.com/auth/cloud-platform"
    ]
)
ee.Initialize(credentials, project=PROJECT_ID)

# PATHS
TEST_CSV_PATH = "Data/test2(test(1)).csv"      
OUTPUT_DIR = "test_sat_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD TEST DATA

df = pd.read_csv(TEST_CSV_PATH)

# Make sure indexing is clean
df = df.reset_index(drop=True)

print("Test samples:", len(df))

# SENTINEL-2 COLLECTION

sentinel = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterDate("2020-01-01", "2021-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
)

# DOWNLOAD FUNCTION

def download_satellite_image(lat, lon, idx, size=224, scale=10):
    save_path = os.path.join(OUTPUT_DIR, f"{idx}.png")

    # Skip if already downloaded
    if os.path.exists(save_path):
        return True

    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(scale * size / 2).bounds()

        image = (
            sentinel
            .filterBounds(point)
            .median()
            .select(["B4", "B3", "B2"])
        )

        # Visualization params (CRITICAL)
        vis_params = {
            "min": 0,
            "max": 3000,
            "bands": ["B4", "B3", "B2"]
        }

        image_vis = image.visualize(**vis_params)

        url = image_vis.getThumbURL({
            "region": region,
            "dimensions": f"{size}x{size}",
            "format": "png"
        })

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(save_path)

        return True

    except Exception as e:
        print(f"[FAILED] index {idx}: {e}")
        return False


# DOWNLOAD LOOP

failed = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    lat = row["lat"]
    lon = row["long"]   # change to "lon" if column name differs

    success = download_satellite_image(lat, lon, idx)
    if not success:
        failed.append(idx)

print("\n✅ Download completed")
print("Failed images:", len(failed))
