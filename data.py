# filepath: c:\Users\Betina Kuriakose\OneDrive\Desktop\Crop_Disease_Bot\data.py
import kagglehub

try:
    path = kagglehub.dataset_download("jawadali1045/20k-multi-class-crop-disease-images")
    print("Path to dataset files:", path)
except Exception as e:
    print("Error downloading dataset:", e)