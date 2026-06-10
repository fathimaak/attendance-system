import urllib.request
import bz2
import os

models = {
    "shape_predictor_68_face_landmarks.dat": 
        "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    "dlib_face_recognition_resnet_model_v1.dat": 
        "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
}

os.makedirs("models", exist_ok=True)

for filename, url in models.items():
    save_path = f"models/{filename}"
    if os.path.exists(save_path):
        print(f"Already exists: {filename}")
        continue
    print(f"Downloading {filename}... (this may take a few minutes)")
    bz2_path = save_path + ".bz2"
    urllib.request.urlretrieve(url, bz2_path)
    print(f"Extracting...")
    with bz2.open(bz2_path, "rb") as f:
        data = f.read()
    with open(save_path, "wb") as f:
        f.write(data)
    os.remove(bz2_path)
    print(f"Done: {filename}")

print("\nAll models ready.")