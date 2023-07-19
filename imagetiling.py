import cv2
import numpy as np
from PIL import Image
from google.colab import drive
import os
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

minimum_tile_dimension = 100 # default 50
max_number_of_files_to_process = 60

# Define your Roboflow model
from roboflow import Roboflow
rf = Roboflow(api_key="uTSQ4N3TMoWrkgf33Lm9")
project = rf.workspace().project("detection-7brqx")
model = project.version(5).model

# Mount Google Drive
drive.mount('/content/gdrive')

# Define the folder path
# folder_path = "https://drive.google.com/drive/folders/1XGyQAVJvriYgqDiQSd6yiuvojhIqHprR?usp=drive_link"
folder_path = "/content/gdrive/MyDrive/keys /MyImages"

def split_tile(image):
    h, w = image.shape[:2]
    h_mid, w_mid = h // 2, w // 2

    # Split the image into four quadrants
    tiles = [
        image[:h_mid, :w_mid],  # top left
        image[:h_mid, w_mid:],  # top right
        image[h_mid:, :w_mid],  # bottom left
        image[h_mid:, w_mid:]  # bottom right
    ]

    return tiles

def recursive_search(tiles):
    next_tiles = []
    print(f'Processing {len(tiles)} tiles.')
    for i, tile in enumerate(tiles):
        # Save the tile temporarily
        cv2.imwrite("temp.jpg", tile)

        # Infer with the Roboflow model
        results = model.predict("temp.jpg", confidence=10, overlap=30).json()
        detections = results['predictions']
        # print(detections)

        keys = [d for d in detections if d['class'] == 'keys-remote-jewelleries']  # replace 'house key' with the actual class name of the house key in your model

        if len(keys) > 0:  # if a key is detected in this tile
            return True
        elif min(tile.shape[:2]) > minimum_tile_dimension:  # if the tile is still large enough, split for the next level
            print(f'Splitting tile. tile.shape[:2]={tile.shape[:2]}')
            next_tiles.extend(split_tile(tile))

    if next_tiles:
        return recursive_search(next_tiles)

    return False

# Store results in a dataframe
results_df = pd.DataFrame(columns=['image_name', 'found_key'])

# Start the search
print(f'Processing {len(os.listdir(folder_path))} files.')
file_count = 0
for filename in os.listdir(folder_path):
    # if filename.startswith('WIN'):
    #    continue
    file_count += 1
    print(f'Processing file {file_count}')
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(folder_path, filename)
        print(img_path)
        img = Image.open(img_path)
        img = np.array(img)
        found = recursive_search([img])
        results_df = results_df.append({'image_name': filename, 'found_key': found}, ignore_index=True)
    if file_count>max_number_of_files_to_process:
        break

# Count how many images a key was found in
key_found_count = results_df['found_key'].sum()
print("Key was found in {} images.".format(key_found_count))
print(results_df)