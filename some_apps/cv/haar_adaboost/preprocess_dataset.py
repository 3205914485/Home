import cv2
import random
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

def read_coordinates(coordinate_file):
    """
    format: image-name Leye-x Leye-y Reye-x Reye-y nose-x nose-y mouth-x mouth-y
    """
    coordinates = {}
    with open(coordinate_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 9:
                filename = parts[0]
                # 1En -> int
                coords = [round(float(coord)) for coord in parts[1:]]
                coordinates[filename] = coords
    return coordinates

def crop_calculate(h0, w0, coordinates):
    leye_x, leye_y, reye_x, reye_y, nose_x, nose_y, mouth_x, mouth_y = coordinates
    center_x, center_y = nose_x, nose_y
    face_h = 2 * (mouth_y - (leye_y + reye_y) / 2)
    face_w = 2 * (reye_x - leye_x)
    upper = max(0, round(center_y - (face_h / 2) * 1.5))
    lower = min(h0, round(center_y + face_h / 2))
    left = max(0, round(center_x - face_w / 2))
    right = min(w0, round(center_x + face_w / 2))
    return upper, lower, left, right

def crop_face(image, coordinates):
    y1, y2, x1, x2 = crop_calculate(h0=image.shape[0], w0=image.shape[1], coordinates=coordinates)
    return image[y1:y2, x1:x2]

def crop_negative_sample(image, face_coordinates, crop_size=(48, 48)):
    h, w = image.shape[:2]
    face_y1, face_y2, face_x1, face_x2 = crop_calculate(h0=h, w0=w, coordinates=face_coordinates)
    available_regions = [
        (0, 0, w, face_y1),  # upper
        (0, face_y2, w, h),  # lower
        (0, face_y1, face_x1, face_y2),  # left
        (face_x2, face_y1, w, face_y2)  # right
    ]
    available_regions = [r for r in available_regions if (r[2]-r[0] >= crop_size[0] and r[3]-r[1] >= crop_size[1])]
    if not available_regions:
        return None  
    selected_region = random.choice(available_regions)
    x1, y1, x2, y2 = selected_region
    x = random.randint(x1, max(x1, x2 - crop_size[0]))   
    y = random.randint(y1, max(y1, y2 - crop_size[1]))  
    return image[y:y+crop_size[1], x:x+crop_size[0]]

def process_images(image_folder, coordinate_file, output_folder, labels_path, index_path, train_frac, valid_frac, test_frac):
    coordinates = read_coordinates(coordinate_file)
    count = 0
    labels = []
    for filename, coords in tqdm(coordinates.items()):
        image_path = os.path.join(image_folder, filename)
        if not os.path.exists(image_path):
            continue  
        image = cv2.imread(image_path)
        face_crop = crop_face(image, coords)
        negative_crop = crop_negative_sample(image, coords)
        face_crop_resized = cv2.resize(face_crop, (48, 48), interpolation=cv2.INTER_AREA)
        path = f'{output_folder}/{count}.jpg'
        cv2.imwrite(path, face_crop_resized)
        labels.append(1)
        count += 1
        if negative_crop is None or negative_crop.size == 0:
            print(f"failed crop , skip : {filename}")
        else:
            path = f'{output_folder}/{count}.jpg'
            cv2.imwrite(path, negative_crop)
            labels.append(0)
            count +=1
    labels_df = pd.DataFrame(data=labels, columns=['label'])
    labels_df.to_csv(os.path.join(labels_path, 'labels.csv'), index=False)
    total = count
    index = np.random.permutation(total)
    train_end = int(total * train_frac)
    valid_end = train_end + int(total * valid_frac)
    train_index, valid_index, test_index = index[:train_end], index[train_end:valid_end], index[valid_end:]
    pd.DataFrame(train_index, columns=['train_index']).to_csv(os.path.join(index_path, 'train_index.csv'), index=False)
    pd.DataFrame(valid_index, columns=['valid_index']).to_csv(os.path.join(index_path, 'valid_index.csv'), index=False)
    pd.DataFrame(test_index, columns=['test_index']).to_csv(os.path.join(index_path, 'test_index.csv'), index=False)

def main():
    image_folder = 'caltech/Caltech_WebFaces'
    coordinate_file = 'caltech/WebFaces_GroundThruth.txt'
    output_folder = 'data/pics'
    labels_path = 'data'
    index_path = 'data'
    train_frac = 0.7
    valid_frac = 0.15
    test_frac = 0.15  
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    process_images(image_folder, coordinate_file, output_folder, labels_path, index_path, train_frac, valid_frac, test_frac)

if __name__ == "__main__":
    main()
