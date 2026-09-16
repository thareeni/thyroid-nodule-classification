# utils/data_utils.py
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_NAMES = ['benign', 'malignant']  # 0: benign, 1: malignant

def parse_voc_annotation(xml_path):
    """Parse Pascal VOC XML annotation file to extract class labels"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        objects = root.findall('object')
        if not objects:
            return None
            
        obj = objects[0]
        class_value = obj.find('name').text
        
        # Handle both string and numeric values
        if class_value == '0' or class_value == 'benign':
            return 0
        elif class_value == '1' or class_value == 'malignant':
            return 1
        else:
            try:
                val = int(class_value)
                if val == 0 or val == 1:
                    return val
            except:
                pass
            return None
            
    except Exception as e:
        return None

def load_dataset_from_voc(dataset_root, split='train', target_size=(224, 224), batch_size=32):
    """Generator that yields batches of images from Pascal VOC dataset"""
    jpeg_dir = Path(dataset_root) / 'JPEGImages'
    annotation_dir = Path(dataset_root) / 'Annotations'
    imageset_file = Path(dataset_root) / 'ImageSets' / 'Main' / f'{split}.txt'
    
    if not imageset_file.exists():
        raise FileNotFoundError(f"ImageSet file not found: {imageset_file}")
    
    with open(imageset_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]
    
    # Create list of (image_path, label) pairs
    image_label_pairs = []
    for img_id in image_ids:
        img_path = jpeg_dir / f'{img_id}.jpg'
        if not img_path.exists():
            img_path = jpeg_dir / f'{img_id}.png'
            
        if img_path.exists():
            xml_path = annotation_dir / f'{img_id}.xml'
            if xml_path.exists():
                label = parse_voc_annotation(xml_path)
                if label is not None:
                    image_label_pairs.append((str(img_path), label))
    
    num_samples = len(image_label_pairs)
    for i in range(0, num_samples, batch_size):
        batch_pairs = image_label_pairs[i:i+batch_size]
        
        batch_images = []
        batch_labels = []
        
        for img_path, label in batch_pairs:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img = img.astype(np.float32) / 255.0
            
            batch_images.append(img)
            batch_labels.append(label)
        
        if batch_images:
            yield np.array(batch_images), np.array(batch_labels)

def get_dataset_info(dataset_root, split='train'):
    """Get dataset statistics without loading images"""
    annotation_dir = Path(dataset_root) / 'Annotations'
    imageset_file = Path(dataset_root) / 'ImageSets' / 'Main' / f'{split}.txt'
    
    if not imageset_file.exists():
        return 0, {0: 0, 1: 0}
    
    with open(imageset_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]
    
    class_counts = {0: 0, 1: 0}
    
    for img_id in image_ids:
        xml_path = annotation_dir / f'{img_id}.xml'
        if xml_path.exists():
            label = parse_voc_annotation(xml_path)
            if label is not None:
                class_counts[label] += 1
    
    total = sum(class_counts.values())
    return total, class_counts