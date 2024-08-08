import os
import cv2
import json
import glob
import shutil
import zipfile
import numpy as np
from PIL import Image
import albumentations as A
from augmentation_pipelines import hflip, vflip, colorize, darken, crop


def augment(
        input_dir, 
        output_dir, 
        annot_file, 
        transformation, 
        suffix='_aug',
        annot_suffix='_aug',
        annot_filename=None,
        pass_annotations=True,
        include_input=False
    ):
    """
    """
    def add_suffix(filename, suffix=suffix):
        name, extension = filename.rsplit('.', 1)
        new_filename = f"{name}{suffix}.{extension}"
        return new_filename

    with open(annot_file) as f:
        annotations = json.load(f)

    if pass_annotations:
        new_images = annotations['images'].copy()
        new_annotations = annotations['annotations'].copy()
    else:
        new_annotations = []
        new_images = []

    if include_input:
        shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)

    for image_info in annotations['images']:
        image_id = image_info['id']
        image_path = os.path.join(input_dir, image_info['file_name'])
        try:
            image = np.array(Image.open(image_path))
        except:
            continue
        
        image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
        bboxes = [ann['bbox'] for ann in image_annotations]
        category_ids = [ann['category_id'] for ann in image_annotations]

        transformed = transformation(image=image, bboxes=bboxes, category_ids=category_ids)
        aug_image = transformed['image']
        aug_bboxes = transformed['bboxes']
        
        aug_image_path = os.path.join(output_dir, add_suffix(image_info['file_name']))
        Image.fromarray(aug_image).save(aug_image_path)
        
        new_image_info = image_info.copy()
        new_image_info['id'] = len(new_images) + 1
        new_image_info['file_name'] = add_suffix(image_info['file_name'])
        new_images.append(new_image_info)
        
        for ann, bbox in zip(image_annotations, aug_bboxes):
            new_ann = ann.copy()
            new_ann['id'] = new_image_info['id'] #
            new_ann['image_id'] = new_image_info['id']
            new_ann['bbox'] = bbox
            new_annotations.append(new_ann) 

    new_annotations_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': annotations['categories']
    }
    
    if annot_filename != None:
        new_annot_file = annot_filename
    else:    
        new_annot_file = add_suffix(annot_file, annot_suffix)
    
    with open(new_annot_file, 'w') as f:
        json.dump(new_annotations_data, f)

    print("Augmentation complete.")

    return new_annot_file


if __name__ == "__main__":
    # Paths
    img_dir = 'images/'                        # original dataset (103 images)
    aug_dir_1 = 'images-augmented-1/'          # geometric (mostly) augmentations
    aug_dir_2 = 'images-augmented-2/'          # color (mostly) augmentations
    aug_dir_final = 'images-augmented-final/'  # cropping and last steps
    annotation_file = 'annotations.json'       # annotations of the original dataset

    os.makedirs(aug_dir_1, exist_ok=True)
    os.makedirs(aug_dir_2, exist_ok=True)
    os.makedirs(aug_dir_final, exist_ok=True)

    # horizontal flip & first step augmentations
    final_annotation_file = augment(
        img_dir, 
        aug_dir_1, 
        annotation_file, 
        hflip, 
        suffix='_hflip', 
        annot_suffix='_GeometricAug',
        pass_annotations=True,
        include_input=True
    )

    # vertical flip & first step augmentations
    augment(
        img_dir, 
        aug_dir_1, 
        final_annotation_file, 
        vflip, 
        suffix='_vflip', 
        annot_suffix='',
        pass_annotations=True,
        include_input=False
    )

    # colorize images & second step augmentations
    final_annotation_file = augment(
        aug_dir_1, 
        aug_dir_2, 
        final_annotation_file, 
        colorize, 
        suffix='_colored', 
        annot_suffix='_ColorAug',
        pass_annotations=True,
        include_input=True
    )

    # de-colorize images & second step augmentations
    augment(
        aug_dir_1, 
        aug_dir_2, 
        final_annotation_file, 
        darken, 
        suffix='_darkened', 
        annot_suffix='',
        pass_annotations=True,
        include_input=False
    )

    # final steps (cropping, resizing, etc.)
    augment(
        aug_dir_2, 
        aug_dir_final, 
        final_annotation_file, 
        crop, 
        suffix='_cropped', 
        annot_filename='annotations.json',
        pass_annotations=True,
        include_input=True
    )

    # file operations & ZIP compression
    shutil.rmtree(img_dir)
    shutil.rmtree(aug_dir_1)
    shutil.rmtree(aug_dir_2)
    os.rename(aug_dir_final, 'images/')

    files = ['images/', 'annotations.json']
    zip_path  = 'tray-track.zip'

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in files:
            zipf.write(file, os.path.basename(file))

    shutil.rmtree('images/')
    
    json_files = glob.glob('*.json')
    for json_file in json_files:
        os.remove(json_file)
