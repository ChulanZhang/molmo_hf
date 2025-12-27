# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""COCO Caption loading script. Based on COCO dataset structure."""

import json
import os
from pathlib import Path

import datasets
import requests
from tqdm import tqdm

_URLS = {
    "annotations": {
        "train": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        "val": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    }
}

_URLS_IMAGES = {
    "train": "http://images.cocodataset.org/zips/train2014.zip",
    "val": "http://images.cocodataset.org/zips/val2014.zip",
}

_SUB_FOLDER_OR_FILE_NAME = {
    "annotations": {
        "train": "captions_train2014.json",
        "val": "captions_val2014.json",
    },
    "images": {
        "train": "train2014",
        "val": "val2014",
    },
}


def download_file(url, filename):
    """Download a file with progress bar."""
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    # Get the total file size
    total_size = int(response.headers.get('content-length', 0))
    
    # Open the local file to write the downloaded content
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


class CocoCaptionBuilder(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def __init__(self, coco_source=None):
        if coco_source is None:
            coco_source = os.getcwd()
        self.coco_source = coco_source
        os.makedirs(self.coco_source, exist_ok=True)

        super().__init__()

    def _info(self):
        features = datasets.Features(
            {
                "image_id": datasets.Value("int64"),
                "image": datasets.Value("string"),
                "captions": [datasets.Value("string")],  # Multiple captions per image
                "caption_id": datasets.Value("int64"),  # ID of the caption annotation
            }
        )
        return datasets.DatasetInfo(
            features=features,
            description="COCO Caption dataset for image captioning task."
        )
    
    def _split_generators(self, dl_manager):
        """Generate splits for train and validation."""
        downloaded_pointer = dl_manager.download(_URLS)
        downloaded_pointer["images"] = dl_manager.download(_URLS_IMAGES)

        # COCO Caption uses COCO 2014 images (train2014, val2014)
        downloaded_pointer["images"] = {
            "train": f"{self.coco_source}/train2014.zip",
            "val": f"{self.coco_source}/val2014.zip",
        }

        data_dir = dl_manager.extract(downloaded_pointer)
        
        # Since train and val annotations are in the same zip file, find the extracted directory
        # The zip file contains both captions_train2014.json and captions_val2014.json
        annotations_base_dir = None
        if "annotations" in data_dir:
            # Get the first available split's extracted directory (both point to same zip)
            for split_name in ["train", "val"]:
                if split_name in data_dir["annotations"]:
                    annotations_base_dir = Path(data_dir["annotations"][split_name])
                    break
        
        gen_kwargs = {}
        for split_name in ["val", "train"]:
            split_gen_kwargs = {}
            for dir_name in list(set(_URLS.keys()) | set(["images"])):
                if dir_name == "annotations":
                    # Annotations are in a zip file, both train and val use the same extracted directory
                    # The zip contains annotations/captions_train2014.json and annotations/captions_val2014.json
                    if annotations_base_dir is not None:
                        # Try annotations subdirectory first (standard COCO structure)
                        ann_file_path = annotations_base_dir / "annotations" / _SUB_FOLDER_OR_FILE_NAME[dir_name][split_name]
                        if not ann_file_path.exists():
                            # Try direct path if annotations subdirectory doesn't exist
                            ann_file_path = annotations_base_dir / _SUB_FOLDER_OR_FILE_NAME[dir_name][split_name]
                        split_gen_kwargs[f"{dir_name}_path"] = ann_file_path
                    else:
                        split_gen_kwargs[f"{dir_name}_path"] = None
                elif dir_name == "images":
                    if split_name in data_dir[dir_name]:
                        # Images: path to directory containing images
                        split_gen_kwargs[f"{dir_name}_path"] = Path(data_dir[dir_name][split_name]) / _SUB_FOLDER_OR_FILE_NAME[dir_name][split_name]
                    else:
                        split_gen_kwargs[f"{dir_name}_path"] = None
            gen_kwargs[split_name] = split_gen_kwargs
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs=gen_kwargs["train"],
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs=gen_kwargs["val"],
            ),
        ]

    def _generate_examples(self, annotations_path, images_path):
        """Generate examples from COCO caption annotations."""
        if annotations_path is None or not annotations_path.exists():
            raise ValueError(f"Annotations file not found: {annotations_path}")
        
        # Load COCO caption annotations
        with open(annotations_path, "r") as f:
            coco_data = json.load(f)
        
        # Group captions by image_id
        # COCO format: {"images": [...], "annotations": [{"image_id": ..., "caption": ..., "id": ...}, ...]}
        image_id_to_captions = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"]
            caption_id = ann["id"]
            
            if image_id not in image_id_to_captions:
                image_id_to_captions[image_id] = []
            image_id_to_captions[image_id].append({
                "caption": caption,
                "caption_id": caption_id
            })
        
        # Create a mapping from image_id to image info for faster lookup
        image_id_to_info = {img["id"]: img for img in coco_data["images"]}
        
        # Generate examples
        for image_id, caption_data_list in image_id_to_captions.items():
            # Get image filename from COCO images list
            image_info = image_id_to_info.get(image_id)
            if image_info is None:
                continue
            
            # COCO image filename format: COCO_{split}_{image_id:0>12}.jpg
            # Extract split from images_path (train2014 or val2014)
            # images_path is a Path object pointing to the extracted directory (e.g., train2014/)
            split_name = images_path.name  # e.g., "train2014" or "val2014"
            image_filename = f"COCO_{split_name}_{image_id:0>12}.jpg"
            image_path = images_path / image_filename
            
            # COCO Caption: Each image has multiple reference captions (typically 5)
            # We create one example per image with all captions for evaluation
            # This matches the standard COCO evaluation where all reference captions are used
            all_captions = [cd["caption"] for cd in caption_data_list]
            # Use the first caption_id as the primary ID for this example
            primary_caption_id = caption_data_list[0]["caption_id"]
            
            yield image_id, dict(
                image_id=image_id,
                image=str(image_path),
                captions=all_captions,  # All reference captions for this image
                caption_id=primary_caption_id,  # Primary caption ID (for compatibility)
            )

