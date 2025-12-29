import json
import os
from collections import defaultdict
from os.path import join, exists

import datasets


class TallyQaBuilder(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version('1.0.0')

    def __init__(self, coco_source=None, *args, **kwargs):
        """
        Args:
            coco_source: Optional path to directory containing local COCO 2014 zip files.
                        If provided, will use local files instead of downloading.
                        Expected files: train2014.zip, val2014.zip
        """
        import logging
        log = logging.getLogger(__name__)
        
        if coco_source is None:
            # Try to use MOLMO_DATA_DIR if available
            if "MOLMO_DATA_DIR" in os.environ:
                from molmo.data.dataset import DATA_HOME
                coco_source = join(DATA_HOME, "downloads")
                log.info(f"MOLMO_DATA_DIR found: {os.environ['MOLMO_DATA_DIR']}")
                log.info(f"Using coco_source: {coco_source}")
            else:
                coco_source = os.getcwd()
                log.warning("MOLMO_DATA_DIR not set, using current directory as coco_source")
        
        # Use absolute path to ensure correct file detection
        self.coco_source = os.path.abspath(coco_source)
        os.makedirs(self.coco_source, exist_ok=True)
        log.info(f"TallyQaBuilder initialized with coco_source: {self.coco_source}")
        super().__init__(*args, **kwargs, dataset_name="tally_qa")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'image': datasets.Image(),
                "image_id": datasets.Value("int32"),
                "image/filename": datasets.Value("string"),
                "questions": datasets.Sequence(datasets.Features({
                    "answer": datasets.Value("int32"),
                    "issimple": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "data_source": datasets.Value("string"),
                    "question_id": datasets.Value("int64"),
                }))
            }),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        import logging
        log = logging.getLogger(__name__)
        
        # Use absolute path to ensure correct file detection
        coco_source_abs = os.path.abspath(self.coco_source)
        
        # Check if local COCO files exist
        train2014_local = join(coco_source_abs, "train2014.zip")
        val2014_local = join(coco_source_abs, "val2014.zip")
        
        log.info(f"Checking for local COCO files in: {coco_source_abs}")
        log.info(f"  train2014.zip: {train2014_local} (exists: {exists(train2014_local)})")
        log.info(f"  val2014.zip: {val2014_local} (exists: {exists(val2014_local)})")
        
        # Use local files if they exist, otherwise download
        image_urls = {}
        if exists(train2014_local):
            image_urls["train2014"] = train2014_local
            log.info(f"Using local train2014.zip: {train2014_local}")
        else:
            image_urls["train2014"] = "http://images.cocodataset.org/zips/train2014.zip"
            log.info("Will download train2014.zip from remote")
            
        if exists(val2014_local):
            image_urls["val2014"] = val2014_local
            log.info(f"Using local val2014.zip: {val2014_local}")
        else:
            image_urls["val2014"] = "http://images.cocodataset.org/zips/val2014.zip"
            log.info("Will download val2014.zip from remote")
        
        # Download remote files first (similar to VQA2 approach)
        # Note: These are NOT COCO files - they are TallyQA-specific files
        remote_urls = {
            "VG_100K": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
            "VG_100K_2": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
            "src": "https://github.com/manoja328/tallyqa/blob/master/tallyqa.zip?raw=true",
        }
        
        log.info("Downloading TallyQA-specific files (VG_100K, VG_100K_2, src)...")
        log.info("Note: COCO files (train2014.zip, val2014.zip) will use local files if available.")
        
        # Download remote files
        downloaded = dl_manager.download(remote_urls)
        
        # Override with local file paths (similar to VQA2 approach)
        # This tells extract() to use local files directly without downloading
        downloaded["train2014"] = image_urls["train2014"]
        downloaded["val2014"] = image_urls["val2014"]
        
        log.info(f"Files to extract:")
        log.info(f"  - train2014: {downloaded['train2014']} (local file, will be extracted)")
        log.info(f"  - val2014: {downloaded['val2014']} (local file, will be extracted)")
        log.info(f"  - VG_100K, VG_100K_2, src: (downloaded from remote)")
        log.info("Starting extraction process (this may show progress bars for large files)...")
        
        # Extract all files (local files should be used directly by extract())
        # Note: The progress bar you see is likely from extracting the large COCO zip files
        # (train2014.zip is 9.73GB, val2014.zip is 5.47GB), not from downloading
        data = dl_manager.extract(downloaded)
        return [
            datasets.SplitGenerator(name=name, gen_kwargs=dict(
                images=data, src=join(data["src"], f"{name}.json")))
            for name in ["train", "test"]
        ]

    def _generate_examples(self, src, images):
        with open(src) as f:
            data = json.load(f)
        grouped_by_image = defaultdict(list)
        for ex in data:
            grouped_by_image[ex["image"]].append(ex)

        for image, questions in grouped_by_image.items():
            image_id = questions[0]["image_id"]
            for q in questions:
                assert q.pop("image_id") == image_id
                assert q.pop("image") == image
                if "issimple" in q:
                    q["issimple"] = int(q["issimple"])
                else:
                    q["issimple"] = -1
            image_src, path = image.split("/")
            image_path = join(images[image_src], image_src, path)
            if not exists(image_path):
                import logging
                log = logging.getLogger(__name__)
                log.warning(f"Image path does not exist: {image_path}, skipping image_id={image_id}")
                continue
            yield image_id, {
                "image_id": image_id,
                "image": image_path,
                "questions": questions,
                "image/filename": image
            }
