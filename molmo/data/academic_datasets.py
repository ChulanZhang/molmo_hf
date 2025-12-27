import json
import logging
import re
from collections import defaultdict
from os.path import exists
from os.path import join

import datasets
import numpy as np

from molmo.data.dataset import DATA_HOME, DatasetBase, Dataset, HfDataset
from molmo.hf_datasets.a_okvqa import AOkVqaBuilder
from molmo.hf_datasets.ai2d import Ai2dDatasetBuilder
from molmo.hf_datasets.android_control import AndroidControlBuilder
from molmo.hf_datasets.clock_bench import ClockBenchBuilder
from molmo.hf_datasets.count_qa import CountQaBuilder
from molmo.hf_datasets.dv_qa import DvQaBuilder
from molmo.hf_datasets.figure_qa import FigureQaBuilder
from molmo.hf_datasets.plot_qa import PlotQaBuilder
from molmo.hf_datasets.tabmwp import TabMwpBuilder
from molmo.hf_datasets.tally_qa import TallyQaBuilder
from molmo.hf_datasets.vqa_v2 import VQAv2BuilderMultiQA
from molmo.hf_datasets.coco_caption import CocoCaptionBuilder

if DATA_HOME is not None:
    DOWNLOADS = join(DATA_HOME, "downloads")
    INFOQA_SOURCE = join(DATA_HOME, "info_qa")
    ST_QA_SRC = join(DATA_HOME, "scene-text")
else:
    DOWNLOADS = None
    INFOQA_SOURCE = None
    ST_QA_SRC = None


class ChartQa(HfDataset):
    """
    ChartQA dataset from HuggingFace M4 project.
    This class loads the ChartQA dataset from HuggingFace (https://huggingface.co/datasets/HuggingFaceM4/ChartQA).

    Args:
        split (str): Dataset split to load. One of "train", "validation", or "test".
        parts (str, optional): Which subset of examples to include. One of:
            - "human": Only human-authored examples
            - "augmented": Only automatically generated examples
            - "both": Both human and augmented examples (default)
        weighted (bool, optional): Whether to apply weighting to balance human/augmented examples. Only valid when parts="both".
            Defaults to False.
    """
    PATH = "HuggingFaceM4/ChartQA"

    def __init__(self, split: str, parts="both", weighted=False, keep_in_memory=False):
        assert split in ["train", "validation", "test"]
        assert parts in ["human", "augmented", "both"]

        if split == "validation":
            split = "val"
        self.updated_split = split
        self.weighted = weighted
        self.parts = parts
        super().__init__(split, keep_in_memory=keep_in_memory)
        if self.parts != "both":
            # Filter out either human or aug datasets
            flags = [int(self.parts == "human")]
            self.dataset = self.dataset.filter(
                lambda x: x in flags,
                input_columns=["human_or_machine"]
            )

    def get(self, item, rng):
        ex = self.dataset[item]
        ex = dict(
            image=ex["image"],
            question=ex["query"],
            answers=ex["label"],
            style="chart_qa",
            metadata=dict(
                is_human=ex['human_or_machine'],
            )
        )
        if self.weighted:
            is_human = ex["metadata"]["is_human"]
            # Weight to balanced human/augmented sets
            if is_human:
                w = 2*20901/(20901+7398)
            else:
                w = 2*7398/(20901+7398)
            ex["weight"] = w
        return ex


class Vqa2(Dataset):
    @classmethod
    def download(cls, n_procs=1):
        import os
        import fsspec
        from fsspec.implementations.http import HTTPFileSystem
        from aiohttp import ClientTimeout
        
        # Set longer timeout for fsspec/aiohttp downloads (in seconds)
        # Default is usually 60 seconds, set to 3600 (1 hour) for large files like COCO
        # This affects the timeout for reading chunks from the HTTP stream
        
        # Store original timeout settings
        original_timeout = os.environ.get("FSSPEC_TIMEOUT")
        original_default_timeout = getattr(HTTPFileSystem, '_default_timeout', None)
        logging.info(f"Vqa2.download called. DOWNLOADS path: {DOWNLOADS}")
        original_client_kwargs = getattr(HTTPFileSystem, 'client_kwargs', {})
        
        try:
            # Set environment variable (may be used by some fsspec implementations)
            os.environ["FSSPEC_TIMEOUT"] = "3600"
            
            # Patch HTTPFileSystem to use longer timeout
            # This increases the timeout for reading data chunks during download
            HTTPFileSystem._default_timeout = 3600
            
            # Also patch client_kwargs to set aiohttp timeout
            # This is the actual timeout used by aiohttp for reading chunks
            if not hasattr(HTTPFileSystem, 'client_kwargs') or HTTPFileSystem.client_kwargs is None:
                HTTPFileSystem.client_kwargs = {}
            HTTPFileSystem.client_kwargs = HTTPFileSystem.client_kwargs.copy()
            HTTPFileSystem.client_kwargs['timeout'] = ClientTimeout(total=3600, connect=60)
            
            # Create DownloadConfig with retry mechanism
            # max_retries helps with network issues, but CRC errors may need cache cleanup
            download_config = datasets.DownloadConfig(
                num_proc=n_procs,
                max_retries=5,  # Add retry mechanism for failed downloads
            )
            
            try:
                logging.info(f"Instantiating VQAv2BuilderMultiQA with source: {DOWNLOADS}")
                builder = VQAv2BuilderMultiQA(DOWNLOADS)
                logging.info(f"Calling download_and_prepare...")
                builder.download_and_prepare(download_config=download_config)
                logging.info(f"download_and_prepare completed.")
            except Exception as e:
                # If we get CRC errors, it means the downloaded file is corrupted
                # Log a helpful message about cleaning cache
                error_msg = str(e)
                if "CRC" in error_msg or "Bad CRC" in error_msg:
                    logging.warning(
                        f"CRC error detected. This usually means the downloaded file is corrupted. "
                        f"Try cleaning the cache and retrying:\n"
                        f"  rm -rf ~/.cache/huggingface/datasets/*vqa*\n"
                        f"  rm -rf {DOWNLOADS}/*train2014*.zip {DOWNLOADS}/*val2014*.zip {DOWNLOADS}/*test2015*.zip\n"
                        f"Then run the download again."
                    )
                raise
        finally:
            # Restore original timeout settings
            if original_timeout is not None:
                os.environ["FSSPEC_TIMEOUT"] = original_timeout
            elif "FSSPEC_TIMEOUT" in os.environ:
                del os.environ["FSSPEC_TIMEOUT"]
            
            if original_default_timeout is not None:
                HTTPFileSystem._default_timeout = original_default_timeout
            elif hasattr(HTTPFileSystem, '_default_timeout'):
                delattr(HTTPFileSystem, '_default_timeout')
            
            # Restore original client_kwargs
            if original_client_kwargs:
                HTTPFileSystem.client_kwargs = original_client_kwargs
            elif hasattr(HTTPFileSystem, 'client_kwargs'):
                HTTPFileSystem.client_kwargs = {}

    def __init__(self, split, multi_question=False):
        assert split in ["train", "validation", "test"]
        self.multi_question = multi_question
        self.dataset = VQAv2BuilderMultiQA(DOWNLOADS).as_dataset(split=split)
        if not self.multi_question:
            flattened_data = []
            for item in self.dataset:
                for q in item["messages"]:
                    flattened_data.append(dict(
                        style=q['style'],
                        question=q["question"],
                        answers=q["answers"],
                        image=item["image"],
                        image_id=item["image_id"],
                        question_id=q["question_id"],
                    ))
            self.dataset = flattened_data

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        ex = self.dataset[item]
        if self.multi_question:
            return dict(
                metadata=dict(image_id=ex["image_id"]),
                image=ex["image"],
                message_list=ex["messages"],
            )
        else:
            return dict(
                style="vqa2",
                answers=ex["answers"],
                metadata=dict(image_id=ex["image_id"], example_id=ex["question_id"]),
                image=ex["image"],
                question=ex["question"],
            )            


class CocoCaption(Dataset):
    """
    COCO Caption dataset for image captioning task.
    
    This dataset loads COCO 2014 images with their captions. Each image has multiple
    reference captions (typically 5) for evaluation.
    
    Args:
        split: Dataset split to load. One of "train", "validation", or "test".
               Note: COCO Caption only has train and validation splits.
    """
    
    @classmethod
    def download(cls, n_procs=1):
        """Download COCO Caption dataset."""
        import os
        import fsspec
        from fsspec.implementations.http import HTTPFileSystem
        from aiohttp import ClientTimeout
        
        # Set longer timeout for fsspec/aiohttp downloads (similar to Vqa2)
        original_timeout = os.environ.get("FSSPEC_TIMEOUT")
        original_default_timeout = getattr(HTTPFileSystem, '_default_timeout', None)
        logging.info(f"CocoCaption.download called. DOWNLOADS path: {DOWNLOADS}")
        original_client_kwargs = getattr(HTTPFileSystem, 'client_kwargs', {})
        
        try:
            # Set environment variable
            os.environ["FSSPEC_TIMEOUT"] = "3600"
            
            # Patch HTTPFileSystem to use longer timeout
            HTTPFileSystem._default_timeout = 3600
            
            # Patch client_kwargs
            if not hasattr(HTTPFileSystem, 'client_kwargs') or HTTPFileSystem.client_kwargs is None:
                HTTPFileSystem.client_kwargs = {}
            HTTPFileSystem.client_kwargs = HTTPFileSystem.client_kwargs.copy()
            HTTPFileSystem.client_kwargs['timeout'] = ClientTimeout(total=3600, connect=60)
            
            # Create DownloadConfig with retry mechanism
            download_config = datasets.DownloadConfig(
                num_proc=n_procs,
                max_retries=5,
            )
            
            try:
                logging.info(f"Instantiating CocoCaptionBuilder with source: {DOWNLOADS}")
                builder = CocoCaptionBuilder(DOWNLOADS)
                logging.info(f"Calling download_and_prepare...")
                builder.download_and_prepare(download_config=download_config)
                logging.info(f"download_and_prepare completed.")
            except Exception as e:
                error_msg = str(e)
                if "CRC" in error_msg or "Bad CRC" in error_msg:
                    logging.warning(
                        f"CRC error detected. This usually means the downloaded file is corrupted. "
                        f"Try cleaning the cache and retrying:\n"
                        f"  rm -rf ~/.cache/huggingface/datasets/*coco*\n"
                        f"  rm -rf {DOWNLOADS}/*train2014*.zip {DOWNLOADS}/*val2014*.zip\n"
                        f"Then run the download again."
                    )
                raise
        finally:
            # Restore original timeout settings
            if original_timeout is not None:
                os.environ["FSSPEC_TIMEOUT"] = original_timeout
            elif "FSSPEC_TIMEOUT" in os.environ:
                del os.environ["FSSPEC_TIMEOUT"]
            
            if original_default_timeout is not None:
                HTTPFileSystem._default_timeout = original_default_timeout
            elif hasattr(HTTPFileSystem, '_default_timeout'):
                delattr(HTTPFileSystem, '_default_timeout')
            
            if original_client_kwargs:
                HTTPFileSystem.client_kwargs = original_client_kwargs
            elif hasattr(HTTPFileSystem, 'client_kwargs'):
                HTTPFileSystem.client_kwargs = {}

    def __init__(self, split):
        assert split in ["train", "validation", "test"]
        # COCO Caption only has train and validation splits
        if split == "test":
            logging.warning("COCO Caption doesn't have a test split, using validation instead")
            split = "validation"
        
        self.dataset = CocoCaptionBuilder(DOWNLOADS).as_dataset(split=split)

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        """Get a single example from the dataset."""
        ex = self.dataset[item]
        return dict(
            style="coco_captioning",  # Matches the style in data_formatter.py
            image=ex["image"],
            captions=ex["captions"],  # List of reference captions
            metadata=dict(
                image_id=ex["image_id"],
                caption_id=ex["caption_id"],
            ),
        )


class AOkVqa(Dataset):
    @classmethod
    def download(cls, n_procs=1):
        AOkVqaBuilder(DOWNLOADS).download_and_prepare()

    def __init__(self, split, direct_answer=False):
        self.split = split
        self.direct_answer = direct_answer
        self.dataset = AOkVqaBuilder(DOWNLOADS).as_dataset(split=split)
        self.style = "a_okvqa_" + ("da" if direct_answer else "mc")
        self.loaded_data = self.load()

    def load(self):
        loaded_data = []
        for example in self.dataset:
            if self.direct_answer:
                if example["difficult_direct_answer"] and self.split in ["validation", "test"]:
                    continue
                out = dict(
                    image=example["image"],
                    question=example["question"],
                    answers=example["direct_answers"],
                    metadata=dict(
                        example_id=example["question_id"]
                    )
                )
            else:
                if example["correct_choice_idx"] is None:
                    out = dict(
                        image=example["image"],
                        question=example["question"],
                        options=example["choices"],
                        metadata=dict(example_id=example["question_id"])
                    )
                else:
                    out = dict(
                        image=example["image"],
                        question=example["question"],
                        options=example["choices"],
                        answer_idx=example["correct_choice_idx"],
                        metadata=dict(example_id=example["question_id"])
                    )
            loaded_data.append(out)
        return loaded_data

    def __len__(self):
        return len(self.loaded_data)

    def get(self, item, rng):
        return dict(**self.loaded_data[item], style=self.style)


class OkVqa(Dataset):
    """
    OK-VQA dataset from HuggingFace M4 project.
    This class loads the OK-VQA dataset from HuggingFace (https://huggingface.co/datasets/HuggingFaceM4/OK-VQA).

    Args:
        split (str): Dataset split to load. One of "train", "validation", or "test".
        multi_question (bool, optional): Whether to group questions by image. Defaults to False.
    """

    PATH = "HuggingFaceM4/OK-VQA"

    @classmethod
    def download(cls, n_procs=1):
        # For datasets 4.4.1+, use load_dataset instead of load_dataset_builder
        try:
            datasets.load_dataset(cls.PATH)
        except RuntimeError as e:
            # If dataset uses loading scripts, try with trust_remote_code=True
            if "Dataset scripts are no longer supported" in str(e) or "loading scripts" in str(e).lower():
                try:
                    datasets.load_dataset(cls.PATH, trust_remote_code=True)
                except Exception:
                    # Fallback: try loading common splits with trust_remote_code
                    for split in ["train", "validation", "test", "val"]:
                        try:
                            datasets.load_dataset(cls.PATH, split=split, trust_remote_code=True)
                        except Exception:
                            continue
            else:
                # Fallback: try loading common splits
                for split in ["train", "validation", "test", "val"]:
                    try:
                        datasets.load_dataset(cls.PATH, split=split)
                    except Exception:
                        continue
        except Exception:
            # Fallback: try loading common splits
            for split in ["train", "validation", "test", "val"]:
                try:
                    datasets.load_dataset(cls.PATH, split=split)
                except Exception:
                    continue

    def __init__(self, split: str, multi_question=False, keep_in_memory=False):
        super().__init__()
        self.multi_question = multi_question
        # For datasets 3.0.0, OK-VQA requires trust_remote_code=True
        # Try loading with trust_remote_code=True first (for datasets 3.0.0)
        try:
            dataset = datasets.load_dataset(
                self.PATH, split=split, keep_in_memory=keep_in_memory, trust_remote_code=True
            )
        except Exception as e:
            # Fallback: try without trust_remote_code (for newer versions that don't need it)
            try:
                dataset = datasets.load_dataset(
                    self.PATH, split=split, keep_in_memory=keep_in_memory
                )
            except Exception as e2:
                error_msg = str(e2)
                # Check if it's a network/download error
                if "ClientPayloadError" in error_msg or "ContentLengthError" in error_msg or "SSL" in error_msg or "timeout" in error_msg.lower():
                    raise RuntimeError(
                        f"Network error while downloading {self.PATH}. "
                        f"This is usually due to network instability or interrupted download. "
                        f"Please try:\n"
                        f"  1. Run the download script first: python scripts/download_data.py okvqa --n_procs 1\n"
                        f"  2. Or retry the experiment later when network is stable.\n"
                        f"  3. If the download keeps failing, you may need to manually download the dataset.\n"
                        f"Original error: {e2}"
                    ) from e2
                else:
                    raise RuntimeError(
                        f"Failed to load dataset {self.PATH} with split {split}. "
                        f"Error: {e2}"
                    ) from e2
        if self.multi_question:
            grouped_by_image = defaultdict(list)
            for ex in dataset:
                grouped_by_image[ex["image_id"]].append(ex)
            data = []
            for image_id, examples in grouped_by_image.items():
                questions = []
                for ex in examples:
                    questions.append(dict(
                        question=ex["question"],
                        answers=[x["raw_answer"] for x in ex["answers"]],
                    ))
                data.append(dict(
                    image=examples[0]["image"],
                    metadata=dict(image_id=image_id),
                    message_list=questions
                ))
            self.data = data
        else:
            self.data = dataset

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        if self.multi_question:
            return dict(ex, style="okvqa")
        else:
            return dict(
                image=ex["image"],
                question=ex["question"],
                answers=[x["raw_answer"] for x in ex["answers"]],
                metadata=dict(
                    example_id=ex["question_id"],
                ),
                style="okvqa",
            )


class TextVqa(HfDataset):
    """
    This class loads the TextVQA dataset from HuggingFace (https://huggingface.co/datasets/facebook/textvqa).
    """
    PATH = "facebook/textvqa"

    @classmethod
    def download(cls, n_procs=1):
        # For datasets 4.4.1+, use load_dataset instead of load_dataset_builder
        try:
            datasets.load_dataset(cls.PATH)
        except RuntimeError as e:
            # If dataset uses loading scripts, try with trust_remote_code=True
            if "Dataset scripts are no longer supported" in str(e) or "loading scripts" in str(e).lower():
                try:
                    datasets.load_dataset(cls.PATH, trust_remote_code=True)
                except Exception:
                    # Fallback: try loading common splits with trust_remote_code
                    for split in ["train", "validation", "test", "val"]:
                        try:
                            datasets.load_dataset(cls.PATH, split=split, trust_remote_code=True)
                        except Exception:
                            continue
            else:
                # Fallback: try loading common splits
                for split in ["train", "validation", "test", "val"]:
                    try:
                        datasets.load_dataset(cls.PATH, split=split)
                    except Exception:
                        continue
        except Exception:
            # Fallback: try loading common splits
            for split in ["train", "validation", "test", "val"]:
                try:
                    datasets.load_dataset(cls.PATH, split=split)
                except Exception:
                    continue

    def __init__(self, split: str, identifier=None, keep_in_memory=False):
        # For datasets 3.0.0, TextVQA requires trust_remote_code=True
        # Override the base class to ensure trust_remote_code is used
        self.split = split
        try:
            # Try with trust_remote_code=True first (for datasets 3.0.0)
            self.dataset = datasets.load_dataset(
                self.PATH, split=split, trust_remote_code=True, keep_in_memory=keep_in_memory
            )
        except Exception as e:
            # Fallback: try without trust_remote_code (for newer versions)
            try:
                self.dataset = datasets.load_dataset(
                    self.PATH, split=split, keep_in_memory=keep_in_memory
                )
            except Exception as e2:
                error_msg = str(e2)
                # Check if it's a network/download error
                if "ClientPayloadError" in error_msg or "ContentLengthError" in error_msg or "SSL" in error_msg or "timeout" in error_msg.lower():
                    raise RuntimeError(
                        f"Network error while downloading {self.PATH}. "
                        f"This is usually due to network instability or interrupted download. "
                        f"Please try:\n"
                        f"  1. Run the download script first: python scripts/download_data.py textvqa --n_procs 1\n"
                        f"  2. Or retry the experiment later when network is stable.\n"
                        f"  3. If the download keeps failing, you may need to manually download the dataset.\n"
                        f"Original error: {e2}"
                    ) from e2
                else:
                    raise RuntimeError(
                        f"Failed to load dataset {self.PATH} with split {split}. "
                        f"Error: {e2}"
                    ) from e2
        # Call parent __init__ but skip the dataset loading part
        # We've already loaded the dataset above
        # HfDataset inherits from Dataset, so we need to call Dataset.__init__ directly
        from molmo.data.dataset import Dataset
        Dataset.__init__(self)

    def get(self, item, rng):
        example = self.dataset[item]
        return dict(
            image=example["image"],
            question=example["question"],
            answers=example.get("answers", []),
            metadata=dict(
                image_url=example["flickr_300k_url"],
                image_id=example["image_id"],
                example_id=example["question_id"],
            ),
            style="text_vqa"
        )


class TallyQa(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        import os
        import fsspec
        from fsspec.implementations.http import HTTPFileSystem
        from aiohttp import ClientTimeout
        
        # Set longer timeout for fsspec/aiohttp downloads (in seconds)
        # Default is usually 60 seconds, set to 10800 (3 hours) for large files like COCO
        # This affects the timeout for reading chunks from the HTTP stream
        # Large files like train2014.zip (9.73GB) and val2014.zip (5.47GB) need very long timeouts
        
        # Store original timeout settings
        original_timeout = os.environ.get("FSSPEC_TIMEOUT")
        original_default_timeout = getattr(HTTPFileSystem, '_default_timeout', None)
        original_client_kwargs = getattr(HTTPFileSystem, 'client_kwargs', {})
        
        try:
            # Set environment variable (may be used by some fsspec implementations)
            # Use 10800 seconds (3 hours) for very large files
            timeout_seconds = 10800
            os.environ["FSSPEC_TIMEOUT"] = str(timeout_seconds)
            
            # Patch HTTPFileSystem to use longer timeout
            # This increases the timeout for reading data chunks during download
            HTTPFileSystem._default_timeout = timeout_seconds
            
            # Also patch client_kwargs to set aiohttp timeout
            # This is the actual timeout used by aiohttp for reading chunks
            if not hasattr(HTTPFileSystem, 'client_kwargs') or HTTPFileSystem.client_kwargs is None:
                HTTPFileSystem.client_kwargs = {}
            HTTPFileSystem.client_kwargs = HTTPFileSystem.client_kwargs.copy()
            # Set total timeout to 3 hours, connect timeout to 5 minutes
            HTTPFileSystem.client_kwargs['timeout'] = ClientTimeout(total=timeout_seconds, connect=300)
            
            # Create DownloadConfig with retry mechanism
            # Increased retries for large files that may have network issues
            download_config = datasets.DownloadConfig(
                num_proc=n_procs,
                max_retries=10,  # Increased retries for large files with network instability
            )
            # Try to use local COCO files if available
            coco_source = None
            if DATA_HOME is not None:
                coco_source = DOWNLOADS
            TallyQaBuilder(coco_source=coco_source).download_and_prepare(download_config=download_config)
        finally:
            # Restore original timeout settings
            if original_timeout is not None:
                os.environ["FSSPEC_TIMEOUT"] = original_timeout
            elif "FSSPEC_TIMEOUT" in os.environ:
                del os.environ["FSSPEC_TIMEOUT"]
            
            if original_default_timeout is not None:
                HTTPFileSystem._default_timeout = original_default_timeout
            elif hasattr(HTTPFileSystem, '_default_timeout'):
                delattr(HTTPFileSystem, '_default_timeout')
            
            # Restore original client_kwargs
            if original_client_kwargs:
                HTTPFileSystem.client_kwargs = original_client_kwargs
            elif hasattr(HTTPFileSystem, 'client_kwargs'):
                HTTPFileSystem.client_kwargs = {}

    def __init__(self, split, coco_source=None):
        """
        Args:
            split: Dataset split ("train" or "test")
            coco_source: Optional path to directory containing local COCO 2014 zip files.
                        If provided, will use local files instead of downloading.
                        Expected files: train2014.zip, val2014.zip
        """
        assert split in ["train", "test"]
        self.dataset = TallyQaBuilder(coco_source=coco_source).as_dataset(split=split)
        super().__init__()

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        ex = self.dataset[item]
        messages = []
        questions = ex["questions"]
        answers = []
        for ix, question in enumerate(questions["question"]):
            answer = str(questions["answer"][ix])
            answers.append(answer)
            messages.append(dict(
                question=question,
                answer=answer,
                style="tally_qa"
            ))
        return dict(
            image=ex["image"],
            message_list=messages,
            metadata=dict(
                image_id=ex["image_id"],
                answers=answers  # Add answers to metadata for evaluation
            )
        )


class AI2D(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        Ai2dDatasetBuilder().download_and_prepare()

    def __init__(self, split, boxes="both"):
        assert split in ["train", "validation", "test"]
        dataset = Ai2dDatasetBuilder().as_dataset(split)
        if boxes == "transparent":
            dataset = dataset.filter(lambda x: not x["abc_label"] or x["has_transparent_box"])
        elif boxes == "opaque":
            dataset = dataset.filter(lambda x: not x["abc_label"] or not x["has_transparent_box"])
        elif boxes == "both":
            pass
        else:
            raise NotImplementedError(boxes)
        self.dataset = dataset

        self.split = split
        self.boxes = boxes
        super().__init__()

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        _ex = dict(self.dataset[item])
        ex = dict(
            image=_ex["image"],
            question=_ex["question"],
            answer_idx=_ex["correct_answer"],
            metadata=dict(
                example_id=_ex["question_id"],
                image_id=_ex["image_id"],
                abc_label=_ex["abc_label"],
                has_transparent_box=_ex["has_transparent_box"]
            ),
        )
        options = _ex["answer_texts"]
        if _ex["abc_label"] and sum(_ex["option_is_abc"]) >= (len(options)-1):
            ex["unlabelled_options"] = [
                opt.upper() if abc else opt
                for opt, abc in zip(options, _ex["option_is_abc"])
            ]
            ex["style"] = "ai2_diagram_no_letter"
        else:
            ex["options"] = options
            ex["style"] = "ai2_diagram"
        return ex


class ScienceQAImageOnly(Dataset):
    """
    This class loads the ScienceQA dataset from HuggingFace (https://huggingface.co/datasets/derek-thomas/ScienceQA).
    """
    PATH = "derek-thomas/ScienceQA"

    @classmethod
    def download(cls, n_procs=1):
        # For datasets 4.4.1+, use load_dataset instead of load_dataset_builder
        try:
            datasets.load_dataset(cls.PATH)
        except RuntimeError as e:
            # If dataset uses loading scripts, try with trust_remote_code=True
            if "Dataset scripts are no longer supported" in str(e) or "loading scripts" in str(e).lower():
                try:
                    datasets.load_dataset(cls.PATH, trust_remote_code=True)
                except Exception:
                    # Fallback: try loading common splits with trust_remote_code
                    for split in ["train", "validation", "test", "val"]:
                        try:
                            datasets.load_dataset(cls.PATH, split=split, trust_remote_code=True)
                        except Exception:
                            continue
            else:
                # Fallback: try loading common splits
                for split in ["train", "validation", "test", "val"]:
                    try:
                        datasets.load_dataset(cls.PATH, split=split)
                    except Exception:
                        continue
        except Exception:
            # Fallback: try loading common splits
            for split in ["train", "validation", "test", "val"]:
                try:
                    datasets.load_dataset(cls.PATH, split=split)
                except Exception:
                    continue

    def __init__(self, split):
        assert split in ["train", "validation", "test"]
        # For datasets 3.0.0, we need trust_remote_code=True for TextVQA
        # Try loading with trust_remote_code=True first (for datasets 3.0.0)
        try:
            self.dataset = datasets.load_dataset(
                self.PATH, split=split, trust_remote_code=True
            ).filter(lambda ex: ex["image"] is not None)
        except Exception as e:
            # Fallback: try without trust_remote_code (for newer datasets versions that don't need it)
            try:
                self.dataset = datasets.load_dataset(
                    self.PATH, split=split
                ).filter(lambda ex: ex["image"] is not None)
            except Exception as e2:
                error_msg = str(e2)
                # Check if it's a network/download error
                if "ClientPayloadError" in error_msg or "ContentLengthError" in error_msg or "SSL" in error_msg:
                    raise RuntimeError(
                        f"Network error while downloading {self.PATH}. "
                        f"This is usually due to network instability. "
                        f"Please try:\n"
                        f"  1. Run the download script first: python scripts/download_data.py textvqa --n_procs 1\n"
                        f"  2. Or retry the experiment later when network is stable.\n"
                        f"Original error: {e2}"
                    ) from e2
                else:
                    raise RuntimeError(
                        f"Failed to load dataset {self.PATH} with split {split}. "
                        f"Error: {e2}"
                    ) from e2
        super().__init__()

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        ex = self.dataset[item]
        question =  ex["question"]
        hint = ex["hint"]
        if hint:
            question = hint + "\n" + hint
        return dict(
            image=ex["image"],
            question=question,
            style="science_qa",
            answer_idx=ex["answer"],
            options=ex["choices"],
        )


class InfoQa(DatasetBase):
    SPLITS = ["train", "validation", "test"]

    @classmethod
    def download(cls, n_procs=1):
        for split in cls.SPLITS:
            if split == "validation":
                filename = "infographicsVQA_val_v1.0_withQT.json"
            else:
                filename = f"infographicsVQA_{split}_v1.0.json"
            if not exists(join(INFOQA_SOURCE, filename)):
                raise ValueError(
                    "InfoQa requires manually downloading https://rrc.cvc.uab.es/?ch=17 (Task 3)"
                    f" please download and unzip the data into `{INFOQA_SOURCE}`"
                )

    def __init__(self, split):
        assert split in self.SPLITS
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            filename = "infographicsVQA_val_v1.0_withQT.json"
        else:
            filename = f"infographicsVQA_{split}_v1.0.json"
        filename = join(INFOQA_SOURCE, filename)
        logging.info(f"Loading docqa data from {filename}")
        with open(filename) as f:
            data = json.load(f)
        out = []
        for ex in data["data"]:
            image_path = join(INFOQA_SOURCE, "images", ex.pop("image_local_name"))
            out.append(dict(
                image=image_path,
                question=ex["question"],
                answers=ex.get("answers", []),
                metadata=dict(example_id=ex["questionId"]),
            ))
        return out

    def get(self, item, rng):
        return dict(**self.data[item], style="info_qa")


class DocQa(HfDataset):
    """
    DocumentVQA dataset from HuggingFace M4 project.
    This class loads the DocumentVQA dataset from HuggingFace (https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA).
    The dataset contains document images paired with questions and answers for visual document understanding tasks.

    Args:
        split (str): Dataset split to load. One of "train", "validation", or "test".
    """
    PATH = "HuggingFaceM4/DocumentVQA"

    def __init__(self, split: str, keep_in_memory=False, **kwargs):
        super().__init__(split, keep_in_memory, **kwargs)

    def get(self, item, rng):
        example = self.dataset[item]
        if self.split == "test":
            for k in ["answers", "question_types"]:
                assert k not in example or example[k] is None
                example[k] = []
        return dict(
                dict(
                image=example["image"],
                question=example["question"],
                answers=example.get("answers"),
                metadata=dict(
                    doc_id=example["docId"],
                    question_types=example.get("question_types"),
                    example_id=example["questionId"],
                )
            ), style="doc_qa")


class SceneTextQa(DatasetBase):

    @classmethod
    def download(cls, n_procs=1):
        if ST_QA_SRC is None:
            raise ValueError(
                "MOLMO_DATA_DIR is not set. Please set it to enable ST-VQA dataset download."
            )
        for split in ["train", "test"]:
            file_path = join(ST_QA_SRC, f"{split}_task_3.json")
            if not exists(file_path):
                raise ValueError(
                    f"SceneTextQa requires manually downloading from https://rrc.cvc.uab.es/?ch=11 (Task 3)\n"
                    f"Please download and unzip the data into `{ST_QA_SRC}`\n"
                    f"Expected file: {file_path}\n"
                    f"Note: The validation split uses the train data, so you only need train_task_3.json and test_task_3.json"
                )

    def __init__(self, split):
        assert split in ["train", "test", "validation"]
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            split = "train"
        # Use the modified split variable for file path, not self.split
        src = join(ST_QA_SRC, f"{split}_task_3.json")
        logging.info(f"Loading scene text data from {src}")
        with open(src) as f:
            data = json.load(f)["data"]
        out = []
        for question in data:
            out.append(dict(
                image=join(ST_QA_SRC, question["file_path"]),
                question=question["question"],
                metadata=dict(example_id=question["question_id"]),
                answers=question.get("answers", []),
            ))
        if self.split in ["train", "validation"]:
            # Custom val split since the data doesn't have one
            out.sort(key=lambda x: x["metadata"]["example_id"])
            np.random.RandomState(63069).shuffle(out)
            if self.split == "train":
                return out[1024:]
            else:
                return out[:1024]
        else:
            return out

    def get(self, item, rng):
        return dict(self.data[item], style="st_qa")


class CountBenchQa(Dataset):

    @classmethod
    def download(self, n_procs=1):
        CountQaBuilder().download_and_prepare()

    def __init__(self):
        self.dataset = CountQaBuilder().as_dataset("test")

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        ex = self.dataset[item]
        return {
            'image': ex["image"],
            'question': ex['question'],
            'style': "point_count",
            'metadata': {
                'count': ex['count'],
                'image_id': ex["example_id"],
                'image_url': ex['image_url'],
            }
        }


class TabWMPDirectAnswer(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        TabMwpBuilder().download_and_prepare()

    def __init__(self, split, include_options: bool):
        self.include_options = include_options
        self._dataset = TabMwpBuilder().as_dataset(split)

    def __len__(self):
        return len(self._dataset)

    def get(self, item, rng):
        ex = self._dataset[item]
        out = dict(
            image=ex["image"],
            question=ex["question"],
            answer=ex["answer"],
            style="tabwmp_da",
            metadata=dict(
                example_id=ex["example_id"]
            )
        )
        if self.include_options and ex["choices"]:
            out["options"] = ex["choices"]
        return out


class FigureQa(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        FigureQaBuilder().download_and_prepare()

    def __init__(self, split, in_memory=False):
        assert split in ["train", "validation1", "test1", "validation2", "test2"]
        self.hf_dataset = FigureQaBuilder().as_dataset(split, in_memory=in_memory)

    def get(self, item, rng):
        example = self.hf_dataset[int(item)]
        qas = example["questions"]
        messages = []
        for q, a in zip(qas["question"], qas["answer"]):
            messages.append(dict(question=q, answer=str(a), style="figure_qa"))
        return dict(image=example["image"], message_list=messages)

    def __len__(self):
        return len(self.hf_dataset)


class PlotQa(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        PlotQaBuilder().download_and_prepare()

    def __init__(self, split, in_memory=False):
        assert split in ["train", "validation", "test"]
        self.hf_dataset = PlotQaBuilder().as_dataset(split, in_memory=in_memory)

    def get(self, item, rng):
        example = self.hf_dataset[int(item)]
        qas = example["questions"]
        messages = []
        for q, a in zip(qas["question"], qas["answer"]):
            messages.append(dict(question=q, answer=a, style="plot_qa"))
        return dict(image=example["image"], message_list=messages)

    def __len__(self):
        return len(self.hf_dataset)


class AndroidControl(Dataset):
    @classmethod
    def download(cls, n_procs=1):
        import os
        import fsspec
        from fsspec.implementations.http import HTTPFileSystem
        from aiohttp import ClientTimeout
        
        # Set longer timeout for fsspec/aiohttp downloads (in seconds)
        # Default is usually 60 seconds, set to 3600 (1 hour) for large files
        # This affects the timeout for reading chunks from the HTTP stream
        
        # Store original timeout settings
        original_timeout = os.environ.get("FSSPEC_TIMEOUT")
        original_default_timeout = getattr(HTTPFileSystem, '_default_timeout', None)
        original_client_kwargs = getattr(HTTPFileSystem, 'client_kwargs', {})
        
        try:
            # Set environment variable (may be used by some fsspec implementations)
            os.environ["FSSPEC_TIMEOUT"] = "3600"
            
            # Patch HTTPFileSystem to use longer timeout
            # This increases the timeout for reading data chunks during download
            HTTPFileSystem._default_timeout = 3600
            
            # Also patch client_kwargs to set aiohttp timeout
            # This is the actual timeout used by aiohttp for reading chunks
            if not hasattr(HTTPFileSystem, 'client_kwargs') or HTTPFileSystem.client_kwargs is None:
                HTTPFileSystem.client_kwargs = {}
            HTTPFileSystem.client_kwargs = HTTPFileSystem.client_kwargs.copy()
            HTTPFileSystem.client_kwargs['timeout'] = ClientTimeout(total=3600, connect=60)
            
            # Create DownloadConfig with retry mechanism
            download_config = datasets.DownloadConfig(
                num_proc=n_procs,
                max_retries=5,  # Add retry mechanism for failed downloads
            )
            
            # Try downloading with retry mechanism
            # For large files, reduce concurrency on retry to avoid network issues
            max_attempts = 3
            current_n_procs = n_procs
            
            for attempt in range(max_attempts):
                try:
                    # On retry, reduce concurrency to avoid overwhelming the network
                    if attempt > 0 and current_n_procs > 2:
                        current_n_procs = max(1, current_n_procs // 2)
                        download_config.num_proc = current_n_procs
                        import logging
                        logging.info(f"Retry attempt {attempt + 1}: Reducing concurrency to {current_n_procs} processes")
                    
                    AndroidControlBuilder().download_and_prepare(download_config=download_config, num_proc=current_n_procs)
                    # If we get here, download succeeded
                    break
                except Exception as e:
                    error_msg = str(e)
                    
                    # Check if it's a network/SSL error
                    is_network_error = (
                        "SSL" in error_msg or 
                        "ContentLength" in error_msg or 
                        "SSLError" in error_msg or
                        "ClientPayloadError" in error_msg or
                        "TimeoutError" in error_msg
                    )
                    
                    if is_network_error and attempt < max_attempts - 1:
                        import logging
                        import time
                        logging.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for AndroidControl: {error_msg[:200]}...\n"
                            f"Retrying in 5 seconds with reduced concurrency..."
                        )
                        time.sleep(5)
                        
                        # Clean partial downloads on retry
                        import glob
                        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
                        android_cache = glob.glob(f"{cache_dir}/*android*")
                        if android_cache:
                            logging.info(f"Cleaning {len(android_cache)} cached files before retry...")
                            import shutil
                            for cache_path in android_cache:
                                try:
                                    if os.path.isdir(cache_path):
                                        shutil.rmtree(cache_path)
                                    else:
                                        os.remove(cache_path)
                                except Exception:
                                    pass
                    else:
                        # Last attempt failed or non-network error
                        if is_network_error:
                            import logging
                            logging.error(
                                f"SSL/ContentLength error detected for AndroidControl after {max_attempts} attempts. "
                                f"This usually means network issues or corrupted download.\n"
                                f"Suggestions:\n"
                                f"  1. Try with lower concurrency: --n_procs 1 or --n_procs 2\n"
                                f"  2. Clean cache and retry: rm -rf ~/.cache/huggingface/datasets/*android*\n"
                                f"  3. Try again later (network may be temporarily unstable)"
                            )
                        raise
        finally:
            # Restore original timeout settings
            if original_timeout is not None:
                os.environ["FSSPEC_TIMEOUT"] = original_timeout
            elif "FSSPEC_TIMEOUT" in os.environ:
                del os.environ["FSSPEC_TIMEOUT"]
            
            if original_default_timeout is not None:
                HTTPFileSystem._default_timeout = original_default_timeout
            elif hasattr(HTTPFileSystem, '_default_timeout'):
                delattr(HTTPFileSystem, '_default_timeout')
            
            # Restore original client_kwargs
            if original_client_kwargs:
                HTTPFileSystem.client_kwargs = original_client_kwargs
            elif hasattr(HTTPFileSystem, 'client_kwargs'):
                HTTPFileSystem.client_kwargs = {}

    def __init__(self, split, mode="all", in_memory=False):
        self.mode = mode
        self.hf_dataset = AndroidControlBuilder().as_dataset(
            "val" if split == "validation" else split, in_memory=in_memory)

    def __len__(self):
        return len(self.hf_dataset)

    def get(self, item, rng):
        ex = self.hf_dataset[item]
        ll, hl_ll, hl, hl_cot = [
            dict(
                prompt="low_level: " + ex["ll_instruction"],
                text=ex["target_action"],
                style="android_control"
            ),
            dict(
                prompt="high_level: " + ex["hl_instruction"] + " low_level: " + ex["ll_instruction"],
                text=ex["target_action"],
                style="android_control"
            ),
            dict(
                prompt="high_level: " + ex["hl_instruction"],
                text=ex["target_action"],
                style="android_control"
            ),
            dict(
                prompt="high_level_cot: " + ex["hl_instruction"],
                text="Plan: " + ex["ll_instruction"] + " Action: " + ex["target_action"],
                style="android_control"
            )
        ]
        example = dict(
            image=ex["image"],
            metadata=dict(
                target_action=ex["target_action"],
                target_box=ex["target_box"],
                ll_instruction=ex["ll_instruction"],
                hl_instruction=ex["hl_instruction"],
            )
        )
        if self.mode == "ll":
            example.update(ll)
        elif self.mode == "hl":
            example.update(hl)
        elif self.mode == "hl_ll":
            example.update(hl_ll)
        elif self.mode == "hl_cot":
            example.update(hl_cot)
        elif self.mode == "all":
            example["message_list"] = [ll, hl_ll, hl, hl_cot]
        else:
            raise NotImplementedError(self.mode)
        return example


class DvQa(Dataset):
    @classmethod
    def download(cls, n_procs=1):
        DvQaBuilder().download_and_prepare()

    def __init__(self, split, in_memory=False):
        self.hf_dataset = DvQaBuilder().as_dataset(split, in_memory=in_memory)

    def __len__(self):
        return len(self.hf_dataset)

    def get(self, item, rng):
        example = self.hf_dataset[int(item)]
        qas = example["questions"]
        messages = []
        for q, a in zip(qas["question"], qas["answer"]):
            messages.append(dict(question=q, answer=a, style="dv_qa"))
        return dict(
            image=example["image"],
            message_list=messages,
            metadata=dict(image_id=example["image_id"]),
        )


class MathVista(HfDataset):
    PATH = "AI4Math/MathVista"

    def __init__(self, split, simplify_question=True, **kwargs):
        super().__init__(split, **kwargs)
        self.simplify_question = simplify_question

    def get(self, item, rng):
        ex = self.dataset[item]
        question: str = ex["question"]
        if self.simplify_question:
            question = question.split("Question:")[-1]
            question = question.split("Hint:")[0].strip()
        out = dict(
            question=question,
            image=ex["decoded_image"],
            metadata=dict(
                example_id=ex["pid"],
                answer=ex["answer"],
                precision=ex["precision"],
                query=ex["question"],
                choices=ex["choices"],
                question_type=ex["question_type"],
                answer_type=ex["answer_type"]
            ),
        )
        if ex["question_type"] == "multi_choice":
            out["options"] = ex["choices"]
            out["style"] = "eval_multiple_choice"
        else:
            out["style"] = "eval_short_answer"
        return out


class RealWorldQa(HfDataset):
    PATH = "xai-org/RealworldQA"

    def __init__(self, mode="no_mc_instruction", in_memory=False):
        super().__init__("test", in_memory)
        self.mode = mode

    def get(self, item, rng):
        ex = self.dataset[item]
        prompt: str = ex["question"]
        if "Please answer directly with a single word or number." in prompt:
            question_type = "short_answer"
        else:
            assert "Please answer directly with only the letter of the correct option and nothing else." in prompt
            question_type = "multiple_choice"
        out = dict(
            image=ex["image"],
            metadata=dict(answer=ex["answer"], prompt=ex["question"], question_type=question_type),
        )
        if self.mode == "plain":
            out.update(style="none", prompt=prompt)
        else:
            if question_type == "short_answer":
                style = "eval_short_answer"
            else:
                style = "eval_multiple_choice"
            if self.mode == "no_instruction":
                if question_type == "short_answer":
                    prompt = prompt.split("\n")[0]
            else:
                if self.mode != "vqa_style_tag":
                    raise NotImplementedError(self.mode)
            out.update(style=style, question=prompt)
        return out


class MMMU(Dataset):
    NAMES = [
        'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
        'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science',
        'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power',
        'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math',
        'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health',
        'Sociology'
    ]

    @classmethod
    def download(cls, n_procs=1):
        for name in cls.NAMES:
            if exists(join(DATA_HOME, "mmmu", name)):
                continue
            # For datasets 4.4.1+, use load_dataset instead of load_dataset_builder
            try:
                datasets.load_dataset("MMMU/MMMU", name=name)
            except Exception:
                # Fallback: try loading common splits
                for split in ["train", "validation", "test", "val"]:
                    try:
                        datasets.load_dataset("MMMU/MMMU", name=name, split=split)
                    except Exception:
                        continue

    def __init__(self, split: str):
        all_parts = []
        for name in self.NAMES:
            all_parts.append(datasets.load_dataset("MMMU/MMMU", name=name, split=split))
        self.data = datasets.concatenate_datasets(all_parts)

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        mc = ex["question_type"] == "multiple-choice"
        out = dict(
            image=ex["image_1"],
            text=ex["answer"],
            question=ex["question"],
            metadata=dict(answer=ex["answer"], example_id=ex["id"], question_type=ex["question_type"]),
            style='a_okvqa_mc' if mc else 'vqa2'
        )
        if mc:
            options = eval(ex["options"])
            if sum((re.match("<img='(.*?)'>", opt) is not None) for opt in options) > 1:
                # Following LLaVa, don't use any images if there are multiple images paths
                # I think the rationale is that this means the image are answer-options
                del out["image"]
            out["options"] = options
        return out


class ClockBench(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        ClockBenchBuilder().download_and_prepare()

    def __init__(self, split):
        assert split in ["coco", "openimg", "movies"]
        dataset = ClockBenchBuilder().as_dataset(split)
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        _ex = dict(self.dataset[item])
        hour, minute = [int(_ex[k]) for k in ["hour", "minute"]]
        if hour == 12:
            hour = 0
        second = -1
        return dict(
            image=np.array(_ex["image"]),
            prompt="What time is being shown?",
            metadata=dict(
                hour=hour,
                minute=minute,
                second=second,
                example_id=_ex["image_id"],
            ),
            style="clocks",
        )
