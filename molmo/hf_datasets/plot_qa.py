import json
from collections import defaultdict
from os.path import join
from typing import List

import datasets


class PlotQaBuilder(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, dataset_name="plot_qa")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'image': datasets.Image(decode=False),
                "image_index": datasets.Value("int64"),
                "questions": datasets.Sequence(datasets.Features({
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "question_id": datasets.Value("int64"),
                }))
            }),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        file_ids = dict(
            train_images="1AYuaPX-Lx7T0GZvnsPgN11Twq2FZbWXL",
            validation_images="1i74NRCEb-x44xqzAovuglex5d583qeiF",
            test_images="1D_WPUy91vOrFl6cJUkE55n3ZuB6Qrc4u",
            train_annotations="1UNvkdq1YJD_ne6D3zbWtoQij37AtfpNp",
            validation_annotations="1y9RwXSye2hnX0e2IlfSK34ESbeVblhH_",
            test_annotations="1OQBkoe_dpvFs-jnWAdRdxzh1-hgNd9bO",
        )
        file_urls = {
            k: f"https://drive.usercontent.google.com/download?id={v}&confirm=t"
            for k, v in file_ids.items()
        }
        downloaded_files = dl_manager.download(file_urls)
        extracted_files = dl_manager.extract(file_urls)
        return [
            datasets.SplitGenerator(name=k, gen_kwargs={"image_dir": extracted_files[f"{k}_images"], "annotations": extracted_files[f"{k}_annotations"]})
            for k in [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]
        ]

    def _generate_examples(self, image_dir, annotations):
        import os
        from glob import glob
        
        with open(annotations, "r") as f:
            data = json.load(f)
        
        # Handle different data formats
        # If data is a dict with "qa_pairs" key, use that
        # If data is a list, use it directly
        if isinstance(data, dict):
            if "qa_pairs" in data:
                qa_pairs = data["qa_pairs"]
            else:
                # If dict but no "qa_pairs", try to find the list of questions
                # Look for any list value in the dict
                qa_pairs = None
                for key, value in data.items():
                    if isinstance(value, list):
                        qa_pairs = value
                        break
                if qa_pairs is None:
                    raise ValueError(f"Could not find qa_pairs in annotations file. Keys: {list(data.keys())}")
        elif isinstance(data, list):
            qa_pairs = data
        else:
            raise ValueError(f"Unexpected data format in annotations file: {type(data)}")
        
        grouped_by_image = defaultdict(list)
        for question in qa_pairs:
            # Ensure question is a dict
            if not isinstance(question, dict):
                # If question is a list, it might be nested incorrectly
                if isinstance(question, list):
                    # Try to process each item in the list
                    for item in question:
                        if isinstance(item, dict):
                            image_index = item.get("image_index")
                            if image_index is not None:
                                grouped_by_image[image_index].append(item)
                continue
            
            image_index = question.get("image_index")
            if image_index is not None:
                grouped_by_image[image_index].append(question)
        
        # Build a mapping from image_index to actual filename
        png_dir = join(image_dir, "png")
        if os.path.exists(png_dir):
            # Get all PNG files and create a mapping
            png_files = glob(join(png_dir, "*.png"))
            # Extract image indices from filenames (assuming format: {index}.png)
            image_index_to_file = {}
            for png_file in png_files:
                basename = os.path.basename(png_file)
                # Try to extract number from filename (e.g., "0.png" -> 0, "150663.png" -> 150663)
                try:
                    idx = int(os.path.splitext(basename)[0])
                    image_index_to_file[idx] = png_file
                except ValueError:
                    # If filename is not a number, skip
                    continue
        
        for image_index, questions_data in grouped_by_image.items():
            questions = []
            
            # Ensure questions_data is a list
            if not isinstance(questions_data, list):
                continue
            
            for q in questions_data:
                # Ensure q is a dict, skip if not
                if not isinstance(q, dict):
                    # If q is a list, try to convert it or skip
                    if isinstance(q, list):
                        # Skip nested lists - this shouldn't happen but handle it gracefully
                        continue
                    continue
                
                # Ensure all required fields are present and are the correct type
                question_text = q.get("question_string", q.get("question", ""))
                answer = q.get("answer", "")
                question_id = q.get("question_id", 0)
                
                # Ensure question_text is a string
                if not isinstance(question_text, str):
                    if question_text is None:
                        question_text = ""
                    else:
                        question_text = str(question_text)
                
                # Ensure answer is converted to string
                if answer is None:
                    answer = ""
                elif not isinstance(answer, str):
                    answer = str(answer)
                
                # Ensure question_id is int64
                try:
                    question_id = int(question_id)
                except (ValueError, TypeError):
                    question_id = 0
                
                # Create a properly formatted question dict
                # This must match the schema exactly: question, answer, question_id
                question_dict = {
                    "question": question_text,
                    "answer": answer,
                    "question_id": question_id,
                }
                
                # Validate the dict structure before appending
                if isinstance(question_dict, dict) and all(
                    isinstance(v, (str, int)) for k, v in question_dict.items()
                ):
                    questions.append(question_dict)
            
            # Skip if no questions
            if not questions:
                continue
            
            # Try to find the image file
            image_path = None
            # First try: direct path with image_index
            direct_path = join(png_dir, str(image_index) + ".png")
            if os.path.exists(direct_path):
                image_path = direct_path
            # Second try: use mapping if available
            elif image_index in image_index_to_file:
                image_path = image_index_to_file[image_index]
            # Third try: search for any file with image_index in name
            else:
                pattern = join(png_dir, f"*{image_index}*.png")
                matches = glob(pattern)
                if matches:
                    image_path = matches[0]
            
            if image_path is None or not os.path.exists(image_path):
                # Skip images that can't be found instead of raising error
                continue
            
            # Final validation: ensure questions is a list of dicts
            # Each dict must have exactly the keys: question, answer, question_id
            validated_questions = []
            for idx, q in enumerate(questions):
                if not isinstance(q, dict):
                    # Skip non-dict items
                    continue
                
                # Ensure all required keys are present and have correct types
                try:
                    validated_q = {
                        "question": str(q.get("question", "")),
                        "answer": str(q.get("answer", "")),
                        "question_id": int(q.get("question_id", 0)),
                    }
                    # Verify all values are the correct type
                    assert isinstance(validated_q["question"], str)
                    assert isinstance(validated_q["answer"], str)
                    assert isinstance(validated_q["question_id"], int)
                    validated_questions.append(validated_q)
                except (ValueError, TypeError, AssertionError) as e:
                    # Skip invalid questions
                    continue
            
            # Skip if no valid questions
            if not validated_questions:
                continue
            
            # Ensure image_index is an integer
            try:
                image_index_int = int(image_index)
            except (ValueError, TypeError):
                continue
            
            # For datasets.Image(decode=False), pass the image path as a string
            # Ensure all values are the correct type
            example = {
                "image": str(image_path),
                "image_index": image_index_int,
                "questions": validated_questions,
            }
            
            # Final check: ensure questions is a list and all elements are dicts
            if not isinstance(example["questions"], list):
                continue
            if not all(isinstance(q, dict) for q in example["questions"]):
                continue
            
            yield image_index_int, example


if __name__ == "__main__":
    PlotQaBuilder().download_and_prepare()
