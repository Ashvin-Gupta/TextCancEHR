import numpy as np 
import pathlib
import sys
notebook_path = pathlib.Path().resolve()  # this is the directory Jupyter started in
print(notebook_path)
repo_root = notebook_path.parents[1]      # move up from src/resources to repo root
sys.path.insert(0, str(repo_root))
from src.data.unified_dataset import UnifiedEHRDataset
from src.pipelines.text_based.token_adaption2 import EHRTokenExtensionStaticTokenizer
import torch

DATA_ROOT = "/data/scratch/qc25022/pancreas/tokenised_data_word_level/cprd_upgi"
SPLITS = ["train", "tuning", "held_out"]  # add any extra splits you used
VOCAB = "/data/scratch/qc25022/pancreas/tokenised_data_word_level/cprd_upgi/vocab.csv"
LABELS ="/data/scratch/qc25022/upgi/master_subject_labels.csv"
MEDICAL = "/data/home/qc25022/CancEHR-Training/src/resources/MedicalDictTranslation2.csv"
LAB = "/data/home/qc25022/CancEHR-Training/src/resources/LabLookUP.csv"
REGION = "/data/home/qc25022/CancEHR-Training/src/resources/RegionLookUp.csv"
TIME = "/data/home/qc25022/CancEHR-Training/src/resources/TimeLookUp.csv"

translator = EHRTokenExtensionStaticTokenizer()
model, tokenizer = translator.extend_tokenizer(
    model_name="unsloth/Qwen3-8B-Base-unsloth-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True
)

def token_length_stats(split, cutoff):
    dataset = UnifiedEHRDataset(
        data_dir=DATA_ROOT,
        vocab_file=VOCAB,
        labels_file=LABELS,
        medical_lookup_file=MEDICAL,
        lab_lookup_file=LAB,
        region_lookup_file=REGION,
        time_lookup_file=TIME,
        cutoff_months=cutoff,
        format="text",
        split=split,
        max_sequence_length=None,
    )
    lengths = []
    for item in dataset:
        if item is None:
            continue
        ids = tokenizer.encode(
            item["text"],
            add_special_tokens=True,
            truncation=False
        )
        lengths.append(len(ids))
    arr = np.array(lengths)
    return {
        "count": arr.size,
        "mean": arr.mean(),
        "p90": np.percentile(arr, 90),
        "p95": np.percentile(arr, 95),
        "max": arr.max(),
        "pct_over_2048": (arr > 2048).mean()
    }

# print(token_length_stats("train", cutoff=12))
print(token_length_stats("train", cutoff=1))

# def describe_lengths(split, cutoff):
#     dataset = UnifiedEHRDataset(
#         data_dir=DATA_ROOT,
#         vocab_file=VOCAB,
#         labels_file=LABELS,
#         medical_lookup_file=MEDICAL,
#         lab_lookup_file=LAB,
#         region_lookup_file=REGION,
#         time_lookup_file=TIME,
#         cutoff_months=cutoff,
#         format="text",
#         split=split,
#         max_sequence_length=None,
#     )
#     lengths_chars = []
#     lengths_tokens = []
#     for item in dataset:
#         if item is None:
#             continue
#         text = item["text"]
#         lengths_chars.append(len(text))
#         lengths_tokens.append(len(text.split()))  # crude word count; swap with tokenizer.encode if desired
#     summary = lambda arr: dict(count=len(arr), mean=np.mean(arr), p95=np.percentile(arr, 95), max=max(arr))
#     return summary(lengths_chars), summary(lengths_tokens)

# for split in ["train", "tuning", "held_out"]:
#     char_stats, token_stats = describe_lengths(split, cutoff=12)
#     print(f"{split} char stats: {char_stats}")
#     print(f"{split} token stats: {token_stats}")