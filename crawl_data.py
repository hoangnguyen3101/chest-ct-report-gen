from huggingface_hub import snapshot_download

REPO_ID = "RadGenome/RadGenome-ChestCT"
OUT_DIR = "../ct2rep"
MAX_WORKERS = 2  # giảm để tránh timeout

patterns = [


    # 2 file region report CSV
    "dataset/radgenome_files/train_region_report.csv",
    "dataset/radgenome_files/validation_region_report.csv",

    # toàn bộ train & valid preprocessed
    "dataset/train_preprocessed/**",
    "dataset/valid_preprocessed/**",

]

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=OUT_DIR,
    local_dir_use_symlinks=False,
    allow_patterns=patterns,
    max_workers=MAX_WORKERS,
)

print("DONE Đã tải train_preprocessed, valid_preprocessed và 2 file region_report.csv")
