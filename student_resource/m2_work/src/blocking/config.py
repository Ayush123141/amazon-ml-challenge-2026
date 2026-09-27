from pathlib import Path


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Amazon ML Challenge\student_resource"
)


# ============================================================
# DATA
# ============================================================

DATA_DIR = PROJECT_ROOT / "dataset"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"


S1_TRAIN = TRAIN_DIR / "train_source1.tsv"
S2_TRAIN = TRAIN_DIR / "train_source2.tsv"
S3_TRAIN = TRAIN_DIR / "train_source3.tsv"

GROUND_TRUTH = (
    TRAIN_DIR /
    "train_ground_truth.tsv"
)


# ============================================================
# M2 OUTPUTS
# ============================================================

M2_DIR = PROJECT_ROOT / "m2_work"

OUTPUT_DIR = (
    M2_DIR /
    "outputs"
)

BLOCKING_100K_DIR = (
    OUTPUT_DIR /
    "blocking_100k"
)

M22_DIR = (
    OUTPUT_DIR /
    "m2_2"
)

REPORT_DIR = (
    M2_DIR /
    "reports"
)

LOG_DIR = (
    M2_DIR /
    "logs"
)


# ============================================================
# EXPERIMENT
# ============================================================

RANDOM_SEED = 42

S1_SAMPLE_SIZE = 100_000

CHUNK_SIZE = 100_000


# ============================================================
# M2.2
# ============================================================

DF_THRESHOLDS = [
    2,
    5,
    10,
    25,
    50,
]