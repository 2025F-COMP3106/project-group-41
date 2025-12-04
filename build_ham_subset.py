import os
import shutil
import pandas as pd


PROJECT_ROOT = os.path.dirname(__file__)

METADATA_CSV = os.path.join(PROJECT_ROOT, "database", "archive", "HAM10000_metadata.csv")
IMAGES_DIRS = [
    os.path.join(PROJECT_ROOT, "database", "archive", "HAM10000_images_part_1"),
    os.path.join(PROJECT_ROOT, "database", "archive", "HAM10000_images_part_2"),
]

OUTPUT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "codebase", "data", "images")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "codebase", "data", "labels.csv")

N_BENIGN = 100       # how many benign images to sample
N_MALIGNANT = 100    # how many malignant images to sample



def find_image_path(filename: str) -> str | None:
    """Search for filename in the HAM10000 image folders."""
    for d in IMAGES_DIRS:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    if not os.path.exists(METADATA_CSV):
        raise FileNotFoundError(f"Metadata CSV not found at {METADATA_CSV}")

    print("Loading metadata from:", METADATA_CSV)
    df = pd.read_csv(METADATA_CSV)

    # Try to use 'benign_malignant' if it exists, otherwise derive from 'dx'
    if "benign_malignant" in df.columns:
        print("Using 'benign_malignant' column to label data...")
        df["bm"] = df["benign_malignant"]
    else:
        print("No 'benign_malignant' column visible, deriving labels from 'dx'...")
        # HAM10000 diagnosis labels
        # mel, bcc, akiec -> malignant; others -> benign (rough but fine for project)
        malignant_dx = {"mel", "bcc", "akiec"}
        df["bm"] = df["dx"].apply(lambda x: "malignant" if x in malignant_dx else "benign")

    # Build filename column (images are named <image_id>.jpg)
    df["filename"] = df["image_id"].astype(str) + ".jpg"

    benign_df = df[df["bm"] == "benign"].head(N_BENIGN)
    malignant_df = df[df["bm"] == "malignant"].head(N_MALIGNANT)

    print(f"Selected {len(benign_df)} benign and {len(malignant_df)} malignant examples.")

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    records = []

    # Copy benign images
    for _, row in benign_df.iterrows():
        fname = row["filename"]
        src = find_image_path(fname)
        if src is None:
            print(f"[WARN] Could not find image file for {fname}, skipping.")
            continue
        dst_name = fname  # keep same name
        dst = os.path.join(OUTPUT_IMAGE_DIR, dst_name)
        shutil.copy(src, dst)
        records.append([dst_name, 0])  # 0 = benign

    # Copy malignant images
    for _, row in malignant_df.iterrows():
        fname = row["filename"]
        src = find_image_path(fname)
        if src is None:
            print(f"[WARN] Could not find image file for {fname}, skipping.")
            continue
        dst_name = fname
        dst = os.path.join(OUTPUT_IMAGE_DIR, dst_name)
        shutil.copy(src, dst)
        records.append([dst_name, 1])  # 1 = malignant

    # Save labels.csv
    labels_df = pd.DataFrame(records, columns=["filepath", "label"])
    labels_df.to_csv(OUTPUT_CSV, index=False)

    print("\n====================================")
    print("SUBSET CREATION COMPLETE")
    print("====================================")
    print(f"Images copied to: {OUTPUT_IMAGE_DIR}")
    print(f"labels.csv saved to: {OUTPUT_CSV}")
    print(f"Total records: {len(labels_df)}")
    print("Class counts:")
    print(labels_df["label"].value_counts())


if __name__ == "__main__":
    main()