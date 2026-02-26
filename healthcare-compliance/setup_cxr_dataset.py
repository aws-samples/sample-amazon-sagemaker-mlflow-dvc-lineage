# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Montgomery County CXR Dataset Setup Script

Downloads the Montgomery County Chest X-Ray dataset from the National
Library of Medicine (NLM), organizes images into class directories
(normal/tuberculosis) based on clinical readings, uploads to S3, and
generates a patient consent manifest with randomly assigned patient IDs.

The dataset contains 138 posterior-anterior chest X-rays from the
Department of Health and Human Services, Montgomery County, Maryland:
  - 80 normal cases
  - 58 cases with tuberculosis manifestations

Source: https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
Citation: Jaeger S, Candemir S, Antani S, et al. Two public chest X-ray
          datasets for computer-aided screening of pulmonary diseases.
          Quant Imaging Med Surg. 2014;4(6):475-477

Usage:
    From the notebook:
        from setup_cxr_dataset import setup_dataset
        raw_data_s3_uri = setup_dataset(bucket, prefix)

    From the command line:
        python setup_cxr_dataset.py --bucket my-bucket --prefix DEMO-cxr-dvc
"""

import os
import uuid
import random
import shutil
import zipfile
import csv
import urllib.request
from pathlib import Path

import boto3


DATASET_URL = "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
DATASET_DIR_NAME = "MontgomerySet"
CLASSES = ["normal", "tuberculosis"]


def parse_labels(dataset_path):
    """Parse ClinicalReadings to determine normal vs tuberculosis labels.

    Each ClinicalReadings text file contains patient metadata and findings.
    Normal cases have "normal" as the last line; TB cases describe
    tuberculosis manifestations.

    Returns:
        dict: {image_filename: label} e.g. {"MCUCXR_0001_0.png": "normal"}
    """
    dataset_path = Path(dataset_path)
    readings_dir = dataset_path / "ClinicalReadings"
    cxr_dir = dataset_path / "CXR_png"

    if not readings_dir.exists():
        raise FileNotFoundError(f"ClinicalReadings directory not found at {readings_dir}")

    labels = {}
    for txt_path in sorted(readings_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        last_line = text.split("\n")[-1].strip().lower()

        img_name = txt_path.stem + ".png"
        if not (cxr_dir / img_name).exists():
            print(f"Warning: no matching image for {txt_path.name}, skipping")
            continue

        if last_line == "normal":
            labels[img_name] = "normal"
        else:
            labels[img_name] = "tuberculosis"

    # Include any images without clinical readings as unlabeled (skip them)
    all_images = {f.name for f in cxr_dir.glob("*.png")}
    unlabeled = all_images - set(labels.keys())
    if unlabeled:
        print(f"Warning: {len(unlabeled)} images without clinical readings, skipping: {unlabeled}")

    return labels


def organize_by_class(dataset_path, labels):
    """Copy images from CXR_png/ into class subdirectories.

    Creates normal/ and tuberculosis/ directories under dataset_path
    with copies of the appropriate images.

    Returns:
        dict: {class_name: count}
    """
    dataset_path = Path(dataset_path)
    cxr_dir = dataset_path / "CXR_png"
    counts = {}

    for cls in CLASSES:
        cls_dir = dataset_path / cls
        cls_dir.mkdir(exist_ok=True)
        counts[cls] = 0

    for img_name, label in sorted(labels.items()):
        src = cxr_dir / img_name
        dst = dataset_path / label / img_name
        if not dst.exists():
            shutil.copy2(src, dst)
        counts[label] += 1

    return counts


def download_dataset(output_dir="."):
    """Download and extract the Montgomery County CXR dataset from NLM.

    Skips download if the dataset directory already exists.

    Returns:
        Path to the extracted dataset directory.
    """
    output_dir = Path(output_dir)
    dataset_path = output_dir / DATASET_DIR_NAME

    if dataset_path.exists():
        print(f"Dataset already exists at {dataset_path}, skipping download")
        return dataset_path

    zip_path = output_dir / "NLM-MontgomeryCXRSet.zip"

    print(f"Downloading Montgomery County CXR dataset from NLM...")
    urllib.request.urlretrieve(DATASET_URL, str(zip_path))

    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    zip_path.unlink()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Expected directory '{DATASET_DIR_NAME}' not found after extraction. "
            f"Contents: {[p.name for p in output_dir.iterdir()]}"
        )

    print(f"Dataset extracted to {dataset_path}")
    return dataset_path


def upload_to_s3(dataset_path, bucket, prefix):
    """Upload chest X-ray images to S3 organized by class.

    Returns:
        S3 URI for the raw data prefix.
    """
    dataset_path = Path(dataset_path)
    s3_client = boto3.client("s3")
    raw_prefix = f"{prefix}/raw-cxr"

    all_images = []
    for cls in CLASSES:
        cls_dir = dataset_path / cls
        if not cls_dir.exists():
            print(f"Warning: class directory {cls_dir} not found, skipping")
            continue
        images = sorted(cls_dir.glob("*.png"))
        all_images.extend((cls, img_path) for img_path in images)

    total = len(all_images)
    print(f"Uploading {total} images to s3://{bucket}/{raw_prefix}")

    for i, (cls, img_path) in enumerate(all_images, 1):
        s3_key = f"{raw_prefix}/{cls}/{img_path.name}"
        s3_client.upload_file(str(img_path), bucket, s3_key)
        if i % 20 == 0 or i == total:
            print(f"  {i}/{total} uploaded ({i * 100 // total}%)")

    raw_data_s3_uri = f"s3://{bucket}/{raw_prefix}"
    print(f"Done: {total} images uploaded to {raw_data_s3_uri}")
    return raw_data_s3_uri


def generate_manifest(dataset_path, output_path="master_manifest.csv", seed=42):
    """Generate a patient consent manifest with random patient IDs.

    Each image is assigned to a patient (1-3 images per patient) with a
    random hex-based patient ID. All patients start with active consent.

    Returns:
        Path to the generated manifest CSV.
    """
    dataset_path = Path(dataset_path)
    rng = random.Random(seed)

    # Collect all class-organized images
    image_records = []
    scan_counter = 0

    for cls in CLASSES:
        cls_dir = dataset_path / cls
        if not cls_dir.exists():
            continue

        images = sorted(cls_dir.glob("*.png"))
        for img_path in images:
            scan_counter += 1
            image_records.append({
                "scan_id": f"SCAN-{scan_counter:04d}",
                "s3_key": f"raw-cxr/{cls}/{img_path.name}",
                "label": cls,
            })

    # Shuffle and assign to patients (1-3 images per patient)
    rng.shuffle(image_records)

    # Use a separate Random for UUIDs to keep patient assignment deterministic
    uuid_rng = random.Random(seed + 1)
    i = 0
    while i < len(image_records):
        patient_id = f"PAT-{uuid.UUID(int=uuid_rng.getrandbits(128)).hex[:6]}"
        num_scans = rng.randint(1, 3)
        for j in range(num_scans):
            if i + j < len(image_records):
                image_records[i + j]["patient_id"] = patient_id
                image_records[i + j]["consent_status"] = "active"
        i += num_scans

    # Write manifest
    output_path = Path(output_path)
    fieldnames = ["patient_id", "scan_id", "s3_key", "label", "consent_status"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(image_records)

    num_patients = len(set(r["patient_id"] for r in image_records))
    label_counts = {}
    for r in image_records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    print(f"Generated manifest: {len(image_records)} scans, {num_patients} patients")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print(f"Saved to {output_path}")

    return output_path


def setup_dataset(bucket, prefix, output_dir="."):
    """Run the full setup: download, label, organize, upload to S3, generate manifest.

    Args:
        bucket: S3 bucket name
        prefix: S3 key prefix (e.g., 'DEMO-cxr-dvc')
        output_dir: Local directory for dataset download and manifest

    Returns:
        S3 URI for the raw data (e.g., 's3://bucket/prefix/raw-cxr')
    """
    output_dir = Path(output_dir)

    print("=" * 60)
    print("Montgomery County CXR Dataset Setup")
    print("=" * 60)

    # Step 1: Download
    dataset_path = download_dataset(output_dir)

    # Step 2: Parse labels from clinical readings
    labels = parse_labels(dataset_path)
    print(f"Parsed labels: {sum(1 for v in labels.values() if v == 'normal')} normal, "
          f"{sum(1 for v in labels.values() if v == 'tuberculosis')} tuberculosis")

    # Step 3: Organize images into class directories
    counts = organize_by_class(dataset_path, labels)
    for cls, count in counts.items():
        print(f"  {cls}: {count} images")

    # Step 4: Upload to S3
    raw_data_s3_uri = upload_to_s3(dataset_path, bucket, prefix)

    # Step 5: Generate manifest
    manifest_path = output_dir / "master_manifest.csv"
    generate_manifest(dataset_path, manifest_path)

    print("=" * 60)
    print("Setup complete!")
    print(f"  Raw data: {raw_data_s3_uri}")
    print(f"  Manifest: {manifest_path}")
    print("=" * 60)

    return raw_data_s3_uri


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup Montgomery County CXR dataset for the demo")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="DEMO-cxr-dvc", help="S3 key prefix")
    parser.add_argument("--output-dir", default=".", help="Local output directory")
    args = parser.parse_args()

    setup_dataset(args.bucket, args.prefix, args.output_dir)
