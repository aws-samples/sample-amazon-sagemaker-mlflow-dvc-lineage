# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Dataset Preprocessing Script

Reads a patient consent manifest, loads raw chest X-ray images from S3
(via a SageMaker Processing input channel), processes images for patients
with active consent, and versions the output with DVC.

Input channels (SageMaker Processing):
    /opt/ml/processing/input/registry/manifest.csv — Patient consent manifest
    /opt/ml/processing/input/raw-data/             — Raw chest X-ray images from S3

The processing code is stateless — the same code handles any manifest.
An opt-out is simply a consent status change in the manifest before
calling the same processing job.
"""

import os
import csv
import argparse
import subprocess
from pathlib import Path

import mlflow

from mlflow_utils import get_or_create_pipeline_run
from PIL import Image
from sklearn.model_selection import train_test_split


REGISTRY_PATH = '/opt/ml/processing/input/registry'
RAW_DATA_PATH = '/opt/ml/processing/input/raw-data'
TARGET_SIZE = (128, 128)


def setup_dvc_repo():
    """Clone and setup DVC repository."""
    dvc_repo_url = os.environ.get('DVC_REPO_URL')
    dvc_repo_name = os.environ.get('DVC_REPO_NAME')

    repo_path = f"/opt/ml/processing/{dvc_repo_name}"

    print(f"Cloning DVC repo: {dvc_repo_url}")
    subprocess.check_call(["git", "clone", dvc_repo_url, repo_path])
    subprocess.check_call(["git", "config", "user.email", "sagemaker@aws.com"], cwd=repo_path)
    subprocess.check_call(["git", "config", "user.name", "SageMaker"], cwd=repo_path)

    return repo_path


def read_patient_registry(registry_dir):
    """Read patient consent manifest from input channel."""
    path = os.path.join(registry_dir, 'manifest.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Registry not found at {path}")

    records = []
    with open(path, 'r') as f:
        for row in csv.DictReader(f):
            records.append(dict(row))

    patients = len(set(r['patient_id'] for r in records))
    active = len(set(
        r['patient_id'] for r in records if r['consent_status'] == 'active'
    ))
    print(f"Patient manifest: {len(records)} scans, {patients} patients, {active} active")
    return records


def load_images_from_manifest(registry, raw_data_dir):
    """Load images for active patients from the raw data input channel.

    Reads manifest records, resolves each scan's local path within the
    raw data input channel, and loads the image with PIL.

    Args:
        registry: List of dicts from read_patient_registry()
        raw_data_dir: Path to the raw data input channel

    Returns:
        dict: {patient_id: [(PIL.Image, label, scan_id), ...]}
    """
    active_records = [r for r in registry if r['consent_status'] == 'active']

    patient_data = {}
    loaded = 0
    skipped = 0

    for record in active_records:
        patient_id = record['patient_id']
        scan_id = record['scan_id']
        s3_key = record['s3_key']
        label = record['label']

        local_path = os.path.join(raw_data_dir, s3_key)

        if not os.path.exists(local_path):
            print(f"Warning: image not found at {local_path}, skipping {scan_id}")
            skipped += 1
            continue

        img = Image.open(local_path).convert('RGB')

        if patient_id not in patient_data:
            patient_data[patient_id] = []
        patient_data[patient_id].append((img, label, scan_id))
        loaded += 1

    print(f"Loaded {loaded} images for {len(patient_data)} patients "
          f"({skipped} skipped)")
    return patient_data


def process_images(patient_data, output_dir, val_split=0.15, test_split=0.15):
    """Process images for active patients into ImageFolder format.

    Performs a patient-level stratified split: each patient (and all their
    scans) goes entirely into train, validation, or test. This prevents
    data leakage between splits.

    Returns (counters, patient_count).
    """
    output_dir = Path(output_dir)

    patient_ids = sorted(patient_data.keys())

    # Determine majority label per patient for stratification
    patient_labels = []
    for pid in patient_ids:
        labels = [label for (_, label, _) in patient_data[pid]]
        majority = max(set(labels), key=labels.count)
        patient_labels.append(majority)

    # Patient-level stratified split: train / (val+test)
    train_patients, valtest_patients, train_labels, valtest_labels = train_test_split(
        patient_ids, patient_labels,
        test_size=(val_split + test_split),
        stratify=patient_labels,
        random_state=42,
    )

    # Split val+test into val / test
    relative_test = test_split / (val_split + test_split)
    val_patients, test_patients, _, _ = train_test_split(
        valtest_patients, valtest_labels,
        test_size=relative_test,
        stratify=valtest_labels,
        random_state=42,
    )

    # Build split assignment map
    split_map = {}
    for pid in train_patients:
        split_map[pid] = 'train'
    for pid in val_patients:
        split_map[pid] = 'validation'
    for pid in test_patients:
        split_map[pid] = 'test'

    enriched_records = []
    counters = {}  # (split_name, class_name) -> count

    for patient_id in patient_ids:
        split_name = split_map[patient_id]

        for img, label, scan_id in patient_data[patient_id]:
            key = (split_name, label)
            counters[key] = counters.get(key, 0) + 1

            dst_path = output_dir / split_name / label / f"{counters[key]:05d}.png"
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Preprocess: resize to model input size
            img.resize(TARGET_SIZE).save(dst_path)

            enriched_records.append({
                'patient_id': patient_id,
                'scan_id': scan_id,
                'file_path': str(dst_path.relative_to(output_dir)),
                'split': split_name,
                'label': label,
            })

    # Write enriched manifest
    enriched_records.sort(key=lambda r: (r['patient_id'], r['scan_id']))
    manifest_path = output_dir / 'manifest.csv'
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'patient_id', 'scan_id', 'file_path', 'split', 'label'
        ])
        writer.writeheader()
        writer.writerows(enriched_records)

    patient_count = len(patient_ids)
    total = sum(counters.values())
    print(f"Processed {total} images for {patient_count} patients")
    for s in ['train', 'validation', 'test']:
        s_total = sum(v for (sn, _), v in counters.items() if sn == s)
        s_patients = sum(1 for pid in patient_ids if split_map[pid] == s)
        print(f"  {s.capitalize()}: {s_total} images, {s_patients} patients")
    print(f"Manifest: {len(enriched_records)} records")

    return counters, patient_count


def version_with_dvc(repo_path, version_tag, pipeline_run_id):
    """Add data to DVC and push to remote."""

    print(f"Versioning data with DVC (tag: {pipeline_run_id})...")

    subprocess.check_call(["dvc", "add", "dataset"], cwd=repo_path)
    subprocess.check_call(["git", "add", "dataset.dvc", ".gitignore"], cwd=repo_path)
    # Commit only if the dataset actually changed. Re-running with identical inputs (same
    # data version / fraction / seed) produces the same DVC hash, in which case the existing
    # commit is reused and just gets the new PIPELINE_RUN_ID tag.
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_path).returncode != 0:
        subprocess.check_call(
            ["git", "commit", "-m", f"Add dataset version {version_tag}"],
            cwd=repo_path
        )
    else:
        print("Dataset unchanged since the last commit; tagging the existing commit")
    subprocess.check_call(["git", "tag", pipeline_run_id], cwd=repo_path)

    print("Pushing data to DVC remote (S3)...")
    subprocess.check_call(["dvc", "push"], cwd=repo_path)

    print("Pushing to git...")
    subprocess.check_call(["git", "push", "origin", "main"], cwd=repo_path)
    subprocess.check_call(["git", "push", "origin", pipeline_run_id], cwd=repo_path)

    commit_id = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_path
    ).decode().strip()

    return commit_id


def log_to_mlflow(counters, patient_count, commit_id, version_tag, pipeline_run_id):
    """Log preprocessing metadata to MLflow."""

    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME')

    if not tracking_uri:
        print("MLFLOW_TRACKING_URI not set, skipping MLflow logging")
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    classes = sorted(set(cls for (_, cls) in counters.keys()))
    train_total = sum(v for (s, _), v in counters.items() if s == 'train')
    val_total = sum(v for (s, _), v in counters.items() if s == 'validation')
    test_total = sum(v for (s, _), v in counters.items() if s == 'test')

    run_name = f"preprocess-{pipeline_run_id}"

    # All stages of one PIPELINE_RUN_ID are nested under a shared parent run
    parent_run_id = get_or_create_pipeline_run(pipeline_run_id, version_tag)
    with mlflow.start_run(run_id=parent_run_id), \
         mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.set_tags({
            "pipeline_run_id": pipeline_run_id,
            "stage": "preprocessing",
            "data_version": version_tag,
        })

        mlflow.log_params({
            "num_classes": len(classes),
            "classes": ",".join(classes),
            "data_version": version_tag,
            "data_git_commit_id": commit_id,
            "total_train_images": train_total,
            "total_val_images": val_total,
            "total_test_images": test_total,
            "patient_count": patient_count,
        })

        print(f"Logged to MLflow run: {run.info.run_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-split", type=float, default=0.15,
        help="Fraction of data to use for validation"
    )
    parser.add_argument(
        "--test-split", type=float, default=0.15,
        help="Fraction of data to use for testing"
    )
    parser.add_argument(
        "--data-version", type=str, default="v1.0",
        help="Version tag for the data"
    )
    args = parser.parse_args()

    pipeline_run_id = os.environ.get('PIPELINE_RUN_ID', args.data_version)

    print(f"Processing dataset from patient manifest")
    print(f"Data version: {args.data_version}")

    # Read patient manifest
    registry = read_patient_registry(REGISTRY_PATH)

    # Load raw images for active patients
    patient_data = load_images_from_manifest(registry, RAW_DATA_PATH)

    # Setup DVC repo
    repo_path = setup_dvc_repo()
    data_dir = Path(repo_path) / "dataset"
    data_dir.mkdir(exist_ok=True)

    # Process images for active patients
    counters, patient_count = process_images(
        patient_data, data_dir,
        val_split=args.val_split, test_split=args.test_split
    )

    # Version with DVC
    commit_id = version_with_dvc(repo_path, args.data_version, pipeline_run_id)

    # Log to MLflow
    log_to_mlflow(counters, patient_count, commit_id, args.data_version, pipeline_run_id)

    print("Preprocessing complete!")
    print(f"Data versioned as: {pipeline_run_id}")
    print(f"Git commit: {commit_id}")


if __name__ == "__main__":
    main()
