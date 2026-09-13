# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
CIFAR-10 Preprocessing Script for DVC Versioning

This script downloads CIFAR-10, optionally samples a fraction of training data,
saves images to disk in ImageFolder format, and versions the data with DVC.

Usage:
    Called by SageMaker FrameworkProcessor with arguments:
    --data-fraction: Fraction of training data to use (0.0-1.0)
    --val-split: Fraction of training data to use for validation
    --data-version: Version tag for DVC (e.g., v1.0)

Environment variables:
    DVC_REPO_URL: AWS CodeCommit repo URL
    DVC_REPO_NAME: Repository name
    MLFLOW_TRACKING_URI: MLflow tracking server ARN
    MLFLOW_EXPERIMENT_NAME: Experiment name
    PIPELINE_RUN_ID: Unique run identifier
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision

import mlflow

from mlflow_utils import get_or_create_pipeline_run


# CIFAR-10 class names (in order of label index)
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def setup_dvc_repo():
    """Clone and setup DVC repository."""
    dvc_repo_url = os.environ.get('DVC_REPO_URL')
    dvc_repo_name = os.environ.get('DVC_REPO_NAME')
    
    repo_path = f"/opt/ml/processing/{dvc_repo_name}"
    
    print(f"Cloning DVC repo: {dvc_repo_url}")
    subprocess.check_call(["git", "clone", dvc_repo_url, repo_path])
    
    # Configure git
    subprocess.check_call(["git", "config", "user.email", "sagemaker@aws.com"], cwd=repo_path)
    subprocess.check_call(["git", "config", "user.name", "SageMaker"], cwd=repo_path)
    
    return repo_path


def download_cifar10():
    """Download CIFAR-10 dataset."""
    print("Downloading CIFAR-10...")
    
    cifar_dir = tempfile.mkdtemp(prefix='cifar10_')
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir,
        train=True,
        download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=cifar_dir,
        train=False,
        download=True
    )
    
    return train_dataset, test_dataset


def save_images(images, labels, output_dir, split_name):
    """Save images to disk in ImageFolder format (split/class_name/00001.png)."""
    counts = {c: 0 for c in CIFAR10_CLASSES}

    for class_name in CIFAR10_CLASSES:
        (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)

    for image, label in zip(images, labels):
        class_name = CIFAR10_CLASSES[label]
        counts[class_name] += 1
        path = output_dir / split_name / class_name / f"{counts[class_name]:05d}.png"
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(path)

    total = sum(counts.values())
    print(f"{split_name.capitalize()} images: {total}")
    return counts


def create_train_val_split(train_dataset, output_dir, data_fraction, val_split):
    """Split training data into train and validation sets using sklearn."""
    images = [img for img, _ in train_dataset]
    labels = [lbl for _, lbl in train_dataset]

    # Subsample if data_fraction < 1.0
    if data_fraction < 1.0:
        images, _, labels, _ = train_test_split(
            images, labels,
            train_size=data_fraction,
            stratify=labels,
            random_state=42,
        )

    # Stratified train/val split
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        images, labels,
        test_size=val_split,
        stratify=labels,
        random_state=42,
    )

    train_counts = save_images(train_imgs, train_lbls, output_dir, 'train')
    val_counts = save_images(val_imgs, val_lbls, output_dir, 'validation')
    return train_counts, val_counts


def save_test_set(test_dataset, output_dir):
    """Save full test set (always 100% for fair evaluation)."""
    images = [img for img, _ in test_dataset]
    labels = [lbl for _, lbl in test_dataset]
    return save_images(images, labels, output_dir, 'test')


def version_with_dvc(repo_path, version_tag, pipeline_run_id):
    """Add data to DVC and push to remote."""
    
    print(f"Versioning data with DVC (tag: {pipeline_run_id})...")
    
    # Add data directory to DVC
    subprocess.check_call(["dvc", "add", "dataset"], cwd=repo_path)
    
    # Git add the .dvc file
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
    
    # Tag with pipeline_run_id (includes timestamp)
    subprocess.check_call(["git", "tag", pipeline_run_id], cwd=repo_path)
    
    # Push to DVC remote
    print("Pushing data to DVC remote (S3)...")
    subprocess.check_call(["dvc", "push"], cwd=repo_path)
    
    # Push to git
    print("Pushing to git...")
    subprocess.check_call(["git", "push", "origin", "main"], cwd=repo_path)
    subprocess.check_call(["git", "push", "origin", pipeline_run_id], cwd=repo_path)
    
    # Get commit ID
    commit_id = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_path
    ).decode().strip()
    
    return commit_id


def log_to_mlflow(data_fraction, train_counts, val_counts, test_counts, 
                  commit_id, version_tag, pipeline_run_id):
    """Log preprocessing metadata to MLflow."""
    
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME')
    
    if not tracking_uri:
        print("MLFLOW_TRACKING_URI not set, skipping MLflow logging")
        return
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
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
            "data_fraction": data_fraction,
            "num_classes": len(CIFAR10_CLASSES),
            "classes": ",".join(CIFAR10_CLASSES),
            "data_version": version_tag,
            "data_git_commit_id": commit_id,
            "total_train_images": sum(train_counts.values()),
            "total_val_images": sum(val_counts.values()),
            "total_test_images": sum(test_counts.values()),
        })
        
        print(f"Logged to MLflow run: {run.info.run_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0.0-1.0)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation"
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default="v1.0",
        help="Version tag for the data"
    )
    args = parser.parse_args()
    
    # Validate data fraction
    if not 0.0 < args.data_fraction <= 1.0:
        print(f"Error: data-fraction must be between 0.0 and 1.0, got {args.data_fraction}")
        sys.exit(1)
    
    print(f"Processing CIFAR-10")
    print(f"Data fraction: {args.data_fraction} ({int(args.data_fraction * 100)}%)")
    print(f"Data version: {args.data_version}")
    
    # Get pipeline run ID from environment
    pipeline_run_id = os.environ.get('PIPELINE_RUN_ID', args.data_version)
    
    # Setup DVC repo
    repo_path = setup_dvc_repo()
    data_dir = Path(repo_path) / "dataset"
    data_dir.mkdir(exist_ok=True)
    
    # Download CIFAR-10
    train_dataset, test_dataset = download_cifar10()
    
    # Process and save images
    print("Processing images...")
    
    # Create train/validation split with data fraction
    train_counts, val_counts = create_train_val_split(
        train_dataset, data_dir, args.data_fraction, args.val_split
    )
    
    # Save full test set (always 100%)
    test_counts = save_test_set(test_dataset, data_dir)
    
    # Version with DVC
    commit_id = version_with_dvc(repo_path, args.data_version, pipeline_run_id)
    
    # Log to MLflow
    log_to_mlflow(
        args.data_fraction, train_counts, val_counts, test_counts,
        commit_id, args.data_version, pipeline_run_id
    )
    
    print("Preprocessing complete!")
    print(f"Data versioned as: {pipeline_run_id}")
    print(f"Git commit: {commit_id}")


if __name__ == "__main__":
    main()
