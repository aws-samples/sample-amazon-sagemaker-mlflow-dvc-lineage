# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
PyTorch MobileNetV3-Small Training Script for Chest X-Ray Classification

This script fine-tunes a pretrained MobileNetV3-Small model on chest X-ray
image data (Montgomery County CXR dataset) that has been versioned with DVC.
It logs metrics, the patient-level manifest, and the model to MLflow.
Registration to the MLflow Model Registry happens in the notebook, after an
inference specification has been attached to the logged model.

Usage:
    Called by SageMaker ModelTrainer with environment variables:
    - MLFLOW_TRACKING_URI: MLflow tracking server ARN
    - MLFLOW_EXPERIMENT_NAME: Experiment name
    - DVC_REPO_URL: AWS CodeCommit repo URL for DVC
    - DATA_VERSION: DVC version tag (e.g., v1.0)
    - PIPELINE_RUN_ID: Unique run identifier
"""

import os
import sys
import csv
import json
import argparse
import subprocess
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature


# SageMaker paths
INPUT_PATH = '/opt/ml/input/data'
OUTPUT_PATH = '/opt/ml/output'
MODEL_PATH = '/opt/ml/model'
DVC_REPO_PATH = tempfile.mkdtemp(prefix='dvc_repo_')
DATASET_PATH = f'{DVC_REPO_PATH}/dataset'


def get_sagemaker_job_tags():
    """Build MLflow tags describing the SageMaker Training job running this script.

    SageMaker injects TRAINING_JOB_ARN and TRAINING_JOB_NAME into the container
    environment, and the training toolkit exposes the job configuration as the
    SM_TRAINING_ENV JSON document. The same tags are set on the MLflow run and
    on the logged model so both can be traced back to the job (and to its
    CloudWatch logs and console page).
    """
    job_arn = os.environ.get('TRAINING_JOB_ARN')
    job_name = os.environ.get('TRAINING_JOB_NAME')
    sm_training_env = json.loads(os.environ.get('SM_TRAINING_ENV', '{}'))

    if not job_name:
        job_name = sm_training_env.get('job_name')
    if not job_name and job_arn:
        job_name = job_arn.rsplit('/', 1)[-1]

    if not job_arn and not job_name:
        return {'sagemaker.runtime': 'local'}

    region = os.environ.get('AWS_REGION') or (job_arn.split(':')[3] if job_arn else None)

    tags = {
        'sagemaker.runtime': 'sagemaker',
        'sagemaker.training_job_name': job_name,
        'mlflow.source.type': 'JOB',
    }
    if job_arn:
        tags['sagemaker.training_job_arn'] = job_arn
    if region and job_name:
        tags['mlflow.source.name'] = (
            f"https://{region}.console.aws.amazon.com/sagemaker/home"
            f"?region={region}#/jobs/{job_name}"
        )
    image = sm_training_env.get('additional_framework_parameters', {}).get('sagemaker_training_image')
    if image:
        tags['sagemaker.container_image'] = image
    instance_type = os.environ.get('SM_CURRENT_INSTANCE_TYPE') or sm_training_env.get('current_instance_type')
    if instance_type:
        tags['sagemaker.instance_type'] = instance_type
    return tags


def fetch_data_from_dvc():
    """Clone DVC repo and pull versioned data."""
    dvc_repo_url = os.environ.get('DVC_REPO_URL')
    pipeline_run_id = os.environ.get('PIPELINE_RUN_ID')
    
    print(f"Cloning repo: {dvc_repo_url}, version tag: {pipeline_run_id}")
    
    subprocess.check_call([
        "git", "clone", "--depth", "1", "--branch", pipeline_run_id,
        dvc_repo_url, DVC_REPO_PATH
    ])
    
    data_git_commit_id = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=DVC_REPO_PATH
    ).decode().strip()
    
    print("Running dvc pull...")
    # Run dvc pull from repo root (where dataset.dvc is)
    subprocess.check_call(["dvc", "pull"], cwd=DVC_REPO_PATH)
    
    return data_git_commit_id


def get_data_transforms():
    """Get transforms for training and validation."""
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def create_model(num_classes):
    """Create MobileNetV3-Small model with custom classifier head."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    
    # Replace the classifier head
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=0)
    args, _ = parser.parse_known_args()
    
    # Environment variables
    dvc_repo_url = os.environ.get('DVC_REPO_URL')
    data_version = os.environ.get('DATA_VERSION')
    pipeline_run_id = os.environ.get('PIPELINE_RUN_ID', data_version)
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'mlruns')
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'cxr-classification')
    # Name of the logged model artifact (also used as the registered model name by the notebook)
    model_name = os.environ.get('MLFLOW_REGISTERED_MODEL_NAME', 'CXR-MobileNetV3')
    
    # Fetch data from DVC
    data_git_commit_id = fetch_data_from_dvc()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    run_name = f"train-{pipeline_run_id}"
    
    print("Starting training...")
    
    try:
        with mlflow.start_run(run_name=run_name) as run:
            # Set tags: lineage tags plus the SageMaker Training job that produced this run
            sagemaker_job_tags = get_sagemaker_job_tags()
            mlflow.set_tags({
                "pipeline_run_id": pipeline_run_id,
                "stage": "training",
                "data_version": data_version,
                "model_type": "mobilenet_v3_small",
                **sagemaker_job_tags,
            })
            
            # Load datasets
            train_transform, val_transform = get_data_transforms()
            
            train_dir = f"{DATASET_PATH}/train"
            val_dir = f"{DATASET_PATH}/validation"
            
            train_dataset = ImageFolder(train_dir, transform=train_transform)
            val_dataset = ImageFolder(val_dir, transform=val_transform)
            
            num_classes = len(train_dataset.classes)
            class_names = train_dataset.classes
            
            print(f"Found {num_classes} classes: {class_names}")
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")
            
            # Log parameters
            params = {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_classes": num_classes,
                "class_names": ",".join(class_names),
                "data_version": data_version,
                "data_git_commit_id": data_git_commit_id,
                "dvc_repo_url": dvc_repo_url,
                "model_architecture": "mobilenet_v3_small",
                "pretrained": True,
                "optimizer": "Adam",
                "device": str(device),
            }

            # Log manifest as artifact (if present in DVC-versioned dataset)
            manifest_path = f"{DATASET_PATH}/manifest.csv"
            if os.path.exists(manifest_path):
                mlflow.log_artifact(manifest_path)
                with open(manifest_path, 'r') as f:
                    reader = csv.DictReader(f)
                    patient_ids = set(row['patient_id'] for row in reader)
                params["patient_count"] = len(patient_ids)
                print(f"Logged manifest: {len(patient_ids)} patients")

            mlflow.log_params(params)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            # Create model
            model = create_model(num_classes)
            model = model.to(device)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            
            # Training loop
            best_val_acc = 0.0
            
            for epoch in range(args.epochs):
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                val_loss, val_acc = validate(model, val_loader, criterion, device)
                
                print(f"Epoch {epoch+1}/{args.epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Log metrics
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                }, step=epoch)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            
            # Log final metrics
            mlflow.log_metrics({
                "final_val_accuracy": best_val_acc,
                "final_train_accuracy": train_acc,
            })
            
            # Prepare sample input for signature
            sample_input = torch.randn(1, 3, 128, 128)
            model.eval()
            with torch.no_grad():
                sample_output = model(sample_input.to(device)).cpu()
            
            signature = infer_signature(
                sample_input.numpy(),
                sample_output.numpy()
            )
            
            # Log the model (do NOT register it here). The notebook attaches an
            # inference.py and a SageMaker inference specification to this logged
            # model and then calls mlflow.register_model(); with the MLflow App in
            # AutoModelRegistrationEnabled mode that single call syncs a deployable
            # Model Package to the SageMaker Model Registry.
            # MLflow >= 3.15 defaults serialization_format to "pt2" (torch.export
            # traced graph), which requires an input_example and fixes the traced
            # input shape. Use "pickle" to save the eager nn.Module.
            model_info = mlflow.pytorch.log_model(
                model,
                name=model_name,
                signature=signature,
                serialization_format="pickle",
                # Tags on the logged model itself (MLflow 3 entity), so the model can be
                # traced to its training job and data version without going via the run
                tags={
                    "pipeline_run_id": pipeline_run_id,
                    "data_version": data_version,
                    "data_git_commit_id": data_git_commit_id,
                    **sagemaker_job_tags,
                },
            )
            
            print(f"Training complete!")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            print(f"Logged model: {model_info.model_id}")
            print(f"SageMaker training job: {sagemaker_job_tags.get('sagemaker.training_job_arn', 'local')}")
            print(f"Run ID: {run.info.run_id}")
    
    except Exception as e:
        trc = traceback.format_exc()
        with open(f'{OUTPUT_PATH}/failure', 'w', encoding='utf-8') as f:
            f.write(f'Exception during training: {e}\n{trc}')
        print(f'Exception during training: {e}\n{trc}')
        sys.exit(255)
    
    sys.exit(0)


if __name__ == '__main__':
    main()
