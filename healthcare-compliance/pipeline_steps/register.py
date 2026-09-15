# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Register step: make the logged model deployable, then register it.

Same pattern as Part 8 of the healthcare notebook, run as a pipeline step:

1. Resolve the MLflow 3 logged model produced by the training run.
2. Upload ``inference.py`` into the logged model's artifact store under
   ``code/`` with ``MlflowClient.log_model_artifacts()``.
3. Log a SageMaker inference specification on the logged model with the
   ``sagemaker-mlflow`` plugin (container image, S3Prefix ModelDataSource
   pointing at the MLflow artifact location, SAGEMAKER_PROGRAM env).
4. ``mlflow.register_model()``. With the MLflow App in
   ``AutoModelRegistrationEnabled`` mode, and the specification logged FIRST,
   the auto-synced SageMaker Model Package is created already deployable.
5. Set the approval status and lineage metadata on the Model Package.
6. Add the lineage edges SageMaker cannot infer by itself: the DVC dataset (pulled
   inside the training container, so invisible to the training job's inputs) and the
   link from the training job to the auto-synced ``MLflow Experiment`` node.
"""

import os
import shutil
import sys
import tempfile
import time

import boto3
import mlflow
from mlflow import MlflowClient
import sagemaker_mlflow

# utils/ sits next to pipeline_steps/ in the working directory that @step uploads
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def register(
    mlflow_run_id: str,
    parent_run_id: str,
    registered_model_name: str,
    inference_image: str,
    model_approval_status: str,
    pipeline_run_id: str,
    data_version: str,
    data_git_commit_id: str,
    training_job_name: str,
    val_accuracy: float,
    patient_count: str = "unknown",
    dvc_remote_uri: str = None,
    dvc_repo_url: str = None,
    processing_job_name: str = None,
) -> dict:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    sm_client = boto3.client("sagemaker")
    region = sm_client.meta.region_name

    # 1. Logged model of the training run
    run = client.get_run(mlflow_run_id)
    logged_model = client.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{mlflow_run_id}'",
        max_results=1,
    )[0]
    print(f"Logged model {logged_model.model_id} at {logged_model.artifact_location}")

    # 2. code/inference.py into the model's artifact store
    inference_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference.py")
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "code"))
        shutil.copy(inference_src, os.path.join(tmp, "code", "inference.py"))
        client.log_model_artifacts(logged_model.model_id, tmp)
    print("Logged code/inference.py")

    # 3. Inference specification (must be logged BEFORE register_model)
    inference_spec = {
        "Containers": [{
            "Image": inference_image,
            "ModelDataSource": {
                "S3DataSource": {
                    "S3Uri": logged_model.artifact_location.rstrip("/") + "/",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "SAGEMAKER_REGION": region,
            },
        }],
        "SupportedContentTypes": ["image/jpeg", "image/png"],
        "SupportedResponseMIMETypes": ["application/json"],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large", "ml.m5.xlarge", "ml.c5.xlarge"],
        "SupportedTransformInstanceTypes": ["ml.m5.xlarge"],
    }
    sagemaker_mlflow.log_inference_specification(
        logged_model.model_id, inference_specification=inference_spec
    )
    print("Logged inference specification")

    # 4. Register; the MLflow App auto-syncs a Model Package and tags the model version
    model_version = mlflow.register_model(f"models:/{logged_model.model_id}", registered_model_name)
    print(f"Registered {model_version.name} v{model_version.version}")

    model_package_arn = None
    for _ in range(40):
        mv = client.get_model_version(model_version.name, model_version.version)
        model_package_arn = mv.tags.get("sagemaker.model_package_arn")
        if model_package_arn:
            break
        time.sleep(3)
    if not model_package_arn:
        raise TimeoutError("SageMaker Model Package was not auto-created within the timeout")
    print(f"Synced Model Package: {model_package_arn}")

    # 5. Approval status + lineage metadata. The spec is already on the package.
    sm_client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus=model_approval_status,
        CustomerMetadataProperties={
            "mlflow_registered_model": f"{model_version.name}/{model_version.version}",
            "mlflow_model_id": logged_model.model_id,
            "mlflow_run_id": mlflow_run_id,
            "pipeline_run_id": pipeline_run_id,
            "data_version": data_version,
            "data_git_commit_id": data_git_commit_id,
            "training_job_name": training_job_name,
            "val_accuracy": f"{val_accuracy:.4f}",
            # Record-level lineage: how many patients' scans went into this model. The full
            # patient manifest is the manifest.csv artifact of the training run.
            "patient_count": str(patient_count),
        },
    )

    described = sm_client.describe_model_package(ModelPackageName=model_package_arn)
    env = described["InferenceSpecification"]["Containers"][0].get("Environment", {})
    if env.get("SAGEMAKER_PROGRAM") != "inference.py":
        raise RuntimeError(f"Inference specification did not round-trip through the sync: {env}")

    result = {
        "model_package_arn": model_package_arn,
        "model_package_group_name": described["ModelPackageGroupName"],
        "model_approval_status": described["ModelApprovalStatus"],
        "registered_model_name": model_version.name,
        "registered_model_version": model_version.version,
        "mlflow_model_id": logged_model.model_id,
    }

    # 6. The two edges SageMaker's automatic lineage cannot produce:
    #    - the DVC dataset: train.py pulls it inside the container, so the training job's
    #      only recorded inputs are its code / sm_drivers channels,
    #    - training job -> MLflow Experiment: the auto-sync links that node to the Model
    #      Package, but not to the job that produced the run.
    #    Deliberately non-fatal: an already registered, approved and deployable Model
    #    Package should not be lost to a lineage permission or timing problem.
    dataset_artifact_arn = None
    if dvc_remote_uri:
        try:
            from utils.lineage_utils import record_run_lineage

            lineage = record_run_lineage(
                mlflow_run_id=mlflow_run_id,
                training_job_name=training_job_name,
                data_version=data_version,
                data_git_commit_id=data_git_commit_id,
                dvc_remote_uri=dvc_remote_uri,
                dvc_repo_url=dvc_repo_url,
                pipeline_run_id=pipeline_run_id,
                processing_job_name=processing_job_name,
                dataset_properties={"patient_count": str(patient_count)},
                sm_client=sm_client,
            )
            dataset_artifact_arn = lineage["dataset_artifact_arn"]
            result["dataset_artifact_arn"] = dataset_artifact_arn
        except Exception as error:  # noqa: BLE001 - lineage must not fail registration
            print(f"WARNING: could not record DVC dataset lineage: {error}")

    # 7. Record the registration as a nested run under the pipeline's parent run, so the
    #    parent shows preprocessing -> training -> evaluation -> registration in one place.
    with mlflow.start_run(run_id=parent_run_id), \
         mlflow.start_run(run_name=f"register-{pipeline_run_id}", nested=True):
        mlflow.set_tags({"pipeline_run_id": pipeline_run_id, "stage": "registration",
                         "data_version": data_version})
        mlflow.log_params({**result, "training_run_id": mlflow_run_id,
                           "data_git_commit_id": data_git_commit_id, "val_accuracy": val_accuracy})
    return result
