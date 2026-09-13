# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Evaluate step: read the training run's metrics back from MLflow.

The training job (source_dir/train.py) already computes validation metrics
on the DVC-versioned validation split and logs them to MLflow. Instead of
re-running inference, this step resolves the MLflow run that belongs to
this pipeline execution (via the shared ``pipeline_run_id`` tag) and returns
its metrics, so a ConditionStep can gate registration on them.
"""

import os

import mlflow


def evaluate(data_version: str, pipeline_execution_id: str, training_job_name: str) -> dict:
    """Resolve the training run of this pipeline execution and return its metrics.

    Args:
        data_version: The ``DataVersion`` pipeline parameter.
        pipeline_execution_id: The pipeline execution id. Together with
            ``data_version`` this forms ``PIPELINE_RUN_ID`` (the same value the
            processing and training jobs receive as an environment variable), set
            as the ``pipeline_run_id`` tag by train.py and used as the DVC git tag.
        training_job_name: Name of the SageMaker Training job (from the
            TrainingStep properties). Recorded for traceability.

    Returns:
        dict with ``pipeline_run_id``, ``parent_run_id``, ``mlflow_run_id``, ``val_accuracy``,
        ``data_version``, ``data_git_commit_id``, ``patient_count`` and ``training_job_name``.
    """
    pipeline_run_id = f"{data_version}-{pipeline_execution_id}"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.get_experiment_by_name(os.environ["MLFLOW_EXPERIMENT_NAME"])

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.pipeline_run_id = '{pipeline_run_id}' and tags.stage = 'training'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise RuntimeError(
            f"No MLflow training run found for pipeline_run_id={pipeline_run_id}"
        )
    run = runs[0]
    # All stages of this pipeline run are nested under one parent run (created by the
    # processing job, see source_dir/mlflow_utils.py); the training run points at it.
    parent_run_id = run.data.tags.get("mlflow.parentRunId")

    result = {
        "pipeline_run_id": pipeline_run_id,
        "parent_run_id": parent_run_id,
        "mlflow_run_id": run.info.run_id,
        "val_accuracy": float(run.data.metrics["final_val_accuracy"]),
        "data_version": run.data.params.get("data_version"),
        "data_git_commit_id": run.data.params.get("data_git_commit_id"),
        "patient_count": run.data.params.get("patient_count", "unknown"),
        "training_job_name": training_job_name,
    }
    print(f"Training run {run.info.run_name}: {result}")

    # Log the evaluation itself as a nested run, next to preprocessing and training
    with mlflow.start_run(run_id=parent_run_id), \
         mlflow.start_run(run_name=f"evaluate-{pipeline_run_id}", nested=True):
        mlflow.set_tags({"pipeline_run_id": pipeline_run_id, "stage": "evaluation",
                         "data_version": result["data_version"]})
        mlflow.log_params({"training_run_id": run.info.run_id, "training_job_name": training_job_name,
                           "data_git_commit_id": result["data_git_commit_id"]})
        mlflow.log_metric("val_accuracy", result["val_accuracy"])
    return result
