# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""MLflow run grouping for one pipeline run id.

Every stage of a pipeline run (preprocessing, training, evaluation,
registration) logs its own MLflow run. To keep them together in the MLflow UI
they are *nested* under one parent run per ``PIPELINE_RUN_ID``:

    <PIPELINE_RUN_ID>            stage=pipeline   (parent)
    ├── preprocess-<id>          stage=preprocessing
    ├── train-<id>               stage=training
    ├── evaluate-<id>            stage=evaluation   (pipeline only)
    └── register-<id>            stage=registration (pipeline only)

The first stage that runs creates the parent; later stages find it by the
``pipeline_run_id`` tag. This works the same whether the stages are started
by hand from a notebook or by a SageMaker AI Pipeline, and needs no run id
to be passed between jobs.

Usage in a job script::

    parent_run_id = get_or_create_pipeline_run(pipeline_run_id, data_version)
    with mlflow.start_run(run_id=parent_run_id):
        with mlflow.start_run(run_name=f"train-{pipeline_run_id}", nested=True) as run:
            ...
"""

import os

import mlflow
from mlflow import MlflowClient


def get_or_create_pipeline_run(pipeline_run_id, data_version=None, tags=None):
    """Return the run id of the parent run grouping all stages of ``pipeline_run_id``.

    The active MLflow experiment (``mlflow.set_experiment``) is used. Creates
    the parent run if it does not exist yet.
    """
    experiment = mlflow.get_experiment(_active_experiment_id())

    existing = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.pipeline_run_id = '{pipeline_run_id}' and tags.stage = 'pipeline'",
        order_by=["attributes.start_time ASC"],
        max_results=1,
        output_format="list",
    )
    if existing:
        return existing[0].info.run_id

    run_tags = {
        "pipeline_run_id": pipeline_run_id,
        "stage": "pipeline",
    }
    if data_version:
        run_tags["data_version"] = data_version
    # Set by the SageMaker AI Pipeline definition (ExecutionVariables); absent for manual runs
    pipeline_execution_arn = os.environ.get("PIPELINE_EXECUTION_ARN")
    if pipeline_execution_arn:
        run_tags["sagemaker.pipeline_execution_arn"] = pipeline_execution_arn
        run_tags["sagemaker.pipeline_name"] = pipeline_execution_arn.split(":pipeline/")[-1].split("/")[0]
    if tags:
        run_tags.update(tags)

    run = MlflowClient().create_run(
        experiment_id=experiment.experiment_id,
        run_name=pipeline_run_id,
        tags=run_tags,
    )
    print(f"Created parent MLflow run {run.info.run_id} for pipeline run id {pipeline_run_id}")
    return run.info.run_id


def _active_experiment_id():
    from mlflow.tracking.fluent import _get_experiment_id

    return _get_experiment_id()
