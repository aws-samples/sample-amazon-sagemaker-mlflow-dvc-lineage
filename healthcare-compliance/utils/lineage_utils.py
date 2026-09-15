# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Lineage helpers that close the two gaps SageMaker's automatic lineage leaves open.

SageMaker records what it mounts, and the Model Registry auto-sync records the MLflow
run that produced a Model Package. Neither of them can see:

1. **The DVC dataset.** ``train.py`` clones the DVC repo and runs ``dvc pull`` *inside*
   the container, so the training job's only lineage inputs are its ``code`` and
   ``sm_drivers`` channels. The processing job that produced the dataset pushes it to
   the DVC remote instead of writing to ``/opt/ml/processing/output``, so it has no
   output artifact either. The graph therefore breaks between preprocessing and
   training.
2. **The training job of an MLflow experiment.** The auto-sync creates an
   ``MLflow Experiment`` action pointing at the Model Package, but nothing links that
   action to the SageMaker Training job that produced the run.

These functions add the missing nodes and edges, so the graph reads end to end::

    raw scans + consent registry
        -> processing job  --Produced-->  DataSet (DVC commit)
        -> training job    --ContributedTo-->  MLflow Experiment action
        -> Model Package   -->  model deployment  -->  endpoint

Everything is idempotent: the DataSet artifact is keyed by the DVC git commit and
reused across runs of the same data version, and associations that already exist are
left alone. Safe to call on every pipeline execution and to re-run as a backfill.

The calling role needs ``sagemaker:CreateArtifact``, ``sagemaker:AddAssociation``,
``sagemaker:ListArtifacts``, ``sagemaker:ListActions``, ``sagemaker:ListTrialComponents``
and ``sagemaker:DescribeTrainingJob`` / ``DescribeProcessingJob``.
"""

import time

import boto3
from botocore.exceptions import ClientError

# SageMaker's own name for the lineage node type it creates for MLflow experiments
MLFLOW_EXPERIMENT_ACTION_TYPE = "MLflow Experiment"


def _sm_client(sm_client=None):
    return sm_client or boto3.client("sagemaker")


def trial_component_arn(job_arn, sm_client=None):
    """Return the lineage node SageMaker auto-creates for a training or processing job.

    ``AddAssociation`` only accepts ``artifact``, ``action``, ``context``, ``experiment``
    and ``experiment-trial-component`` ARNs -- a raw training-job ARN is rejected. The
    trial component named ``<job-name>-aws-training-job`` is how the job itself appears
    in the graph.

    Args:
        job_arn: ARN of a SageMaker Training or Processing job
        sm_client: Optional boto3 SageMaker client

    Returns:
        The trial component ARN, or None if SageMaker did not create one
    """
    sm = _sm_client(sm_client)
    summaries = sm.list_trial_components(SourceArn=job_arn)["TrialComponentSummaries"]
    return summaries[0]["TrialComponentArn"] if summaries else None


def training_job_arn(training_job_name, sm_client=None):
    """Resolve a training job name to its ARN."""
    sm = _sm_client(sm_client)
    return sm.describe_training_job(TrainingJobName=training_job_name)["TrainingJobArn"]


def processing_job_arn(processing_job_name, sm_client=None):
    """Resolve a processing job name to its ARN."""
    sm = _sm_client(sm_client)
    return sm.describe_processing_job(ProcessingJobName=processing_job_name)["ProcessingJobArn"]


def find_processing_job_of_training_job(training_job_name, sm_client=None):
    """Find the Processing job that ran in the same pipeline execution as a training job.

    Pipeline steps are tagged with ``sagemaker:pipeline-execution-arn``, and the
    execution's step list names the Processing job. (``list_processing_jobs`` with
    ``NameContains`` is not reliable for this: it filters a page at a time, so the job
    can be missing from the result even though it exists.)

    Args:
        training_job_name: Name of a SageMaker Training job
        sm_client: Optional boto3 SageMaker client

    Returns:
        The Processing job name, or None if the training job did not run in a pipeline
        or the execution had no processing step
    """
    sm = _sm_client(sm_client)
    job_arn = training_job_arn(training_job_name, sm)
    tags = {t["Key"]: t["Value"] for t in sm.list_tags(ResourceArn=job_arn)["Tags"]}
    execution_arn = tags.get("sagemaker:pipeline-execution-arn")
    if not execution_arn:
        return None

    steps = sm.list_pipeline_execution_steps(
        PipelineExecutionArn=execution_arn
    )["PipelineExecutionSteps"]
    for step in steps:
        processing_job = step.get("Metadata", {}).get("ProcessingJob", {})
        if processing_job.get("Arn"):
            return processing_job["Arn"].split("/")[-1]
    return None


def find_mlflow_experiment_action(mlflow_run_id, timeout=0, sm_client=None):
    """Find the ``MLflow Experiment`` lineage action for an MLflow run.

    The Model Registry auto-sync creates one action per registered model version and
    stores the MLflow run id in ``Source.SourceId``. It is created asynchronously, a
    moment after the Model Package appears, so pass a ``timeout`` when calling this
    right after ``mlflow.register_model()``.

    Args:
        mlflow_run_id: The MLflow run id (32-char hex)
        timeout: Seconds to keep polling for the action (0 = single attempt)
        sm_client: Optional boto3 SageMaker client

    Returns:
        The action ARN, or None if the sync did not create one (which happens when the
        MLflow App role lacks the lineage permissions -- the Model Package then carries
        ``mlflow.lineage_status=failed``)
    """
    sm = _sm_client(sm_client)
    deadline = time.time() + timeout

    while True:
        paginator = sm.get_paginator("list_actions")
        for page in paginator.paginate(ActionType=MLFLOW_EXPERIMENT_ACTION_TYPE):
            for action in page["ActionSummaries"]:
                if action.get("Source", {}).get("SourceId") == mlflow_run_id:
                    return action["ActionArn"]
        if time.time() >= deadline:
            return None
        time.sleep(3)


def get_or_create_dvc_dataset_artifact(
    dvc_remote_uri,
    data_git_commit_id,
    dvc_repo_url,
    data_version,
    pipeline_run_id=None,
    extra_properties=None,
    sm_client=None,
):
    """Get or create the DataSet artifact standing for one DVC data version.

    ``SourceUri`` is the DVC remote plus the git commit as a fragment. It is the dedup
    key SageMaker uses, it is unique per data version, and it stays resolvable: the
    commit is what ``git checkout <commit> && dvc pull`` needs to reconstruct the data.

    Args:
        dvc_remote_uri: S3 URI of the DVC remote, e.g. ``s3://<bucket>/DEMO-cxr-dvc``
        data_git_commit_id: DVC repo commit of this data version (from the MLflow run's
            ``data_git_commit_id`` parameter)
        dvc_repo_url: DVC repo URL, e.g. ``codecommit::us-west-2://cxr-dvc-demo``
        data_version: Human-readable version, e.g. ``v2.0``
        pipeline_run_id: Optional ``PIPELINE_RUN_ID`` (the git tag of the commit)
        extra_properties: Optional extra artifact properties, e.g. ``patient_count``
        sm_client: Optional boto3 SageMaker client

    Returns:
        The artifact ARN
    """
    sm = _sm_client(sm_client)
    source_uri = f"{dvc_remote_uri.rstrip('/')}#{data_git_commit_id}"

    existing = sm.list_artifacts(SourceUri=source_uri)["ArtifactSummaries"]
    if existing:
        return existing[0]["ArtifactArn"]

    properties = {
        "data_version": data_version,
        "data_git_commit_id": data_git_commit_id,
        "dvc_repo_url": dvc_repo_url,
    }
    if pipeline_run_id:
        properties["pipeline_run_id"] = pipeline_run_id
    properties.update(extra_properties or {})

    response = sm.create_artifact(
        # ArtifactName takes alphanumerics and hyphens only, so a version like "v2.0"
        # cannot be used as-is; the commit prefix is unique anyway.
        ArtifactName=f"dvc-dataset-{data_git_commit_id[:12]}",
        ArtifactType="DataSet",
        Source={
            "SourceUri": source_uri,
            "SourceTypes": [
                {"SourceIdType": "Custom", "Value": f"dvc-commit:{data_git_commit_id}"}
            ],
        },
        Properties=properties,
    )
    return response["ArtifactArn"]


def associate(source_arn, destination_arn, association_type, sm_client=None):
    """Add an association, treating "already exists" as success.

    Returns:
        True if the edge was created, False if it was already there
    """
    sm = _sm_client(sm_client)
    try:
        sm.add_association(
            SourceArn=source_arn,
            DestinationArn=destination_arn,
            AssociationType=association_type,
        )
        return True
    except ClientError as error:
        # AddAssociation is not idempotent: a duplicate raises ValidationException
        if "must be unique" in str(error):
            return False
        raise


def record_run_lineage(
    mlflow_run_id,
    training_job_name,
    data_version,
    data_git_commit_id,
    dvc_remote_uri,
    dvc_repo_url,
    pipeline_run_id=None,
    processing_job_name=None,
    dataset_properties=None,
    action_timeout=60,
    sm_client=None,
    verbose=True,
):
    """Add the DVC dataset node and the MLflow-experiment edge for one run.

    Creates (or reuses) the DataSet artifact for the data version and wires up:

    - ``processing job --Produced--> DataSet``          (if ``processing_job_name`` given)
    - ``DataSet --ContributedTo--> training job``
    - ``training job --ContributedTo--> MLflow Experiment action``

    Args:
        mlflow_run_id: MLflow run id of the training run
        training_job_name: Name of the SageMaker Training job that produced the run
        data_version: Data version, e.g. ``v2.0``
        data_git_commit_id: DVC repo commit of that data version
        dvc_remote_uri: S3 URI of the DVC remote
        dvc_repo_url: DVC repo URL
        pipeline_run_id: Optional ``PIPELINE_RUN_ID``
        processing_job_name: Optional name of the Processing job that built the dataset.
            If omitted and the training job ran as a pipeline step, it is discovered from
            the pipeline execution.
        dataset_properties: Optional extra artifact properties, e.g. ``patient_count``
        action_timeout: Seconds to wait for the auto-synced MLflow Experiment action
        sm_client: Optional boto3 SageMaker client
        verbose: Print what was created

    Returns:
        dict with ``dataset_artifact_arn``, ``mlflow_action_arn`` (may be None),
        ``training_job_trial_component`` and ``edges`` (list of created/existing edges)
    """
    sm = _sm_client(sm_client)

    dataset_arn = get_or_create_dvc_dataset_artifact(
        dvc_remote_uri=dvc_remote_uri,
        data_git_commit_id=data_git_commit_id,
        dvc_repo_url=dvc_repo_url,
        data_version=data_version,
        pipeline_run_id=pipeline_run_id,
        extra_properties=dataset_properties,
        sm_client=sm,
    )

    train_tc = trial_component_arn(training_job_arn(training_job_name, sm), sm)
    if not train_tc:
        raise RuntimeError(f"No lineage node for training job {training_job_name}")

    edges = []

    # For a pipeline-produced run the Processing job is discoverable from the execution
    if not processing_job_name:
        processing_job_name = find_processing_job_of_training_job(training_job_name, sm)

    if processing_job_name:
        preprocess_tc = trial_component_arn(
            processing_job_arn(processing_job_name, sm), sm
        )
        if preprocess_tc:
            created = associate(preprocess_tc, dataset_arn, "Produced", sm)
            edges.append(("processing job -> dataset", "created" if created else "exists"))

    created = associate(dataset_arn, train_tc, "ContributedTo", sm)
    edges.append(("dataset -> training job", "created" if created else "exists"))

    action_arn = find_mlflow_experiment_action(mlflow_run_id, action_timeout, sm)
    if action_arn:
        created = associate(train_tc, action_arn, "ContributedTo", sm)
        edges.append(("training job -> MLflow experiment", "created" if created else "exists"))

    if verbose:
        print(f"DataSet artifact: {dataset_arn}")
        print(f"  {data_version} / DVC commit {data_git_commit_id[:12]}")
        if action_arn:
            print(f"MLflow Experiment action: {action_arn.split('/')[-1]}")
        else:
            print("MLflow Experiment action: not found "
                  "(model version not registered yet, or the sync skipped lineage)")
        for edge, state in edges:
            print(f"  [{state}] {edge}")

    return {
        "dataset_artifact_arn": dataset_arn,
        "mlflow_action_arn": action_arn,
        "training_job_trial_component": train_tc,
        "edges": edges,
    }
