# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Audit query functions for healthcare compliance.

Query MLflow to answer compliance questions like:
- "Which models were trained using patient X's scans?"
- "Verify that patient X was excluded from all models after date Y"
"""

import csv
from datetime import datetime

import mlflow
from mlflow import MlflowClient


def find_models_with_patient(patient_id, experiment_name=None):
    """
    Find all training runs that included a patient's data.

    Downloads each run's manifest artifact from MLflow and checks
    if the patient is listed.

    Args:
        patient_id: Patient ID to search for (e.g., 'PAT-00023')
        experiment_name: Optional MLflow experiment name to filter

    Returns:
        List of dicts with: run_id, run_name, start_time, data_version,
        data_git_commit_id, patient_found (bool)
    """
    client = MlflowClient()

    filter_string = 'tags.stage = "training"'
    if experiment_name:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Experiment '{experiment_name}' not found")
            return []
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=["start_time DESC"]
        )
    else:
        runs = client.search_runs(
            filter_string=filter_string,
            order_by=["start_time DESC"]
        )

    results = []
    for run in runs:
        try:
            manifest_path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path="manifest.csv"
            )
        except Exception:
            # Run doesn't have a manifest artifact
            continue

        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            patient_found = any(row['patient_id'] == patient_id for row in reader)

        if patient_found:
            results.append({
                'run_id': run.info.run_id,
                'run_name': run.info.run_name,
                'start_time': run.info.start_time,
                'data_version': run.data.params.get('data_version', 'unknown'),
                'data_git_commit_id': run.data.params.get('data_git_commit_id', 'unknown'),
            })

    print(f"Patient {patient_id} found in {len(results)} training run(s)")
    for r in results:
        print(f"  - {r['run_name']} (data version: {r['data_version']})")

    return results


def verify_patient_excluded_after_date(patient_id, after_date, experiment_name=None):
    """
    Verify that a patient is excluded from all models trained after a date.

    Args:
        patient_id: Patient ID to check (e.g., 'PAT-00023')
        after_date: Date string in 'YYYY-MM-DD' format
        experiment_name: Optional MLflow experiment name to filter

    Returns:
        dict with: status ('PASSED'/'FAILED'), checked_runs, violations
    """
    client = MlflowClient()
    cutoff_ms = int(datetime.strptime(after_date, "%Y-%m-%d").timestamp() * 1000)

    filter_string = 'tags.stage = "training"'
    kwargs = {"filter_string": filter_string, "order_by": ["start_time DESC"]}
    if experiment_name:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Experiment '{experiment_name}' not found")
            return {"status": "ERROR", "checked_runs": 0, "violations": []}
        kwargs["experiment_ids"] = [experiment.experiment_id]

    runs = client.search_runs(**kwargs)
    runs_after = [r for r in runs if r.info.start_time >= cutoff_ms]

    violations = []
    for run in runs_after:
        try:
            manifest_path = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id, artifact_path="manifest.csv"
            )
        except Exception:
            continue

        with open(manifest_path, 'r') as f:
            if any(row['patient_id'] == patient_id for row in csv.DictReader(f)):
                violations.append({
                    'run_id': run.info.run_id,
                    'run_name': run.info.run_name,
                    'data_version': run.data.params.get('data_version', 'unknown'),
                })

    status = "PASSED" if not violations else "FAILED"
    if violations:
        for v in violations:
            print(f"  VIOLATION: {v['run_name']} (data version: {v['data_version']})")
    else:
        print(f"  Patient not found in any post-{after_date} models")

    return {"status": status, "checked_runs": len(runs_after), "violations": violations}


def get_patients_in_model(run_id):
    """
    List all patient IDs in a specific model's training data.

    Args:
        run_id: MLflow run ID

    Returns:
        List of unique patient IDs
    """
    try:
        manifest_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="manifest.csv"
        )
    except Exception:
        print(f"No manifest artifact found for run {run_id}")
        return []

    patient_ids = set()
    with open(manifest_path, 'r') as f:
        for row in csv.DictReader(f):
            patient_ids.add(row['patient_id'])

    patient_list = sorted(patient_ids)
    print(f"Run {run_id}: {len(patient_list)} patients in training data")
    return patient_list