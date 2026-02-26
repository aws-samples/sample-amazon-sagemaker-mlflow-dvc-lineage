# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Audit query example functions for healthcare compliance.

Query MLflow to answer compliance questions like:
- "Which models were trained using patient X's scans?"
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