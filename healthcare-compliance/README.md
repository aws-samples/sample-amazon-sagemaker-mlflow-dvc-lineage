# Healthcare: Record-Level Lineage and Patient Opt-Out

This variant extends the [foundational dataset-level lineage pattern](../foundational/) with **record-level lineage**. The foundational pattern tells you *which dataset version* trained a model, but not which individual records are inside it — you'd need to reconstruct the dataset to find out. This variant adds a **manifest** (a structured index of every record in each dataset version) logged as an MLflow artifact on every training run, making individual records queryable without pulling the full dataset. Combined with a **consent registry** that controls which records can enter the ML pipeline, you can answer audit questions like "which models used record X?" and handle record exclusion requests with full audit trails.

> **Disclaimer:** This demo illustrates technical patterns for data lineage and record-level traceability. It does not constitute a compliant system on its own. Achieving regulatory compliance (HIPAA, FDA, GDPR, etc.) requires additional infrastructure controls, legal review, and organizational processes. See the [Production Considerations](../README.md#production-considerations-for-regulated-environments) section for guidance on what to layer on top.

This demo uses the [Montgomery County CXR Dataset](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#702702-tuberculosis-chest-x-ray-image-data-sets) — 138 chest X-rays across 2 classes (normal, tuberculosis) from the National Library of Medicine, with randomly assigned patient IDs. The patterns (consent manifest, opt-out workflow, audit queries) apply to any regulated domain: replace "patient" with customer, transaction, user, etc.

## What You'll Learn

- Use a **patient consent manifest** to control which records enter the ML pipeline
- Handle **patient opt-out requests** by updating the manifest and reprocessing
- Include a **record-level manifest** linking individual data records to dataset versions
- Run **audit queries**: "Which models were trained on patient X's data?"
- **Verify exclusion**: confirm a patient's data is excluded from all models after their opt-out date
- Run the whole flow as one **SageMaker AI Pipeline** with the consent registry as a parameter, every stage grouped under one MLflow parent run

## Quick Start

1. Open `healthcare_example_record_level_lineage.ipynb` in Amazon SageMaker Studio or a SageMaker AI notebook instance
2. Follow the notebook cells sequentially — the notebook will:
   - Download the Montgomery CXR dataset from NLM, upload raw images to S3, and generate a patient manifest
   - Process the dataset with all consented patients (v1.0) and train a model
   - Simulate a patient opting out
   - Reprocess and retrain without that patient's data (v2.0)
   - Run audit queries to verify the patient's data was excluded
   - Pick the best consent-compliant model (only runs on the latest data version are eligible), register it and deploy it to a SageMaker AI endpoint
   - Run the same flow as a SageMaker AI Pipeline (Part 9)

## Files in This Directory

| File | Description |
|------|-------------|
| `healthcare_example_record_level_lineage.ipynb` | Main notebook |
| `setup_cxr_dataset.py` | Dataset setup script (download, S3 upload, manifest generation) |
| `utils/audit_queries.py` | MLflow audit query functions |
| `utils/manifest_utils.py` | Registry and manifest I/O utilities |
| `utils/lineage_utils.py` | Adds the DVC dataset node and the training job → MLflow Experiment edge that SageMaker's automatic lineage cannot infer |
| `pipeline_steps/evaluate.py`, `pipeline_steps/register.py` | `@step` functions of the Part 9 pipeline |
| `pipeline_steps/inference.py` | Inference handlers uploaded to the logged model by the register step |
| `img/` | MLflow and SageMaker Studio screenshots used in the notebook |

## Note on Deployed Models

This demo shows how to exclude a record and retrain, but does not automate the invalidation of previously deployed models. In production, after a record exclusion you would also use the audit queries to identify any deployed endpoints serving models trained on that record, and flag them for retraining or retirement.

## Part 9: As a SageMaker AI Pipeline

Part 9 of the notebook puts the same stages together as one parameterized SageMaker AI Pipeline, reusing the DVC repository, raw data, consent registry, MLflow App and experiment from the earlier parts:

| Step | Type | What it does |
|---|---|---|
| `preprocess-dvc` | `ProcessingStep` | Same `FrameworkProcessor` + `preprocessing_healthcare.py`: applies the registry at `RegistryS3Uri`, patient-level split, writes `manifest.csv`, `dvc add/push`, git tag = `PIPELINE_RUN_ID` |
| `train-mlflow` | `TrainingStep` | Same `ModelTrainer` + `train.py`: `dvc pull` at that tag, train, log run, model and manifest to MLflow (tagged with the training job) |
| `evaluate-mlflow` | `@step` | Reads `final_val_accuracy` and `patient_count` from the MLflow run of this execution ([`pipeline_steps/evaluate.py`](./pipeline_steps/evaluate.py)) |
| `check-val-accuracy` | `ConditionStep` | `val_accuracy >= MinValAccuracy` → register, else `FailStep` |
| `register-model` | `@step` | Logs `code/inference.py` and the inference specification on the logged model, `mlflow.register_model()`, approves the auto-synced Model Package with `patient_count` in its metadata ([`pipeline_steps/register.py`](./pipeline_steps/register.py)) |

The consent registry is a pipeline parameter: when a patient opts out, update the registry, upload it under a new prefix, and start the pipeline with that `RegistryS3Uri` and a new `DataVersion`. `PIPELINE_RUN_ID = <DataVersion>-<PipelineExecutionId>` is the DVC git tag, the MLflow run name suffix and `pipeline_run_id` tag, and is written to the Model Package metadata. In MLflow, all stages of one `PIPELINE_RUN_ID` are nested under a parent run of that name (`source_dir/mlflow_utils.py`). The Part 7 audit queries work unchanged on pipeline-produced runs.

## Cleanup

The notebook includes cleanup cells at the end. To fully remove all resources:

1. **Delete the SageMaker AI endpoints** — done by the cleanup cells at the end of Part 8 and Part 9
2. **Delete the pipeline** (optional, removes its execution history):
   ```python
   pipeline.delete()
   ```
3. **Delete the Model Package group** created by the MLflow sync (optional, name `CXR-MobileNetV3-<suffix>`):
   ```bash
   aws sagemaker list-model-package-groups --name-contains CXR-MobileNetV3
   ```
4. **Delete the MLflow App** (optional):
   ```python
   sm_client.delete_mlflow_app(Arn=mlflow_app_arn)
   ```
5. **Delete the AWS CodeCommit repository**:
   ```bash
   aws codecommit delete-repository --repository-name cxr-dvc-demo
   ```
6. **Delete S3 data** (DVC cache, raw images, consent registries, training output, pipeline step artifacts; MLflow artifacts live under `workspaces/` in the same bucket):
   ```bash
   aws s3 rm s3://<your-bucket>/DEMO-cxr-dvc --recursive
   aws s3 rm s3://<your-bucket>/cxr-train --recursive
   aws s3 rm s3://<your-bucket>/cxr-dvc-mlflow-pipeline --recursive
   ```

## Prerequisites

See the [root README](../README.md) for full prerequisites and architecture details.
