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

## Quick Start

1. Open `healthcare_example_record_level_lineage.ipynb` in Amazon SageMaker Studio or a SageMaker AI notebook instance
2. Follow the notebook cells sequentially — the notebook will:
   - Download the Montgomery CXR dataset from NLM, upload raw images to S3, and generate a patient manifest
   - Process the dataset with all consented patients (v1.0) and train a model
   - Simulate a patient opting out
   - Reprocess and retrain without that patient's data (v2.0)
   - Run audit queries to verify the patient's data was excluded
   - Deploy the retrained model to a SageMaker AI endpoint

## Files in This Directory

| File | Description |
|------|-------------|
| `healthcare_example_record_level_lineage.ipynb` | Main notebook |
| `setup_cxr_dataset.py` | Dataset setup script (download, S3 upload, manifest generation) |
| `utils/audit_queries.py` | MLflow audit query functions |
| `utils/manifest_utils.py` | Registry and manifest I/O utilities |

## Note on Deployed Models

This demo shows how to exclude a record and retrain, but does not automate the invalidation of previously deployed models. In production, after a record exclusion you would also use the audit queries to identify any deployed endpoints serving models trained on that record, and flag them for retraining or retirement.

## Cleanup

The notebook includes cleanup cells at the end. To fully remove all resources:

1. **Delete the SageMaker AI endpoint** — run the cleanup cell in the notebook
2. **Delete the MLflow App** (optional):
   ```python
   sm_client.delete_mlflow_app(Arn=mlflow_app_arn)
   ```
3. **Delete the AWS CodeCommit repository**:
   ```bash
   aws codecommit delete-repository --repository-name cxr-dvc-demo
   ```
4. **Delete S3 data** (DVC cache, raw images, and MLflow artifacts):
   ```bash
   aws s3 rm s3://<your-bucket>/DEMO-cxr-dvc --recursive
   ```

## Prerequisites

See the [root README](../README.md) for full prerequisites and architecture details.
