# Foundational: Dataset-Level Lineage with DVC and MLflow

This variant demonstrates **dataset-level lineage** — tracing every trained model back to the exact version of the dataset it was trained on. It simulates new training data being incrementally added over time: train with 5% of CIFAR-10, then expand to 10% and retrain, with DVC versioning each dataset and MLflow linking every model to its data version.

With dataset-level lineage, every model links to a DVC commit hash that points to the exact processed dataset in Amazon S3. You can reproduce any model's training data with `dvc pull`. However, you don't have structured metadata about which individual records are inside each dataset — to find out, you'd need to reconstruct the dataset and inspect it. If you need to query which specific records trained a model (e.g., for opt-out requests or audits), see the [healthcare-compliance](../healthcare-compliance/) variant, which adds a record-level manifest.

## What You'll Learn

- Version processed datasets with DVC and store them in S3
- Track experiments and link models to specific data versions with MLflow
- Compare model performance across different data versions
- Deploy a model from MLflow registry to an Amazon SageMaker AI endpoint

## Quick Start

1. Open `foundational_dataset_level_lineage.ipynb` in Amazon SageMaker Studio or a SageMaker AI notebook instance
2. Follow the notebook cells sequentially
3. The notebook will:
   - Configure a DVC repository backed by S3
   - Set up a managed MLflow tracking server
   - Run two experiments with different data fractions (5% and 10%)
   - Compare results in MLflow
   - Deploy the best model to a SageMaker AI endpoint

## Cleanup

The notebook includes cleanup cells at the end. To fully remove all resources:

1. **Delete the SageMaker AI endpoint** — run the cleanup cell in the notebook
2. **Delete the MLflow App** (optional):
   ```python
   sm_client.delete_mlflow_app(Arn=mlflow_app_arn)
   ```
3. **Delete the AWS CodeCommit repository**:
   ```bash
   aws codecommit delete-repository --repository-name <your-dvc-repo-name>
   ```
4. **Delete S3 data** (DVC cache and MLflow artifacts):
   ```bash
   aws s3 rm s3://<your-bucket>/DEMO-cifar10-dvc --recursive
   ```

## Prerequisites

See the [root README](../README.md) for full prerequisites and architecture details.