# Foundational: Dataset-Level Lineage with DVC and MLflow

This variant demonstrates **dataset-level lineage** — tracing every trained model back to the exact version of the dataset it was trained on. It simulates new training data being incrementally added over time: train with 5% of CIFAR-10, then expand to 10% and retrain, with DVC versioning each dataset and MLflow linking every model to its data version.

With dataset-level lineage, every model links to a DVC commit hash that points to the exact processed dataset in Amazon S3. You can reproduce any model's training data with `dvc pull`. However, you don't have structured metadata about which individual records are inside each dataset — to find out, you'd need to reconstruct the dataset and inspect it. If you need to query which specific records trained a model (e.g., for opt-out requests or audits), see the [healthcare-compliance](../healthcare-compliance/) variant, which adds a record-level manifest.

## What You'll Learn

- Version processed datasets with DVC and store them in S3
- Track experiments and link models to specific data versions with MLflow
- Compare model performance across different data versions
- Deploy a model from MLflow registry to an Amazon SageMaker AI endpoint
- Run the whole flow as one SageMaker AI Pipeline, with every stage grouped under one MLflow parent run

## Quick Start

1. Open `foundational_dataset_level_lineage.ipynb` in Amazon SageMaker Studio or a SageMaker AI notebook instance
2. Follow the notebook cells sequentially
3. The notebook will:
   - Configure a DVC repository backed by S3
   - Set up a managed MLflow tracking server
   - Run two experiments with different data fractions (5% and 10%)
   - Compare results in MLflow
   - Deploy the best model to a SageMaker AI endpoint
   - Run the same flow as a SageMaker AI Pipeline (Part 4)

## Part 4: As a SageMaker AI Pipeline

Part 4 of the notebook puts the same stages together as one parameterized SageMaker AI Pipeline, reusing the DVC repository, MLflow App and experiment from Parts 1 and 2:

| Step | Type | What it does |
|---|---|---|
| `preprocess-dvc` | `ProcessingStep` | Same `FrameworkProcessor` + `preprocessing_foundational.py`: samples `DataFraction` of CIFAR-10, `dvc add/push`, git tag = `PIPELINE_RUN_ID` |
| `train-mlflow` | `TrainingStep` | Same `ModelTrainer` + `train.py`: `dvc pull` at that tag, train, log run + model to MLflow (tagged with the training job) |
| `evaluate-mlflow` | `@step` | Reads `final_val_accuracy` from the MLflow run of this execution ([`pipeline_steps/evaluate.py`](./pipeline_steps/evaluate.py)) |
| `check-val-accuracy` | `ConditionStep` | `val_accuracy >= MinValAccuracy` → register, else `FailStep` |
| `register-model` | `@step` | Logs `code/inference.py` and the inference specification on the logged model, `mlflow.register_model()`, approves the auto-synced Model Package ([`pipeline_steps/register.py`](./pipeline_steps/register.py)) |

`PIPELINE_RUN_ID = <DataVersion>-<PipelineExecutionId>` is the DVC git tag, the MLflow run name suffix and `pipeline_run_id` tag, and is written to the Model Package metadata, so every artifact of an execution can be traced back to the exact data. In MLflow, all stages of one `PIPELINE_RUN_ID` are nested under a parent run of that name (`source_dir/mlflow_utils.py`), for manual runs and pipeline executions alike. Deployment stays outside the pipeline; the notebook shows how to deploy the resulting Model Package.

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