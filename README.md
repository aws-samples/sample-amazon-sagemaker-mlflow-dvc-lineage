# End-to-End ML Traceability with DVC and MLflow on Amazon SageMaker AI

This sample demonstrates how to integrate [DVC](https://dvc.org/) (Data Version Control) with [MLflow on Amazon SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html) for data lineage and model tracking. 

Fully managed MLflow on Amazon SageMaker AI makes it easier to track experiments and monitor performance of models and AI applications using a single tool. The workflow uses SageMaker AI Processing and Training jobs to version datasets with DVC and track models with MLflow — enabling full traceability from production models back to the exact data they were trained on.

## Why Data Lineage Matters

In regulated industries and enterprise ML, you need to answer questions like:
- *"Which data was used to train the model currently in production?"*
- *"Was this customer's data included in any of our models?"*
- *"Can we reproduce the exact model we deployed 6 months ago?"*

### How DVC + MLflow Solves This

- Every model in MLflow links to a specific DVC commit hash
- That commit hash points to the exact dataset version in Amazon S3
- You can trace any model back to its training data, identify affected models, and retrain with corrected datasets

This pattern applies to healthcare, autonomous vehicles, e-commerce, content moderation, and any ML team debugging model degradation across data versions.


## Getting Started

This repo includes two notebooks that build on the same architecture. Pick the one that fits your use case:

| Notebook | Description | Key Question It Answers |
|----------|-------------|------------------------|
| **[Foundational](./foundational/foundational_dataset_level_lineage.ipynb)** | **Dataset-level lineage.** Every model links to the exact dataset version (via DVC commit hash) it was trained on. You can reproduce any model's training data with `dvc pull`. However, you don't have structured metadata about which individual records are inside each dataset version — to find out, you'd need to reconstruct the dataset and inspect its contents. | *"Which dataset version trained this model?"* |
| **[Healthcare Compliance](./healthcare-compliance/healthcare_example_record_level_lineage.ipynb)** | **Record-level lineage.** Builds on the foundational pattern by adding a **manifest** — a structured index of every individual record in each dataset version — logged as an MLflow artifact on every training run. This makes individual records queryable without reconstructing the dataset. Combined with a **consent registry** that controls which records enter the pipeline, you can answer audit questions and handle record exclusion requests. | *"Which specific records trained this model, and can I exclude one?"* |

The healthcare notebook uses a CSV as the registry for simplicity, but in production this would be a central database (e.g., a consent management platform or DynamoDB table) that the processing job queries directly.

Both notebooks share the same training script (`source_dir/train.py`) and follow the same architecture, with separate preprocessing scripts for each use case.

### Prerequisites

- An AWS Account
- Python 3.11 (tested with 3.11.14)
- An IAM user/role with permissions for:
  - Amazon SageMaker AI (Processing, Training, MLflow App, Endpoints)
  - Amazon S3
  - AWS CodeCommit
  - IAM (to create execution roles if needed)

#### IAM Role Requirements

The notebook uses `sagemaker.core.helper.session_helper.get_execution_role()` to retrieve the IAM role for SageMaker AI jobs.

**If running locally or outside Amazon SageMaker Studio:** Your IAM role must have a trust relationship allowing `sagemaker.amazonaws.com` to assume it:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

## Architecture

![Architecture Diagram](img/architecture_diagram.png)

## Experiment Tracking

After running the notebook, you can compare experiments in the MLflow UI. To access the UI, see [Launch the MLflow UI using a presigned URL](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-launch-ui.html).

![MLflow Experiment Comparison](img/mlflow_experiment.png)

Click into any run to see training/validation loss curves, hyperparameters, and the DVC data version linking the exact dataset:

![MLflow Training Run Details](img/mlflow_training_run.png)

Models are registered in the MLflow Model Registry with version history and links to the training run that produced each model. With the MLflow App in `AutoModelRegistrationEnabled` mode, each registration also creates a deployable Model Package in the SageMaker Model Registry:

![MLflow Registered Model](img/mlflow_registered_model.png)

## What You'll Build

- **Model**: PyTorch MobileNetV3-Small fine-tuned for image classification (chest X-ray: normal vs tuberculosis; foundational: CIFAR-10)
- **Data Versioning**: Train with different data versions and trace each model back to its exact training data via DVC commit hashes
- **Experiment Comparison**: Compare model performance across data versions in MLflow

### Components

- **AWS CodeCommit** - Git repository for DVC metadata
- **Amazon S3** - Storage backend for DVC data files and MLflow artifacts
- **SageMaker AI MLflow App** - Managed MLflow tracking server
- **SageMaker AI Processing** - Data preprocessing with DVC integration
- **SageMaker AI Training** - Model training with MLflow logging (CPU instances)
- **SageMaker Model Registry** - Model Packages auto-synced from the MLflow Model Registry, carrying an inference specification logged with `sagemaker-mlflow`
- **SageMaker AI Endpoints** - Model deployment from the registered Model Package (`Model` → `EndpointConfig` → `Endpoint`)

## Project Structure

```
├── foundational/                           # Start here
│   ├── README.md
│   └── foundational_dataset_level_lineage.ipynb               # Foundational notebook (data fraction comparison)
├── healthcare-compliance/                  # Extended use case
│   ├── README.md
│   ├── healthcare_example_record_level_lineage.ipynb # Patient consent/opt-out workflow
│   ├── setup_cxr_dataset.py               # Dataset download, S3 upload, manifest generation
│   └── utils/                              # Audit query and manifest utilities
├── source_dir/                             # Shared SageMaker AI job code
│   ├── preprocessing_foundational.py       # Data-fraction sampling
│   ├── preprocessing_healthcare.py         # Patient consent registry processing
│   ├── train.py                            # MobileNetV3 training with MLflow logging
│   └── requirements.txt                    # Dependencies for SageMaker AI jobs
├── img/                                    # MLflow UI screenshots
└── requirements.txt                        # Local development dependencies
```

## Cleanup

Each notebook includes cleanup cells at the end to delete endpoints and other resources. See the individual READMEs for full cleanup instructions:

- [Foundational cleanup](./foundational/README.md#cleanup)
- [Healthcare compliance cleanup](./healthcare-compliance/README.md#cleanup)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Production Considerations for Regulated Environments

DVC and MLflow provide traceability and experiment tracking, but are not tamper-evident on their own. In a regulated deployment (e.g. FDA 21 CFR Part 11, HIPAA), you would layer on infrastructure-level controls such as:

- **S3 Object Lock** (compliance mode) on DVC remotes and MLflow artifact stores to prevent modification or deletion of versioned data and model artifacts
- **AWS CloudTrail** for independent, append-only logging of all access to storage and training infrastructure
- **IAM policies** enforcing least-privilege access to production buckets, MLflow tracking servers, and Git repositories

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.

### Trained Model License

Models trained using this repository are also licensed under MIT-0. See [MODEL_LICENSE.md](MODEL_LICENSE.md) for details on:
- License terms for trained model weights
- Base model attribution (MobileNetV3-Small / ImageNet)
- Training data attribution (Montgomery County CXR, public domain)
- Model card template

## Software Bill of Materials

A Software Bill of Materials (SBOM) is provided in [SBOM.json](SBOM.json) following the CycloneDX 1.5 specification. It documents:
- All Python dependencies and their licenses
- Pretrained model components
- AWS services used in the workflow
