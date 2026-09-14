# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""SageMaker inference handlers for the MobileNetV3 image classifier.

Uploaded by the register step to <logged model artifact location>/code/ so the
S3Prefix ModelDataSource downloads it to /opt/ml/model/code/inference.py on
the endpoint, next to the MLflow model files (MLmodel, data/model.pth).
"""

import io
import json
import os

import torch
from PIL import Image
from torchvision import transforms

# Must match the validation transform in source_dir/train.py
TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def model_fn(model_dir):
    # Load the pickled nn.Module directly with torch. This keeps mlflow out of
    # the serving container and sidesteps mlflow.pytorch.load_model's symlink
    # path check, which rejects /opt/ml/model on SageMaker endpoints.
    model_path = os.path.join(model_dir, "data", "model.pth")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    content_type = (request_content_type or "").split(";")[0].strip()
    if content_type in ("image/jpeg", "image/png", "application/x-image"):
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        return TRANSFORM(image).unsqueeze(0)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    with torch.no_grad():
        logits = model(input_data)
        return torch.softmax(logits, dim=1)


def output_fn(prediction, accept):
    return json.dumps(prediction.tolist()), "application/json"
