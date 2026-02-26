# Model License

## Trained Model License

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

### License Terms

Models trained using this repository are licensed under the MIT-0 License, consistent with the source code license.

Permission is hereby granted, free of charge, to any person obtaining a copy of the trained model weights and associated documentation files (the "Model"), to deal in the Model without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Model, and to permit persons to whom the Model is furnished to do so.

THE MODEL IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE MODEL OR THE USE OR OTHER DEALINGS IN THE MODEL.

### Base Model Attribution

This project fine-tunes MobileNetV3-Small:

- **Base Model**: MobileNetV3-Small
- **Pretrained Weights**: ImageNet-1K (IMAGENET1K_V1)
- **Implementation Source**: PyTorch torchvision library (BSD 3-Clause License)
- **Original Architecture**: Google Research (Apache License 2.0)

Note: The MobileNet architecture was originally developed by Google and released under Apache 2.0. The torchvision implementation and pretrained weights used in this project are provided by PyTorch under BSD 3-Clause.

### Training Data

Models may be trained on either dataset depending on which notebook is used:

**Foundational notebook:**
- **Dataset**: CIFAR-10
- **Source**: https://www.cs.toronto.edu/~kriz/cifar.html
- **Citation**: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009
- **License**: Available for research purposes

**Healthcare compliance notebook:**
- **Dataset**: Montgomery County Chest X-Ray Dataset
- **Source**: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#702702-tuberculosis-chest-x-ray-image-data-sets
- **Citation**: Jaeger S, Candemir S, Antani S, Wang YX, Lu PX, Thoma G. Two public chest X-ray datasets for computer-aided screening of pulmonary diseases. Quant Imaging Med Surg. 2014;4(6):475-477
- **License**: Public domain (US Government work)

### Model Card

When deploying models trained with this repository, consider documenting:

| Field | Foundational | Healthcare |
|-------|-------------|------------|
| Model Name | CIFAR10-MobileNetV3 | CXR-MobileNetV3 |
| Model Version | Tracked via MLflow | Tracked via MLflow |
| Model Type | Image Classification | Image Classification |
| Architecture | MobileNetV3-Small (fine-tuned) | MobileNetV3-Small (fine-tuned) |
| Training Data | CIFAR-10 (versioned via DVC) | Montgomery CXR (versioned via DVC) |
| Input Format | RGB images, 96x96 pixels | RGB images, 128x128 pixels |
| Output Format | 10-class probability distribution | 2-class probability distribution (normal, tuberculosis) |
| Intended Use | Educational/demonstration purposes | Educational/demonstration purposes |
| Limitations | Trained on small images; may not generalize to high-resolution inputs | Demo only; not validated for clinical use |

### Data Lineage

Each trained model is linked to its exact training data via:
- DVC commit hash (stored in MLflow parameters)
- Git tag for data version
- Full reproducibility through DVC + MLflow integration
