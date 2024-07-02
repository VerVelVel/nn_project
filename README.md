# Sports and Blood Cell Image Classification Project

## Overview

This project offers two main functionalities:

1. **Sports Image Classification**
2. **Blood Cell Image Classification**

### Sports Image Classification

- Users can upload their sports images and receive a classification.
- The model is trained using the 100 Sports Image Classification dataset.
- Various ResNet architectures (ResNet18/50/101/152) are utilized.
- Pretrained models from `torchvision.models` are employed, with the last layer replaced and all parameters frozen except the classification layer.

### Blood Cell Image Classification

- Users can classify images of blood cells.
- The training data is sourced from the blood cell images dataset.
- The model can be developed from scratch or similar to the sports image classification task.
- The dataset for this task can be downloaded using:

    ```sh
    kaggle datasets download -d paultimothymooney/blood-cells
    ```

    The relevant files are located in `dataset2-master/dataset2-master/images`.

## Deployment

- The service is deployed on Streamlit servers.

## Additional Features

- **Image Upload via URL**: Users can upload images by pasting a URL, with the image displayed alongside the classification result.
- **Batch Image Upload**: Users can upload multiple images at once.
- **Model Response Time Visualization**: Display the time taken by the model to classify the image.

### Sports Image Classification Model Info

- **Model**: Pretrained ResNet152 with the last layer replaced.
- **Classes**: 100
- **Training Dataset Size**: 13,492 images
- **Training Time**: 15 epochs = 40 minutes (batch_size=32)
- **F1 Score**: 0.695 (train), 0.840 (valid)

### Blood Cell Classification Model Info

- **Model**: ResNet18, trained from scratch.
- **Classes**: 4
- **Training Dataset Size**: 9,957 images
- **Training Time**: 15 epochs = 40 minutes
- **F1 Score**: 0.915 (train), 0.849 (valid)
- 
## Access the Project

You can access the project [here](https://nnproject-6x4qxdtt9hcxuk3thchrn9.streamlit.app/).

