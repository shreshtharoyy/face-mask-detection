# Face Mask Detection using Computer Vision

This project focuses on building a computer vision–based model to detect
whether a person is wearing a face mask or not. The model was trained
using a YOLOv8 architecture with transfer learning in a GPU-enabled
Google Colab environment.

The goal of this project is to understand the complete workflow of a
deep learning–based computer vision pipeline, including dataset
preparation, model training, and evaluation.

---

## Dataset
The dataset was sourced from Roboflow and prepared in YOLO format.  
It was forked and augmented to improve data diversity before training.

- Source: Roboflow
- Format: YOLOv8
- Classes:
  - with_mask
  - without_mask
  - incorrect_mask

After augmentation, the dataset was downloaded and uploaded to Google
Drive. The training pipeline accessed the dataset directly from Google
Drive within the Google Colab environment.

Dataset link: https://universe.roboflow.com/your-workspace/face-mask-jk4nr-nvjph

---

## Model Training
- Architecture: YOLOv8 (pretrained)
- Framework: Ultralytics YOLO
- Training approach: Transfer learning
- Epochs: 50
- Image size: 640
- Batch size: 16
- Hardware: NVIDIA T4 GPU (Google Colab)

The model was initialized using pretrained YOLOv8 weights and fine-tuned
on the face mask dataset. Training and validation were handled using the
built-in Ultralytics training pipeline.

---

## Model Evaluation
The trained model was evaluated on a validation dataset containing
806 images and 1255 labeled instances. Standard object detection metrics
were used for evaluation.

- Precision: 0.83
- Recall: 0.74
- mAP@50: 0.81
- mAP@50–95: 0.56

Class-wise evaluation showed strong performance for the `without_mask`
class, while relatively lower recall was observed for the
`incorrect_mask` class due to fewer training samples. Overall, the model
demonstrates good generalization with no significant overfitting or
underfitting observed.

Inference speed averaged approximately 8 ms per image, reflecting the
efficiency of the YOLO architecture.

---

## Files in this Repository
- `face_mask_detection.ipynb`  
  Contains the complete workflow including dataset loading, model
  training, validation, and evaluation.

---

## Tools & Technologies
- Python
- Ultralytics YOLOv8
- Roboflow
- OpenCV
- Google Colab (NVIDIA T4 GPU)
