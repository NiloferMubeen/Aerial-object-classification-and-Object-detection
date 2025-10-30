# Aerial-Object Classification & Object Detection  
> A dual-task framework for classifying and localising objects in aerial imagery.  

## 📌 Project Summary  
This project implements two key tasks on aerial (drone/UAV/satellite-style) imagery:  
- **Image Classification** – determining an overall class label for an input aerial image.  
- **Object Detection** – locating and identifying individual objects in the image via bounding boxes.  
The codebase is designed to support experimentation with different backbones, augmentations, and datasets tailored to aerial-view challenges.

## 📂 Repository Structure  
/
├── data/ # Dataset files (raw images, annotations)
├── classification/ # Scripts & notebooks for the classification task
├── detection/ # Scripts & notebooks for the detection task
├── models/ # Saved model checkpoints
├── utils/ # Utility modules (data loading, transforms, metrics)
├── requirements.txt # Python dependencies
└── README.md # This file


🔧 Key Features

Dual-task support: classification + detection.

Modular scripts and pipelines for flexibility (swap backbones, augmentations, datasets).

Visualisation utilities (confusion matrices, bounding-box overlay) for qualitative assessment.

Focus on aerial imagery: designed to handle the unique challenges of overhead views.




https://github.com/user-attachments/assets/0851843b-4dad-4efa-b930-76d8c5b5ce42

