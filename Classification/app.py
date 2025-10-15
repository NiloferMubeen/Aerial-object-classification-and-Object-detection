import cv2
import torch
from PIL import Image
import torch.nn as nn
import streamlit as st
from ultralytics import YOLO
from torchvision import transforms, models
from streamlit_option_menu import option_menu


import numpy as np
from PIL import Image


# --- Page settings ---
st.set_page_config(page_title="Bird vs Drone Classifier", layout="centered")

with st.sidebar:
    selected = option_menu("Main Menu", ["Classification", 'Object Detection'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)


if selected == "Classification":
                st.title("Bird vs Drone Image Classifier")
                st.write("Upload an image and the model to classify it.")



                # --- Upload model ---
                model_file = st.file_uploader("Upload your trained model (.pth file)", type=["pth"])
                if model_file:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    num_classes = 2

                    # Load pretrained ResNet model
                    model = models.resnet18(pretrained=False)
                    in_features = model.fc.in_features
                    model.fc = nn.Linear(in_features, num_classes)
                    model.load_state_dict(torch.load(model_file, map_location=device))
                    model.to(device)
                    model.eval()

                    st.success("Model loaded successfully!")

                # --- Upload image ---
                image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
                if image_file and model_file:
                    image = Image.open(image_file).convert("RGB")
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                    # --- Image transforms ---
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
                    ])
                    input_tensor = transform(image).unsqueeze(0).to(device)

                    # --- Prediction ---
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        _, pred = torch.max(outputs, 1)

                    classes = ["bird", "drone"]
                    st.write(f"Predicted Class: **{classes[pred.item()]}**")


if selected == "Object Detection":
      
    

            st.title("🕊️ Bird vs Drone Object Detection with YOLOv8")

            # -------------------------------
            # 2. Upload image
            # -------------------------------
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

            # -------------------------------
            # 3. Load YOLOv8 model
            # -------------------------------
            @st.cache_resource
            def load_model():
                model = YOLO(r"C:\Users\HAI\GUVI\Aerial_object\Obj_detection\runs\detect\train\weights\best.pt")  # path to your trained YOLOv8 model
                return model

            if uploaded_file:
                # Open image
                img = Image.open(uploaded_file).convert("RGB")
                img_array = np.array(img)

                st.image(img, caption="Uploaded Image", use_column_width=True)
                
                # Load model
                model = load_model()
                
                # -------------------------------
                # 4. Run inference
                # -------------------------------
                results = model.predict(img_array, imgsz=640)
                

                # Draw bounding boxes on the image
                annotated_frame = results[0].plot()  # YOLOv8 returns annotated image

                # Convert to PIL image for Streamlit display
                annotated_img = Image.fromarray(annotated_frame)
                
                st.image(annotated_img, caption="Detection Result")
                
                # -------------------------------
                # 5. Show detection details
                # -------------------------------
                st.subheader("Detection Details")
                for r in results:
                    boxes = r.boxes
                    if len(boxes) == 0:
                        st.write("No objects detected.")
                    else:
                        for i, box in enumerate(boxes):
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            name = r.names[cls_id]
                            st.write(f"{i+1}. {name} - Confidence: {conf:.2f}")

