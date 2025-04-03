import cv2
import os
import numpy as np
from ultralytics import YOLO
from openvino.runtime import Core


# -----------------------------
# Load Models
# -----------------------------
# 1. Detection model
detection_model = YOLO("best.pt", task="detect")

# 2. DeepMAR attribute model (converted to OpenVINO)
ie = Core()
# Replace with your DeepMAR model’s XML and BIN files
attr_model_xml = "onnx_models/openvino_model/model.xml"
attr_model_bin = "onnx_models/openvino_model/model.bin"
attr_model = ie.read_model(model=attr_model_xml, weights=attr_model_bin)
compiled_attr_model = ie.compile_model(model=attr_model, device_name="CPU")
input_layer_attr = compiled_attr_model.inputs[0]
output_layer_attr = compiled_attr_model.outputs[0]

# List of attribute names from your training dataset.
att_list = [
    "accessoryHat",
    "hairLong",
    "hairShort",
    "upperBodyShortSleeve",
    "upperBodyBlack",
    "upperBodyBlue",
    "upperBodyBrown",
    "upperBodyGreen",
    "upperBodyGrey",
    "upperBodyOrange",
    "upperBodyPink",
    "upperBodyPurple",
    "upperBodyRed",
    "upperBodyWhite",
    "upperBodyYellow",
    "upperBodyLongSleeve",
    "lowerBodyShorts",
    "lowerBodyShortSkirt",
    "lowerBodyBlack",
    "lowerBodyBlue",
    "lowerBodyBrown",
    "lowerBodyGreen",
    "lowerBodyGrey",
    "lowerBodyOrange",
    "lowerBodyPink",
    "lowerBodyPurple",
    "lowerBodyRed",
    "lowerBodyWhite",
    "lowerBodyYellow",
    "lowerBodyLongSkirt",
    "footwearLeatherShoes",
    "footwearSandals",
    "footwearShoes",
    "footwearSneaker",
    "carryingBackpack",
    "carryingMessengerBag",
    "carryingLuggageCase",
    "carryingSuitcase",
    "personalLess30",
    "personalLess45",
    "personalLess60",
    "personalLarger60",
    "personalLess15",
    "personalMale",
    "personalFemale",
]

# -----------------------------
# Video Input Setup & Output Directory
# -----------------------------
# video_path = "/home/hungtrieu07/dev/video_data/1740730604383.mp4"  # or 0 for webcam
# video_path = "/home/hungtrieu07/dev/video_data/1740730890898.mp4"
# video_path = "/home/hungtrieu07/dev/video_data/1740730950413.mp4"
# video_path = "/home/hungtrieu07/dev/video_data/1740731094984.mp4"
video_path = "/home/hungtrieu07/dev/video_data/1740731190026.mp4"
# video_path = "/home/hungtrieu07/dev/video_data/test_video.avi"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# Create an output directory if saving frames/images
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_count = 0

# -----------------------------
# Preprocessing Settings for DeepMAR
# -----------------------------
# These are DeepMAR’s normalization parameters.
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# -----------------------------
# Main Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video processing complete.")
        break

    frame_count += 1

    # Run YOLO detection on the frame
    detection_results = detection_model(frame)
    # Base annotated frame for display
    # annotated_frame = detection_results[0].plot()

    # Process each detection result
    for result in detection_results:
        for box in result.boxes:
            # Get bounding box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Check if the detection is a person (adjust if necessary)
            if int(box.cls) == 3:
                # Crop the detected person from the original frame
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue  # Skip empty crops

                # Resize the cropped image to 224x224 (width x height)
                person_crop_resized = cv2.resize(person_crop, (224, 224))

                # Preprocess for DeepMAR:
                # Convert from BGR (OpenCV) to RGB
                person_rgb = cv2.cvtColor(person_crop_resized, cv2.COLOR_BGR2RGB)
                # Convert to float and scale to [0, 1]
                person_float = person_rgb.astype(np.float32) / 255.0
                # Normalize using DeepMAR’s mean and std
                person_norm = (person_float - mean) / std
                # Rearrange from HWC to CHW format
                input_tensor = np.transpose(person_norm, (2, 0, 1))
                # Add batch dimension: (1, C, H, W)
                input_tensor = np.expand_dims(input_tensor, axis=0)

                # Run attribute inference with OpenVINO
                attr_probs = compiled_attr_model([input_tensor])[output_layer_attr][0]

                # print("Attribute scores:", attr_probs)
                # Convert raw logits to probabilities using the sigmoid function
                attr_probs = 1 / (1 + np.exp(-attr_probs))

                # print("Attribute scores:", attr_probs)

                # Build a list of text strings for attributes with score >= 0.7
                attr_texts = []
                for idx, score in enumerate(attr_probs):
                    if score >= 0.7:  # Adjust threshold if needed
                        attr_texts.append(f"{att_list[idx]}: {score:.3f}")

                # Print to terminal (this should already show newlines correctly)
                text_to_draw = "\n".join(attr_texts)
                print(text_to_draw)

                # Draw attribute text on the frame, one line per attribute
                y_offset = 15  # Space between lines
                font_scale = 0.5
                font_thickness = 2
                for i, attr_text in enumerate(attr_texts):
                    y_pos = y1 - 10 - (len(attr_texts) - i - 1) * y_offset  # Adjust y-position for each line
                    cv2.putText(frame, attr_text, (x1, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

    # Resize annotated frame back to original dimensions (if needed)
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Display the frame
    cv2.imshow("Inference", frame)
    if cv2.waitKey(fps) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# import numpy as np
# from openvino.runtime import Core

# ie = Core()
# attr_model_xml = "ov_exported_model/person-attr/model.xml"
# attr_model_bin = "ov_exported_model/person-attr/model.bin"
# attr_model = ie.read_model(model=attr_model_xml, weights=attr_model_bin)
# compiled_attr_model = ie.compile_model(model=attr_model, device_name="CPU")
# output_layer_attr = compiled_attr_model.outputs[0]

# # Dummy input (1, 3, 224, 224)
# input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
# output = compiled_attr_model([input_tensor])[output_layer_attr]
# print("Output shape:", output.shape)
# print("Raw output sample:", output[0][:10])  # First 10 values