from ultralytics import YOLO
import cv2
import os

# Load the pretrained YOLOv8 model
model = YOLO("yolov8m_tuned.pt")

# Open the video capture (replace 'input_video.mp4' with your video file path)
cap = cv2.VideoCapture("./1.mp4")

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object to save the processed video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (frame_width, frame_height))

# Create directory to save frames if it doesn't exist
frame_dir = './frames'
if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)

frame_count = 0  # Initialize frame counter

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break  # End of video
    
    # Run inference on the current frame
    results = model(frame)

    # Access the first result (since we are processing one frame at a time)
    result = results[0]

    # Get the predicted boxes, labels, and confidences from the result
    boxes = result.boxes.xywh  # Bounding boxes in xywh format (x, y, w, h)
    labels = result.names  # Class names (e.g., '10S', 'JS', etc.)
    confidences = result.boxes.conf  # Confidence scores for each prediction

    # Loop through each prediction and draw bounding boxes on the frame
    for i in range(len(boxes)):
        box = boxes[i]
        confidence = confidences[i]
        label = labels[int(result.boxes.cls[i])]  # Get the class label based on the index
        
        # Convert xywh to xy coordinates (top-left and bottom-right corners)
        x1 = int(box[0] - box[2]/2)
        y1 = int(box[1] - box[3]/2)
        x2 = int(box[0] + box[2]/2)
        y2 = int(box[1] + box[3]/2)
        
        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
        
        # Add label and confidence on the frame
        text = f"{label} {confidence:.2f}"
        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save the current frame as an image
    frame_filename = os.path.join(frame_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)  # Save frame as JPG

    # Write the processed frame to the output video
    out.write(frame)

    # Display the current frame with bounding boxes
    cv2.imshow("Detected Video", frame)

    # Press 'q' to exit the video preview
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1  # Increment the frame counter

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
