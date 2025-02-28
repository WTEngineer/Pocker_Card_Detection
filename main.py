from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("yolov8s_playing_cards.pt")

results = model("card.jpeg")
# results = model("card2.png")

result = results[0]

boxes = result.boxes.xywh 
labels = result.names 
confidences = result.boxes.conf  

img = cv2.imread("card.jpeg")
# img = cv2.imread("card2.png")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
ax = plt.gca()

for i in range(len(boxes)):
    box = boxes[i]
    confidence = confidences[i]
    label = labels[int(result.boxes.cls[i])]
    
    rect = plt.Rectangle(
        (box[0] - box[2]/2, box[1] - box[3]/2),
        box[2], box[3], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    ax.text(
        box[0] - box[2]/2, box[1] - box[3]/2, 
        f"{label} {confidence:.2f}", color='r', 
        fontsize=12, verticalalignment='bottom', horizontalalignment='left'
    )

plt.show()
