import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import wandb

# === Step 1: WandB Setup ===

wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key is None:
    raise ValueError("WANDB_API_KEY environment variable not set.")


wandb.login(key=wandb_api_key)

# Initialize a WandB run with some configuration parameters
wandb.init(project='distance_classification_project', config={
    "n_clusters": 2,
    "scaleFactor": 1.05,
    "minNeighbors": 4,
    "minSize": [25, 25],
    "maxSize": [50, 50]
})

# === Step 2: Read and Process the Faculty Image ===
img = cv2.imread('Plaksha_Faculty.jpg')
if img is None:
    print("Error: Plaksha_Faculty.jpg not found")
    exit(1)

# Convert to grayscale for face detection
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the Haar cascade classifier 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces (adjust parameters as needed)
faces_rect = face_cascade.detectMultiScale(gray_img, 1.05, 4, minSize=(25, 25), maxSize=(50, 50))

# === Step 3: Annotate Detected Faces ===
text = "Face Detected"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (0, 0, 255)  # Red (in BGR)
font_thickness = 1

for (x, y, w, h) in faces_rect:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(img, text, (x, y - 10), font, font_scale, font_color, font_thickness)

# Instead of displaying, save the annotated image
annotated_image_filename = "faces_detected.jpg"
cv2.imwrite(annotated_image_filename, img)
print(f"Annotated image saved as {annotated_image_filename}")

# === Step 4: Extract Hue-Saturation Features & Perform Clustering ===
# Convert the image to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue_saturation = []
face_images = []  # To store individual face images

for (x, y, w, h) in faces_rect:
    face = img_hsv[y:y+h, x:x+w]
    hue = np.mean(face[:, :, 0])
    saturation = np.mean(face[:, :, 1])
    hue_saturation.append((hue, saturation))
    face_images.append(face)

hue_saturation = np.array(hue_saturation)

# Perform k-Means clustering on the hue-saturation features
kmeans = KMeans(n_clusters=2, n_init=10).fit(hue_saturation)

# Plot the clustered faces with custom markers using Matplotlib
fig, ax = plt.subplots(figsize=(12, 6))
for i, (x, y, w, h) in enumerate(faces_rect):
    im = OffsetImage(cv2.cvtColor(cv2.resize(face_images[i], (20, 20)), cv2.COLOR_HSV2RGB))
    ab = AnnotationBbox(im, (hue_saturation[i, 0], hue_saturation[i, 1]), frameon=False, pad=0)
    ax.add_artist(ab)
    plt.plot(hue_saturation[i, 0], hue_saturation[i, 1], 'o', markersize=5)

plt.xlabel("Hue")
plt.ylabel("Saturation")
plt.title("Face Clustering Based on Hue and Saturation")
plt.grid(True)

# Save the clustering plot as an image
clustering_plot_filename = "clustering_plot.png"
plt.savefig(clustering_plot_filename)
plt.close(fig)
print(f"Clustering plot saved as {clustering_plot_filename}")

# === Step 5: Log Outputs to WandB ===
wandb.log({
    "Detected Faces": len(faces_rect),
    "Annotated Image": wandb.Image(annotated_image_filename),
    "Clustering Plot": wandb.Image(clustering_plot_filename)
})

# Finish the WandB run
wandb.finish()

