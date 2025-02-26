import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# -------------------------------
# Part 1: Face Detection on Plaksha_Faculty.jpg
# -------------------------------

# Read the image
img = cv2.imread('Plaksha_Faculty.jpg')
if img is None:
    print("Error: Plaksha_Faculty.jpg not found.")
    exit(1)

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the Haar cascade classifier file (ensure the XML file is in your folder)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces (parameters can be adjusted)
faces_rect = face_cascade.detectMultiScale(gray_img, 1.05, 4, minSize=(25,25), maxSize=(50,50))

# Define text and font parameters for annotation
text = "Face Detected"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (0, 0, 255)  # Red in BGR
font_thickness = 2

# Draw rectangles and annotate each detected face
for (x, y, w, h) in faces_rect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(img, text, (x, y - 10), font, font_scale, font_color, font_thickness)

# Save the annotated image
annotated_image_filename = "faces_detected.jpg"
cv2.imwrite(annotated_image_filename, img)
print(f"Annotated image saved as {annotated_image_filename}")

# -------------------------------
# Part 2: Clustering Based on Hue and Saturation Features
# -------------------------------

# Re-read the original image for feature extraction (to avoid modifications)
original_img = cv2.imread('Plaksha_Faculty.jpg')
img_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
hue_saturation = []
face_images = []  # To store the face regions

# Extract the mean hue and saturation values for each detected face
for (x, y, w, h) in faces_rect:
    face = img_hsv[y:y+h, x:x+w]
    hue = np.mean(face[:, :, 0])
    saturation = np.mean(face[:, :, 1])
    hue_saturation.append((hue, saturation))
    face_images.append(face)

hue_saturation = np.array(hue_saturation)

# Perform k-Means clustering on the hue-saturation features (using 2 clusters)
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

clustering_plot_filename = "clustering_plot.png"
plt.savefig(clustering_plot_filename)
plt.close(fig)
print(f"Clustering plot saved as {clustering_plot_filename}")

# -------------------------------
# Part 3: Scatter Plot with Cluster Centroids
# -------------------------------

cluster_0_points = []
cluster_1_points = []

for i, (x, y, w, h) in enumerate(faces_rect):
    if kmeans.labels_[i] == 0:
        cluster_0_points.append((hue_saturation[i, 0], hue_saturation[i, 1]))
    else:
        cluster_1_points.append((hue_saturation[i, 0], hue_saturation[i, 1]))

cluster_0_points = np.array(cluster_0_points)
cluster_1_points = np.array(cluster_1_points)

plt.scatter(cluster_0_points[:, 0], cluster_0_points[:, 1], color='green', label='Cluster 0')
plt.scatter(cluster_1_points[:, 0], cluster_1_points[:, 1], color='blue', label='Cluster 1')

centroid_0 = np.mean(cluster_0_points, axis=0)
centroid_1 = np.mean(cluster_1_points, axis=0)

plt.scatter(centroid_0[0], centroid_0[1], color='black', marker='x', s=100, label='Centroid 0')
plt.scatter(centroid_1[0], centroid_1[1], color='black', marker='D', s=100, label='Centroid 1')

plt.xlabel("Hue")
plt.ylabel("Saturation")
plt.title("Cluster Scatter Plot with Centroids")
plt.legend()
plt.grid(True)

scatter_plot_filename = "cluster_scatter_plot.png"
plt.savefig(scatter_plot_filename)
plt.close()
print(f"Cluster scatter plot saved as {scatter_plot_filename}")

# -------------------------------
# Part 4: Process the Template Image (Dr_Shashi_Tharoor.jpg)
# -------------------------------

template_img = cv2.imread('Dr_Shashi_Tharoor.jpg')
if template_img is None:
    print("Error: Dr_Shashi_Tharoor.jpg not found.")
    exit(1)

template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
template_faces = face_cascade.detectMultiScale(template_gray, 1.05, 4, minSize=(25,25), maxSize=(50,50))

for (x, y, w, h) in template_faces:
    cv2.rectangle(template_img, (x, y), (x+w, y+h), (0, 255, 0), 3)

template_annotated_filename = "template_faces.jpg"
cv2.imwrite(template_annotated_filename, template_img)
print(f"Template image with detected face(s) saved as {template_annotated_filename}")

template_hsv = cv2.cvtColor(template_img, cv2.COLOR_BGR2HSV)
template_hue = np.mean(template_hsv[:, :, 0])
template_saturation = np.mean(template_hsv[:, :, 1])
template_label = kmeans.predict([[template_hue, template_saturation]])[0]

# -------------------------------
# Part 5: Visualize Template Image with Clusters
# -------------------------------

fig, ax = plt.subplots(figsize=(12, 6))
for i, (x, y, w, h) in enumerate(faces_rect):
    color = 'red' if kmeans.labels_[i] == 0 else 'blue'
    im = OffsetImage(cv2.cvtColor(cv2.resize(face_images[i], (20, 20)), cv2.COLOR_HSV2RGB))
    ab = AnnotationBbox(im, (hue_saturation[i, 0], hue_saturation[i, 1]), frameon=False, pad=0)
    ax.add_artist(ab)
    plt.plot(hue_saturation[i, 0], hue_saturation[i, 1], 'o', markersize=5, color=color)

if template_label == 0:
    color = 'red'
else:
    color = 'blue'
im = OffsetImage(cv2.cvtColor(cv2.resize(template_img, (20, 20)), cv2.COLOR_BGR2RGB))
ab = AnnotationBbox(im, (template_hue, template_saturation), frameon=False, pad=0)
ax.add_artist(ab)

plt.xlabel("Hue")
plt.ylabel("Saturation")
plt.title("Template Face and Cluster Visualization")
plt.grid(True)

template_cluster_plot_filename = "template_cluster_plot.png"
plt.savefig(template_cluster_plot_filename)
plt.close(fig)
print(f"Template cluster plot saved as {template_cluster_plot_filename}")

# -------------------------------
# Part 6: Final Scatter Plot with Centroids and Template Marker
# -------------------------------

cluster_0_points = []
cluster_1_points = []

for i, (x, y, w, h) in enumerate(faces_rect):
    if kmeans.labels_[i] == 0:
        cluster_0_points.append((hue_saturation[i, 0], hue_saturation[i, 1]))
    else:
        cluster_1_points.append((hue_saturation[i, 0], hue_saturation[i, 1]))

cluster_0_points = np.array(cluster_0_points)
cluster_1_points = np.array(cluster_1_points)

plt.scatter(cluster_0_points[:, 0], cluster_0_points[:, 1], color='green', label='Cluster 0')
plt.scatter(cluster_1_points[:, 0], cluster_1_points[:, 1], color='blue', label='Cluster 1')

centroid_0 = np.mean(cluster_0_points, axis=0)
centroid_1 = np.mean(cluster_1_points, axis=0)

plt.scatter(centroid_0[0], centroid_0[1], color='black', marker='x', s=100, label='Centroid 0')
plt.scatter(centroid_1[0], centroid_1[1], color='black', marker='D', s=100, label='Centroid 1')

plt.plot(template_hue, template_saturation, marker='o', color='violet', markersize=10, label='Template')

plt.xlabel("Hue")
plt.ylabel("Saturation")
plt.title("Final Cluster Scatter Plot with Template")
plt.legend()
plt.grid(True)

final_scatter_filename = "final_scatter_plot.png"
plt.savefig(final_scatter_filename)
plt.close()
print(f"Final scatter plot saved as {final_scatter_filename}")

