import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch
from scipy.spatial.distance import cdist


# ======================================================
# 1. GENERATE IMAGE USING GENERATIVE AI
# ======================================================

def generate_image(prompt="a pencil sketch of a futuristic city"):
    print("[AI] Generating image from text prompt...")

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    ).to("cuda")

    result = pipe(prompt).images[0]
    result.save("generated.png")

    print("[AI] Image saved as generated.png")
    return "generated.png"


# ======================================================
# 2. COMPUTER VISION — EDGE & STROKE EXTRACTION
# ======================================================

def extract_strokes(image_path):
    print("[CV] Extracting edges and contours...")

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 80, 160)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    # Convert contours to simplified stroke paths
    strokes = []
    for contour in contours:
        if len(contour) > 8:  # ignore tiny noise strokes
            path = contour.squeeze()
            strokes.append(path)

    print(f"[CV] Extracted {len(strokes)} stroke paths")
    return strokes, edges


# ======================================================
# 3. AI-STYLE STROKE PLANNING (NO ML REQUIRED)
# ======================================================

def order_strokes(strokes):
    print("[AI] Planning stroke order...")

    if not strokes:
        return []

    # Compute centroids
    centroids = np.array([np.mean(s, axis=0) for s in strokes])

    # Start at the stroke nearest to the top-left
    start_idx = np.argmin(centroids[:, 0] + centroids[:, 1])

    ordered = [strokes[start_idx]]
    remaining = centroids.tolist()
    strokes_left = strokes.copy()

    # Remove chosen stroke
    remaining.pop(start_idx)
    strokes_left.pop(start_idx)

    # Greedy nearest-neighbor ordering
    current = centroids[start_idx]

    while strokes_left:
        remaining_centroids = np.array([np.mean(s, axis=0) for s in strokes_left])
        dists = cdist([current], remaining_centroids).flatten()
        next_idx = np.argmin(dists)

        ordered.append(strokes_left[next_idx])
        current = remaining_centroids[next_idx]

        strokes_left.pop(next_idx)

    print("[AI] Stroke planning complete.")
    return ordered


# ======================================================
# 4. DRAWING SIMULATOR (MATPLOTLIB)
# ======================================================

def simulate_drawing(ordered_strokes):
    print("[SIM] Rendering simulated drawing...")

    plt.figure(figsize=(8, 8))

    for stroke in ordered_strokes:
        stroke = np.array(stroke)
        plt.plot(stroke[:, 0], -stroke[:, 1])  # invert Y for drawing aesthetics

    plt.title("AI Drawing Simulation")
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.show()


# ======================================================
# 5. FULL PIPELINE
# ======================================================

def run_pipeline(prompt):
    # Generate image
    path = generate_image(prompt)

    # Extract raw strokes
    strokes, edges = extract_strokes(path)

    # Plan stroke sequence
    ordered = order_strokes(strokes)

    # Simulate drawing on screen
    simulate_drawing(ordered)


# ======================================================
# EXECUTE
# ======================================================

if __name__ == "__main__":
    prompt = "a pencil sketch of a serene mountain landscape in minimalist style"
    run_pipeline(prompt)
