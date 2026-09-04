from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from .severity import calculate_severity


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"

CONF_THRESHOLD = 0.20
IMAGE_SIZE = 640


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))


# ============================================================
# DAMAGE DETECTION + SEVERITY
# ============================================================

def detect_damage(image):
    """
    Detect road damage and calculate road-condition severity.

    Parameters
    ----------
    image : str, pathlib.Path, or supported image input
        Image path or image supported by Ultralytics.

    Returns
    -------
    dict
        Structured road-damage detection and severity result.
    """

    # --------------------------------------------------------
    # Get original image dimensions
    # --------------------------------------------------------

    with Image.open(image) as img:
        image_width, image_height = img.size

    # --------------------------------------------------------
    # YOLO inference
    # --------------------------------------------------------

    results = model.predict(
        source=image,
        conf=CONF_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device=0,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            detections.append({
                "class_id": class_id,
                "class": model.names[class_id],
                "confidence": round(confidence, 4),
                "bbox": [
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2), 2),
                    round(float(y2), 2)
                ]
            })

    # --------------------------------------------------------
    # Calculate severity
    # --------------------------------------------------------

    severity = calculate_severity(
        detections=detections,
        image_width=image_width,
        image_height=image_height
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "damage_detected": len(detections) > 0,
        "damage_count": len(detections),
        "detections": detections,
        "severity": severity
    }


# ============================================================
# SIMPLE COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print("Model loaded successfully!")
    print(f"Model: {MODEL_PATH}")

    test_image = (
        BASE_DIR.parent
        / "datasets"
        / "processed"
        / "rdd4"
        / "images"
        / "val"
        / "India_India_000043.jpg"
    )

    print("\nTesting image:")
    print(test_image)

    if not test_image.exists():

        print("\nTest image not found!")
        print("Change the test_image path in this file.")

    else:

        output = detect_damage(str(test_image))

        output["image"] = test_image.name

        print("\nDetection + Severity result:")
        print(output)