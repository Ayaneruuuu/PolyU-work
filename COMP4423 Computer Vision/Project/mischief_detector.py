from ultralytics import YOLO
from pathlib import Path
import cv2
import math

MODEL_PATH = "merged_pet_household_yolov8/runs/detect/pet_mischief_yolov8n_final/weights/best.pt"
INPUT_DIR = "demo_images"
OUTPUT_DIR = "mischief_outputs"


PET_CLASSES = {"cat", "dog"}
HIGH_RISK_OBJECTS = {"glass", "vase", "plate", "fork", "spoon"}
MEDIUM_RISK_OBJECTS = {"book", "box", "backpack", "fan"}
LOW_RISK_OBJECTS = {"chair", "table"}


def box_center(box):
    """Return center point of a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def normalized_distance(box1, box2, image_width, image_height):
    """
    Calculate distance between two bounding box centers.
    The distance is normalized by image diagonal, so it works for different image sizes.
    """
    c1 = box_center(box1)
    c2 = box_center(box2)

    distance = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    diagonal = math.sqrt(image_width ** 2 + image_height ** 2)

    return distance / diagonal


def boxes_overlap(box1, box2):
    """Check whether two boxes overlap."""
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)

    return inter_x2 > inter_x1 and inter_y2 > inter_y1


def calculate_mischief(detections, image_width, image_height):
    """
    detections: list of dictionaries:
    {
        "class_name": str,
        "confidence": float,
        "box": [x1, y1, x2, y2]
    }
    """

    pets = [d for d in detections if d["class_name"] in PET_CLASSES]
    objects = [d for d in detections if d["class_name"] not in PET_CLASSES]

    if not pets:
        return {
            "risk_level": "LOW RISK",
            "score": 0,
            "message": "No pet detected in the scene."
        }

    highest_score = 10
    best_message = "Pet detected, but no risky object is nearby."
    final_risk = "LOW RISK"

    for pet in pets:
        for obj in objects:
            pet_name = pet["class_name"]
            obj_name = obj["class_name"]

            dist = normalized_distance(
                pet["box"], obj["box"], image_width, image_height
            )

            overlap = boxes_overlap(pet["box"], obj["box"])

            # High-risk objects
            if obj_name in HIGH_RISK_OBJECTS:
                if overlap:
                    score = 95
                    risk = "HIGH RISK"
                    message = f"{pet_name} is touching or overlapping with {obj_name}."
                elif dist < 0.32:
                    score = 85
                    risk = "HIGH RISK"
                    message = f"{pet_name} is very close to {obj_name}."
                elif dist < 0.45:
                    score = 70
                    risk = "MEDIUM RISK"
                    message = f"{pet_name} is near {obj_name}."
                else:
                    score = 20
                    risk = "LOW RISK"
                    message = f"{pet_name} is far from {obj_name}."

            # Medium-risk objects
            elif obj_name in MEDIUM_RISK_OBJECTS:
                if overlap:
                    score = 70
                    risk = "MEDIUM RISK"
                    message = f"{pet_name} is interacting with {obj_name}."
                elif dist < 0.25:
                    score = 55
                    risk = "MEDIUM RISK"
                    message = f"{pet_name} is close to {obj_name}."
                else:
                    score = 15
                    risk = "LOW RISK"
                    message = f"{pet_name} is not close to risky objects."

            # Low-risk objects
            else:
                if overlap or dist < 0.25:
                    score = 30
                    risk = "LOW RISK"
                    message = f"{pet_name} is near {obj_name}, but this is low risk."
                else:
                    score = 10
                    risk = "LOW RISK"
                    message = "No risky spatial relationship detected."

            if score > highest_score:
                highest_score = score
                best_message = message
                final_risk = risk

    return {
        "risk_level": final_risk,
        "score": highest_score,
        "message": best_message
    }


def draw_results(image, detections, risk_result):
    """Draw bounding boxes and risk message on image."""

    for d in detections:
        x1, y1, x2, y2 = map(int, d["box"])
        label = f'{d["class_name"]} {d["confidence"]:.2f}'

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    risk_text = f'{risk_result["risk_level"]} | Score: {risk_result["score"]}'
    message_text = risk_result["message"]

    cv2.rectangle(image, (10, 10), (850, 90), (0, 0, 0), -1)
    cv2.putText(
        image,
        risk_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255) if risk_result["risk_level"] == "HIGH RISK" else (0, 255, 255),
        2
    )
    cv2.putText(
        image,
        message_text,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return image


def main():
    model = YOLO(MODEL_PATH)

    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + list(input_path.glob("*.jpeg"))

    if not image_files:
        print("No images found in demo_images folder.")
        return

    for img_file in image_files:
        image = cv2.imread(str(img_file))

        if image is None:
            print(f"Could not read image: {img_file}")
            continue

        height, width = image.shape[:2]

        results = model(str(img_file), conf=0.25, imgsz=640)

        detections = []

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": xyxy
                })

        risk_result = calculate_mischief(detections, width, height)

        output_image = draw_results(image, detections, risk_result)

        save_path = output_path / img_file.name
        cv2.imwrite(str(save_path), output_image)

        print(f"Saved: {save_path}")
        print(risk_result)


if __name__ == "__main__":
    main()