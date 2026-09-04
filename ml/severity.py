
import numpy as np


TYPE_IMPACT = {
    "D00": 0.50,
    "D10": 0.50,
    "D20": 1.00,
    "D40": 0.90
}

# Calibration values from validation dataset
EXTENT_P50 = 0.0694596724
EXTENT_P90 = 0.2847822312

CONFIDENCE_P50 = 0.39845
CONFIDENCE_P90 = 0.664475

COUNT_P99 = 10


def percentile_score(value, p50, p90):
    if value <= 0:
        return 0.0

    if value <= p50:
        return (value / p50) * 50.0

    if value >= p90:
        return 100.0

    return 50.0 + (
        (value - p50) / (p90 - p50)
    ) * 50.0


def count_score(count):
    if count <= 0:
        return 0.0

    return min(
        100.0,
        100.0 * np.log1p(count) / np.log1p(COUNT_P99)
    )


def reliability_score(confidence):
    return percentile_score(
        confidence,
        CONFIDENCE_P50,
        CONFIDENCE_P90
    )


def calculate_union_area(boxes):
    """
    Calculate union area of bounding boxes.
    """

    if not boxes:
        return 0.0

    xs = sorted(
        set(
            [b[0] for b in boxes] +
            [b[2] for b in boxes]
        )
    )

    ys = sorted(
        set(
            [b[1] for b in boxes] +
            [b[3] for b in boxes]
        )
    )

    union_area = 0.0

    for i in range(len(xs) - 1):

        x1 = xs[i]
        x2 = xs[i + 1]

        if x2 <= x1:
            continue

        for j in range(len(ys) - 1):

            y1 = ys[j]
            y2 = ys[j + 1]

            if y2 <= y1:
                continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            for bx1, by1, bx2, by2 in boxes:

                if (
                    bx1 <= cx <= bx2
                    and
                    by1 <= cy <= by2
                ):
                    union_area += (
                        (x2 - x1) *
                        (y2 - y1)
                    )
                    break

    return union_area


def calculate_severity(
    detections,
    image_width,
    image_height
):
    """
    Calculate road-condition severity.

    detections format:

    [
        {
            "class": "D20",
            "confidence": 0.82,
            "bbox": [x1, y1, x2, y2]
        }
    ]
    """

    if not detections:
        return {
            "severity_score": 0.0,
            "severity": "NO DAMAGE",
            "damage_count": 0,
            "damage_types": [],
            "damage_extent": 0.0,
            "extent_score": 0.0,
            "spread_score": 0.0,
            "reliability_score": 0.0,
            "type_score": 0.0
        }

    image_area = image_width * image_height

    boxes = []

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        boxes.append(
            (
                float(x1),
                float(y1),
                float(x2),
                float(y2)
            )
        )

    union_area = calculate_union_area(boxes)

    damage_extent = union_area / image_area

    # Extent: 45%
    extent_score = percentile_score(
        damage_extent,
        EXTENT_P50,
        EXTENT_P90
    )

    # Spread: 25%
    damage_count = len(detections)

    count_component = count_score(damage_count)

    damage_types = sorted(
        set(
            detection["class"]
            for detection in detections
        )
    )

    type_diversity = min(
        100.0,
        (len(damage_types) / 4.0) * 100.0
    )

    spread_score = (
        0.70 * count_component
        +
        0.30 * type_diversity
    )

    # Reliability: 20%
    mean_confidence = np.mean(
        [
            float(detection["confidence"])
            for detection in detections
        ]
    )

    reliability = reliability_score(
        mean_confidence
    )

    # Type impact: 10%
    type_impact = max(
        TYPE_IMPACT.get(
            detection["class"],
            0.50
        )
        for detection in detections
    )

    type_score = type_impact * 100.0

    # Final score
    severity_score = (
        0.45 * extent_score
        +
        0.25 * spread_score
        +
        0.20 * reliability
        +
        0.10 * type_score
    )

    severity_score = float(
        np.clip(
            severity_score,
            0.0,
            100.0
        )
    )

    # Classification
    if severity_score <= 30:
        severity = "LOW"

    elif severity_score <= 60:
        severity = "MEDIUM"

    else:
        severity = "HIGH"

    return {
    "severity_score": float(round(severity_score, 2)),
    "severity": severity,
    "damage_count": int(damage_count),
    "damage_types": damage_types,
    "damage_extent": float(round(damage_extent, 6)),
    "extent_score": float(round(extent_score, 2)),
    "spread_score": float(round(spread_score, 2)),
    "reliability_score": float(round(reliability, 2)),
    "type_score": float(round(type_score, 2))
}