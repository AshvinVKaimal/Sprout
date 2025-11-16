#!/usr/bin/env python3
"""
Automatic human measurements from a single image containing a person and a 15 cm scale placed alongside.

Assumptions (see README_measurements.md for details):
- The 15 cm scale is a visible, roughly rectangular high-contrast object in the image (vertical or horizontal).
- The scale is fully visible and not occluded.
- The person is mostly standing upright and fully visible (head to feet) in the image.

This script attempts a robust automated pipeline using OpenCV + MediaPipe (for pose landmarks).
It outputs estimated height (cm), head circumference (cm) and wrist circumference (cm).
"""

import argparse
import math
import sys
import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    sys.exit("Error: mediapipe is required. Install with `pip install mediapipe opencv-python`.\n" + str(e))


def detect_scale_pixel_length(image_gray, orig_img):
    """Detect the 15 cm scale in the image and return its pixel length (longer side).
    Strategy:
    - Canny -> find contours -> find rectangular-ish contours
    - For each candidate, compute minAreaRect and pick the one whose aspect ratio
      is large (long thin object) and reasonable area.
    Returns pixel_length or None if not found.
    """
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image_gray.shape[:2]
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.0005 * w * h:  # skip tiny
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        if short_side <= 0:
            continue
        aspect = long_side / (short_side + 1e-8)
        # Prefer elongated objects and not too close to image size
        if aspect > 3.0 and long_side > 0.05 * max(w, h) and long_side < 0.9 * max(w, h):
            candidates.append((area, long_side, rect))

    if not candidates:
        # fallback: try largest contour's long side
        if contours:
            best_cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(best_cnt)
            long_side = max(rect[1][0], rect[1][1])
            return long_side
        return None

    # pick candidate with largest area
    candidates.sort(key=lambda x: x[0], reverse=True)
    _area, pixel_length, rect = candidates[0]
    return pixel_length


def estimate_head_top_y(landmarks, image_h):
    # landmarks are normalized (x,y) in mediapipe
    ys = []
    for name in ['LEFT_EYE', 'RIGHT_EYE', 'NOSE', 'LEFT_EAR', 'RIGHT_EAR']:
        lm = getattr(mp.solutions.pose.PoseLandmark, name)
        y = landmarks[lm].y
        ys.append(y)
    # top candidate is minimum y among these minus a small offset (normalized)
    min_y = min(ys)
    # offset as fraction of distance between nose and shoulders to approximate top of head
    try:
        shoulder_y = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y +
                      landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y) / 2.0
        offset = max(0.03, 0.12 * abs(shoulder_y - min_y))
    except Exception:
        offset = 0.05
    top_y = max(0.0, min_y - offset)
    return top_y * image_h


def estimate_feet_y(landmarks, image_h):
    # use ankle and heel landmarks; take the max (lowest in image)
    ys = []
    for name in ['LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL']:
        lm = getattr(mp.solutions.pose.PoseLandmark, name)
        ys.append(landmarks[lm].y)
    feet_y = max(ys)
    return min(image_h, feet_y * image_h)


def landmark_point_to_pixel(lm, image_w, image_h):
    return int(lm.x * image_w), int(lm.y * image_h)


def measure_head_and_wrists(landmarks, image, pixel_per_cm):
    h, w = image.shape[:2]
    # Head width: use distance between left and right ear (if available) or eyes
    left_ear = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR]
    right_ear = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR]
    left_eye = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE]
    right_eye = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE]

    # prefer ear-to-ear
    if left_ear.visibility > 0.3 and right_ear.visibility > 0.3:
        lx, ly = landmark_point_to_pixel(left_ear, w, h)
        rx, ry = landmark_point_to_pixel(right_ear, w, h)
        head_width_px = math.hypot(rx - lx, ry - ly)
    else:
        lx, ly = landmark_point_to_pixel(left_eye, w, h)
        rx, ry = landmark_point_to_pixel(right_eye, w, h)
        head_width_px = math.hypot(rx - lx, ry - ly) * 1.6  # eyes-to-approximate ear distance

    head_width_cm = head_width_px / pixel_per_cm
    # approximate head circumference: assume roughly circular/elliptical
    head_circumference_cm = math.pi * head_width_cm * 1.02

    # Wrist circumference: crop small region around wrist and estimate thickness
    wrist_circumferences = []
    for lm_name in ['LEFT_WRIST', 'RIGHT_WRIST']:
        lm = getattr(mp.solutions.pose.PoseLandmark, lm_name)
        lmobj = landmarks[lm]
        if lmobj.visibility < 0.25:
            continue
        cx, cy = landmark_point_to_pixel(lmobj, w, h)
        size = int(max(40, 0.06 * max(w, h)))
        x0 = max(0, cx - size)
        x1 = min(w, cx + size)
        y0 = max(0, cy - size)
        y1 = min(h, cy + size)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # find largest contour and its minAreaRect width (short side -> thickness)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # fallback: use a small fixed pixel width
            wrist_px = max(6, 0.02 * max(w, h))
        else:
            c = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            (rw, rh) = rect[1]
            wrist_px = min(rw, rh)
        wrist_circ_cm = math.pi * (wrist_px / pixel_per_cm) * 1.05
        wrist_circumferences.append(wrist_circ_cm)

    wrist_circ = None
    if wrist_circumferences:
        wrist_circ = sum(wrist_circumferences) / len(wrist_circumferences)

    return head_circumference_cm, wrist_circ


def process_image(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Unable to open image: {path}")
    orig = img.copy()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pixel_length = detect_scale_pixel_length(gray, orig)
    if pixel_length is None:
        raise RuntimeError("Could not detect the 15 cm scale automatically.")

    pixel_per_cm = pixel_length / 15.0

    # Run MediaPipe pose
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        # convert BGR->RGB
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            raise RuntimeError("Pose landmarks not detected. Ensure a full-body person is visible.")
        landmarks = results.pose_landmarks.landmark

        # estimate top of head and feet in pixel coords
        top_y = estimate_head_top_y(landmarks, h)
        feet_y = estimate_feet_y(landmarks, h)
        pixel_height = max(0.0, feet_y - top_y)
        height_cm = pixel_height / pixel_per_cm

        head_circ_cm, wrist_circ_cm = measure_head_and_wrists(landmarks, img, pixel_per_cm)

    return {
        'height_cm': float(height_cm),
        'head_circumference_cm': float(head_circ_cm),
        'wrist_circumference_cm': float(wrist_circ_cm) if wrist_circ_cm is not None else None,
        'pixel_per_cm': float(pixel_per_cm)
    }


def main():
    parser = argparse.ArgumentParser(description='Automatic human measurements from a single image with a 15 cm scale.')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    res = process_image(args.image)
    print('Results:')
    print(f"  Estimated height: {res['height_cm']:.1f} cm")
    print(f"  Head circumference (approx): {res['head_circumference_cm']:.1f} cm")
    if res['wrist_circumference_cm']:
        print(f"  Wrist circumference (approx): {res['wrist_circumference_cm']:.1f} cm")
    else:
        print('  Wrist circumference: not detected')
    print(f"  Pixel per cm (scale) : {res['pixel_per_cm']:.3f} px/cm")


if __name__ == '__main__':
    main()
