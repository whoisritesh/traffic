import argparse
import time
from pathlib import Path
import cv2
from ultralytics import YOLO


def is_rider_on_motorcycle(head_box, moto_boxes) -> bool:
    """Return True only if the detected head overlaps or sits right above a motorcycle."""
    hx1, hy1, hx2, hy2 = head_box
    h_cx = (hx1 + hx2) / 2.0
    h_cy = (hy1 + hy2) / 2.0
    h_height = max(hy2 - hy1, 1)

    for mx1, my1, mx2, my2 in moto_boxes:
        m_width = max(mx2 - mx1, 1)
        # Expand motorcycle box horizontally by 25% and upward by 2.2x head height to cover the rider's head
        pad_x = m_width * 0.25
        top_y = my1 - (h_height * 2.2)

        if (mx1 - pad_x) <= h_cx <= (mx2 + pad_x) and top_y <= h_cy <= my2:
            return True
    return False


def run_detection(source: str, weights: str, moto_weights: str, output_path: str, conf: float = 0.45, show: bool = True, filter_moto: bool = True):
    helmet_model = YOLO(weights)
    moto_model = YOLO(moto_weights)  # COCO class 3 = motorcycle

    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    colors = {
        0: (0, 200, 0),    # Green: With Helmet
        1: (0, 0, 255),    # Red: Without Helmet
    }

    if show:
        cv2.namedWindow("YOLOv8 Highway Helmet Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv8 Highway Helmet Detection", 1280, 720)

    prev_time = time.time()
    print(f"Processing '{source}' (filtering out walking pedestrians)... Press 'q' to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect motorcycles (COCO class 3)
        moto_results = moto_model.predict(frame, classes=[3], conf=0.30, verbose=False)[0]
        moto_boxes = [tuple(map(int, b.xyxy[0])) for b in moto_results.boxes]

        # 2. Detect helmets and keep ONLY riders associated with a motorcycle
        results = helmet_model.predict(frame, conf=conf, verbose=False)[0]

        helmet_count = 0
        no_helmet_count = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Skip walking pedestrians not on a motorcycle
            if filter_moto and not is_rider_on_motorcycle((x1, y1, x2, y2), moto_boxes):
                continue

            label_name = helmet_model.names[cls_id]
            if cls_id == 0:
                helmet_count += 1
            else:
                no_helmet_count += 1

            color = colors.get(cls_id, (255, 200, 0))
            label = f"{label_name} {score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

        curr_time = time.time()
        current_fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time

        cv2.rectangle(frame, (15, 15), (540, 62), (20, 20, 20), -1)
        hud_text = f"Helmet: {helmet_count} | No Helmet: {no_helmet_count} | FPS: {current_fps:.1f}"
        cv2.putText(frame, hud_text, (25, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)

        writer.write(frame)

        if show:
            cv2.imshow("YOLOv8 Highway Helmet Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done! Annotated video saved to: {Path(output_path).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Highway Helmet Detection (Riders Only)")
    parser.add_argument("--source", type=str, default="sample_videos/traffic.mp4", help="Path to video or '0' for webcam")
    parser.add_argument("--weights", type=str, default="models/helmet_best.pt", help="Path to trained helmet weights")
    parser.add_argument("--moto-weights", type=str, default="yolov8n.pt", help="YOLOv8 weights for motorcycle filtering")
    parser.add_argument("--output", type=str, default="outputs/helmet_detected.mp4", help="Output video path")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--no-moto-filter", action="store_true", help="Disable motorcycle filter for indoor webcam demos")
    args = parser.parse_args()

    run_detection(args.source, args.weights, args.moto_weights, args.output, args.conf, filter_moto=not args.no_moto_filter)
