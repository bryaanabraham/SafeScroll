import cv2
import os
from insightface.utils import face_align
from insightface.app import FaceAnalysis

def detect_faces_from_image_path(image_path):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    face_analyzer = FaceAnalysis(name='antelopev2', providers=providers)
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Failed to load image: {image_path}")

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = []

    faces = face_analyzer.get(rgb_frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        display_img = frame[y1:y2, x1:x2]

        if face.kps is not None:
            aligned = face_align.norm_crop(rgb_frame, face.kps)
            face_img = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
        else:
            face_img = display_img

        results.append({
            "face_img": face_img,
            "display_img": display_img,
            "bbox": (x1, y1, x2 - x1, y2 - y1),
            "embedding": face.embedding.astype("float32"),
            "det_score": float(face.det_score)
        })

    return results

def draw_faces(image_path, results, output_path=None):
    image = cv2.imread(image_path)

    for r in results:
        x, y, w, h = r["bbox"]
        score = r["det_score"]

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{score:.2f}",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_faces{ext}"

    cv2.imwrite(output_path, image)
    print(f"Saved to {output_path}")
