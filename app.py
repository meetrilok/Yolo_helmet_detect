"""
Flask front-end for single-class helmet detection.
• Upload multiple images; each comes back annotated.
• Shows “Detected: helmet” when ≥1 detection, else “No helmet detected”.
"""

from flask import Flask, render_template, request
import cv2, numpy as np, base64

from detector import YOLOOnnxDetector

# ------------------------------------------------------------------
MODEL_PATH = "model/best.onnx"     # path to your helmet ONNX
TEMPLATE = "indel.html"            # file in templates/

LABEL = "helmet"
BOX_COLOR = (0, 255, 0)            # green
# ------------------------------------------------------------------

app = Flask(__name__)
detector = YOLOOnnxDetector(MODEL_PATH)


def annotate(img, boxes, scores):
    """Draw green boxes + score label on BGR image."""
    for (x1, y1, x2, y2), conf in zip(boxes, scores):
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
        txt = f"{LABEL}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1),
                      BOX_COLOR, thickness=-1)
        cv2.putText(img, txt, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return img


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        results = []
        for file in request.files.getlist("images"):
            buf = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

            boxes, scores = detector(img)
            annotated = annotate(img, boxes, scores) if boxes.size else img

            _, enc = cv2.imencode(".jpg", annotated)
            img_b64 = base64.b64encode(enc).decode()
            caption = [LABEL] if boxes.size else ["No helmet detected"]
            results.append((img_b64, caption))

        return render_template(TEMPLATE, results=results)

    return render_template(TEMPLATE)


if __name__ == "__main__":
    # use 0.0.0.0 for LAN access; set debug=False in production
    app.run(host="0.0.0.0", port=5000, debug=True)
