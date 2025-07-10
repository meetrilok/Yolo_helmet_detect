from flask import Flask, render_template, request
import cv2, numpy as np, base64
from detector import YOLOOnnxDetector

MODEL_PATH   = "model/best.onnx"
TEMPLATE_HTML = "indel.html"

app = Flask(__name__)
det = YOLOOnnxDetector(MODEL_PATH)

LABEL = "helmet"
BOX_COLOR = (0, 255, 0)

def draw_boxes(img, boxes, scores):
    for (x1, y1, x2, y2), conf in zip(boxes, scores):
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
        txt = f"{LABEL}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), BOX_COLOR, -1)
        cv2.putText(img, txt, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        results = []
        for f in request.files.getlist("images"):
            nparr = np.frombuffer(f.read(), np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            boxes, scores = det(img)
            annotated = draw_boxes(img, boxes, scores) if boxes.size else img

            _, enc = cv2.imencode(".jpg", annotated)
            img_b64 = base64.b64encode(enc).decode()
            caption = [LABEL] if boxes.size else ["No helmet detected"]
            results.append((img_b64, caption))

        return render_template(TEMPLATE_HTML, results=results)

    return render_template(TEMPLATE_HTML)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
