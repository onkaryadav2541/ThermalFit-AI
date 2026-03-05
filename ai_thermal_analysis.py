import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import winsound
from fpdf import FPDF
from datetime import datetime
import os


def main():
    # --- PATH CONFIGURATION ---
    video_path = r'C:\Users\Onkar\OneDrive\Desktop\Screen Recordings\with film tight sakshi.mp4'
    model_path = r'C:\Users\Onkar\PycharmProjects\Quantative_testing_method_for_respiratory_mask\runs\pose\runs\pose\thermal_mask_model\weights\best.pt'
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")

    yolo_model = YOLO(model_path)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    cap = cv2.VideoCapture(video_path)
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]

    # --- TRACKING & CALIBRATION ---
    LEAK_THRESHOLD = 140
    total_frames, leak_frames_count = 0, 0
    leak_log = {name: 0 for name in
                ["Nose_Bridge", "Left_Cheek", "Right_Cheek", "Left_Chin", "Right_Chin", "Center_Chin"]}
    captured_proofs = {}
    face_shape, max_shift = "Calculating...", 0
    initial_nose = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()

        # 1. YOLO: DYNAMIC MASK LOCK
        results = yolo_model(frame, conf=0.5, verbose=False)
        mask_box = None
        for r in results:
            if len(r.boxes) > 0:
                mask_box = r.boxes.xyxy[0].cpu().numpy().astype(int)

        # 2. MEDIAPIPE: ADVANCED FACE STRUCTURE & SHIFT
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_res = face_mesh.process(rgb)
        if mp_res.multi_face_landmarks:
            f = mp_res.multi_face_landmarks[0].landmark
            # Enhanced Shape Logic: Width vs Height vs Temple
            jaw_w = abs(f[234].x - f[454].x)
            f_h = abs(f[10].y - f[152].y)
            ratio = jaw_w / f_h
            if ratio > 0.92:
                face_shape = "Square/Round"
            elif ratio < 0.80:
                face_shape = "Oval/Long"
            else:
                face_shape = "Heart/Diamond"

            nose = np.array([f[1].x * w, f[1].y * h])
            if initial_nose is None:
                initial_nose = nose
            else:
                shift = np.linalg.norm(nose - initial_nose)
                max_shift = max(max_shift, shift)

        # 3. DYNAMIC U-SHAPE ANCHORING (FIXED ALIGNMENT)
        if mask_box is not None:
            x1, y1, x2, y2 = mask_box
            wb, hb = x2 - x1, y2 - y1

            # Recalculated to follow the mask's actual edges in a U-shape
            points = {
                "Nose_Bridge": (x1 + int(wb * 0.5), y1 + int(hb * 0.05)),
                "Left_Cheek": (x1 + int(wb * 0.1), y1 + int(hb * 0.45)),
                "Right_Cheek": (x2 - int(wb * 0.1), y1 + int(hb * 0.45)),
                "Left_Chin": (x1 + int(wb * 0.25), y2 - int(hb * 0.15)),  # Raised for U-curve
                "Right_Chin": (x2 - int(wb * 0.25), y2 - int(hb * 0.15)),  # Raised for U-curve
                "Center_Chin": (x1 + int(wb * 0.5), y2 - 5)  # Absolute bottom
            }

            frame_leak = False
            for name, (px, py) in points.items():
                px, py = np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)
                # Area sensing for sensitivity
                roi = gray[max(0, py - 2):min(h, py + 2), max(0, px - 2):min(w, px + 2)]
                val = np.max(roi) if roi.size > 0 else 0

                if val > LEAK_THRESHOLD:
                    leak_log[name] += 1
                    frame_leak = True
                    cv2.circle(display_frame, (px, py), 8, (0, 0, 255), -1)  # RED for LEAK
                    if name not in captured_proofs:
                        ss_path = os.path.join(downloads_path, f"{name}_proof.jpg")
                        cv2.imwrite(ss_path, display_frame)
                        captured_proofs[name] = ss_path
                else:
                    cv2.circle(display_frame, (px, py), 8, (0, 255, 0), -1)  # GREEN for STABLE

            if frame_leak:
                leak_frames_count += 1
                winsound.Beep(1200, 40)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 120, 0), 2)

        cv2.putText(display_frame, f"Face: {face_shape}", (20, 40), 0, 0.7, (0, 255, 255), 2)
        cv2.imshow("Quantitative Fit Test", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # --- PENALTY FIT SCORE & PDF REPORT ---
    l_rate = (leak_frames_count / total_frames) * 100 if total_frames > 0 else 0
    final_score = max(0, 100 - (l_rate * 3.0) - (max_shift * 0.5))  # Penalizes movement and leaks

    # Save PDF to Downloads with automatic numbering
    count, pdf_path = 1, os.path.join(downloads_path, f"{video_base_name}_Report.pdf")
    while os.path.exists(pdf_path):
        pdf_path = os.path.join(downloads_path, f"{video_base_name}_Report_{count}.pdf")
        count += 1

    pdf = FPDF()
    pdf.add_page();
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "QUANTITATIVE MASK FIT REPORT", ln=True, align='C')
    pdf.ln(5);
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Face Morphology: {face_shape} | Max Shift: {max_shift:.1f}px", ln=True)
    pdf.set_text_color(255, 0, 0) if final_score < 85 else pdf.set_text_color(0, 100, 0)
    pdf.cell(200, 10, f"FINAL FIT SCORE: {final_score:.2f} / 100", ln=True)
    pdf.set_text_color(0, 0, 0);
    pdf.ln(10)

    pdf.cell(200, 10, "LEAK EVIDENCE GALLERY:", ln=True)
    for name, path in captured_proofs.items():
        pdf.cell(200, 8, f"Detected Leak Point: {name}", ln=True)
        pdf.image(path, w=100);
        pdf.ln(5)

    pdf.output(pdf_path)
    print(f"\nPDF Report successfully saved to Downloads: {pdf_path}")
    cap.release();
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()