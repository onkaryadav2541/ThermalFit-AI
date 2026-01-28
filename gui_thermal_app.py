import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Videos\Captures\nutonguy.mp4'

# PHYSICS
T_MIN = 22.0
T_MAX = 35.0
LEAK_THRESHOLD_SCORE = 45

# UI COLORS
COLOR_BG = "#1a1a1a"
COLOR_SIDEBAR = "#2b2b2b"
COLOR_ACCENT = "#1f6aa5"
COLOR_RED = "#cf352e"
COLOR_GREEN = "#2ecf5f"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")


class ThermalFitApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("ThermalFit-AI: Master's Thesis Interface")
        self.geometry("1400x850")  # BIG WINDOW
        self.configure(fg_color=COLOR_BG)

        # STATE VARIABLES
        self.cap = None
        self.mode = None  # "MANUAL" or "AI"
        self.is_running = False
        self.manual_points = []
        self.session_scores = []

        # SENSOR DATA
        self.point_names = ["Nose Bridge", "Left Cheek", "Right Cheek", "Left Chin", "Right Chin"]
        self.sensor_data = {name: deque(maxlen=10) for name in self.point_names}

        # AI SETUP
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        # AI Sensor Indices (Approximations for Nose, Cheeks, Chin)
        self.ai_indices = [6, 123, 352, 172, 397]

        self.setup_ui()

    def setup_ui(self):
        # GRID LAYOUT (2 Columns: Sidebar, Main Video)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- 1. LEFT SIDEBAR (Controls) ---
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0, fg_color=COLOR_SIDEBAR)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        # Logo / Title
        self.lbl_title = ctk.CTkLabel(self.sidebar, text="THERMAL FIT\nANALYSIS", font=("Roboto", 28, "bold"))
        self.lbl_title.pack(pady=(30, 10))

        self.lbl_subtitle = ctk.CTkLabel(self.sidebar, text="Quantitative Leak Detection", font=("Roboto", 12))
        self.lbl_subtitle.pack(pady=(0, 20))

        # Mode Buttons
        self.btn_manual = ctk.CTkButton(self.sidebar, text="🎯 MANUAL MODE", command=self.start_manual_mode,
                                        height=50, font=("Roboto", 16, "bold"), fg_color=COLOR_ACCENT)
        self.btn_manual.pack(fill="x", padx=20, pady=10)

        self.btn_ai = ctk.CTkButton(self.sidebar, text="🤖 AI AUTO MODE", command=self.start_ai_mode,
                                    height=50, font=("Roboto", 16, "bold"), fg_color="#5A2E8A")
        self.btn_ai.pack(fill="x", padx=20, pady=10)

        self.btn_stop = ctk.CTkButton(self.sidebar, text="⏹ STOP / REPORT", command=self.stop_analysis,
                                      height=40, fg_color=COLOR_RED, hover_color="#8a1c17")
        self.btn_stop.pack(fill="x", padx=20, pady=(20, 10))

        # Instructions Box
        self.txt_console = ctk.CTkTextbox(self.sidebar, height=150, fg_color="#111111", text_color="#00FF00")
        self.txt_console.pack(fill="x", padx=20, pady=20)
        self.log("System Ready.\nSelect a Mode to begin.")

        # Status Indicator
        self.lbl_status = ctk.CTkLabel(self.sidebar, text="STANDBY", font=("Roboto", 24, "bold"), text_color="gray")
        self.lbl_status.pack(pady=20, side="bottom")

        # --- 2. MAIN AREA (Video + Dashboard) ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        # Video Screen (Big!)
        self.video_label = ctk.CTkLabel(self.main_frame, text="", fg_color="black", corner_radius=10)
        self.video_label.pack(expand=True, fill="both")

        # Bind Click Event for Manual Mode
        self.video_label.bind("<Button-1>", self.on_video_click)

        # Dashboard (Bottom Bars)
        self.dashboard_frame = ctk.CTkFrame(self.main_frame, height=150, fg_color="#222222")
        self.dashboard_frame.pack(fill="x", pady=(10, 0))

        self.bars = {}
        for i, name in enumerate(self.point_names):
            # Column for each sensor
            frame = ctk.CTkFrame(self.dashboard_frame, fg_color="transparent")
            frame.pack(side="left", expand=True, fill="both", padx=5, pady=5)

            ctk.CTkLabel(frame, text=name, font=("Arial", 11)).pack()

            # Progress Bar
            bar = ctk.CTkProgressBar(frame, orientation="vertical", height=80, width=20)
            bar.set(0)
            bar.pack(pady=5)

            # Value Label
            val = ctk.CTkLabel(frame, text="0%", font=("Arial", 14, "bold"))
            val.pack()

            self.bars[name] = {'bar': bar, 'val': val}

    # --- CORE LOGIC ---
    def log(self, message):
        self.txt_console.configure(state="normal")
        self.txt_console.insert("end", "\n> " + message)
        self.txt_console.see("end")
        self.txt_console.configure(state="disabled")

    def calculate_score(self, intensity):
        if np.isnan(intensity): return 0
        fraction = intensity / 255.0
        real_temp = T_MIN + (fraction * (T_MAX - T_MIN))
        score = ((real_temp - T_MIN) / (T_MAX - T_MIN)) * 100
        return int(np.clip(score, 0, 100))

    # --- MODES ---
    def start_manual_mode(self):
        self.stop_analysis()  # Reset first
        self.mode = "MANUAL"
        self.manual_points = []
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self.is_running = True
        self.log("MANUAL MODE STARTED")
        self.log("Step 1: Click 5 points on the mask.")
        self.update_loop()

    def start_ai_mode(self):
        self.stop_analysis()
        self.mode = "AI"
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self.is_running = True
        self.log("AI MODE STARTED")
        self.log("Auto-detecting face mesh...")
        self.update_loop()

    def stop_analysis(self):
        self.is_running = False
        if self.cap: self.cap.release()

        # Show Final Report
        if self.session_scores:
            avg_score = int(sum(self.session_scores) / len(self.session_scores))
            fit_factor = 100 - avg_score
            result = "PASSED" if avg_score < LEAK_THRESHOLD_SCORE else "FAILED"

            # Popup Report
            report = f"SESSION REPORT\n\nResult: {result}\nFit Factor: {fit_factor}/100\nAvg Leak Score: {avg_score}%"
            self.log("-" * 20)
            self.log(f"FINAL RESULT: {result}")
            tk_msg = ctk.CTkLabel(self.main_frame, text=report, width=400, height=200,
                                  fg_color="#333333", corner_radius=20, font=("Arial", 20))
            tk_msg.place(relx=0.5, rely=0.5, anchor="center")

        self.session_scores = []
        self.video_label.configure(image=None)
        self.lbl_status.configure(text="STANDBY", text_color="gray")

    def on_video_click(self, event):
        if self.mode == "MANUAL" and len(self.manual_points) < 5:
            # Scale click to image logic is handled inside update loop usually,
            # but for GUI we need to match the displayed size.
            # Simple approach: We capture the click, but we only "lock" it
            # if we are in the 'Frozen' state of manual mode.
            pass  # Logic moved to Update Loop for simpler syncing

    # --- VIDEO LOOP ---
    def update_loop(self):
        if not self.is_running or not self.cap: return

        # 1. READ FRAME
        # If in Manual Mode and NOT finished clicking, keep showing the FIRST frame
        if self.mode == "MANUAL" and len(self.manual_points) < 5:
            # Create a "Pause" effect by setting frame pos to 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        else:
            # Normal Playback
            ret, frame = self.cap.read()

        if not ret:
            self.stop_analysis()
            return

        # 2. RESIZE FOR GUI (Big Screen)
        # We target a fixed display size, e.g., 800x600 for consistency
        display_w, display_h = 960, 720
        frame = cv2.resize(frame, (display_w, display_h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        worst_score = 0

        # --- PROCESSING ---

        # MODE A: MANUAL (Click Phase)
        if self.mode == "MANUAL" and len(self.manual_points) < 5:
            cv2.putText(frame, f"CLICK {5 - len(self.manual_points)} POINTS", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)
            cv2.putText(frame, f"Next: {self.point_names[len(self.manual_points)]}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Handle Click (We need to grab the mouse position relative to the label)
            # CTK doesn't give easy coordinates, so we use the global mouse position
            # and subtract the widget offset. This is tricky.
            # EASIER: We check 'ctk' pointer events.

            # Draw existing dots
            for pt in self.manual_points:
                cv2.circle(frame, pt, 8, (0, 0, 255), -1)

        # MODE B: MANUAL (Running Phase)
        elif self.mode == "MANUAL":
            for i, (mx, my) in enumerate(self.manual_points):
                name = self.point_names[i]
                roi = gray[my - 2:my + 3, mx - 2:mx + 3]
                if roi.size > 0:
                    val = self.calculate_score(np.mean(roi))
                else:
                    val = 0

                self.update_sensor(name, val, frame, (mx, my))
                if val > worst_score: worst_score = val

        # MODE C: AI AUTO
        elif self.mode == "AI":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                mesh = results.multi_face_landmarks[0]
                for i, idx in enumerate(self.ai_indices):
                    name = self.point_names[i]
                    lm = mesh.landmark[idx]
                    x, y = int(lm.x * display_w), int(lm.y * display_h)

                    # Clamp
                    x = max(2, min(x, display_w - 3))
                    y = max(2, min(y, display_h - 3))

                    roi = gray[y - 2:y + 3, x - 2:x + 3]
                    val = self.calculate_score(np.mean(roi)) if roi.size > 0 else 0

                    self.update_sensor(name, val, frame, (x, y))
                    if val > worst_score: worst_score = val
            else:
                cv2.putText(frame, "SEARCHING FOR FACE...", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Update Status Label
        if self.mode != "MANUAL" or len(self.manual_points) == 5:
            self.session_scores.append(worst_score)
            if worst_score > LEAK_THRESHOLD_SCORE:
                self.lbl_status.configure(text="LEAK DETECTED", text_color=COLOR_RED)
            else:
                self.lbl_status.configure(text="SEAL SECURE", text_color=COLOR_GREEN)

        # Convert to Image for GUI
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk

        # Recursive Call (30ms = ~30 FPS)
        self.after(30, self.update_loop)

    def update_sensor(self, name, score, frame, pos):
        # Update Data
        self.sensor_data[name].append(score)
        avg = int(sum(self.sensor_data[name]) / len(self.sensor_data[name]))

        # Update GUI Bar
        self.bars[name]['bar'].set(avg / 100)
        self.bars[name]['val'].configure(text=f"{avg}%")

        # Color Logic
        if avg > LEAK_THRESHOLD_SCORE:
            self.bars[name]['bar'].configure(progress_color=COLOR_RED)
            cv2.circle(frame, pos, 6, (0, 0, 255), -1)
        else:
            self.bars[name]['bar'].configure(progress_color=COLOR_GREEN)
            cv2.circle(frame, pos, 6, (0, 255, 0), -1)

    # Click Handling Wrapper
    def on_video_click(self, event):
        if self.mode == "MANUAL" and len(self.manual_points) < 5:
            # event.x and event.y are relative to the label
            # Since we resize the frame to match the label/window size roughly,
            # we can assume 1:1 or close enough for this demo
            self.manual_points.append((event.x, event.y))
            self.log(f"Point Locked: {event.x}, {event.y}")

            if len(self.manual_points) == 5:
                self.log("All points set. Starting Analysis...")


if __name__ == "__main__":
    app = ThermalFitApp()
    app.mainloop()