import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import threading

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\Users\Onkar\OneDrive\Desktop\Screen Recordings\without film loose elvis.mp4'

# PHYSICS (Synced with Master)
T_MIN = 22.0
T_MAX = 35.0
LEAK_THRESHOLD_SCORE = 45

# VISUAL SETTINGS
SENSING_AREA = 6  # 12x12 pixel box
DOT_RADIUS = 12  # Big dots

# COLORS
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
        self.geometry("1600x900")  # Maximized Window
        self.configure(fg_color=COLOR_BG)

        # STATE VARIABLES
        self.cap = None
        self.mode = None  # "MANUAL" or "AI"
        self.is_running = False
        self.manual_points = []
        self.session_scores = []
        self.video_width = 0
        self.video_height = 0

        # SENSOR DATA (Updated to 6 Points)
        self.point_names = [
            "Nose Bridge",
            "Left Cheek",
            "Right Cheek",
            "Left Chin",
            "Right Chin",
            "Center Chin"
        ]
        self.sensor_data = {name: deque(maxlen=10) for name in self.point_names}

        # Leak Counters for Report
        self.leak_counters = {name: 0 for name in self.point_names}
        self.total_frames = 0

        # AI SETUP
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        # AI Indices (Approximations for 6 points)
        self.ai_indices = [6, 123, 352, 172, 397, 152]

        self.setup_ui()

    def setup_ui(self):
        # GRID LAYOUT
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT SIDEBAR ---
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color=COLOR_SIDEBAR)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        # Logo
        self.lbl_title = ctk.CTkLabel(self.sidebar, text="THERMAL FIT\nANALYZER", font=("Roboto", 28, "bold"))
        self.lbl_title.pack(pady=(30, 10))

        self.lbl_subtitle = ctk.CTkLabel(self.sidebar, text="Quantitative Leak Detection", font=("Roboto", 12))
        self.lbl_subtitle.pack(pady=(0, 20))

        # Controls
        self.btn_manual = ctk.CTkButton(self.sidebar, text="🎯 STATIC SENSORS (MANUAL)", command=self.start_manual_mode,
                                        height=50, font=("Roboto", 14, "bold"), fg_color=COLOR_ACCENT)
        self.btn_manual.pack(fill="x", padx=20, pady=10)

        self.btn_ai = ctk.CTkButton(self.sidebar, text="🤖 AI TRACKING (AUTO)", command=self.start_ai_mode,
                                    height=50, font=("Roboto", 14, "bold"), fg_color="#5A2E8A")
        self.btn_ai.pack(fill="x", padx=20, pady=10)

        self.btn_stop = ctk.CTkButton(self.sidebar, text="⏹ STOP & GENERATE REPORT", command=self.stop_analysis,
                                      height=40, fg_color=COLOR_RED, hover_color="#8a1c17")
        self.btn_stop.pack(fill="x", padx=20, pady=(20, 10))

        # Console Log
        self.txt_console = ctk.CTkTextbox(self.sidebar, height=150, fg_color="#111111", text_color="#00FF00")
        self.txt_console.pack(fill="x", padx=20, pady=20)
        self.log("System Ready.\nSelect Mode to begin.")

        # Status
        self.lbl_status = ctk.CTkLabel(self.sidebar, text="STANDBY", font=("Roboto", 24, "bold"), text_color="gray")
        self.lbl_status.pack(pady=20, side="bottom")

        # --- MAIN AREA ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        # Video Screen (Auto-Resizing)
        self.video_frame = ctk.CTkFrame(self.main_frame, fg_color="black")
        self.video_frame.pack(expand=True, fill="both")

        self.video_label = ctk.CTkLabel(self.video_frame, text="", fg_color="black")
        self.video_label.pack(expand=True, fill="both")

        # Bind Click
        self.video_label.bind("<Button-1>", self.on_video_click)

        # Dashboard (Bars)
        self.dashboard_frame = ctk.CTkFrame(self.main_frame, height=140, fg_color="#222222")
        self.dashboard_frame.pack(fill="x", pady=(10, 0))

        self.bars = {}
        for i, name in enumerate(self.point_names):
            frame = ctk.CTkFrame(self.dashboard_frame, fg_color="transparent")
            frame.pack(side="left", expand=True, fill="both", padx=5, pady=5)

            ctk.CTkLabel(frame, text=name, font=("Arial", 10)).pack()

            bar = ctk.CTkProgressBar(frame, orientation="vertical", height=70, width=15)
            bar.set(0)
            bar.pack(pady=5)

            val = ctk.CTkLabel(frame, text="0%", font=("Arial", 12, "bold"))
            val.pack()

            self.bars[name] = {'bar': bar, 'val': val}

    # --- LOGIC ---
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

    def start_manual_mode(self):
        self.stop_analysis()
        self.mode = "MANUAL"
        self.manual_points = []
        self.leak_counters = {name: 0 for name in self.point_names}
        self.total_frames = 0

        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self.is_running = True

        # Get real video dimensions for scaling
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.log("MANUAL MODE STARTED")
        self.log("Click 6 Points on the mask.")
        self.update_loop()

    def start_ai_mode(self):
        self.stop_analysis()
        self.mode = "AI"
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self.is_running = True

        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.log("AI AUTO MODE STARTED")
        self.update_loop()

    def stop_analysis(self):
        self.is_running = False
        if self.cap: self.cap.release()

        # Generate Report
        if self.total_frames > 5:
            self.show_report_popup()

        self.video_label.configure(image=None)
        self.lbl_status.configure(text="STANDBY", text_color="gray")

    def show_report_popup(self):
        # Calculate Stats
        failed_locs = []
        report_text = "SESSION LEAK REPORT\n------------------\n"

        for name in self.point_names:
            leak_percent = (self.leak_counters[name] / self.total_frames) * 100
            status = "FAIL" if leak_percent > 5 else "PASS"
            if status == "FAIL": failed_locs.append(name)
            report_text += f"{name}: {status} ({int(leak_percent)}%)\n"

        avg_fit = int(sum(self.session_scores) / len(self.session_scores))
        final_score = 100 - avg_fit

        report_text += f"\nFIT FACTOR: {final_score}/100"

        # Color
        bg_color = COLOR_GREEN if not failed_locs else COLOR_RED

        # Popup Window
        top = ctk.CTkToplevel(self)
        top.geometry("400x500")
        top.title("Final Report")

        ctk.CTkLabel(top, text="CERTIFICATE OF ANALYSIS", font=("Arial", 20, "bold")).pack(pady=20)
        ctk.CTkLabel(top, text=report_text, font=("Consolas", 14), justify="left").pack(pady=10)
        ctk.CTkButton(top, text="CLOSE", command=top.destroy, fg_color=bg_color).pack(pady=20)

    # --- VIDEO LOOP ---
    def update_loop(self):
        if not self.is_running or not self.cap: return

        # 1. READ FRAME
        if self.mode == "MANUAL" and len(self.manual_points) < 6:
            # FREEZE FRAME FOR CLICKING
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        else:
            # PLAY VIDEO
            ret, frame = self.cap.read()

        if not ret:
            self.stop_analysis()
            return

        # 2. SCALING (FIT TO SCREEN)
        # Get label size to fit video perfectly
        screen_w = self.video_label.winfo_width()
        screen_h = self.video_label.winfo_height()

        # Protect against startup zero-size
        if screen_w < 100: screen_w = 800
        if screen_h < 100: screen_h = 600

        # Resize frame to fit label
        frame_resized = cv2.resize(frame, (screen_w, screen_h))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Scale Factors (To map clicks on Resized Image -> Back to Logic)
        # Actually, simpler: We do logic ON the resized image for the GUI

        worst_score = 0

        # --- MODE A: CLICKING ---
        if self.mode == "MANUAL" and len(self.manual_points) < 6:
            cv2.putText(frame_resized, f"CLICK {6 - len(self.manual_points)} POINTS", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame_resized, f"Next: {self.point_names[len(self.manual_points)]}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            for pt in self.manual_points:
                cv2.circle(frame_resized, pt, DOT_RADIUS, (0, 0, 255), -1)

        # --- MODE B: ANALYSIS ---
        elif self.mode == "MANUAL":
            for i, (mx, my) in enumerate(self.manual_points):
                name = self.point_names[i]

                # Safety Clip
                mx = np.clip(mx, SENSING_AREA, screen_w - SENSING_AREA)
                my = np.clip(my, SENSING_AREA, screen_h - SENSING_AREA)

                # Measure 12x12 Box
                roi = gray[my - SENSING_AREA:my + SENSING_AREA, mx - SENSING_AREA:mx + SENSING_AREA]

                if roi.size > 0:
                    val = self.calculate_score(np.mean(roi))
                else:
                    val = 0

                self.update_sensor(name, val, frame_resized, (mx, my))
                if val > worst_score: worst_score = val

                if val > LEAK_THRESHOLD_SCORE:
                    self.leak_counters[name] += 1

        # --- MODE C: AI ---
        elif self.mode == "AI":
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                mesh = results.multi_face_landmarks[0]
                for i, idx in enumerate(self.ai_indices):
                    name = self.point_names[i]
                    lm = mesh.landmark[idx]
                    x, y = int(lm.x * screen_w), int(lm.y * screen_h)

                    x = np.clip(x, SENSING_AREA, screen_w - SENSING_AREA)
                    y = np.clip(y, SENSING_AREA, screen_h - SENSING_AREA)

                    roi = gray[y - SENSING_AREA:y + SENSING_AREA, x - SENSING_AREA:x + SENSING_AREA]
                    val = self.calculate_score(np.mean(roi)) if roi.size > 0 else 0

                    self.update_sensor(name, val, frame_resized, (x, y))
                    if val > worst_score: worst_score = val
                    if val > LEAK_THRESHOLD_SCORE: self.leak_counters[name] += 1

        # Update Status
        if self.mode != "MANUAL" or len(self.manual_points) == 6:
            self.total_frames += 1
            self.session_scores.append(worst_score)

            if worst_score > LEAK_THRESHOLD_SCORE:
                self.lbl_status.configure(text=f"LEAK DETECTED ({worst_score}%)", text_color=COLOR_RED)
            else:
                self.lbl_status.configure(text="SEAL SECURE", text_color=COLOR_GREEN)

        # Show Image
        img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk

        self.after(30, self.update_loop)

    def update_sensor(self, name, score, frame, pos):
        self.sensor_data[name].append(score)
        avg = int(sum(self.sensor_data[name]) / len(self.sensor_data[name]))

        self.bars[name]['bar'].set(avg / 100)
        self.bars[name]['val'].configure(text=f"{avg}%")

        color = (0, 0, 255) if avg > LEAK_THRESHOLD_SCORE else (0, 255, 0)
        if avg > LEAK_THRESHOLD_SCORE:
            self.bars[name]['bar'].configure(progress_color=COLOR_RED)
        else:
            self.bars[name]['bar'].configure(progress_color=COLOR_GREEN)

        cv2.circle(frame, pos, DOT_RADIUS, color, -1)
        cv2.circle(frame, pos, 3, (255, 255, 255), -1)

    def on_video_click(self, event):
        if self.mode == "MANUAL" and len(self.manual_points) < 6:
            # event.x/y are exactly where the user clicked on the RESIZED image
            # Since we do logic on the resized image too, we can use them directly!
            self.manual_points.append((event.x, event.y))
            self.log(f"Point Locked: {event.x}, {event.y}")

            if len(self.manual_points) == 6:
                self.log("Starting Static Analysis...")
                # REWIND VIDEO LOGIC
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


if __name__ == "__main__":
    app = ThermalFitApp()
    app.mainloop()