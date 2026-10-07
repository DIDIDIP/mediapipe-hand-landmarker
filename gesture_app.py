"""Custom gesture trainer GUI: collect -> train -> recognize, all in one window.

Usage:
    python gesture_app.py
"""
import csv
import threading
import time
import tkinter as tk
from collections import Counter, deque
from tkinter import messagebox, scrolledtext, ttk

import cv2
import joblib
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont, ImageTk

from gesture_common import (CLF_PATH, DATA_PATH, NUM_FEATURES, create_landmarker,
                            draw_hands, landmarks_to_features)
from train_gestures import train

COUNTDOWN_SEC = 3
UI_FONT = ("Malgun Gothic", 10)
HEADER = ["label"] + [f"f{i}" for i in range(NUM_FEATURES)]


def load_font(size):
    try:
        return ImageFont.truetype("malgun.ttf", size)  # supports Korean labels
    except OSError:
        return ImageFont.load_default(size)


def read_rows():
    if not DATA_PATH.exists():
        return []
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return list(reader)


class GestureApp:
    def __init__(self, root):
        self.root = root
        root.title("Custom Gesture Trainer")
        root.option_add("*Font", UI_FONT)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.font_big = load_font(48)
        self.font_mid = load_font(28)
        self.font_small = load_font(20)

        self.counts = Counter(row[0] for row in read_rows())
        self.labels = sorted(self.counts)
        self.recording = None        # label being recorded
        self.record_start = 0.0      # time recording actually starts (after countdown)
        self.remaining = 0
        self.buffer = []             # samples not yet written to CSV
        self.frame_idx = 0
        self.clf = joblib.load(CLF_PATH) if CLF_PATH.exists() else None
        self.history = {}
        self.training = False

        self.cap = None
        self.landmarker = create_landmarker(num_hands=2)
        self.t0 = time.monotonic()
        self.last_ts = -1

        self.build_ui()
        self.open_camera(0)
        self.refresh_label_list()
        root.bind("<space>", self.on_space)
        self.loop()

    # ---------------------------------------------------------------- UI
    def build_ui(self):
        self.video = ttk.Label(self.root)
        self.video.grid(row=0, column=0, padx=8, pady=8, sticky="n")

        side = ttk.Frame(self.root)
        side.grid(row=0, column=1, padx=(0, 8), pady=8, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        cam = ttk.Frame(side)
        cam.pack(fill="x", pady=(0, 6))
        ttk.Label(cam, text="카메라 번호").pack(side="left")
        self.cam_var = tk.IntVar(value=0)
        ttk.Spinbox(cam, from_=0, to=9, width=4, textvariable=self.cam_var).pack(side="left", padx=4)
        ttk.Button(cam, text="변경", command=lambda: self.open_camera(self.cam_var.get())).pack(side="left")

        self.tabs = ttk.Notebook(side)
        self.tabs.pack(fill="both", expand=True)
        self.tabs.add(self.build_collect_tab(), text=" 1. 데이터 수집 ")
        self.tabs.add(self.build_train_tab(), text=" 2. 학습 ")
        self.tabs.add(self.build_infer_tab(), text=" 3. 실시간 인식 ")
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self.stop_recording())

        self.status = tk.StringVar()
        ttk.Label(side, textvariable=self.status, foreground="gray").pack(fill="x", pady=(6, 0))

    def build_collect_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)

        add = ttk.Frame(tab)
        add.pack(fill="x")
        self.new_label = tk.StringVar()
        entry = ttk.Entry(add, textvariable=self.new_label)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda e: self.add_label())
        ttk.Button(add, text="제스처 추가", command=self.add_label).pack(side="left", padx=(4, 0))

        self.tree = ttk.Treeview(tab, columns=("count",), height=8, selectmode="browse")
        self.tree.heading("#0", text="제스처")
        self.tree.heading("count", text="샘플 수")
        self.tree.column("#0", width=180)
        self.tree.column("count", width=80, anchor="e")
        self.tree.pack(fill="both", expand=True, pady=8)

        opts = ttk.Frame(tab)
        opts.pack(fill="x")
        ttk.Label(opts, text="한 번에 녹화할 개수").pack(side="left")
        self.samples_var = tk.IntVar(value=200)
        ttk.Spinbox(opts, from_=20, to=2000, increment=20, width=6,
                    textvariable=self.samples_var).pack(side="left", padx=4)

        self.rec_btn = ttk.Button(tab, text="● 녹화 시작  (Space)", command=self.toggle_recording)
        self.rec_btn.pack(fill="x", pady=(8, 4))
        self.progress = ttk.Progressbar(tab, maximum=1)
        self.progress.pack(fill="x")

        ttk.Button(tab, text="선택한 제스처 데이터 삭제", command=self.delete_label).pack(fill="x", pady=(12, 0))
        ttk.Label(tab, wraplength=280, foreground="gray", justify="left", text=(
            "팁: 녹화 중에 손을 조금씩 돌리고, 앞뒤로 움직이고, 기울여 주세요.\n"
            "'none'(아무 제스처 아님)도 하나 모아 두면 오인식이 줄어요.")).pack(fill="x", pady=(12, 0))
        return tab

    def build_train_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)
        self.train_btn = ttk.Button(tab, text="▶ 학습 시작", command=self.start_training)
        self.train_btn.pack(fill="x")
        self.train_bar = ttk.Progressbar(tab)
        self.train_bar.pack(fill="x", pady=6)
        self.log = scrolledtext.ScrolledText(tab, width=58, height=22, font=("Consolas", 9),
                                             wrap="none")
        self.log.pack(fill="both", expand=True)
        return tab

    def build_infer_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)
        self.result_var = tk.StringVar(value="-")
        ttk.Label(tab, textvariable=self.result_var, font=("Malgun Gothic", 28, "bold"),
                  anchor="center").pack(fill="x", pady=(0, 8))

        self.threshold = tk.DoubleVar(value=0.7)
        self.smooth = tk.DoubleVar(value=5)
        self.add_slider(tab, "확신도 기준 (낮으면 '?')", self.threshold, 0.3, 0.99, "{:.2f}")
        self.add_slider(tab, "흔들림 보정 (프레임 수)", self.smooth, 1, 15, "{:.0f}")

        ttk.Label(tab, text="제스처별 확률 (첫 번째 손)").pack(anchor="w", pady=(10, 2))
        self.proba_frame = ttk.Frame(tab)
        self.proba_frame.pack(fill="x")
        self.proba_bars = {}
        self.model_msg = ttk.Label(tab, foreground="gray", wraplength=280)
        self.model_msg.pack(fill="x", pady=(10, 0))
        self.rebuild_proba_bars()
        return tab

    def add_slider(self, parent, text, var, lo, hi, fmt):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        value = ttk.Label(row, width=5, anchor="e")
        ttk.Label(row, text=text).pack(side="top", anchor="w")

        def on_move(_=None):
            value.config(text=fmt.format(var.get()))

        scale = ttk.Scale(row, from_=lo, to=hi, variable=var, command=on_move)
        scale.pack(side="left", fill="x", expand=True)
        value.pack(side="left")
        on_move()

    def rebuild_proba_bars(self):
        for child in self.proba_frame.winfo_children():
            child.destroy()
        self.proba_bars = {}
        if self.clf is None:
            self.model_msg.config(text="학습된 모델이 없어요. 2. 학습 탭에서 먼저 학습하세요.")
            return
        self.model_msg.config(text="")
        for name in self.clf.classes_:
            row = ttk.Frame(self.proba_frame)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=str(name), width=12).pack(side="left")
            bar = ttk.Progressbar(row, maximum=1)
            bar.pack(side="left", fill="x", expand=True)
            self.proba_bars[str(name)] = bar

    # ---------------------------------------------------------- camera
    def open_camera(self, index):
        if self.cap is not None:
            self.cap.release()
        # The first open sometimes fails right after another app released the camera
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_DSHOW):
            self.cap = cv2.VideoCapture(index, backend)
            if self.cap.isOpened():
                break
            self.cap.release()
            time.sleep(0.3)
        if not self.cap.isOpened():
            self.status.set(f"카메라 {index}번을 열 수 없어요.")
        else:
            self.status.set(f"카메라 {index}번 사용 중")

    def loop(self):
        ok, frame = (self.cap.read() if self.cap is not None else (False, None))
        if ok:
            self.process(cv2.flip(frame, 1))
        self.root.after(10, self.loop)

    def process(self, frame):
        self.frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int((time.monotonic() - self.t0) * 1000), self.last_ts + 1)
        self.last_ts = ts
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        draw_hands(frame, result)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        tab = self.tabs.index(self.tabs.select())
        if tab == 0:
            self.collect_step(result, draw, img.size)
        elif tab == 2:
            self.infer_step(result, draw, img.size)

        self.photo = ImageTk.PhotoImage(img)
        self.video.config(image=self.photo)

    @staticmethod
    def text(draw, xy, msg, font, color, anchor="la"):
        draw.text(xy, msg, font=font, fill=color, anchor=anchor,
                  stroke_width=3, stroke_fill="black")

    # ---------------------------------------------------------- collect
    def refresh_label_list(self, select=None):
        select = select or self.selected_label()
        self.tree.delete(*self.tree.get_children())
        for name in self.labels:
            self.tree.insert("", "end", iid=name, text=name, values=(self.counts[name],))
        if select in self.labels:
            self.tree.selection_set(select)
        elif self.labels:
            self.tree.selection_set(self.labels[0])

    def selected_label(self):
        sel = self.tree.selection() if hasattr(self, "tree") else ()
        return sel[0] if sel else None

    def add_label(self):
        name = self.new_label.get().strip()
        if not name:
            return
        if "," in name or '"' in name:
            messagebox.showwarning("이름 오류", "제스처 이름에 쉼표(,)나 따옴표는 쓸 수 없어요.")
            return
        if name not in self.labels:
            self.labels.append(name)
        self.new_label.set("")
        self.refresh_label_list(select=name)

    def delete_label(self):
        name = self.selected_label()
        if not name or self.recording:
            return
        if not messagebox.askyesno("삭제 확인", f"'{name}' 제스처의 샘플 {self.counts[name]}개를 모두 삭제할까요?"):
            return
        rows = [r for r in read_rows() if r[0] != name]
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DATA_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerows(rows)
        self.counts.pop(name, None)
        self.labels.remove(name)
        self.refresh_label_list()
        self.status.set(f"'{name}' 삭제 완료")

    def on_space(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Spinbox, tk.Text)):
            return
        if self.tabs.index(self.tabs.select()) == 0:
            self.toggle_recording()

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
            return
        name = self.selected_label()
        if not name:
            messagebox.showinfo("제스처 선택", "먼저 제스처를 추가하고 목록에서 선택하세요.")
            return
        try:
            self.remaining = max(1, self.samples_var.get())
        except tk.TclError:
            self.remaining = 200
            self.samples_var.set(200)
        self.recording = name
        self.progress.config(maximum=self.remaining, value=0)
        self.record_start = time.monotonic() + COUNTDOWN_SEC
        self.rec_btn.config(text="■ 녹화 중지  (Space)")

    def stop_recording(self):
        if self.buffer:
            new_file = not DATA_PATH.exists()
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            with DATA_PATH.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(HEADER)
                writer.writerows(self.buffer)
            self.status.set(f"'{self.recording}' 샘플 {len(self.buffer)}개 저장")
            self.buffer = []
        self.recording = None
        self.rec_btn.config(text="● 녹화 시작  (Space)")

    def collect_step(self, result, draw, size):
        w, h = size
        if not self.recording:
            self.text(draw, (10, 10), "제스처를 선택하고 Space로 녹화", self.font_small, "white")
            return

        wait = self.record_start - time.monotonic()
        if wait > 0:
            self.text(draw, (w // 2, h // 2), str(int(wait) + 1), self.font_big, "yellow", "mm")
            self.text(draw, (w // 2, h // 2 + 50), f"'{self.recording}' 준비", self.font_mid, "yellow", "mm")
            return

        if result.hand_landmarks and self.frame_idx % 2 == 0:
            feats = landmarks_to_features(result.hand_landmarks[0],
                                          result.handedness[0][0].category_name)
            self.buffer.append([self.recording] + [f"{v:.5f}" for v in feats])
            self.counts[self.recording] += 1
            self.remaining -= 1
            self.progress.step(1)
            self.tree.set(self.recording, "count", self.counts[self.recording])

        self.text(draw, (10, 10), f"● REC  {self.recording}  ({self.remaining} 남음)",
                  self.font_small, "red")
        if not result.hand_landmarks:
            self.text(draw, (10, 40), "손이 안 보여요!", self.font_small, "red")
        if self.remaining <= 0:
            self.stop_recording()

    # ------------------------------------------------------------ train
    def start_training(self):
        if self.training:
            return
        self.training = True
        self.train_btn.config(state="disabled")
        self.train_bar.config(mode="indeterminate")
        self.train_bar.start(10)
        self.log.delete("1.0", "end")
        self.log.insert("end", "학습 중...\n")
        threading.Thread(target=self.train_worker, daemon=True).start()

    def train_worker(self):
        try:
            report, ok = train(), True
        except Exception as e:  # show any error in the log instead of crashing
            report, ok = f"오류: {e}", False
        self.root.after(0, self.train_done, report, ok)

    def train_done(self, report, ok):
        self.training = False
        self.train_bar.stop()
        self.train_bar.config(mode="determinate", value=0)
        self.train_btn.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("end", report)
        if ok:
            self.clf = joblib.load(CLF_PATH)
            self.history.clear()
            self.rebuild_proba_bars()
            self.status.set("학습 완료! 3. 실시간 인식 탭에서 확인하세요.")

    # ------------------------------------------------------------ infer
    def infer_step(self, result, draw, size):
        if self.clf is None:
            self.text(draw, (10, 10), "학습된 모델이 없어요", self.font_small, "red")
            return
        w, h = size
        shown = []
        for i, (landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
            side = handedness[0].category_name
            proba = self.clf.predict_proba([landmarks_to_features(landmarks, side)])[0]
            best = proba.argmax()
            name = str(self.clf.classes_[best]) if proba[best] >= self.threshold.get() else "?"

            hist = self.history.setdefault(side, deque(maxlen=15))
            hist.append(name)
            recent = list(hist)[-max(1, round(self.smooth.get())):]
            voted = Counter(recent).most_common(1)[0][0]
            shown.append(voted)

            x = int(min(lm.x for lm in landmarks) * w)
            y = min(int(max(lm.y for lm in landmarks) * h) + 10, h - 40)
            self.text(draw, (x, y), f"{voted} {proba[best]:.2f}", self.font_mid, "lime")

            if i == 0:
                for cls, p in zip(self.clf.classes_, proba):
                    if str(cls) in self.proba_bars:
                        self.proba_bars[str(cls)].config(value=float(p))

        self.result_var.set("  /  ".join(shown) if shown else "-")
        if not shown:
            for bar in self.proba_bars.values():
                bar.config(value=0)

    # ------------------------------------------------------------ close
    def on_close(self):
        self.stop_recording()
        if self.cap is not None:
            self.cap.release()
        self.landmarker.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    GestureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
