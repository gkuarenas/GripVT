"""
Standalone camera calibration tool for GripVT.

Controls:
  Space  — capture current frame (corners must be detected to accept)
  Esc    — quit
"""
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import customtkinter as ctk
import numpy as np
import openpyxl
from PIL import Image

# ── Checkerboard constants ────────────────────────────────────────────────────
# Edit SQUARE_MM if you reprint at a different physical size.
INNER_COLS: int   = 9       # inner corners along the long axis of the board
INNER_ROWS: int   = 6       # inner corners along the short axis
SQUARE_MM:  float = 25.0    # physical size of one printed square (mm)
MIN_IMAGES: int   = 15      # minimum accepted captures to enable calibration

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE             = Path(__file__).parent
RESULTS_ROOT      = _HERE / "results"
CHECKERBOARD_PATH       = _HERE / "checkerboard_9x6_25mm_A4.png"
CHECKERBOARD_SMALL_PATH = _HERE / "checkerboard_9x6_5mm_A4.png"


# ── Checkerboard generator ────────────────────────────────────────────────────

def _generate_checkerboard() -> np.ndarray:
    """Render a 10×7 checkerboard at 300 DPI on an A4 landscape canvas."""
    DPI       = 300
    MM_PER_IN = 25.4
    canvas_w  = round(297 / MM_PER_IN * DPI)   # 3508 px
    canvas_h  = round(210 / MM_PER_IN * DPI)   # 2480 px
    sq_px     = round(SQUARE_MM / MM_PER_IN * DPI)

    n_cols_sq = INNER_COLS + 1
    n_rows_sq = INNER_ROWS + 1
    board_w   = n_cols_sq * sq_px
    board_h   = n_rows_sq * sq_px
    margin_x  = (canvas_w - board_w) // 2
    margin_y  = (canvas_h - board_h) // 2

    img = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    for row in range(n_rows_sq):
        for col in range(n_cols_sq):
            if (row + col) % 2 == 0:
                x1 = margin_x + col * sq_px
                y1 = margin_y + row * sq_px
                img[y1:y1 + sq_px, x1:x1 + sq_px] = 0
    return img


def _generate_checkerboard_small() -> np.ndarray:
    """Render a 10×7 checkerboard with 5 mm squares at 300 DPI on A4 portrait.
    Use this board when calibrating at close working distances (~44 mm)."""
    DPI       = 300
    MM_PER_IN = 25.4
    # A4 portrait: 210 × 297 mm
    canvas_w  = round(210 / MM_PER_IN * DPI)   # 2480 px
    canvas_h  = round(297 / MM_PER_IN * DPI)   # 3508 px
    sq_px     = round(5.0 / MM_PER_IN * DPI)   # ~59 px

    n_cols_sq = INNER_COLS + 1
    n_rows_sq = INNER_ROWS + 1
    board_w   = n_cols_sq * sq_px
    board_h   = n_rows_sq * sq_px
    margin_x  = (canvas_w - board_w) // 2
    margin_y  = (canvas_h - board_h) // 2

    img = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    for row in range(n_rows_sq):
        for col in range(n_cols_sq):
            if (row + col) % 2 == 0:
                x1 = margin_x + col * sq_px
                y1 = margin_y + row * sq_px
                img[y1:y1 + sq_px, x1:x1 + sq_px] = 0
    return img


# ── App ───────────────────────────────────────────────────────────────────────

class CalibrationApp(ctk.CTk):
    _FEED_W   = 800
    _FEED_H   = 450
    _FREEZE_S = 1.5     # seconds to hold the corner-overlay frame after capture
    _CTRL_W   = 360

    def __init__(self) -> None:
        super().__init__()
        self.title("GripVT — Camera Calibration")
        self.geometry("1200x680")
        self.resizable(False, False)

        # camera state
        self._cap:          cv2.VideoCapture | None = None
        self._running:      bool = False
        self._freeze_until: float = 0.0
        self._frozen_frame: np.ndarray | None = None
        self._img_size:     tuple[int, int] | None = None   # (width, height) raw

        # capture state
        self._raw_frames: list[np.ndarray] = []
        self._obj_pts:    list[np.ndarray] = []
        self._img_pts:    list[np.ndarray] = []

        # 3-D object-point template — shape (1, N, 3) required by cv2.fisheye
        self._objp = np.zeros((1, INNER_ROWS * INNER_COLS, 3), np.float32)
        self._objp[0, :, :2] = (
            np.mgrid[0:INNER_COLS, 0:INNER_ROWS].T.reshape(-1, 2) * SQUARE_MM
        )

        self._current_ctk_image: ctk.CTkImage | None = None  # holds PIL reference

        # undistortion state (populated from loaded calibration)
        self._undist_map1: np.ndarray | None = None
        self._undist_map2: np.ndarray | None = None
        self._show_undistorted: bool = False

        self._build_ui()
        self.bind("<space>", self._on_space)
        self.bind("<Escape>", lambda _: self.destroy())
        self.after(0, self._load_latest_calibration)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        # ── feed panel ────────────────────────────────────────────────────────
        feed_bg = ctk.CTkFrame(self)
        feed_bg.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)

        self._feed_label = ctk.CTkLabel(
            feed_bg, text="No feed",
            width=self._FEED_W, height=self._FEED_H,
        )
        self._feed_label.place(relx=0.5, rely=0.5, anchor="center")

        # ── control panel ─────────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(self, width=self._CTRL_W)
        ctrl.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        ctrl.grid_propagate(False)

        ctk.CTkLabel(ctrl, text="Camera Calibration", font=("", 16, "bold")).pack(
            pady=(18, 12)
        )

        # camera index
        idx_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        idx_row.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkLabel(idx_row, text="Camera index:").pack(side="left")
        self._idx_entry = ctk.CTkEntry(idx_row, width=60)
        self._idx_entry.insert(0, "0")
        self._idx_entry.pack(side="left", padx=(8, 0))

        self._open_btn = ctk.CTkButton(
            ctrl, text="Open Camera", command=self._toggle_camera
        )
        self._open_btn.pack(fill="x", padx=16, pady=(0, 8))

        self._undist_btn = ctk.CTkButton(
            ctrl, text="Show Undistorted: OFF",
            command=self._toggle_undistort, state="disabled",
            fg_color="gray30", hover_color="gray40",
        )
        self._undist_btn.pack(fill="x", padx=16, pady=(0, 12))

        ctk.CTkLabel(ctrl, text="─" * 42, text_color="gray50").pack(pady=(0, 10))

        self._count_label = ctk.CTkLabel(ctrl, text="Accepted: 0")
        self._count_label.pack()

        ctk.CTkLabel(
            ctrl, text="Press SPACE to capture", text_color="gray60"
        ).pack(pady=(4, 12))

        self._calib_btn = ctk.CTkButton(
            ctrl, text="Run Calibration",
            command=self._start_calibration,
        )
        self._calib_btn.pack(fill="x", padx=16, pady=(0, 12))

        ctk.CTkLabel(ctrl, text="─" * 42, text_color="gray50").pack(pady=(0, 6))

        ctk.CTkLabel(ctrl, text="Status", anchor="w").pack(padx=16, fill="x")
        self._log = ctk.CTkTextbox(ctrl, wrap="word", state="disabled")
        self._log.pack(fill="both", expand=True, padx=16, pady=(4, 16))

    # ── undistort toggle ──────────────────────────────────────────────────────

    def _toggle_undistort(self) -> None:
        self._show_undistorted = not self._show_undistorted
        if self._show_undistorted:
            self._undist_btn.configure(
                text="Show Undistorted: ON",
                fg_color=["#2a7fd4", "#1a6bbf"], hover_color=["#1a6bbf", "#155a9e"],
            )
        else:
            self._undist_btn.configure(
                text="Show Undistorted: OFF",
                fg_color="gray30", hover_color="gray40",
            )

    # ── camera ────────────────────────────────────────────────────────────────

    def _toggle_camera(self) -> None:
        if self._running:
            self._running = False
            if self._cap:
                self._cap.release()
                self._cap = None
            self._open_btn.configure(text="Open Camera")
            self._feed_label.configure(image=None, text="No feed")
            self._log_msg("Camera closed.")
            return

        try:
            idx = int(self._idx_entry.get())
        except ValueError:
            self._log_msg("Invalid camera index.")
            return

        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            self._log_msg(f"Could not open camera {idx}.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._cap = cap
        self._running = True
        self._open_btn.configure(text="Close Camera")
        self._log_msg(f"Camera {idx} opened at {actual_w}×{actual_h}.")
        self._update_feed()

    def _update_feed(self) -> None:
        if not self._running or self._cap is None:
            return

        if time.time() < self._freeze_until and self._frozen_frame is not None:
            self._show_frame(self._frozen_frame)
        else:
            ret, frame = self._cap.read()
            if ret:
                if self._img_size is None:
                    h, w = frame.shape[:2]
                    self._img_size = (w, h)
                display = (
                    cv2.remap(frame, self._undist_map1, self._undist_map2, cv2.INTER_LINEAR)
                    if self._show_undistorted and self._undist_map1 is not None
                    else frame
                )
                self._show_frame(display)

        self.after(33, self._update_feed)   # ~30 fps

    def _show_frame(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize(
            (self._FEED_W, self._FEED_H), Image.BILINEAR
        )
        self._current_ctk_image = ctk.CTkImage(
            light_image=pil, size=(self._FEED_W, self._FEED_H)
        )
        self._feed_label.configure(image=self._current_ctk_image, text="")

    # ── capture ───────────────────────────────────────────────────────────────

    def _on_space(self, _event=None) -> None:
        if not self._running or self._cap is None:
            return

        ret, frame = self._cap.read()
        if not ret:
            self._log_msg("Frame grab failed.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, (INNER_COLS, INNER_ROWS))
        if not found:
            self._log_msg("No corners detected — frame discarded.")
            return

        self._raw_frames.append(frame.copy())
        self._obj_pts.append(self._objp.copy())
        # fisheye expects imagePoints as (1, N, 2)
        self._img_pts.append(corners.reshape(1, -1, 2))

        n = len(self._raw_frames)
        self._count_label.configure(text=f"Accepted: {n}")

        vis = frame.copy()
        cv2.drawChessboardCorners(vis, (INNER_COLS, INNER_ROWS), corners, True)
        self._frozen_frame = vis
        self._freeze_until = time.time() + self._FREEZE_S
        self._log_msg(f"Captured #{n}")

    # ── calibration ───────────────────────────────────────────────────────────

    def _start_calibration(self) -> None:
        if not self._raw_frames:
            self._log_msg("No captured frames — press SPACE to capture first.")
            return
        self._calib_btn.configure(state="disabled", text="Calibrating…")
        self._log_msg("Running calibration…")
        threading.Thread(target=self._calibrate_thread, daemon=True).start()

    def _calibrate_thread(self) -> None:
        try:
            result = self._run_calibration()
            self.after(0, lambda: self._on_calibration_done(result))
        except Exception as exc:
            self.after(0, lambda: self._log_msg(f"ERROR: {exc}"))
            self.after(0, lambda: self._calib_btn.configure(
                state="normal", text="Run Calibration"
            ))

    def _run_calibration(self) -> dict:
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = RESULTS_ROOT / f"{ts}_calibration"
        raw_dir     = session_dir / "raw_images"
        undist_dir  = session_dir / "undist_images"
        raw_dir.mkdir(parents=True)
        undist_dir.mkdir(parents=True)

        for i, frame in enumerate(self._raw_frames):
            cv2.imwrite(str(raw_dir / f"raw_{i + 1:03d}.png"), frame)

        n_im  = len(self._raw_frames)
        K     = np.zeros((3, 3))
        D     = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n_im)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n_im)]
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            self._obj_pts, self._img_pts, self._img_size,
            K, D, rvecs, tvecs,
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW,
        )

        per_rms: list[float] = []
        for objp, imgp, rvec, tvec in zip(self._obj_pts, self._img_pts, rvecs, tvecs):
            proj, _ = cv2.fisheye.projectPoints(
                objp.reshape(-1, 1, 3), rvec.reshape(1, 3), tvec.reshape(1, 3), K, D
            )
            err = float(np.linalg.norm(
                imgp.reshape(-1, 2) - proj.reshape(-1, 2), axis=1
            ).mean())
            per_rms.append(err)

        calib = {
            "model": "fisheye",
            "K": K.tolist(),
            "D": D.tolist(),
            "image_size": list(self._img_size),
            "rms_px": round(float(rms), 6),
            "square_mm": SQUARE_MM,
            "n_images": len(self._raw_frames),
        }
        with open(session_dir / "calibration.json", "w") as f:
            json.dump(calib, f, indent=2)

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Reprojection Errors"
        ws.append(["image", "rms_px"])
        for i, err in enumerate(per_rms):
            ws.append([f"raw_{i + 1:03d}.png", round(err, 6)])
        ws.append(["OVERALL", round(float(rms), 6)])
        wb.save(session_dir / "reprojection_errors.xlsx")

        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, self._img_size, np.eye(3), balance=0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K_new, self._img_size, cv2.CV_16SC2
        )
        for i, frame in enumerate(self._raw_frames):
            undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            cv2.imwrite(str(undist_dir / f"undist_{i + 1:03d}.png"), undist)

        return {
            "rms": float(rms),
            "session_dir": str(session_dir),
            "n": len(self._raw_frames),
        }

    def _on_calibration_done(self, result: dict) -> None:
        rms = result["rms"]
        quality = (
            "Excellent" if rms < 0.5 else
            "Acceptable" if rms < 1.0 else
            "Poor — consider retaking images"
        )
        self._log_msg(f"RMS: {rms:.4f} px — {quality}")
        self._log_msg(f"Images used: {result['n']}")
        self._log_msg(f"Saved to: {result['session_dir']}")
        self._calib_btn.configure(state="normal", text="Run Calibration")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_latest_calibration(self) -> None:
        if not RESULTS_ROOT.exists():
            self._log_msg("No results folder found yet.")
            return

        sessions = sorted(
            [d for d in RESULTS_ROOT.iterdir()
             if d.is_dir() and d.name.endswith("_calibration")],
            key=lambda d: d.name,
            reverse=True,
        )
        if not sessions:
            self._log_msg("No previous calibration found.")
            return

        calib_path = sessions[0] / "calibration.json"
        if not calib_path.exists():
            self._log_msg(f"Latest session has no calibration.json: {sessions[0].name}")
            return

        with open(calib_path) as f:
            data = json.load(f)

        K      = data["K"]
        D      = np.array(data["D"]).flatten().tolist()
        rms    = data.get("rms_px", float("nan"))
        n      = data.get("n_images", "?")
        w, h   = data.get("image_size", ["?", "?"])
        fx, fy = K[0][0], K[1][1]
        cx, cy = K[0][2], K[1][2]

        quality = (
            "Excellent" if rms < 0.5 else
            "Acceptable" if rms < 1.0 else
            "Poor"
        )

        # build remap maps so the live feed can optionally show undistorted output
        # alpha=0: new camera matrix is scaled so the remapped image has no black
        # borders — the FOV shrinks slightly but the full frame is valid pixels.
        K_arr    = np.array(data["K"], dtype=float)
        D_arr    = np.array(data["D"], dtype=float)
        img_size = tuple(data["image_size"])   # (width, height)
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_arr, D_arr, img_size, np.eye(3), balance=0
        )
        self._undist_map1, self._undist_map2 = cv2.fisheye.initUndistortRectifyMap(
            K_arr, D_arr, np.eye(3), K_new, img_size, cv2.CV_16SC2
        )
        self._undist_btn.configure(state="normal")

        self._log_msg("── Last calibration ──────────────")
        self._log_msg(f"Session : {sessions[0].name}")
        self._log_msg(f"Images  : {n}    Resolution: {w}×{h} px")
        self._log_msg(f"RMS     : {rms:.4f} px  ({quality})")
        self._log_msg(f"fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
        self._log_msg("D (k1 k2 k3 k4):")
        self._log_msg("  " + "  ".join(f"{v:+.6f}" for v in D))
        self._log_msg("Undistort preview: ready")
        self._log_msg("──────────────────────────────────")

    def _log_msg(self, msg: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def destroy(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()
        super().destroy()


def main() -> None:
    if not CHECKERBOARD_PATH.exists():
        img = _generate_checkerboard()
        pil = Image.fromarray(img)
        pil.save(str(CHECKERBOARD_PATH), dpi=(300, 300))
        print(f"Checkerboard saved → {CHECKERBOARD_PATH}")

    if not CHECKERBOARD_SMALL_PATH.exists():
        img = _generate_checkerboard_small()
        pil = Image.fromarray(img)
        pil.save(str(CHECKERBOARD_SMALL_PATH), dpi=(300, 300))
        print(f"Small checkerboard saved → {CHECKERBOARD_SMALL_PATH}")

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    CalibrationApp().mainloop()


if __name__ == "__main__":
    main()
