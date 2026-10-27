#!/usr/bin/env python3
"""
Video Tool - Multi-tab video processing utility (PyQt6)
Tabs:
  1. Video Merge
  2. Video Convert
  3. VFold Merge
  4. Video Crop   (interactive crop box + aspect ratio lock)
"""

import os
import sys
import subprocess
import threading
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QRect, QPoint, QSize, QTimer, QUrl
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QAction, QDragEnterEvent, QDropEvent
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
    QRadioButton, QButtonGroup, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTextEdit, QProgressBar, QSplitter, QFrame, QCheckBox,
    QFormLayout, QSizePolicy, QAbstractItemView, QScrollArea
)

# Optional OpenCV for frame extraction (fallback to ffmpeg if missing)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ---------------------------------------------------------------------------
# Portable paths (works with source + PyInstaller)
# ---------------------------------------------------------------------------
def app_dir() -> Path:
    """Folder of the .exe (frozen) or of this script (dev)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_DIR = app_dir()
LOG_DIR = APP_DIR / "logs"
DATA_DIR = APP_DIR / "data"
for _d in (LOG_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv",
                    ".m4v", ".ts", ".mts", ".m2ts", ".3gp", ".mpg", ".mpeg"}

DEFAULT_VIDEOS_DIR = Path.home() / "Videos"
if not DEFAULT_VIDEOS_DIR.exists():
    DEFAULT_VIDEOS_DIR = APP_DIR / "videos"
    DEFAULT_VIDEOS_DIR.mkdir(exist_ok=True)

ASPECT_PRESETS = {
    "Free": None,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "4:5": 4 / 5,
    "21:9": 21 / 9,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def collect_videos_from_paths(paths: list) -> list[Path]:
    videos = []
    for p in paths:
        path = Path(p)
        if path.is_file() and is_video(path):
            videos.append(path.resolve())
        elif path.is_dir():
            for f in sorted(path.rglob("*")):
                if is_video(f):
                    videos.append(f.resolve())
    seen = set()
    unique = []
    for v in videos:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


class CancelToken:
    """Shared stop flag: 'none' | 'after' | 'now'."""
    def __init__(self):
        self._state = "none"   # none / after / now
        self._lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None

    def request_after(self):
        with self._lock:
            if self._state == "none":
                self._state = "after"

    def request_now(self):
        with self._lock:
            self._state = "now"
        # Kill running ffmpeg ASAP
        proc = self.process
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
            except Exception:
                pass

    def should_stop_now(self) -> bool:
        with self._lock:
            return self._state == "now"

    def should_stop_after(self) -> bool:
        with self._lock:
            return self._state in ("after", "now")

    def reset(self):
        with self._lock:
            self._state = "none"
            self.process = None


def run_ffmpeg(cmd: list[str], log_callback=None, cancel: Optional[CancelToken] = None) -> tuple[bool, str]:
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if cancel is not None:
            cancel.process = process
        output_lines = []
        for line in process.stdout:
            if cancel and cancel.should_stop_now():
                try:
                    process.terminate()
                except Exception:
                    pass
                return False, "Cancelled by user"
            line = line.rstrip()
            output_lines.append(line)
            if log_callback:
                log_callback(line)
        process.wait()
        if cancel and cancel.should_stop_now():
            return False, "Cancelled by user"
        if process.returncode == 0:
            return True, "Success"
        return False, "\n".join(output_lines[-40:])
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Dual file logger for bulk jobs  (lives in APP_DIR/logs – portable)
# ---------------------------------------------------------------------------
class JobLogger:
    """
    Writes two log files under APP_DIR/logs:
      - full_<job>_<stamp>.log
      - summary_<job>_<stamp>.txt   (Time, Status, File, Error)
    """

    def __init__(self, job_name: str = "job"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_name)
        self.full_path = LOG_DIR / f"full_{safe}_{stamp}.log"
        self.summary_path = LOG_DIR / f"summary_{safe}_{stamp}.txt"
        self._full = open(self.full_path, "w", encoding="utf-8")
        self._summary = open(self.summary_path, "w", encoding="utf-8")
        self._summary.write("Time\tStatus\tFile\tError\n")
        self._full.write(f"=== Video Tool job: {job_name}  started {datetime.now().isoformat()} ===\n")
        self._full.write(f"App dir  : {APP_DIR}\n")
        self._full.write(f"Full log : {self.full_path}\n")
        self._full.write(f"Summary  : {self.summary_path}\n\n")
        self._full.flush()

    def line(self, text: str):
        self._full.write(text + ("" if text.endswith("\n") else "\n"))
        self._full.flush()

    def result(self, path: Path | str, ok: bool, error: str = "", status: str | None = None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st = status if status else ("OK" if ok else "FAILED")
        err = (error or "").replace("\n", " ").replace("\t", " ")[:300]
        self._summary.write(f"{ts}\t{st}\t{path}\t{err}\n")
        self._summary.flush()
        self.line(f"[RESULT] {st}: {path}" + (f"  → {err}" if err else ""))

    def close(self):
        self._full.write(f"\n=== finished {datetime.now().isoformat()} ===\n")
        self._full.close()
        self._summary.close()

    def paths_message(self) -> str:
        return f"Logs saved:\n  Full   : {self.full_path}\n  Summary: {self.summary_path}"


def probe_has_audio(path: Path) -> bool:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


def get_video_size(path: Path) -> tuple[int, int]:
    """Return (width, height) of first video stream."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        parts = r.stdout.strip().split(",")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 0, 0


def get_video_duration(path: Path) -> float:
    """Return duration in seconds (0.0 on failure)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def get_video_fps(path: Path) -> float:
    """Return frames-per-second of first video stream (25.0 fallback)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        rate = r.stdout.strip()
        if "/" in rate:
            num, den = rate.split("/", 1)
            den_f = float(den)
            if den_f:
                return float(num) / den_f
        return float(rate) if rate else 25.0
    except Exception:
        return 25.0


def extract_frame(path: Path, time_sec: float = 0.5) -> Optional[QImage]:
    """Extract a single frame as QImage. Prefer OpenCV, fallback to ffmpeg."""
    if HAS_CV2:
        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return None
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_no = int(time_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            return QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        except Exception:
            pass

    # Fallback: ffmpeg → temp png
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    try:
        cmd = [
            "ffmpeg", "-y", "-ss", str(time_sec), "-i", str(path),
            "-frames:v", "1", "-q:v", "2", tmp.name
        ]
        ok, _ = run_ffmpeg(cmd)
        if ok and Path(tmp.name).exists():
            img = QImage(tmp.name)
            if not img.isNull():
                return img
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
class FFmpegWorker(QThread):
    """Runs an ffmpeg command (or any callable) off the GUI thread."""
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(bool, str)   # success, message

    def __init__(self, cmd: list[str] | None = None, func=None,
                 cancel: Optional[CancelToken] = None, parent=None):
        super().__init__(parent)
        self.cmd = cmd
        self.func = func   # callable(log_cb) -> (ok, msg)
        self.cancel = cancel

    def run(self):
        try:
            if self.func is not None:
                ok, msg = self.func(lambda line: self.log.emit(line))
            else:
                ok, msg = run_ffmpeg(self.cmd, lambda line: self.log.emit(line),
                                    cancel=self.cancel)
            self.finished_ok.emit(ok, msg)
        except Exception as e:
            self.finished_ok.emit(False, str(e))

    def request_stop_now(self):
        if self.cancel:
            self.cancel.request_now()

    def request_stop_after(self):
        if self.cancel:
            self.cancel.request_after()


# ---------------------------------------------------------------------------
# Interactive Crop Widget
# ---------------------------------------------------------------------------
class CropView(QLabel):
    """Shows a video frame and an interactive crop rectangle with aspect lock."""
    crop_changed = pyqtSignal(int, int, int, int)  # x, y, w, h in source pixels

    HANDLE = 10  # handle size in widget pixels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#1e1e1e; border:1px solid #444;")
        self.setMouseTracking(True)

        self._source: Optional[QImage] = None
        self._src_w = 0
        self._src_h = 0
        self._scale = 1.0          # source → widget scale
        self._offset = QPoint(0, 0)  # top-left of image inside widget

        # Crop rect in *source* coordinates
        self._crop = QRect()
        self._aspect: Optional[float] = None  # None = free

        self._drag_mode = None   # None | "move" | "n" | "s" | "e" | "w" | "nw" ...
        self._drag_start = QPoint()
        self._crop_start = QRect()

    def set_image(self, image: QImage):
        self._source = image
        self._src_w = image.width()
        self._src_h = image.height()
        # Default crop = full frame
        self._crop = QRect(0, 0, self._src_w, self._src_h)
        self._update_pixmap()
        self.crop_changed.emit(self._crop.x(), self._crop.y(),
                               self._crop.width(), self._crop.height())

    def set_aspect(self, ratio: Optional[float]):
        self._aspect = ratio
        if ratio and not self._crop.isNull():
            self._apply_aspect_to_crop()
            self.update()
            self.crop_changed.emit(self._crop.x(), self._crop.y(),
                                   self._crop.width(), self._crop.height())

    def crop_rect(self) -> QRect:
        return QRect(self._crop)

    def _apply_aspect_to_crop(self):
        if not self._aspect or self._crop.isNull():
            return
        cx = self._crop.center().x()
        cy = self._crop.center().y()
        w = self._crop.width()
        h = int(w / self._aspect)
        if h > self._src_h:
            h = self._src_h
            w = int(h * self._aspect)
        if w > self._src_w:
            w = self._src_w
            h = int(w / self._aspect)
        x = max(0, min(cx - w // 2, self._src_w - w))
        y = max(0, min(cy - h // 2, self._src_h - h))
        self._crop = QRect(x, y, w, h)

    def _update_pixmap(self):
        if self._source is None:
            self.clear()
            return
        # Fit image into widget keeping aspect
        ww, wh = self.width(), self.height()
        if ww < 10 or wh < 10:
            return
        sx = ww / self._src_w
        sy = wh / self._src_h
        self._scale = min(sx, sy)
        disp_w = int(self._src_w * self._scale)
        disp_h = int(self._src_h * self._scale)
        self._offset = QPoint((ww - disp_w) // 2, (wh - disp_h) // 2)
        scaled = self._source.scaled(disp_w, disp_h,
                                     Qt.AspectRatioMode.IgnoreAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(scaled))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def _src_to_widget(self, p: QPoint) -> QPoint:
        return QPoint(int(p.x() * self._scale) + self._offset.x(),
                      int(p.y() * self._scale) + self._offset.y())

    def _widget_to_src(self, p: QPoint) -> QPoint:
        x = int((p.x() - self._offset.x()) / self._scale)
        y = int((p.y() - self._offset.y()) / self._scale)
        return QPoint(max(0, min(x, self._src_w)), max(0, min(y, self._src_h)))

    def _crop_widget_rect(self) -> QRect:
        tl = self._src_to_widget(self._crop.topLeft())
        br = self._src_to_widget(self._crop.bottomRight())
        return QRect(tl, br).normalized()

    def _hit_test(self, pos: QPoint) -> Optional[str]:
        r = self._crop_widget_rect()
        hs = self.HANDLE
        # corners first
        if QRect(r.left() - hs//2, r.top() - hs//2, hs, hs).contains(pos):
            return "nw"
        if QRect(r.right() - hs//2, r.top() - hs//2, hs, hs).contains(pos):
            return "ne"
        if QRect(r.left() - hs//2, r.bottom() - hs//2, hs, hs).contains(pos):
            return "sw"
        if QRect(r.right() - hs//2, r.bottom() - hs//2, hs, hs).contains(pos):
            return "se"
        # edges
        if QRect(r.left() + hs, r.top() - hs//2, r.width() - 2*hs, hs).contains(pos):
            return "n"
        if QRect(r.left() + hs, r.bottom() - hs//2, r.width() - 2*hs, hs).contains(pos):
            return "s"
        if QRect(r.left() - hs//2, r.top() + hs, hs, r.height() - 2*hs).contains(pos):
            return "w"
        if QRect(r.right() - hs//2, r.top() + hs, hs, r.height() - 2*hs).contains(pos):
            return "e"
        if r.contains(pos):
            return "move"
        return None

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self._source is None:
            return
        mode = self._hit_test(event.position().toPoint())
        if mode:
            self._drag_mode = mode
            self._drag_start = event.position().toPoint()
            self._crop_start = QRect(self._crop)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._drag_mode is None:
            mode = self._hit_test(pos)
            cursors = {
                "nw": Qt.CursorShape.SizeFDiagCursor,
                "se": Qt.CursorShape.SizeFDiagCursor,
                "ne": Qt.CursorShape.SizeBDiagCursor,
                "sw": Qt.CursorShape.SizeBDiagCursor,
                "n": Qt.CursorShape.SizeVerCursor,
                "s": Qt.CursorShape.SizeVerCursor,
                "e": Qt.CursorShape.SizeHorCursor,
                "w": Qt.CursorShape.SizeHorCursor,
                "move": Qt.CursorShape.SizeAllCursor,
            }
            self.setCursor(cursors.get(mode, Qt.CursorShape.ArrowCursor))
            return

        delta = pos - self._drag_start
        dx = int(delta.x() / self._scale)
        dy = int(delta.y() / self._scale)
        r = QRect(self._crop_start)

        if self._drag_mode == "move":
            r.translate(dx, dy)
            # clamp
            if r.left() < 0:
                r.moveLeft(0)
            if r.top() < 0:
                r.moveTop(0)
            if r.right() > self._src_w:
                r.moveRight(self._src_w)
            if r.bottom() > self._src_h:
                r.moveBottom(self._src_h)
        else:
            # resize
            if "n" in self._drag_mode:
                r.setTop(r.top() + dy)
            if "s" in self._drag_mode:
                r.setBottom(r.bottom() + dy)
            if "w" in self._drag_mode:
                r.setLeft(r.left() + dx)
            if "e" in self._drag_mode:
                r.setRight(r.right() + dx)

            r = r.normalized()
            # enforce min size
            if r.width() < 16:
                r.setWidth(16)
            if r.height() < 16:
                r.setHeight(16)

            # aspect lock
            if self._aspect:
                if self._drag_mode in ("e", "w"):
                    new_h = int(r.width() / self._aspect)
                    cy = r.center().y()
                    r.setHeight(new_h)
                    r.moveCenter(QPoint(r.center().x(), cy))
                elif self._drag_mode in ("n", "s"):
                    new_w = int(r.height() * self._aspect)
                    cx = r.center().x()
                    r.setWidth(new_w)
                    r.moveCenter(QPoint(cx, r.center().y()))
                else:  # corner
                    # keep the opposite corner fixed, adjust to aspect
                    if self._drag_mode in ("se", "ne"):
                        r.setWidth(int(r.height() * self._aspect))
                    else:
                        r.setHeight(int(r.width() / self._aspect))

            # clamp to source
            r = r.intersected(QRect(0, 0, self._src_w, self._src_h))

        self._crop = r
        self.update()
        self.crop_changed.emit(self._crop.x(), self._crop.y(),
                               self._crop.width(), self._crop.height())

    def mouseReleaseEvent(self, event):
        self._drag_mode = None

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._source is None or self._crop.isNull():
            return
        painter = QPainter(self)
        r = self._crop_widget_rect()

        # Dim outside crop
        painter.setBrush(QColor(0, 0, 0, 140))
        painter.setPen(Qt.PenStyle.NoPen)
        # top
        painter.drawRect(0, 0, self.width(), r.top())
        # bottom
        painter.drawRect(0, r.bottom(), self.width(), self.height() - r.bottom())
        # left
        painter.drawRect(0, r.top(), r.left(), r.height())
        # right
        painter.drawRect(r.right(), r.top(), self.width() - r.right(), r.height())

        # Crop border
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(0, 200, 255), 2))
        painter.drawRect(r)

        # Handles
        painter.setBrush(QColor(0, 200, 255))
        hs = self.HANDLE
        for pt in (r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight(),
                   QPoint(r.center().x(), r.top()), QPoint(r.center().x(), r.bottom()),
                   QPoint(r.left(), r.center().y()), QPoint(r.right(), r.center().y())):
            painter.drawRect(pt.x() - hs//2, pt.y() - hs//2, hs, hs)

        # Size label
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Segoe UI", 9))
        label = f"{self._crop.width()} × {self._crop.height()}"
        painter.drawText(r.adjusted(6, 6, -6, -6), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, label)
        painter.end()


# ---------------------------------------------------------------------------
# Base tab helpers
# ---------------------------------------------------------------------------
class DropListWidget(QListWidget):
    """List widget that accepts file/folder drops."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path:
                    self.addItem(path)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class LogBox(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 9))
        self.setMaximumHeight(160)

    def append_line(self, text: str):
        self.append(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


# ---------------------------------------------------------------------------
# Tab: Video Merge
# ---------------------------------------------------------------------------
class MergeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: Optional[FFmpegWorker] = None
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        # Input
        grp = QGroupBox("Input Videos / Folders (drag & drop or Add)")
        v = QVBoxLayout(grp)
        self.list = DropListWidget()
        v.addWidget(self.list)
        btn_row = QHBoxLayout()
        btn_row.addWidget(QPushButton("Add Files…", clicked=self._add_files))
        btn_row.addWidget(QPushButton("Add Folder…", clicked=self._add_folder))
        btn_row.addWidget(QPushButton("Remove Selected", clicked=self._remove))
        btn_row.addWidget(QPushButton("Clear All", clicked=self._clear))
        btn_row.addStretch()
        v.addLayout(btn_row)
        layout.addWidget(grp)

        # Options
        opt = QGroupBox("Options")
        form = QFormLayout(opt)
        self.mode_fast = QRadioButton("Fast Merge (stream copy – same codec required)")
        self.mode_full = QRadioButton("Full Encode Merge (re-encode – works with mixed codecs)")
        self.mode_fast.setChecked(True)
        form.addRow(self.mode_fast)
        form.addRow(self.mode_full)
        self.out_name = QLineEdit()
        self.out_name.setPlaceholderText("optional – auto name if empty")
        form.addRow("Output name:", self.out_name)
        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("empty → Videos folder")
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse_dir(self.out_dir))
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.out_dir)
        dir_row.addWidget(browse)
        form.addRow("Output folder:", dir_row)
        layout.addWidget(opt)

        btn_row = QHBoxLayout()
        self.btn = QPushButton("▶  Start Merge")
        self.btn.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn.clicked.connect(self._start)
        self.btn_stop_now = QPushButton("Stop ASAP")
        self.btn_stop_now.setEnabled(False)
        self.btn_stop_now.setStyleSheet("color:#c00;")
        self.btn_stop_now.clicked.connect(self._stop_now)
        btn_row.addWidget(self.btn)
        btn_row.addWidget(self.btn_stop_now)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.log = LogBox()
        layout.addWidget(self.log)
        self._cancel: Optional[CancelToken] = None

    def _stop_now(self):
        if self._cancel:
            self._cancel.request_now()
            self.log.append_line("… Stopping ASAP")

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select videos", "",
            "Video files (*" + " *".join(VIDEO_EXTENSIONS) + ");;All files (*.*)"
        )
        for f in files:
            self.list.addItem(f)

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            self.list.addItem(d)

    def _remove(self):
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def _clear(self):
        self.list.clear()

    def _browse_dir(self, line: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            line.setText(d)

    def _start(self):
        items = [self.list.item(i).text() for i in range(self.list.count())]
        if not items:
            QMessageBox.warning(self, "No input", "Add at least one video or folder.")
            return
        videos = collect_videos_from_paths(items)
        if len(videos) < 2:
            QMessageBox.warning(self, "Not enough videos",
                                f"Found only {len(videos)} video(s). Need at least 2.")
            return

        mode = "fast" if self.mode_fast.isChecked() else "full"
        out_name = self.out_name.text().strip()
        out_dir = self.out_dir.text().strip()
        out_path = Path(out_dir) if out_dir else DEFAULT_VIDEOS_DIR
        out_path.mkdir(parents=True, exist_ok=True)

        if not out_name:
            out_name = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        if not out_name.lower().endswith((".mp4", ".mkv", ".mov")):
            out_name += ".mp4"
        output = out_path / out_name

        if output.exists():
            if QMessageBox.question(self, "Overwrite?", f"{output} already exists. Overwrite?") != QMessageBox.StandardButton.Yes:
                return

        self._cancel = CancelToken()
        self.btn.setEnabled(False)
        self.btn_stop_now.setEnabled(True)
        self.log.clear()
        self.log.append_line(f"Starting {mode} merge of {len(videos)} videos…\n→ {output}\n")

        self._merge_output = output
        cancel = self._cancel
        self._worker = FFmpegWorker(
            func=lambda log_cb: do_merge(videos, output, mode, log_callback=log_cb, cancel=cancel),
            cancel=cancel,
            parent=self
        )
        self._worker.log.connect(self.log.append_line)
        self._worker.finished_ok.connect(self._merge_finished)
        self._worker.start()

    def _merge_finished(self, ok: bool, msg: str):
        self.btn.setEnabled(True)
        self.btn_stop_now.setEnabled(False)
        self._cancel = None
        if ok:
            self.log.append_line(f"\n✓ Done → {self._merge_output}\n")
            QMessageBox.information(self, "Success", f"Merged video saved to:\n{self._merge_output}")
        else:
            self.log.append_line(f"\n✗ Failed:\n{msg}\n")
            if "Cancelled" in msg:
                QMessageBox.information(self, "Cancelled", "Merge was cancelled.")
            else:
                QMessageBox.critical(self, "Merge Failed", msg[:800])


def do_merge(videos: list[Path], output: Path, mode: str, log_callback=None,
             cancel: Optional[CancelToken] = None) -> tuple[bool, str]:
    """Shared merge implementation used by Merge and VFold tabs."""
    if cancel and cancel.should_stop_now():
        return False, "Cancelled by user"
    if mode == "fast":
        tmpd = tempfile.mkdtemp(prefix="vmerge_")
        try:
            list_file = Path(tmpd) / "concat.txt"
            with open(list_file, "w", encoding="utf-8") as fh:
                for v in videos:
                    path_str = v.resolve().as_posix().replace("'", r"'\''")
                    fh.write(f"file '{path_str}'\n")
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-c", "copy", str(output)
            ]
            return run_ffmpeg(cmd, log_callback, cancel=cancel)
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)

    # ----- Full Encode: normalize resolution / fps / audio then concat -----
    # Pick target size = largest width among inputs (keeps quality of the biggest clip)
    sizes = [get_video_size(v) for v in videos]
    target_w, target_h = 0, 0
    for w, h in sizes:
        if w * h > target_w * target_h:
            target_w, target_h = w, h
    if target_w == 0:
        target_w, target_h = 1920, 1080
    # Ensure even
    target_w -= target_w % 2
    target_h -= target_h % 2

    if log_callback:
        log_callback(f"Full encode target resolution: {target_w}x{target_h}")

    n = len(videos)
    inputs: list[str] = []
    for v in videos:
        inputs.extend(["-i", str(v)])

    # Build filter: scale+pad each video, normalize audio, then concat
    # Also handle clips that may lack audio by generating silence (duration matched later by concat)
    has_audio = [probe_has_audio(v) for v in videos]
    any_audio = any(has_audio)

    filter_parts = []
    for i in range(n):
        # Video: scale to fit inside target, pad to exact size, force SAR=1, set fps
        filter_parts.append(
            f"[{i}:v:0]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v{i}];"
        )
        if any_audio:
            if has_audio[i]:
                filter_parts.append(
                    f"[{i}:a:0]aformat=sample_rates=48000:channel_layouts=stereo,"
                    f"aresample=async=1:first_pts=0[a{i}];"
                )
            else:
                # Silent audio – duration will be driven by the video stream in concat
                filter_parts.append(
                    f"anullsrc=channel_layout=stereo:sample_rate=48000[a{i}];"
                )

    if any_audio:
        concat_in = "".join(f"[v{i}][a{i}]" for i in range(n))
        filter_parts.append(f"{concat_in}concat=n={n}:v=1:a=1[v][a]")
        map_args = ["-map", "[v]", "-map", "[a]"]
        audio_codec = ["-c:a", "aac", "-b:a", "128k"]
    else:
        concat_in = "".join(f"[v{i}]" for i in range(n))
        filter_parts.append(f"{concat_in}concat=n={n}:v=1:a=0[v]")
        map_args = ["-map", "[v]"]
        audio_codec = []

    filter_str = "".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_str,
        *map_args,
        "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-threads", "4",
        *audio_codec,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output)
    ]
    return run_ffmpeg(cmd, log_callback, cancel=cancel)


# ---------------------------------------------------------------------------
# Tab: Video Convert
# ---------------------------------------------------------------------------
class ConvertTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        grp = QGroupBox("Input Videos / Folders (drag & drop or Add)")
        v = QVBoxLayout(grp)
        self.list = DropListWidget()
        v.addWidget(self.list)
        btn_row = QHBoxLayout()
        btn_row.addWidget(QPushButton("Add Files…", clicked=self._add_files))
        btn_row.addWidget(QPushButton("Add Folder…", clicked=self._add_folder))
        btn_row.addWidget(QPushButton("Remove Selected", clicked=self._remove))
        btn_row.addWidget(QPushButton("Clear All", clicked=self._clear))
        btn_row.addStretch()
        v.addLayout(btn_row)
        layout.addWidget(grp)

        opt = QGroupBox("Conversion Options")
        form = QFormLayout(opt)

        self.fmt = QComboBox()
        self.fmt.addItems(["mp4", "mkv", "mov", "webm", "avi"])
        form.addRow("Output format:", self.fmt)

        self.res = QComboBox()
        self.res.addItems(["original", "3840x2160", "2560x1440", "1920x1080",
                           "1280x720", "854x480", "640x360", "custom"])
        form.addRow("Resolution:", self.res)

        self.custom_res = QLineEdit()
        self.custom_res.setPlaceholderText("e.g. 1280x720")
        form.addRow("Custom (W×H):", self.custom_res)

        self.scale_fit = QRadioButton("Fit (keep aspect, pad if needed)")
        self.scale_stretch = QRadioButton("Stretch (ignore aspect)")
        self.scale_fit.setChecked(True)
        form.addRow(self.scale_fit)
        form.addRow(self.scale_stretch)

        self.crf = QSpinBox()
        self.crf.setRange(15, 35)
        self.crf.setValue(23)
        form.addRow("Quality (CRF):", self.crf)

        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("empty → Videos folder")
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse_dir(self.out_dir))
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.out_dir)
        dir_row.addWidget(browse)
        form.addRow("Output folder:", dir_row)
        layout.addWidget(opt)

        btn_row = QHBoxLayout()
        self.btn = QPushButton("▶  Start Convert")
        self.btn.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn.clicked.connect(self._start)
        self.btn_stop_after = QPushButton("Stop after current")
        self.btn_stop_after.setEnabled(False)
        self.btn_stop_after.clicked.connect(self._stop_after)
        self.btn_stop_now = QPushButton("Stop ASAP")
        self.btn_stop_now.setEnabled(False)
        self.btn_stop_now.setStyleSheet("color:#c00;")
        self.btn_stop_now.clicked.connect(self._stop_now)
        btn_row.addWidget(self.btn)
        btn_row.addWidget(self.btn_stop_after)
        btn_row.addWidget(self.btn_stop_now)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.log = LogBox()
        layout.addWidget(self.log)
        self._cancel: Optional[CancelToken] = None

    def _stop_after(self):
        if self._cancel:
            self._cancel.request_after()
            self.log.append_line("… Will stop after current video finishes")

    def _stop_now(self):
        if self._cancel:
            self._cancel.request_now()
            self.log.append_line("… Stopping ASAP (killing ffmpeg)")

    def _set_running(self, running: bool):
        self.btn.setEnabled(not running)
        self.btn_stop_after.setEnabled(running)
        self.btn_stop_now.setEnabled(running)

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select videos", "",
            "Video files (*" + " *".join(VIDEO_EXTENSIONS) + ");;All files (*.*)"
        )
        for f in files:
            self.list.addItem(f)

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            self.list.addItem(d)

    def _remove(self):
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def _clear(self):
        self.list.clear()

    def _browse_dir(self, line: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            line.setText(d)

    def _start(self):
        items = [self.list.item(i).text() for i in range(self.list.count())]
        if not items:
            QMessageBox.warning(self, "No input", "Add at least one video or folder.")
            return
        videos = collect_videos_from_paths(items)
        if not videos:
            QMessageBox.warning(self, "No videos", "No video files found.")
            return

        fmt = self.fmt.currentText()
        res = self.res.currentText()
        custom = self.custom_res.text().strip()
        scale_mode = "fit" if self.scale_fit.isChecked() else "stretch"
        crf = self.crf.value()
        out_dir = self.out_dir.text().strip()
        out_path = Path(out_dir) if out_dir else DEFAULT_VIDEOS_DIR
        out_path.mkdir(parents=True, exist_ok=True)

        vf = None
        if res != "original":
            if res == "custom":
                if "x" not in custom.lower():
                    QMessageBox.critical(self, "Invalid", "Custom resolution must be like 1280x720")
                    return
                w, h = custom.lower().split("x")
            else:
                w, h = res.split("x")
            w, h = w.strip(), h.strip()
            if scale_mode == "stretch":
                vf = f"scale={w}:{h}"
            else:
                vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"

        self._cancel = CancelToken()
        self._set_running(True)
        self.log.clear()
        self.log.append_line(f"Converting {len(videos)} video(s) → {fmt}  (CRF={crf})\n")

        cancel = self._cancel

        def convert_job(log_cb):
            jlog = JobLogger("convert")
            def both(line):
                jlog.line(line)
                log_cb(line)

            both(f"Converting {len(videos)} video(s) → {fmt}  CRF={crf}")
            both(f"Output folder: {out_path}\n")

            ok_count = 0
            processed = 0
            for i, vid in enumerate(videos, 1):
                if cancel.should_stop_after() and processed > 0:
                    both(f"Stopped by user after {processed} file(s).")
                    break
                if cancel.should_stop_now():
                    both("Stopped ASAP by user.")
                    break

                stem = vid.stem
                out_file = out_path / f"{stem}_converted.{fmt}"
                c = 1
                while out_file.exists():
                    out_file = out_path / f"{stem}_converted_{c}.{fmt}"
                    c += 1
                both(f"[{i}/{len(videos)}] {vid.name} → {out_file.name}")

                if fmt == "webm":
                    cmd = ["ffmpeg", "-y", "-i", str(vid)]
                    if vf:
                        cmd += ["-vf", vf]
                    cmd += ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(crf),
                            "-c:a", "libopus", "-b:a", "128k", str(out_file)]
                else:
                    cmd = ["ffmpeg", "-y", "-i", str(vid)]
                    if vf:
                        cmd += ["-vf", vf]
                    cmd += [
                        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-threads", "4",
                        "-c:a", "aac", "-b:a", "128k",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(out_file)
                    ]
                ok, msg = run_ffmpeg(cmd, both, cancel=cancel)
                processed += 1
                if ok:
                    ok_count += 1
                    jlog.result(vid, True)
                else:
                    both(f"  FAILED: {msg[:200]}")
                    jlog.result(vid, False, msg)
                    if cancel.should_stop_now():
                        break

            jlog.close()
            return True, f"{ok_count}/{processed}|{jlog.full_path}|{jlog.summary_path}"

        self._conv_total = len(videos)
        self._worker = FFmpegWorker(func=convert_job, cancel=cancel, parent=self)
        self._worker.log.connect(self.log.append_line)
        self._worker.finished_ok.connect(self._convert_finished)
        self._worker.start()

    def _convert_finished(self, ok: bool, msg: str):
        self._set_running(False)
        self._cancel = None
        # msg format: "3/5|/path/full.log|/path/summary.txt"
        parts = msg.split("|")
        count_part = parts[0] if parts else msg
        full_log = parts[1] if len(parts) > 1 else ""
        summary_log = parts[2] if len(parts) > 2 else ""

        self.log.append_line(f"\nFinished: {count_part} succeeded")
        if full_log:
            self.log.append_line(f"Full log   : {full_log}")
            self.log.append_line(f"Summary log: {summary_log}")

        try:
            ok_count = int(count_part.split("/")[0])
            total = int(count_part.split("/")[1])
        except Exception:
            ok_count, total = 0, self._conv_total

        extra = f"\n\nSummary log (failed files):\n{summary_log}" if summary_log else ""
        if ok_count == total:
            QMessageBox.information(self, "Success", f"All {ok_count} video(s) converted.{extra}")
        else:
            QMessageBox.warning(self, "Partial",
                                f"{ok_count} succeeded, {total - ok_count} failed.{extra}")


# ---------------------------------------------------------------------------
# Tab: VFold Merge
# ---------------------------------------------------------------------------
class VFoldTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        info = QLabel("Select a parent folder that contains subfolders.\n"
                      "Each subfolder’s videos will be merged into <subfoldername>_merge.mp4")
        layout.addWidget(info)

        grp = QGroupBox("Parent Folder")
        h = QHBoxLayout(grp)
        self.path = QLineEdit()
        self.path.setPlaceholderText("Drop a folder here or Browse…")
        self.path.setAcceptDrops(True)
        # simple drop support
        self.path.installEventFilter(self)
        h.addWidget(self.path)
        h.addWidget(QPushButton("Browse…", clicked=self._browse))
        h.addWidget(QPushButton("Scan", clicked=self._scan))
        layout.addWidget(grp)

        self.tree = QListWidget()
        self.tree.setMinimumHeight(120)
        layout.addWidget(QLabel("Detected subfolders with videos:"))
        layout.addWidget(self.tree)

        opt = QGroupBox("Options")
        form = QFormLayout(opt)
        self.mode_fast = QRadioButton("Fast Merge (stream copy)")
        self.mode_full = QRadioButton("Full Encode Merge")
        self.mode_fast.setChecked(True)
        form.addRow(self.mode_fast)
        form.addRow(self.mode_full)
        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("empty → same as parent folder")
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse_dir(self.out_dir))
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.out_dir)
        dir_row.addWidget(browse)
        form.addRow("Output folder:", dir_row)
        layout.addWidget(opt)

        # Skip already-done option
        self.skip_existing = QCheckBox("Skip subfolder if <name>_merge.mp4 already exists")
        self.skip_existing.setChecked(True)
        layout.addWidget(self.skip_existing)

        btn_row = QHBoxLayout()
        self.btn = QPushButton("▶  Start VFold Merge")
        self.btn.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn.clicked.connect(self._start)
        self.btn_stop_after = QPushButton("Stop after current folder")
        self.btn_stop_after.setEnabled(False)
        self.btn_stop_after.clicked.connect(self._stop_after)
        self.btn_stop_now = QPushButton("Stop ASAP")
        self.btn_stop_now.setEnabled(False)
        self.btn_stop_now.setStyleSheet("color:#c00;")
        self.btn_stop_now.clicked.connect(self._stop_now)
        btn_row.addWidget(self.btn)
        btn_row.addWidget(self.btn_stop_after)
        btn_row.addWidget(self.btn_stop_now)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.log = LogBox()
        layout.addWidget(self.log)
        self._cancel: Optional[CancelToken] = None

    def _stop_after(self):
        if self._cancel:
            self._cancel.request_after()
            self.log.append_line("… Will stop after current subfolder finishes")

    def _stop_now(self):
        if self._cancel:
            self._cancel.request_now()
            self.log.append_line("… Stopping ASAP (killing ffmpeg)")

    def _set_running(self, running: bool):
        self.btn.setEnabled(not running)
        self.btn_stop_after.setEnabled(running)
        self.btn_stop_now.setEnabled(running)

    def eventFilter(self, obj, event):
        if obj is self.path and event.type() == event.Type.DragEnter:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return True
        if obj is self.path and event.type() == event.Type.Drop:
            urls = event.mimeData().urls()
            if urls:
                p = urls[0].toLocalFile()
                if Path(p).is_dir():
                    self.path.setText(p)
                    self._scan()
            return True
        return super().eventFilter(obj, event)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select parent folder")
        if d:
            self.path.setText(d)
            self._scan()

    def _browse_dir(self, line: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            line.setText(d)

    def _scan(self):
        self.tree.clear()
        parent = Path(self.path.text().strip())
        if not parent.is_dir():
            return
        for sub in sorted(parent.iterdir()):
            if sub.is_dir():
                vids = [f for f in sorted(sub.iterdir()) if is_video(f)]
                if vids:
                    self.tree.addItem(f"{sub.name}  ({len(vids)} videos)")
                    self.tree.item(self.tree.count()-1).setData(Qt.ItemDataRole.UserRole, str(sub))

    def _start(self):
        parent = Path(self.path.text().strip())
        if not parent.is_dir():
            QMessageBox.warning(self, "No folder", "Select a valid parent folder.")
            return
        if self.tree.count() == 0:
            QMessageBox.warning(self, "Nothing", "No subfolders with videos. Click Scan.")
            return

        mode = "fast" if self.mode_fast.isChecked() else "full"
        out_dir = self.out_dir.text().strip()
        out_base = Path(out_dir) if out_dir else parent
        out_base.mkdir(parents=True, exist_ok=True)
        skip_existing = self.skip_existing.isChecked()

        self._cancel = CancelToken()
        self._set_running(True)
        self.log.clear()
        self.log.append_line(f"Starting VFold merge ({mode})…\n")
        if skip_existing:
            self.log.append_line("Skip existing outputs: ON\n")

        # Capture list of subfolders on main thread
        subs = []
        for i in range(self.tree.count()):
            item = self.tree.item(i)
            subs.append(Path(item.data(Qt.ItemDataRole.UserRole)))

        cancel = self._cancel

        def vfold_job(log_cb):
            jlog = JobLogger("vfold")
            def both(line):
                jlog.line(line)
                log_cb(line)

            both(f"Starting VFold merge ({mode})")
            both(f"Parent / output base: {out_base}")
            both(f"Skip existing: {skip_existing}\n")

            ok_count = 0
            skipped = 0
            processed = 0
            total = len(subs)
            for sub in subs:
                if cancel.should_stop_after() and processed > 0:
                    both(f"Stopped by user after {processed} subfolder(s).")
                    break
                if cancel.should_stop_now():
                    both("Stopped ASAP by user.")
                    break

                vids = [f for f in sorted(sub.iterdir()) if is_video(f)]
                # Exclude already-produced merge files inside the subfolder
                vids = [v for v in vids if not v.name.endswith("_merge.mp4")]
                if not vids:
                    both(f"  SKIP (no videos): {sub.name}")
                    jlog.result(sub, False, "No video files found in subfolder", status="SKIPPED")
                    skipped += 1
                    continue

                out_name = f"{sub.name}_merge.mp4"
                output = out_base / out_name

                if skip_existing and output.exists():
                    both(f"  SKIP (already exists): {out_name}")
                    jlog.result(sub, True, "Output already exists", status="SKIPPED")
                    skipped += 1
                    ok_count += 1  # treat as done
                    continue

                both(f"  Merging {sub.name} ({len(vids)} videos) → {out_name}")

                if len(vids) == 1:
                    cmd = ["ffmpeg", "-y", "-i", str(vids[0]), "-c", "copy", str(output)]
                    ok, msg = run_ffmpeg(cmd, both, cancel=cancel)
                else:
                    ok, msg = do_merge(vids, output, mode, log_callback=both, cancel=cancel)

                processed += 1
                if ok:
                    ok_count += 1
                    both(f"    ✓ {sub.name}")
                    jlog.result(sub, True)
                else:
                    both(f"    ✗ {sub.name}: {msg[:150]}")
                    jlog.result(sub, False, msg)
                    if cancel.should_stop_now():
                        break

            jlog.close()
            return True, f"{ok_count}/{total}|{jlog.full_path}|{jlog.summary_path}|{skipped}"

        self._vfold_total = len(subs)
        self._worker = FFmpegWorker(func=vfold_job, cancel=cancel, parent=self)
        self._worker.log.connect(self.log.append_line)
        self._worker.finished_ok.connect(self._vfold_finished)
        self._worker.start()

    def _vfold_finished(self, ok: bool, msg: str):
        self._set_running(False)
        self._cancel = None
        parts = msg.split("|")
        count_part = parts[0] if parts else msg
        full_log = parts[1] if len(parts) > 1 else ""
        summary_log = parts[2] if len(parts) > 2 else ""
        skipped = parts[3] if len(parts) > 3 else "0"

        self.log.append_line(f"\nFinished: {count_part}  (skipped: {skipped})")
        if full_log:
            self.log.append_line(f"Full log   : {full_log}")
            self.log.append_line(f"Summary log: {summary_log}")

        try:
            ok_count = int(count_part.split("/")[0])
            total = int(count_part.split("/")[1])
        except Exception:
            ok_count, total = 0, self._vfold_total

        extra = f"\n\nSummary log (failed/skipped):\n{summary_log}" if summary_log else ""
        if ok_count == total:
            QMessageBox.information(self, "Success", f"All {ok_count} subfolder(s) merged.{extra}")
        else:
            QMessageBox.warning(self, "Partial",
                                f"{ok_count} succeeded, {total - ok_count} failed/skipped.{extra}")


# ---------------------------------------------------------------------------
# Tab: Video Crop
# ---------------------------------------------------------------------------
class CropTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path: Optional[Path] = None
        self.src_w = 0
        self.src_h = 0
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        # Top: file select
        top = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select or drop a video…")
        self.path_edit.setAcceptDrops(True)
        self.path_edit.installEventFilter(self)
        top.addWidget(self.path_edit)
        top.addWidget(QPushButton("Browse…", clicked=self._browse))
        top.addWidget(QPushButton("Load Preview", clicked=self._load_preview))
        layout.addLayout(top)

        # Main area: crop view + controls
        split = QHBoxLayout()

        self.crop_view = CropView()
        self.crop_view.setMinimumSize(480, 360)
        self.crop_view.crop_changed.connect(self._on_crop_changed)
        split.addWidget(self.crop_view, stretch=3)

        # Right controls
        controls = QVBoxLayout()
        asp_grp = QGroupBox("Aspect Ratio")
        asp_l = QVBoxLayout(asp_grp)
        self.aspect_combo = QComboBox()
        for name in ASPECT_PRESETS:
            self.aspect_combo.addItem(name)
        self.aspect_combo.currentTextChanged.connect(self._on_aspect_changed)
        asp_l.addWidget(self.aspect_combo)
        controls.addWidget(asp_grp)

        geo_grp = QGroupBox("Crop Geometry (source pixels)")
        form = QFormLayout(geo_grp)
        self.spin_x = QSpinBox(); self.spin_x.setRange(0, 99999)
        self.spin_y = QSpinBox(); self.spin_y.setRange(0, 99999)
        self.spin_w = QSpinBox(); self.spin_w.setRange(16, 99999)
        self.spin_h = QSpinBox(); self.spin_h.setRange(16, 99999)
        for s in (self.spin_x, self.spin_y, self.spin_w, self.spin_h):
            s.valueChanged.connect(self._on_spin_changed)
        form.addRow("X:", self.spin_x)
        form.addRow("Y:", self.spin_y)
        form.addRow("Width:", self.spin_w)
        form.addRow("Height:", self.spin_h)
        controls.addWidget(geo_grp)

        out_grp = QGroupBox("Output")
        form2 = QFormLayout(out_grp)
        self.out_name = QLineEdit()
        self.out_name.setPlaceholderText("optional")
        form2.addRow("Name:", self.out_name)
        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("empty → Videos folder")
        browse = QPushButton("…")
        browse.clicked.connect(lambda: self._browse_dir(self.out_dir))
        row = QHBoxLayout()
        row.addWidget(self.out_dir)
        row.addWidget(browse)
        form2.addRow("Folder:", row)
        controls.addWidget(out_grp)

        self.btn = QPushButton("▶  Crop Video")
        self.btn.setStyleSheet("font-weight:bold; padding:8px;")
        self.btn.clicked.connect(self._start_crop)
        controls.addWidget(self.btn)
        self.btn_stop_now = QPushButton("Stop ASAP")
        self.btn_stop_now.setEnabled(False)
        self.btn_stop_now.setStyleSheet("color:#c00;")
        self.btn_stop_now.clicked.connect(self._stop_now)
        controls.addWidget(self.btn_stop_now)
        controls.addStretch()

        right = QWidget()
        right.setLayout(controls)
        right.setFixedWidth(240)
        split.addWidget(right)
        layout.addLayout(split)

        self.log = LogBox()
        layout.addWidget(self.log)

        self._updating_spins = False
        self._cancel: Optional[CancelToken] = None

    def _stop_now(self):
        if self._cancel:
            self._cancel.request_now()
            self.log.append_line("… Stopping ASAP")

    def eventFilter(self, obj, event):
        if obj is self.path_edit:
            if event.type() == event.Type.DragEnter and event.mimeData().hasUrls():
                event.acceptProposedAction()
                return True
            if event.type() == event.Type.Drop:
                urls = event.mimeData().urls()
                if urls:
                    p = urls[0].toLocalFile()
                    if is_video(Path(p)):
                        self.path_edit.setText(p)
                        self._load_preview()
                return True
        return super().eventFilter(obj, event)

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select video", "",
            "Video files (*" + " *".join(VIDEO_EXTENSIONS) + ");;All files (*.*)"
        )
        if f:
            self.path_edit.setText(f)
            self._load_preview()

    def _browse_dir(self, line: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            line.setText(d)

    def _load_preview(self):
        path = Path(self.path_edit.text().strip())
        if not is_video(path):
            QMessageBox.warning(self, "Invalid", "Please select a valid video file.")
            return
        self.video_path = path
        self.log.append_line(f"Loading preview: {path.name}")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            img = extract_frame(path, time_sec=1.0)
            if img is None or img.isNull():
                QMessageBox.critical(self, "Error", "Could not extract a frame from the video.")
                return
            self.src_w, self.src_h = img.width(), img.height()
            # Also try ffprobe for exact size (in case of SAR etc.)
            pw, ph = get_video_size(path)
            if pw and ph:
                self.src_w, self.src_h = pw, ph
            self.crop_view.set_image(img)
            self.spin_x.setMaximum(self.src_w)
            self.spin_y.setMaximum(self.src_h)
            self.spin_w.setMaximum(self.src_w)
            self.spin_h.setMaximum(self.src_h)
            self.log.append_line(f"  Size: {self.src_w}×{self.src_h}")
        finally:
            QApplication.restoreOverrideCursor()

    def _on_aspect_changed(self, name: str):
        ratio = ASPECT_PRESETS.get(name)
        self.crop_view.set_aspect(ratio)

    def _on_crop_changed(self, x, y, w, h):
        self._updating_spins = True
        self.spin_x.setValue(x)
        self.spin_y.setValue(y)
        self.spin_w.setValue(w)
        self.spin_h.setValue(h)
        self._updating_spins = False

    def _on_spin_changed(self):
        if self._updating_spins or self.video_path is None:
            return
        # Push spin values into crop view
        x = self.spin_x.value()
        y = self.spin_y.value()
        w = self.spin_w.value()
        h = self.spin_h.value()
        # Simple set – CropView will clamp on next interaction
        self.crop_view._crop = QRect(x, y, w, h)
        self.crop_view.update()

    def _start_crop(self):
        if self.video_path is None:
            QMessageBox.warning(self, "No video", "Load a video first.")
            return
        r = self.crop_view.crop_rect()
        if r.width() < 16 or r.height() < 16:
            QMessageBox.warning(self, "Too small", "Crop region is too small.")
            return

        # Make even dimensions (required by most codecs)
        x, y, w, h = r.x(), r.y(), r.width(), r.height()
        w -= w % 2
        h -= h % 2
        if w < 16 or h < 16:
            QMessageBox.warning(self, "Too small", "Crop region too small after even adjustment.")
            return

        out_name = self.out_name.text().strip()
        out_dir = self.out_dir.text().strip()
        out_path = Path(out_dir) if out_dir else DEFAULT_VIDEOS_DIR
        out_path.mkdir(parents=True, exist_ok=True)
        if not out_name:
            out_name = f"{self.video_path.stem}_crop.mp4"
        if not out_name.lower().endswith((".mp4", ".mkv", ".mov")):
            out_name += ".mp4"
        output = out_path / out_name

        if output.exists():
            if QMessageBox.question(self, "Overwrite?", f"{output} exists. Overwrite?") != QMessageBox.StandardButton.Yes:
                return

        self._cancel = CancelToken()
        self.btn.setEnabled(False)
        self.btn_stop_now.setEnabled(True)
        self.log.clear()
        self.log.append_line(f"Cropping {self.video_path.name}")
        self.log.append_line(f"  Region: x={x} y={y} w={w} h={h}")
        self.log.append_line(f"  → {output}\n")

        # crop filter: crop=w:h:x:y
        cmd = [
            "ffmpeg", "-y", "-i", str(self.video_path),
            "-vf", f"crop={w}:{h}:{x}:{y}",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-threads", "4",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output)
        ]

        self._crop_output = output
        self._worker = FFmpegWorker(cmd=cmd, cancel=self._cancel, parent=self)
        self._worker.log.connect(self.log.append_line)
        self._worker.finished_ok.connect(self._crop_finished)
        self._worker.start()

    def _crop_finished(self, ok: bool, msg: str):
        self.btn.setEnabled(True)
        self.btn_stop_now.setEnabled(False)
        self._cancel = None
        if ok:
            self.log.append_line(f"\n✓ Done → {self._crop_output}")
            QMessageBox.information(self, "Success", f"Cropped video saved to:\n{self._crop_output}")
        else:
            self.log.append_line(f"\n✗ Failed:\n{msg}")
            QMessageBox.critical(self, "Crop Failed", msg[:800])


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Tab: Video Convert-S  (Convert + segment by time / frames)
# ---------------------------------------------------------------------------
class ConvertSTab(QWidget):
    """Same as Video Convert, plus extract a segment by time or frames
    (start / middle / end / custom)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        grp = QGroupBox("Input Videos / Folders (drag & drop or Add)")
        v = QVBoxLayout(grp)
        self.list = DropListWidget()
        v.addWidget(self.list)
        btn_row = QHBoxLayout()
        btn_row.addWidget(QPushButton("Add Files…", clicked=self._add_files))
        btn_row.addWidget(QPushButton("Add Folder…", clicked=self._add_folder))
        btn_row.addWidget(QPushButton("Remove Selected", clicked=self._remove))
        btn_row.addWidget(QPushButton("Clear All", clicked=self._clear))
        btn_row.addStretch()
        v.addLayout(btn_row)
        layout.addWidget(grp)

        # --- Segment options ---
        seg = QGroupBox("Segment (extract part of video)")
        seg_form = QFormLayout(seg)

        self.seg_mode = QComboBox()
        self.seg_mode.addItems([
            "Full video (no trim)",
            "Start of video",
            "Middle of video",
            "End of video",
            "Custom time (seconds)",
            "Custom frames",
        ])
        self.seg_mode.currentIndexChanged.connect(self._on_seg_mode)
        seg_form.addRow("Mode:", self.seg_mode)

        self.seg_duration = QDoubleSpinBox()
        self.seg_duration.setRange(0.1, 86400.0)
        self.seg_duration.setDecimals(2)
        self.seg_duration.setValue(5.0)
        self.seg_duration.setSuffix(" s")
        seg_form.addRow("Length (for Start/Middle/End):", self.seg_duration)

        self.seg_start = QDoubleSpinBox()
        self.seg_start.setRange(0.0, 86400.0)
        self.seg_start.setDecimals(3)
        self.seg_start.setValue(0.0)
        self.seg_start.setSuffix(" s")
        seg_form.addRow("Start time (custom time):", self.seg_start)

        self.seg_end = QDoubleSpinBox()
        self.seg_end.setRange(0.0, 86400.0)
        self.seg_end.setDecimals(3)
        self.seg_end.setValue(5.0)
        self.seg_end.setSuffix(" s")
        seg_form.addRow("End time (custom time, 0=use length):", self.seg_end)

        self.seg_frame_start = QSpinBox()
        self.seg_frame_start.setRange(0, 99999999)
        self.seg_frame_start.setValue(0)
        seg_form.addRow("Start frame (custom frames):", self.seg_frame_start)

        self.seg_frame_count = QSpinBox()
        self.seg_frame_count.setRange(1, 99999999)
        self.seg_frame_count.setValue(150)
        seg_form.addRow("Frame count (custom frames):", self.seg_frame_count)

        self.seg_accurate = QCheckBox("Frame-accurate seek (slower, better for short clips)")
        self.seg_accurate.setChecked(True)
        seg_form.addRow(self.seg_accurate)

        layout.addWidget(seg)
        self._on_seg_mode()  # set initial enable state

        # --- Same convert options as Convert tab ---
        opt = QGroupBox("Conversion Options")
        form = QFormLayout(opt)

        self.fmt = QComboBox()
        self.fmt.addItems(["mp4", "mkv", "mov", "webm", "avi"])
        form.addRow("Output format:", self.fmt)

        self.res = QComboBox()
        self.res.addItems(["original", "3840x2160", "2560x1440", "1920x1080",
                           "1280x720", "854x480", "640x360", "custom"])
        form.addRow("Resolution:", self.res)

        self.custom_res = QLineEdit()
        self.custom_res.setPlaceholderText("e.g. 1280x720")
        form.addRow("Custom (W×H):", self.custom_res)

        self.scale_fit = QRadioButton("Fit (keep aspect, pad if needed)")
        self.scale_stretch = QRadioButton("Stretch (ignore aspect)")
        self.scale_fit.setChecked(True)
        form.addRow(self.scale_fit)
        form.addRow(self.scale_stretch)

        self.crf = QSpinBox()
        self.crf.setRange(15, 35)
        self.crf.setValue(23)
        form.addRow("Quality (CRF):", self.crf)

        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("empty → Videos folder")
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse_dir(self.out_dir))
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.out_dir)
        dir_row.addWidget(browse)
        form.addRow("Output folder:", dir_row)
        layout.addWidget(opt)

        btn_row = QHBoxLayout()
        self.btn = QPushButton("▶  Start Convert-S")
        self.btn.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn.clicked.connect(self._start)
        self.btn_stop_after = QPushButton("Stop after current")
        self.btn_stop_after.setEnabled(False)
        self.btn_stop_after.clicked.connect(self._stop_after)
        self.btn_stop_now = QPushButton("Stop ASAP")
        self.btn_stop_now.setEnabled(False)
        self.btn_stop_now.setStyleSheet("color:#c00;")
        self.btn_stop_now.clicked.connect(self._stop_now)
        btn_row.addWidget(self.btn)
        btn_row.addWidget(self.btn_stop_after)
        btn_row.addWidget(self.btn_stop_now)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.log = LogBox()
        layout.addWidget(self.log)
        self._cancel: Optional[CancelToken] = None

    def _on_seg_mode(self):
        idx = self.seg_mode.currentIndex()
        # 0=full, 1=start, 2=middle, 3=end, 4=custom time, 5=custom frames
        self.seg_duration.setEnabled(idx in (1, 2, 3))
        self.seg_start.setEnabled(idx == 4)
        self.seg_end.setEnabled(idx == 4)
        self.seg_frame_start.setEnabled(idx == 5)
        self.seg_frame_count.setEnabled(idx == 5)

    def _stop_after(self):
        if self._cancel:
            self._cancel.request_after()
            self.log.append_line("… Will stop after current video finishes")

    def _stop_now(self):
        if self._cancel:
            self._cancel.request_now()
            self.log.append_line("… Stopping ASAP (killing ffmpeg)")

    def _set_running(self, running: bool):
        self.btn.setEnabled(not running)
        self.btn_stop_after.setEnabled(running)
        self.btn_stop_now.setEnabled(running)

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select videos", "",
            "Video files (*" + " *".join(VIDEO_EXTENSIONS) + ");;All files (*.*)"
        )
        for f in files:
            self.list.addItem(f)

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            self.list.addItem(d)

    def _remove(self):
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def _clear(self):
        self.list.clear()

    def _browse_dir(self, line: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            line.setText(d)

    def _resolve_segment(self, path: Path) -> tuple[Optional[float], Optional[float], Optional[int], Optional[int], str]:
        """
        Returns (ss, t, start_frame, frame_count, label).
        ss/t in seconds; frame fields set only for frame mode.
        None means not used.
        """
        idx = self.seg_mode.currentIndex()
        if idx == 0:
            return None, None, None, None, "full"

        duration = get_video_duration(path)
        if duration <= 0:
            return None, None, None, None, "full"  # fallback

        if idx == 1:  # start
            length = min(self.seg_duration.value(), duration)
            return 0.0, length, None, None, f"start {length:.2f}s"

        if idx == 2:  # middle
            length = min(self.seg_duration.value(), duration)
            ss = max(0.0, (duration - length) / 2.0)
            return ss, length, None, None, f"middle {length:.2f}s @ {ss:.2f}s"

        if idx == 3:  # end
            length = min(self.seg_duration.value(), duration)
            ss = max(0.0, duration - length)
            return ss, length, None, None, f"end {length:.2f}s @ {ss:.2f}s"

        if idx == 4:  # custom time
            ss = self.seg_start.value()
            end = self.seg_end.value()
            if end > ss:
                t = end - ss
            else:
                t = self.seg_duration.value()
            t = min(t, max(0.0, duration - ss))
            return ss, t, None, None, f"time {ss:.3f}s + {t:.3f}s"

        # idx == 5 custom frames
        fps = get_video_fps(path)
        start_f = self.seg_frame_start.value()
        count = self.seg_frame_count.value()
        ss = start_f / fps if fps > 0 else 0.0
        t = count / fps if fps > 0 else float(count)
        return ss, t, start_f, count, f"frames {start_f}+{count} (~{t:.2f}s)"

    def _start(self):
        items = [self.list.item(i).text() for i in range(self.list.count())]
        if not items:
            QMessageBox.warning(self, "No input", "Add at least one video or folder.")
            return
        videos = collect_videos_from_paths(items)
        if not videos:
            QMessageBox.warning(self, "No videos", "No video files found.")
            return

        fmt = self.fmt.currentText()
        res = self.res.currentText()
        custom = self.custom_res.text().strip()
        scale_mode = "fit" if self.scale_fit.isChecked() else "stretch"
        crf = self.crf.value()
        out_dir = self.out_dir.text().strip()
        out_path = Path(out_dir) if out_dir else DEFAULT_VIDEOS_DIR
        out_path.mkdir(parents=True, exist_ok=True)
        accurate = self.seg_accurate.isChecked()

        vf = None
        if res != "original":
            if res == "custom":
                if "x" not in custom.lower():
                    QMessageBox.critical(self, "Invalid", "Custom resolution must be like 1280x720")
                    return
                w, h = custom.lower().split("x")
            else:
                w, h = res.split("x")
            w, h = w.strip(), h.strip()
            if scale_mode == "stretch":
                vf = f"scale={w}:{h}"
            else:
                vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"

        # Capture UI settings on main thread (do not read widgets from worker)
        seg_idx = self.seg_mode.currentIndex()
        seg_len = self.seg_duration.value()
        seg_start_t = self.seg_start.value()
        seg_end_t = self.seg_end.value()
        seg_fstart = self.seg_frame_start.value()
        seg_fcount = self.seg_frame_count.value()
        mode_name = self.seg_mode.currentText()

        self._cancel = CancelToken()
        self._set_running(True)
        self.log.clear()
        self.log.append_line(f"Convert-S: {len(videos)} video(s) → {fmt}  CRF={crf}")
        self.log.append_line(f"Segment mode: {mode_name}\n")

        cancel = self._cancel

        def resolve_segment(path: Path):
            """Pure function using captured settings (safe off main thread)."""
            if seg_idx == 0:
                return None, None, None, None, "full"
            duration = get_video_duration(path)
            if duration <= 0:
                return None, None, None, None, "full"
            if seg_idx == 1:  # start
                length = min(seg_len, duration)
                return 0.0, length, None, None, f"start {length:.2f}s"
            if seg_idx == 2:  # middle
                length = min(seg_len, duration)
                ss = max(0.0, (duration - length) / 2.0)
                return ss, length, None, None, f"middle {length:.2f}s @ {ss:.2f}s"
            if seg_idx == 3:  # end
                length = min(seg_len, duration)
                ss = max(0.0, duration - length)
                return ss, length, None, None, f"end {length:.2f}s @ {ss:.2f}s"
            if seg_idx == 4:  # custom time
                ss = seg_start_t
                if seg_end_t > ss:
                    t = seg_end_t - ss
                else:
                    t = seg_len
                t = min(t, max(0.0, duration - ss))
                return ss, t, None, None, f"time {ss:.3f}s + {t:.3f}s"
            # frames
            fps = get_video_fps(path)
            ss = seg_fstart / fps if fps > 0 else 0.0
            t = seg_fcount / fps if fps > 0 else float(seg_fcount)
            return ss, t, seg_fstart, seg_fcount, f"frames {seg_fstart}+{seg_fcount} (~{t:.2f}s)"

        def convert_s_job(log_cb):
            jlog = JobLogger("convert_s")
            def both(line):
                jlog.line(line)
                log_cb(line)

            both(f"Convert-S: {len(videos)} video(s) → {fmt}  CRF={crf}")
            both(f"Segment: {mode_name}")
            both(f"Output folder: {out_path}\n")

            ok_count = 0
            processed = 0
            for i, vid in enumerate(videos, 1):
                if cancel.should_stop_after() and processed > 0:
                    both(f"Stopped by user after {processed} file(s).")
                    break
                if cancel.should_stop_now():
                    both("Stopped ASAP by user.")
                    break

                ss, t, start_f, fcount, label = resolve_segment(vid)
                stem = vid.stem
                suffix = "" if label == "full" else "_seg"
                out_file = out_path / f"{stem}{suffix}_converted.{fmt}"
                c = 1
                while out_file.exists():
                    out_file = out_path / f"{stem}{suffix}_converted_{c}.{fmt}"
                    c += 1

                both(f"[{i}/{len(videos)}] {vid.name} → {out_file.name}  ({label})")

                cmd = ["ffmpeg", "-y"]
                if ss is not None and t is not None:
                    # Frame mode is converted to seconds via fps; same seek path
                    if accurate:
                        cmd += ["-i", str(vid), "-ss", f"{ss:.6f}", "-t", f"{t:.6f}"]
                    else:
                        cmd += ["-ss", f"{ss:.6f}", "-i", str(vid), "-t", f"{t:.6f}"]
                else:
                    cmd += ["-i", str(vid)]

                if vf:
                    cmd += ["-vf", vf]

                if fmt == "webm":
                    cmd += ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(crf),
                            "-c:a", "libopus", "-b:a", "128k", str(out_file)]
                else:
                    cmd += [
                        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-threads", "4",
                        "-c:a", "aac", "-b:a", "128k",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(out_file)
                    ]

                ok, msg = run_ffmpeg(cmd, both, cancel=cancel)
                processed += 1
                if ok:
                    ok_count += 1
                    jlog.result(vid, True)
                else:
                    both(f"  FAILED: {msg[:200]}")
                    jlog.result(vid, False, msg)
                    if cancel.should_stop_now():
                        break

            jlog.close()
            return True, f"{ok_count}/{processed}|{jlog.full_path}|{jlog.summary_path}"

        self._conv_total = len(videos)
        self._worker = FFmpegWorker(func=convert_s_job, cancel=cancel, parent=self)
        self._worker.log.connect(self.log.append_line)
        self._worker.finished_ok.connect(self._finished)
        self._worker.start()

    def _finished(self, ok: bool, msg: str):
        self._set_running(False)
        self._cancel = None
        parts = msg.split("|")
        count_part = parts[0] if parts else msg
        full_log = parts[1] if len(parts) > 1 else ""
        summary_log = parts[2] if len(parts) > 2 else ""

        self.log.append_line(f"\nFinished: {count_part} succeeded")
        if full_log:
            self.log.append_line(f"Full log   : {full_log}")
            self.log.append_line(f"Summary log: {summary_log}")

        try:
            ok_count = int(count_part.split("/")[0])
            total = int(count_part.split("/")[1])
        except Exception:
            ok_count, total = 0, self._conv_total

        extra = f"\n\nSummary log:\n{summary_log}" if summary_log else ""
        if ok_count == total and total > 0:
            QMessageBox.information(self, "Success", f"All {ok_count} video(s) processed.{extra}")
        else:
            QMessageBox.warning(self, "Partial",
                                f"{ok_count} succeeded, {total - ok_count} failed/stopped.{extra}")



# ---------------------------------------------------------------------------
# Tab: Vid1S  – single video, multiple segment clips in a queue
# ---------------------------------------------------------------------------
class Vid1STab(QWidget):
    """
    Pick one video, add multiple time ranges to a queue
    (e.g. 1–6 s, then 10–18 s), then process → one output file per range.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path: Optional[Path] = None
        self.duration = 0.0
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        # Source video
        src = QGroupBox("Source video (single file)")
        sh = QHBoxLayout(src)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select or drop one video…")
        self.path_edit.setAcceptDrops(True)
        self.path_edit.installEventFilter(self)
        sh.addWidget(self.path_edit)
        sh.addWidget(QPushButton("Browse…", clicked=self._browse))
        sh.addWidget(QPushButton("Probe", clicked=self._probe))
        layout.addWidget(src)

        self.info_lbl = QLabel("Duration: —")
        layout.addWidget(self.info_lbl)

        # Range input
        rng = QGroupBox("Add segment to queue")
        form = QFormLayout(rng)

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0.0, 86400.0)
        self.start_spin.setDecimals(3)
        self.start_spin.setSuffix(" s")
        self.start_spin.setValue(0.0)
        form.addRow("Start:", self.start_spin)

        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0.0, 86400.0)
        self.end_spin.setDecimals(3)
        self.end_spin.setSuffix(" s")
        self.end_spin.setValue(5.0)
        form.addRow("End:", self.end_spin)

        self.clip_name = QLineEdit()
        self.clip_name.setPlaceholderText("optional label (e.g. intro)")
        form.addRow("Label:", self.clip_name)

        add_row = QHBoxLayout()
        add_row.addWidget(QPushButton("＋  Add to queue", clicked=self._add_range))
        add_row.addWidget(QPushButton("Use full duration as End", clicked=self._use_full_end))
        add_row.addStretch()
        form.addRow(add_row)
        layout.addWidget(rng)

        # Queue
        qgrp = QGroupBox("Queue (each row → one output file)")
        qv = QVBoxLayout(qgrp)
        self.queue = QListWidget()
        self.queue.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        qv.addWidget(self.queue)
        qb = QHBoxLayout()
        qb.addWidget(QPushButton("Remove selected", clicked=self._remove_selected))
        qb.addWidget(QPushButton("Clear queue", clicked=self._clear_queue))
        qb.addWidget(QPushButton("Move Up", clicked=lambda: self._move(-1)))
        qb.addWidget(QPushButton("Move Down", clicked=lambda: self._move(1)))
        qb.addStretch()
        qv.addLayout(qb)
        layout.addWidget(qgrp)

        # Convert options
        opt = QGroupBox("Output options")
        form2 = QFormLayout(opt)

        self.fmt = QComboBox()
        self.fmt.addItems(["mp4", "mkv", "mov", "webm", "avi"])
        form2.addRow("Format:", self.fmt)

        self.res = QComboBox()
        self.res.addItems(["original", "3840x2160", "2560x1440", "1920x1080",
                           "1280x720", "854x480", "640x360", "custom"])
        form2.addRow("Resolution:", self.res)

        self.custom_res = QLineEdit()
        self.custom_res.setPlaceholderText("e.g. 1280x720")
        form2.addRow("Custom (W×H):", self.custom_res)

        self.scale_fit = QRadioButton("Fit (keep aspect, pad)")
        self.scale_stretch = QRadioButton("Stretch")
        self.scale_fit.setChecked(True)
        form2.addRow(self.scale_fit)
        form2.addRow(self.scale_stretch)

        self.crf = QSpinBox()
        self.crf.setRange(15, 35)
        self.crf.setValue(23)
        form2.addRow("CRF:", self.crf)

        self.accurate = QCheckBox("Frame-accurate seek (slower)")
        self.accurate.setChecked(True)
        form2.addRow(self.accurate)

        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("empty → Videos folder")
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse_dir(self.out_dir))
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.out_dir)
        dir_row.addWidget(browse)
        form2.addRow("Output folder:", dir_row)
        layout.addWidget(opt)

        # Actions
        btn_row = QHBoxLayout()
        self.btn = QPushButton("▶  Process queue")
        self.btn.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn.clicked.connect(self._start)
        self.btn_stop_after = QPushButton("Stop after current")
        self.btn_stop_after.setEnabled(False)
        self.btn_stop_after.clicked.connect(self._stop_after)
        self.btn_stop_now = QPushButton("Stop ASAP")
        self.btn_stop_now.setEnabled(False)
        self.btn_stop_now.setStyleSheet("color:#c00;")
        self.btn_stop_now.clicked.connect(self._stop_now)
        btn_row.addWidget(self.btn)
        btn_row.addWidget(self.btn_stop_after)
        btn_row.addWidget(self.btn_stop_now)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.log = LogBox()
        layout.addWidget(self.log)
        self._cancel: Optional[CancelToken] = None

    def eventFilter(self, obj, event):
        if obj is self.path_edit:
            if event.type() == event.Type.DragEnter and event.mimeData().hasUrls():
                event.acceptProposedAction()
                return True
            if event.type() == event.Type.Drop:
                urls = event.mimeData().urls()
                if urls:
                    p = urls[0].toLocalFile()
                    if is_video(Path(p)):
                        self.path_edit.setText(p)
                        self._probe()
                return True
        return super().eventFilter(obj, event)

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select video", "",
            "Video files (*" + " *".join(VIDEO_EXTENSIONS) + ");;All files (*.*)"
        )
        if f:
            self.path_edit.setText(f)
            self._probe()

    def _browse_dir(self, line: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            line.setText(d)

    def _probe(self):
        path = Path(self.path_edit.text().strip())
        if not is_video(path):
            self.info_lbl.setText("Duration: —  (invalid video)")
            self.video_path = None
            self.duration = 0.0
            return
        self.video_path = path
        self.duration = get_video_duration(path)
        w, h = get_video_size(path)
        fps = get_video_fps(path)
        self.info_lbl.setText(
            f"Duration: {self.duration:.3f} s   |   {w}×{h}   |   {fps:.3f} fps   |   {path.name}"
        )
        self.start_spin.setMaximum(max(self.duration, 1.0))
        self.end_spin.setMaximum(max(self.duration, 1.0))
        if self.end_spin.value() > self.duration > 0:
            self.end_spin.setValue(min(5.0, self.duration))

    def _use_full_end(self):
        if self.duration > 0:
            self.end_spin.setValue(self.duration)

    def _add_range(self):
        if self.video_path is None:
            QMessageBox.warning(self, "No video", "Select and probe a video first.")
            return
        start = self.start_spin.value()
        end = self.end_spin.value()
        if end <= start:
            QMessageBox.warning(self, "Invalid range", "End must be greater than Start.")
            return
        if self.duration > 0 and start >= self.duration:
            QMessageBox.warning(self, "Invalid range", "Start is past the end of the video.")
            return
        if self.duration > 0:
            end = min(end, self.duration)

        label = self.clip_name.text().strip()
        length = end - start
        display = f"{start:.3f}s → {end:.3f}s  ({length:.3f}s)"
        if label:
            display = f"[{label}]  {display}"

        item = QListWidgetItem(display)
        item.setData(Qt.ItemDataRole.UserRole, (start, end, label))
        self.queue.addItem(item)
        self.clip_name.clear()
        self.start_spin.setValue(end)
        self.end_spin.setValue(min(end + length, self.duration if self.duration > 0 else end + length))

    def _remove_selected(self):
        for item in self.queue.selectedItems():
            self.queue.takeItem(self.queue.row(item))

    def _clear_queue(self):
        self.queue.clear()

    def _move(self, direction: int):
        row = self.queue.currentRow()
        if row < 0:
            return
        new_row = row + direction
        if new_row < 0 or new_row >= self.queue.count():
            return
        item = self.queue.takeItem(row)
        self.queue.insertItem(new_row, item)
        self.queue.setCurrentRow(new_row)

    def _stop_after(self):
        if self._cancel:
            self._cancel.request_after()
            self.log.append_line("… Will stop after current clip finishes")

    def _stop_now(self):
        if self._cancel:
            self._cancel.request_now()
            self.log.append_line("… Stopping ASAP")

    def _set_running(self, running: bool):
        self.btn.setEnabled(not running)
        self.btn_stop_after.setEnabled(running)
        self.btn_stop_now.setEnabled(running)

    def _start(self):
        if self.video_path is None or not is_video(self.video_path):
            QMessageBox.warning(self, "No video", "Select a valid source video first.")
            return
        if self.queue.count() == 0:
            QMessageBox.warning(self, "Empty queue", "Add at least one segment to the queue.")
            return

        clips = []
        for i in range(self.queue.count()):
            item = self.queue.item(i)
            start, end, label = item.data(Qt.ItemDataRole.UserRole)
            clips.append((start, end, label or f"clip{i+1}"))

        fmt = self.fmt.currentText()
        res = self.res.currentText()
        custom = self.custom_res.text().strip()
        scale_mode = "fit" if self.scale_fit.isChecked() else "stretch"
        crf = self.crf.value()
        accurate = self.accurate.isChecked()
        out_dir = self.out_dir.text().strip()
        out_path = Path(out_dir) if out_dir else DEFAULT_VIDEOS_DIR
        out_path.mkdir(parents=True, exist_ok=True)
        src = self.video_path

        vf = None
        if res != "original":
            if res == "custom":
                if "x" not in custom.lower():
                    QMessageBox.critical(self, "Invalid", "Custom resolution must be like 1280x720")
                    return
                w, h = custom.lower().split("x")
            else:
                w, h = res.split("x")
            w, h = w.strip(), h.strip()
            if scale_mode == "stretch":
                vf = f"scale={w}:{h}"
            else:
                vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"

        self._cancel = CancelToken()
        self._set_running(True)
        self.log.clear()
        self.log.append_line(f"Vid1S: {src.name}  →  {len(clips)} clip(s)  format={fmt}\n")

        cancel = self._cancel

        def job(log_cb):
            jlog = JobLogger("vid1s")
            def both(line):
                jlog.line(line)
                log_cb(line)

            both(f"Source: {src}")
            both(f"Clips : {len(clips)}  format={fmt}  CRF={crf}")
            both(f"Output: {out_path}\n")

            ok_count = 0
            processed = 0
            stem = src.stem

            for i, (start, end, label) in enumerate(clips, 1):
                if cancel.should_stop_after() and processed > 0:
                    both(f"Stopped by user after {processed} clip(s).")
                    break
                if cancel.should_stop_now():
                    both("Stopped ASAP by user.")
                    break

                t = end - start
                safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
                out_file = out_path / f"{stem}_{safe_label}_{start:.2f}-{end:.2f}.{fmt}"
                c = 1
                while out_file.exists():
                    out_file = out_path / f"{stem}_{safe_label}_{start:.2f}-{end:.2f}_{c}.{fmt}"
                    c += 1

                both(f"[{i}/{len(clips)}] {start:.3f}s → {end:.3f}s  ({t:.3f}s)  [{label}] → {out_file.name}")

                cmd = ["ffmpeg", "-y"]
                if accurate:
                    cmd += ["-i", str(src), "-ss", f"{start:.6f}", "-t", f"{t:.6f}"]
                else:
                    cmd += ["-ss", f"{start:.6f}", "-i", str(src), "-t", f"{t:.6f}"]

                if vf:
                    cmd += ["-vf", vf]

                if fmt == "webm":
                    cmd += ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(crf),
                            "-c:a", "libopus", "-b:a", "128k", str(out_file)]
                else:
                    cmd += [
                        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-threads", "4",
                        "-c:a", "aac", "-b:a", "128k",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(out_file)
                    ]

                ok, msg = run_ffmpeg(cmd, both, cancel=cancel)
                processed += 1
                if ok:
                    ok_count += 1
                    jlog.result(out_file, True)
                else:
                    both(f"  FAILED: {msg[:200]}")
                    jlog.result(out_file, False, msg)
                    if cancel.should_stop_now():
                        break

            jlog.close()
            return True, f"{ok_count}/{processed}|{jlog.full_path}|{jlog.summary_path}"

        self._total = len(clips)
        self._worker = FFmpegWorker(func=job, cancel=cancel, parent=self)
        self._worker.log.connect(self.log.append_line)
        self._worker.finished_ok.connect(self._finished)
        self._worker.start()

    def _finished(self, ok: bool, msg: str):
        self._set_running(False)
        self._cancel = None
        parts = msg.split("|")
        count_part = parts[0] if parts else msg
        full_log = parts[1] if len(parts) > 1 else ""
        summary_log = parts[2] if len(parts) > 2 else ""

        self.log.append_line(f"\nFinished: {count_part} succeeded")
        if full_log:
            self.log.append_line(f"Full log   : {full_log}")
            self.log.append_line(f"Summary log: {summary_log}")

        try:
            ok_count = int(count_part.split("/")[0])
            total = int(count_part.split("/")[1])
        except Exception:
            ok_count, total = 0, getattr(self, "_total", 0)

        extra = f"\n\nSummary:\n{summary_log}" if summary_log else ""
        if ok_count == total and total > 0:
            QMessageBox.information(self, "Success", f"All {ok_count} clip(s) created.{extra}")
        else:
            QMessageBox.warning(self, "Partial",
                                f"{ok_count} succeeded, {total - ok_count} failed/stopped.{extra}")



class VideoToolApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Tool")
        self.resize(1000, 720)
        self.setMinimumSize(860, 600)

        tabs = QTabWidget()
        tabs.addTab(MergeTab(), "Video Merge")
        tabs.addTab(ConvertTab(), "Video Convert")
        tabs.addTab(ConvertSTab(), "Video Convert-S")
        tabs.addTab(Vid1STab(), "Vid1S")
        tabs.addTab(VFoldTab(), "VFold Merge")
        tabs.addTab(CropTab(), "Video Crop")
        self.setCentralWidget(tabs)

        # Status
        self.status = self.statusBar()
        self.status.showMessage("Ready")


# ---------------------------------------------------------------------------
def main():
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found in PATH.")
        sys.exit(1)
    if shutil.which("ffprobe") is None:
        print("ERROR: ffprobe not found in PATH.")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = VideoToolApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
