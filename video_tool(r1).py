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
# Constants
# ---------------------------------------------------------------------------
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv",
                    ".m4v", ".ts", ".mts", ".m2ts", ".3gp", ".mpg", ".mpeg"}

DEFAULT_VIDEOS_DIR = Path.home() / "Videos"
if not DEFAULT_VIDEOS_DIR.exists():
    DEFAULT_VIDEOS_DIR = Path.cwd() / "videos"
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


def run_ffmpeg(cmd: list[str], log_callback=None) -> tuple[bool, str]:
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
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            if log_callback:
                log_callback(line)
        process.wait()
        if process.returncode == 0:
            return True, "Success"
        return False, "\n".join(output_lines[-40:])
    except Exception as e:
        return False, str(e)


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

    def __init__(self, cmd: list[str] | None = None, func=None, parent=None):
        super().__init__(parent)
        self.cmd = cmd
        self.func = func   # alternative: callable that returns (ok, msg)

    def run(self):
        try:
            if self.func is not None:
                ok, msg = self.func(lambda line: self.log.emit(line))
            else:
                ok, msg = run_ffmpeg(self.cmd, lambda line: self.log.emit(line))
            self.finished_ok.emit(ok, msg)
        except Exception as e:
            self.finished_ok.emit(False, str(e))


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

        # Action
        self.btn = QPushButton("▶  Start Merge")
        self.btn.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn.clicked.connect(self._start)
        layout.addWidget(self.btn)

        self.log = LogBox()
        layout.addWidget(self.log)

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

        self.btn.setEnabled(False)
        self.log.clear()
        self.log.append_line(f"Starting {mode} merge of {len(videos)} videos…\n→ {output}\n")

        self._merge_output = output
        self._worker = FFmpegWorker(
            func=lambda log_cb: do_merge(videos, output, mode, log_callback=log_cb),
            parent=self
        )
        self._worker.log.connect(self.log.append_line)
        self._worker.finished_ok.connect(self._merge_finished)
        self._worker.start()

    def _merge_finished(self, ok: bool, msg: str):
        self.btn.setEnabled(True)
        if ok:
            self.log.append_line(f"\n✓ Done → {self._merge_output}\n")
            QMessageBox.information(self, "Success", f"Merged video saved to:\n{self._merge_output}")
        else:
            self.log.append_line(f"\n✗ Failed:\n{msg}\n")
            QMessageBox.critical(self, "Merge Failed", msg[:800])


def do_merge(videos: list[Path], output: Path, mode: str, log_callback=None) -> tuple[bool, str]:
    """Shared merge implementation used by Merge and VFold tabs."""
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
            return run_ffmpeg(cmd, log_callback)
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)
    else:
        n = len(videos)
        inputs = []
        for v in videos:
            inputs.extend(["-i", str(v)])
        filter_parts = [f"[{i}:v:0][{i}:a:0]" for i in range(n)]
        filter_str = "".join(filter_parts) + f"concat=n={n}:v=1:a=1[v][a]"
        cmd = [
            "ffmpeg", "-y", *inputs,
            "-filter_complex", filter_str,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-threads", "4",
            "-c:a", "aac", "-b:a", "128k",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output)
        ]
        return run_ffmpeg(cmd, log_callback)


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

        self.btn = QPushButton("▶  Start Convert")
        self.btn.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn.clicked.connect(self._start)
        layout.addWidget(self.btn)

        self.log = LogBox()
        layout.addWidget(self.log)

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

        self.btn.setEnabled(False)
        self.log.clear()
        self.log.append_line(f"Converting {len(videos)} video(s) → {fmt}  (CRF={crf})\n")

        def convert_job(log_cb):
            ok_count = 0
            for i, vid in enumerate(videos, 1):
                stem = vid.stem
                out_file = out_path / f"{stem}_converted.{fmt}"
                c = 1
                while out_file.exists():
                    out_file = out_path / f"{stem}_converted_{c}.{fmt}"
                    c += 1
                log_cb(f"[{i}/{len(videos)}] {vid.name} → {out_file.name}")

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
                ok, msg = run_ffmpeg(cmd, log_cb)
                if ok:
                    ok_count += 1
                else:
                    log_cb(f"  FAILED: {msg[:200]}")
            return True, f"{ok_count}/{len(videos)}"

        self._conv_total = len(videos)
        self._worker = FFmpegWorker(func=convert_job, parent=self)
        self._worker.log.connect(self.log.append_line)
        self._worker.finished_ok.connect(self._convert_finished)
        self._worker.start()

    def _convert_finished(self, ok: bool, msg: str):
        self.btn.setEnabled(True)
        # msg is like "3/3"
        self.log.append_line(f"\nFinished: {msg} succeeded")
        try:
            ok_count = int(msg.split("/")[0])
            total = int(msg.split("/")[1])
        except Exception:
            ok_count, total = 0, self._conv_total
        if ok_count == total:
            QMessageBox.information(self, "Success", f"All {ok_count} video(s) converted.")
        else:
            QMessageBox.warning(self, "Partial", f"{ok_count} succeeded, {total - ok_count} failed.")


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

        self.btn = QPushButton("▶  Start VFold Merge")
        self.btn.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn.clicked.connect(self._start)
        layout.addWidget(self.btn)

        self.log = LogBox()
        layout.addWidget(self.log)

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

        self.btn.setEnabled(False)
        self.log.clear()
        self.log.append_line(f"Starting VFold merge ({mode})…\n")

        # Capture list of subfolders on main thread
        subs = []
        for i in range(self.tree.count()):
            item = self.tree.item(i)
            subs.append(Path(item.data(Qt.ItemDataRole.UserRole)))

        def vfold_job(log_cb):
            ok_count = 0
            total = len(subs)
            for sub in subs:
                vids = [f for f in sorted(sub.iterdir()) if is_video(f)]
                if not vids:
                    continue
                out_name = f"{sub.name}_merge.mp4"
                output = out_base / out_name
                log_cb(f"  Merging {sub.name} ({len(vids)} videos) → {out_name}")

                if len(vids) == 1:
                    cmd = ["ffmpeg", "-y", "-i", str(vids[0]), "-c", "copy", str(output)]
                    ok, msg = run_ffmpeg(cmd, log_cb)
                else:
                    ok, msg = do_merge(vids, output, mode, log_callback=log_cb)

                if ok:
                    ok_count += 1
                    log_cb(f"    ✓ {sub.name}")
                else:
                    log_cb(f"    ✗ {sub.name}: {msg[:150]}")
            return True, f"{ok_count}/{total}"

        self._vfold_total = len(subs)
        self._worker = FFmpegWorker(func=vfold_job, parent=self)
        self._worker.log.connect(self.log.append_line)
        self._worker.finished_ok.connect(self._vfold_finished)
        self._worker.start()

    def _vfold_finished(self, ok: bool, msg: str):
        self.btn.setEnabled(True)
        self.log.append_line(f"\nFinished: {msg}")
        try:
            ok_count = int(msg.split("/")[0])
            total = int(msg.split("/")[1])
        except Exception:
            ok_count, total = 0, self._vfold_total
        if ok_count == total:
            QMessageBox.information(self, "Success", f"All {ok_count} subfolder(s) merged.")
        else:
            QMessageBox.warning(self, "Partial", f"{ok_count} succeeded, {total - ok_count} failed.")


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
        controls.addStretch()

        right = QWidget()
        right.setLayout(controls)
        right.setFixedWidth(240)
        split.addWidget(right)
        layout.addLayout(split)

        self.log = LogBox()
        layout.addWidget(self.log)

        self._updating_spins = False

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

        self.btn.setEnabled(False)
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

        self._crop_output = output  # keep for finished handler
        self._worker = FFmpegWorker(cmd=cmd, parent=self)
        self._worker.log.connect(self.log.append_line)
        self._worker.finished_ok.connect(self._crop_finished)
        self._worker.start()

    def _crop_finished(self, ok: bool, msg: str):
        self.btn.setEnabled(True)
        if ok:
            self.log.append_line(f"\n✓ Done → {self._crop_output}")
            QMessageBox.information(self, "Success", f"Cropped video saved to:\n{self._crop_output}")
        else:
            self.log.append_line(f"\n✗ Failed:\n{msg}")
            QMessageBox.critical(self, "Crop Failed", msg[:800])


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class VideoToolApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Tool")
        self.resize(1000, 720)
        self.setMinimumSize(860, 600)

        tabs = QTabWidget()
        tabs.addTab(MergeTab(), "Video Merge")
        tabs.addTab(ConvertTab(), "Video Convert")
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
