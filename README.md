# Video-Tool-R1
Python Based Video tool for All video function like split, convert, merge etc.
# Video Tool (PyQt6) – Portable

Multi-tab desktop video utility. Designed to be packaged with PyInstaller.

## Tabs

| Tab | Description |
|-----|-------------|
| **Video Merge** | Merge multiple videos. Fast / Full Encode. Stop ASAP. |
| **Video Convert** | Format / resolution / aspect. Stop ASAP or after current file. |
| **Video Convert-S** | Same as Convert + extract segment: Start / Middle / End / custom time / frames. |
| **Vid1S** | One video → multiple time ranges in a queue → one output file per range. |
| **Vid1F** | One video → multiple **frame** ranges in a queue → one output file per range. |
| **VFold Merge** | Parent folder → each subfolder becomes `<name>_merge.mp4`. Skip existing. Stop after folder / ASAP. |
| **Video Crop** | Interactive crop + aspect lock. Stop ASAP. |

## Stop controls

- **Stop ASAP** – kills the current ffmpeg process immediately.
- **Stop after current** (Convert / VFold) – finishes the current video or subfolder, then stops.

## VFold: skip already done

Checkbox **Skip subfolder if `<name>_merge.mp4` already exists** (on by default).
Re-running the same parent folder only processes new / missing outputs.

## Portable paths (PyInstaller-ready)

Everything lives next to the executable / script:

```
VideoTool.exe          (or video_tool.py)
logs/                  full_*.log  +  summary_*.txt
data/                  (reserved for future state/db)
```

No writes into the user profile for logs.

## Requirements

```bash
pip install PyQt6 opencv-python
# ffmpeg must be on PATH (or place ffmpeg.exe next to the tool)
```

## Run

```bash
python video_tool.py
```

## PyInstaller one-file example

```bash
pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed ^
  --name VideoTool ^
  --add-data "logs;logs" ^
  video_tool.py
```

Place `ffmpeg.exe` / `ffprobe.exe` in the same folder as `VideoTool.exe`.

## Logs

| File | Contents |
|------|----------|
| `logs/full_<job>_<stamp>.log` | Complete ffmpeg output |
| `logs/summary_<job>_<stamp>.txt` | Time · Status · Path · Error (easy to review failures/skips) |
