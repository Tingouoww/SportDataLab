
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import cv2
import sys
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont

def draw_chinese_text(img, text, position, font_path="C:/Windows/Fonts/msjh.ttc",
                      font_size=28, color=(255,255,255)):
    """
    在 OpenCV 影像上繪製中文文字
    img: OpenCV BGR numpy array
    text: 中文文字
    position: (x, y)
    font_path: TrueType 字體檔路徑
    """
    if not text:
        return img
    # 轉成 PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    # 畫字
    draw.text(position, text, font=font, fill=color[::-1])  # PIL 用 RGB, OpenCV 用 BGR

    # 轉回 OpenCV 格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def load_data(set1_path: Path, rallyseg_path: Path):
    df = pd.read_csv(set1_path, encoding='utf-8')
    rally_seg = pd.read_csv(rallyseg_path)
    return df, rally_seg

def build_rally_index(rally_seg: pd.DataFrame):
    """
    把 RallySeg 轉換成一個查表 dict
    """
    idx = {}
    for i, row in rally_seg.iterrows():
        idx[i+1] = (int(row["Start"]), int(row["End"]))
    return idx

def extract_events(df: pd.DataFrame, rally_id: int) -> pd.DataFrame:
    """
    從 set1.csv 中，抓出指定 rally_id 的所有球事件
    """
    sub = df[df["rally"] == rally_id].copy()
    sub.sort_values(["ball_round"], inplace=True)
    for c in ["frame_num", "end_frame_num", "player_location_x", "player_location_y",
              "opponent_location_x", "opponent_location_y"]:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce") # 資料強制轉成數字
    return sub

def draw_marker(img, x, y, color, label=None, radius=8, thickness=2):
    """
    在影片上標記一個點
    """
    if x is None or y is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(y, float) and np.isnan(y)):
        return
    try:
        x_i, y_i = int(x), int(y)
    except Exception:
        return
    cv2.circle(img, (x_i, y_i), radius, color, thickness, lineType=cv2.LINE_AA)
    cv2.circle(img, (x_i, y_i), 2, color, -1, lineType=cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (x_i+10, y_i-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def guess_writer_attempts(out_path: Path):
    """
    決定輸出影片用的編碼器（fourcc）
    """
    ext = out_path.suffix.lower()
    attempts = []
    if ext in [".mp4", ".m4v", ".mov"]:
        attempts = [
            ("mp4v", out_path),
            ("avc1", out_path),
            ("H264", out_path),
            ("XVID", out_path.with_suffix(".avi")),
            ("MJPG", out_path.with_suffix(".avi")),
        ]
    else:
        if ext != ".avi":
            out_path = out_path.with_suffix(".avi")
        attempts = [("XVID", out_path), ("MJPG", out_path)]
    return attempts

def open_video_writer(out_path: Path, fps: float, size: tuple[int,int]):
    for fourcc_str, path in guess_writer_attempts(out_path):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_str), fps, size)
        if writer.isOpened():
            return writer, path, fourcc_str
    return None, None, None

def build_intervals(events: pd.DataFrame, seg_start: int) -> List[Dict[str, Any]]:
    intervals = []
    for _, row in events.iterrows():
        try:
            g0 = int(row["frame_num"]); g1 = int(row["end_frame_num"])
        except Exception:
            continue
        intervals.append({
            "ball_round": int(row["ball_round"]),
            "global_start": g0,
            "global_end": g1,
            "local_start": g0 - seg_start,
            "local_end": g1 - seg_start,
            "player": str(row.get("player", "")),
            "type": str(row.get("type", "")),
            "player_x": None if pd.isna(row.get("player_location_x")) else float(row.get("player_location_x")),
            "player_y": None if pd.isna(row.get("player_location_y")) else float(row.get("player_location_y")),
            "oppo_x": None if pd.isna(row.get("opponent_location_x")) else float(row.get("opponent_location_x")),
            "oppo_y": None if pd.isna(row.get("opponent_location_y")) else float(row.get("opponent_location_y")),
        })
    return intervals

def annotate_rally_segment(
    video_path: Path,
    set1_path: Path,
    rallyseg_path: Path,
    rally_id: int,
    out_path: Path,
    video_mode: str = "clip",  # 預設改為 clip（因多數人先剪片段）
    draw_opponent: bool = True,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    fps_scale: float = 1.0,   # 當前影片FPS / 資料表FPS
    offset: int = 0,          # 幀偏移；正值表示片段比資料晚開始
    verbose: bool = True,
):
    df, rally_seg = load_data(set1_path, rallyseg_path)
    rally_index = build_rally_index(rally_seg)
    if rally_id not in rally_index:
        raise ValueError(f"rally_id={rally_id} 不在 RallySeg 內。")

    seg_start, seg_end = rally_index[rally_id]
    if seg_end <= seg_start:
        raise ValueError(f"Rally {rally_id}: End({seg_end}) <= Start({seg_start})")

    events = extract_events(df, rally_id)
    intervals = build_intervals(events, seg_start)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if video_mode == "main":
        # 直接用 global frame 對齊
        start_frame = seg_start
        end_frame = min(seg_end, total) if total > 0 else seg_end
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        def is_active(idx):
            # idx 是 global frame
            return [it for it in intervals if it["global_start"] <= idx < it["global_end"]]
        def show_local(idx):  # 僅作顯示
            return idx - seg_start
        mode_note = "MAIN"
    elif video_mode == "clip":
        # 片段影片：以 local frame 對齊；支援 fps_scale 與 offset
        start_frame = 0
        end_frame = total if total > 0 else max((it["local_end"] for it in intervals), default=0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # 把資料表的 local 區間縮放後再平移：scaled = round(local * fps_scale - offset)
        scaled_intervals = []
        for it in intervals:
            s = int(round(it["local_start"] * fps_scale - offset))
            e = int(round(it["local_end"]   * fps_scale - offset))
            if e <= s:
                e = s + 1  # 至少一幀，避免空區間
            it2 = dict(it)
            it2["scaled_start"] = s
            it2["scaled_end"] = e
            scaled_intervals.append(it2)
        def is_active(idx):
            # idx 是目前片段的 frame index
            return [it for it in scaled_intervals if it["scaled_start"] <= idx < it["scaled_end"]]
        def show_local(idx):
            # 反推回原始 local 幀號（純顯示用途）
            return int(round((idx + offset) / max(fps_scale, 1e-9)))
        mode_note = "CLIP"
    else:
        raise ValueError("--video-mode 必須是 main 或 clip")

    writer, real_out, used_fourcc = open_video_writer(out_path, fps, (width, height))
    if writer is None:
        raise RuntimeError("無法建立輸出影片，請改用 .avi 或安裝對應的編碼器。")

    if verbose:
        print(f"[INFO] 模式: {mode_note}  Rally {rally_id}")
        print(f"[INFO] 影片: {video_path}  fps={fps:.3f} size=({width}x{height}) total={total}")
        print(f"[INFO] 區間: [{start_frame}, {end_frame})  寫出: {real_out}  fourcc={used_fourcc}")
        if video_mode == "clip":
            print(f"[INFO] fps_scale={fps_scale}  offset={offset} (scaled_start = round(local*fps_scale - offset))")

    COLOR_PLAYER = (36, 255, 12)
    COLOR_OPPO   = (0, 215, 255)
    COLOR_INFO   = (255, 255, 255)

    current = start_frame
    frames_written = 0

    while current < end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        overlay = frame.copy()
        active = is_active(current)

        info_lines = [f"Rally: {rally_id} [{mode_note}]  VideoFrame: {current}  LocalFrame: {show_local(current)}"]

        if active:
            a = active[0]
            info_lines.append(f"Ball: {a['ball_round']}  Player: {a['player']}  Type: {a['type']}")

            px, py = a["player_x"], a["player_y"]
            if px is not None and py is not None and not (np.isnan(px) or np.isnan(py)):
                draw_marker(overlay, int(px*scale_x), int(py*scale_y), COLOR_PLAYER, label="Player")

            ox, oy = a["oppo_x"], a["oppo_y"]
            if ox is not None and oy is not None and not (np.isnan(ox) or np.isnan(oy)):
                draw_marker(overlay, int(ox*scale_x), int(oy*scale_y), COLOR_OPPO, label="Opponent")

        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        y = 30
        for line in info_lines:
            frame = draw_chinese_text(frame, line, (250, y), font_path="C:/Windows/Fonts/msjh.ttc", font_size=28, color=(255,255,255))
            # cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_INFO, 2, cv2.LINE_AA)
            y += 28

        writer.write(frame)
        frames_written += 1
        current += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if verbose:
        print(f"[DONE] 寫入幀數: {frames_written}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True, help="影片路徑（主影片或片段）")
    p.add_argument("--rally", type=int, required=True, help="rally id（1-based）")
    p.add_argument("--out", type=Path, required=True, help="輸出影片路徑（.mp4 或 .avi）")
    p.add_argument("--set1", type=Path, default=Path(__file__).parent / "set1.csv")
    p.add_argument("--rallyseg", type=Path, default=Path(__file__).parent / "RallySeg.csv")
    p.add_argument("--video-mode", type=str, choices=["main", "clip"], default="clip",
                   help="主影片=main；若輸入的是已剪好的 rally 片段，請用 clip（預設）")
    p.add_argument("--no-opponent", action="store_true", help="不要畫對手位置")
    p.add_argument("--scale-x", type=float, default=1.0)
    p.add_argument("--scale-y", type=float, default=1.0)
    p.add_argument("--fps-scale", type=float, default=1.0, help="片段FPS / 資料表FPS，例如 30/60=0.5")
    p.add_argument("--offset", type=int, default=0, help="幀偏移：正值=片段比資料晚開始，負值=片段比資料早開始")
    p.add_argument("--quiet", action="store_true")

    args = p.parse_args()

    annotate_rally_segment(
        video_path=args.video,
        set1_path=args.set1,
        rallyseg_path=args.rallyseg,
        rally_id=args.rally,
        out_path=args.out,
        video_mode=args.video_mode,
        draw_opponent=(not args.no_opponent),
        scale_x=args.scale_x,
        scale_y=args.scale_y,
        fps_scale=args.fps_scale,
        offset=args.offset,
        verbose=(not args.quiet),
    )

if __name__ == "__main__":
    main()
