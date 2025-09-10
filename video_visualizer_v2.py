#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rally Video Visualizer (v3: 面板可依座標高亮 + 格子標籤 + 累計計數 + 可輸出影片)
- local_frame = set1.frame_num - RallySeg.Start
- RallySeg.csv 欄位：Score / Start / End（或用參數覆寫）
- 預設以 set1 的 player_location_x / player_location_y 畫點（可用參數覆寫）
- --panel-mode: event / position（預設 position）
- --panel-counts: 顯示累計計數（依當前幀 rows 的所有點，每幀累計）
"""

import argparse
import os
import sys
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import pandas as pd

# ============= 可調整預設值 =============
DEFAULT_PANEL_GRID = (3, 3)
DEFAULT_EVENT_TO_CELL = {
    "clear": (0, 1),
    "drop": (1, 1),
    "net":  (2, 1),
    "smash":(0, 1),
    "drive":(1, 1),
}
GRID_ROW_LABELS = ["後場", "中場", "前場"]  # 0=上排(後場) 2=下排(前場)
GRID_COL_LABELS = ["左", "中", "右"]
# =====================================


# ============= 工具函式 =============

def parse_color_to_bgr(s: str) -> Tuple[int, int, int]:
    if not isinstance(s, str):
        return (0, 255, 255)
    name = s.strip().lower()
    table = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "orange": (0, 165, 255),
    }
    if name in table:
        return table[name]
    hexs = name.lstrip("#")
    if len(hexs) == 6:
        r = int(hexs[0:2], 16)
        g = int(hexs[2:4], 16)
        b = int(hexs[4:6], 16)
        return (b, g, r)
    return (0, 255, 255)


def _auto_pick_column(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in preferred:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


# ============= 讀檔 & 對齊 =============

def load_positions_csv(
    csv_path: str,
    frame_col: Optional[str] = None,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    label_col: Optional[str] = None,
    event_col: Optional[str] = None,
    color_col: Optional[str] = None,
    rally_id_col: Optional[str] = None,
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Positions CSV not found: {csv_path}")
    df_raw = pd.read_csv(csv_path)

    if frame_col is None:
        frame_col = _auto_pick_column(df_raw, ["frame_num", "frame"])
    if x_col is None:
        x_col = _auto_pick_column(df_raw, ["player_location_x", "hit_x", "landing_x", "x", "cx", "Xpix", "posX", "u"])
    if y_col is None:
        y_col = _auto_pick_column(df_raw, ["player_location_y", "hit_y", "landing_y", "y", "cy", "Ypix", "posY", "v"])

    missing = [n for n, v in [("frame", frame_col), ("x", x_col), ("y", y_col)] if v is None]
    if missing:
        raise ValueError(f"[positions] 找不到必要欄位：{missing}；現有欄位={list(df_raw.columns)}")

    df = df_raw.rename(columns={frame_col: "frame_main", x_col: "x", y_col: "y"})
    df["frame_main"] = pd.to_numeric(df["frame_main"], errors="raise").astype(int)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])

    if label_col and label_col in df.columns:
        df = df.rename(columns={label_col: "label"})
    else:
        guess_label = _auto_pick_column(df_raw, ["player", "server", "type", "rally"])
        df["label"] = df_raw.get(guess_label, pd.Series([""] * len(df_raw)))

    if event_col and event_col in df.columns:
        df = df.rename(columns={event_col: "event"})
    else:
        df["event"] = ""

    if color_col and color_col in df.columns:
        df["color_bgr"] = df[color_col].apply(parse_color_to_bgr)
    else:
        df["color_bgr"] = None

    if rally_id_col and rally_id_col in df.columns:
        df = df.rename(columns={rally_id_col: "rally_id"})
        df["rally_id"] = df["rally_id"].astype(str)
    else:
        df["rally_id"] = ""

    return df


def load_rally_segments_csv(
    csv_path: str,
    rally_id_col: str = "Score",
    start_col: str = "Start",
    end_col: str = "End",
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"RallySeg CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for c in [rally_id_col, start_col, end_col]:
        if c not in df.columns:
            raise ValueError(f"[RallySeg] 缺少必要欄位：{c}")

    df = df.rename(columns={rally_id_col: "rally_id", start_col: "start_main", end_col: "end_main"})
    df["rally_id"] = df["rally_id"].astype(str)
    df["start_main"] = pd.to_numeric(df["start_main"], errors="raise").astype(int)
    df["end_main"] = pd.to_numeric(df["end_main"], errors="raise").astype(int)
    return df


def map_positions_to_local_frames(pos: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    pos = pos.copy()
    if (pos.get("rally_id", "") != "").any():
        merged = pos.merge(seg[["rally_id", "start_main", "end_main"]], on="rally_id", how="left", validate="m:1")
    else:
        merged_list = []
        for _, r in pos.iterrows():
            fm = int(r["frame_main"])
            cand = seg[(seg["start_main"] <= fm) & (fm <= seg["end_main"])]
            if len(cand) > 0:
                rr = r.to_dict()
                rr["rally_id"] = str(cand.iloc[0]["rally_id"])
                rr["start_main"] = int(cand.iloc[0]["start_main"])
                rr["end_main"] = int(cand.iloc[0]["end_main"])
                merged_list.append(rr)
        if not merged_list:
            merged = pos.copy()
            merged["local_frame"] = -1
            return merged
        merged = pd.DataFrame(merged_list)

    merged["local_frame"] = (merged["frame_main"] - merged["start_main"]).astype(int)
    merged = merged[merged["local_frame"] >= 0]
    return merged


def _resolve_rally_id(seg_df: pd.DataFrame, rally_id_arg: str) -> str:
    rid = str(rally_id_arg)
    if (seg_df["rally_id"] == rid).any():
        return rid
    if rid.isdigit():
        idx = int(rid) - 1
        if 0 <= idx < len(seg_df):
            return str(seg_df.iloc[idx]["rally_id"])
    return rid


# ============= 視覺化繪製 =============

def draw_points(frame_img: np.ndarray, rows: pd.DataFrame, radius: int = 6, thickness: int = -1, show_label: bool = True) -> None:
    h, w = frame_img.shape[:2]
    for _, r in rows.iterrows():
        x, y = int(r["x"]), int(r["y"])
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        color = r["color_bgr"] if isinstance(r.get("color_bgr", None), tuple) else (0, 255, 255)
        cv2.circle(frame_img, (x, y), radius, color, thickness)
        if show_label and r.get("label", ""):
            cv2.putText(frame_img, str(r["label"]), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_trails(frame_img: np.ndarray, df_local: pd.DataFrame, current_local_frame: int, trail_len: int = 20, key_col: str = "label") -> None:
    start = max(current_local_frame - trail_len, 0)
    sub = df_local[(df_local["local_frame"] >= start) & (df_local["local_frame"] <= current_local_frame)].copy()
    if key_col not in sub.columns:
        sub[key_col] = ""
    for label, g in sub.groupby(key_col):
        g = g.sort_values("local_frame")
        pts = g[["x", "y"]].values.astype(int)
        if len(pts) < 2:
            continue
        color = (0, 255, 255)
        if "color_bgr" in g.columns and isinstance(g["color_bgr"].iloc[-1], tuple):
            color = g["color_bgr"].iloc[-1]
        for i in range(len(pts) - 1):
            p1, p2 = tuple(pts[i]), tuple(pts[i + 1])
            cv2.line(frame_img, p1, p2, color, 2)


def _cell_from_xy(x: int, y: int, W: int, H: int, grid: Tuple[int, int]) -> Tuple[int, int]:
    rows, cols = grid
    c = int(np.clip((x / max(W, 1e-6)) * cols, 0, cols - 1))
    r = int(np.clip((y / max(H, 1e-6)) * rows, 0, rows - 1))
    return (r, c)


def decide_highlight_cell_event(event_str: str, event_to_cell: Dict[str, Tuple[int, int]] = DEFAULT_EVENT_TO_CELL) -> Optional[Tuple[int, int]]:
    if not event_str:
        return None
    s = event_str.lower()
    for key, cell in event_to_cell.items():
        if key in s:
            return cell
    return None


def draw_zone_panel(
    frame_img: np.ndarray,
    panel_size: Tuple[int, int] = (240, 240),
    grid: Tuple[int, int] = DEFAULT_PANEL_GRID,
    highlight_cell: Optional[Tuple[int, int]] = None,
    counts: Optional[np.ndarray] = None,
    anchor: str = "br",
    margin: int = 12,
    label: str = "區域指示",
) -> None:
    H, W = frame_img.shape[:2]
    pw, ph = panel_size
    if anchor == "br":
        x0, y0 = W - pw - margin, H - ph - margin
    elif anchor == "tr":
        x0, y0 = W - pw - margin, margin
    elif anchor == "bl":
        x0, y0 = margin, H - ph - margin
    else:
        x0, y0 = margin, margin

    panel = np.full((ph, pw, 3), (24, 24, 24), dtype=np.uint8)
    rows, cols = grid
    cw, ch = pw // cols, ph // rows

    # 標題條
    cv2.rectangle(panel, (0, 0), (pw, 24), (40, 40, 40), -1)
    cv2.putText(panel, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # 畫格子 + 標籤 + 計數
    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * cw, r * ch
            x2, y2 = x1 + cw, y1 + ch
            # 背景
            if highlight_cell is not None and (r, c) == highlight_cell:
                cv2.rectangle(panel, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (0, 255, 255), -1)
                cv2.rectangle(panel, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), (0, 0, 0), 2)
            else:
                cv2.rectangle(panel, (x1, y1), (x2, y2), (64, 64, 64), 1)

            # 角落標籤（行/列）
            row_name = GRID_ROW_LABELS[r] if r < len(GRID_ROW_LABELS) else f"R{r+1}"
            col_name = GRID_COL_LABELS[c] if c < len(GRID_COL_LABELS) else f"C{c+1}"
            tag = f"{row_name}-{col_name}"
            cv2.putText(panel, tag, (x1 + 6, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (220, 220, 220) if highlight_cell != (r, c) else (0, 0, 0), 1, cv2.LINE_AA)

            # 累計次數
            if counts is not None and r < counts.shape[0] and c < counts.shape[1]:
                center = (x1 + cw // 2 - 8, y1 + ch // 2 + 6)
                text = str(int(counts[r, c]))
                cv2.putText(panel, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255) if highlight_cell != (r, c) else (0, 0, 0), 2, cv2.LINE_AA)

    # 疊到主畫面
    roi = frame_img[y0:y0 + ph, x0:x0 + pw]
    blended = cv2.addWeighted(panel, 0.92, roi, 0.08, 0)
    frame_img[y0:y0 + ph, x0:x0 + pw] = blended


def heatmap_accumulate(heatmap: np.ndarray, rows: pd.DataFrame, sigma: int = 16, strength: float = 1.0) -> None:
    H, W = heatmap.shape[:2]
    for _, r in rows.iterrows():
        x, y = int(r["x"]), int(r["y"])
        if x < 0 or y < 0 or x >= W or y >= H:
            continue
        size = max(9, sigma * 3)
        x1, y1 = max(0, x - size), max(0, y - size)
        x2, y2 = min(W - 1, x + size), min(H - 1, y + size)
        patch = heatmap[y1:y2 + 1, x1:x2 + 1]
        px, py = x - x1, y - y1
        gx = cv2.getGaussianKernel(ksize=2 * size + 1, sigma=sigma)
        gy = cv2.getGaussianKernel(ksize=2 * size + 1, sigma=sigma)
        g = gy @ gx.T
        g = g / (g.max() + 1e-6)
        g_crop = g[(size - py):(size - py + patch.shape[0]), (size - px):(size - px + patch.shape[1])]
        if g_crop.shape == patch.shape:
            patch += (g_crop * strength).astype(np.float32)


def overlay_heatmap(frame_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> None:
    hm_norm = heatmap.copy()
    if hm_norm.max() > 0:
        hm_norm = (hm_norm / hm_norm.max()) * 255.0
    hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(hm_color, alpha, frame_img, 1 - alpha, 0)
    frame_img[:, :] = blended


# ============= 主程式 =============

def main():
    ap = argparse.ArgumentParser(description="Rally 分段影片視覺化（面板可依座標/事件高亮，可輸出影片）")
    ap.add_argument("--video", required=True, help="rally 分段影片路徑（mp4/avi）")
    ap.add_argument("--positions", required=True, help="set1（或同等）座標 CSV（含主影片 frame）")
    ap.add_argument("--rallyseg", default="RallySeg.csv", help="RallySeg.csv（含 Start/End）")
    ap.add_argument("--rally-id", required=True, help="要播的 rally（Score 如 1_00_01，或整數序號 1 表示第一列）")

    # 欄位對應（可覆寫）
    ap.add_argument("--pos-frame-col", default="", help="positions 的主影片幀欄位（留空自動偵測，set1 預設 frame_num）")
    ap.add_argument("--pos-x-col", default="", help="positions 的 x 欄位（留空自動偵測，set1 預設 player_location_x）")
    ap.add_argument("--pos-y-col", default="", help="positions 的 y 欄位（留空自動偵測，set1 預設 player_location_y）")
    ap.add_argument("--pos-label-col", default="", help="positions 的 label 欄位（可留空）")
    ap.add_argument("--pos-event-col", default="", help="positions 的 event 欄位（可留空）")
    ap.add_argument("--pos-color-col", default="", help="positions 的 color 欄位（可留空）")
    ap.add_argument("--pos-rally-col", default="", help="positions 的 rally_id 欄（若沒有可留空）")

    ap.add_argument("--seg-rally-col", default="Score", help="RallySeg 的 rally 欄位名（預設 Score）")
    ap.add_argument("--seg-start-col", default="Start", help="RallySeg 的主影片起始幀欄位名（預設 Start）")
    ap.add_argument("--seg-end-col", default="End", help="RallySeg 的主影片結束幀欄位名（預設 End）")

    # 顯示/面板/輸出
    ap.add_argument("--show-label", action="store_true", help="顯示 label")
    ap.add_argument("--trail", type=int, default=20, help="軌跡往回幾幀")
    ap.add_argument("--panel", action="store_true", help="顯示右下角區域面板")
    ap.add_argument("--panel-anchor", default="br", choices=["br", "tr", "bl", "tl"], help="面板位置")
    ap.add_argument("--panel-mode", default="position", choices=["position", "event"], help="面板高亮方式（預設 position）")
    ap.add_argument("--panel-counts", action="store_true", help="在格子中顯示累計次數")
    ap.add_argument("--heatmap", action="store_true", help="顯示熱點圖（持續累積）")
    ap.add_argument("--resize-width", type=int, default=0, help="視窗寬度縮放（0 = 不縮放）")
    ap.add_argument("--no-window", action="store_true", help="不顯示視窗（純輸出）")
    # 影片輸出
    ap.add_argument("--out-video", default="", help="輸出影片路徑（如 output.mp4；留空則不輸出）")
    ap.add_argument("--fourcc", default="mp4v", help="輸出編碼器 FourCC（如 mp4v、XVID、avc1）")
    ap.add_argument("--fps", type=float, default=0.0, help="輸出 FPS；0 表示沿用輸入影片 FPS")

    args = ap.parse_args()

    # 讀資料
    df_pos = load_positions_csv(
        csv_path=args.positions,
        frame_col=args.pos_frame_col or None,
        x_col=args.pos_x_col or None,
        y_col=args.pos_y_col or None,
        label_col=args.pos_label_col if args.pos_label_col else None,
        event_col=args.pos_event_col if args.pos_event_col else None,
        color_col=args.pos_color_col if args.pos_color_col else None,
        rally_id_col=args.pos_rally_col if args.pos_rally_col else None,
    )
    df_seg = load_rally_segments_csv(
        csv_path=args.rallyseg,
        rally_id_col=args.seg_rally_col,
        start_col=args.seg_start_col,
        end_col=args.seg_end_col,
    )
    df_local_all = map_positions_to_local_frames(df_pos, df_seg)

    # 解析 rally-id
    target_rally_id = _resolve_rally_id(df_seg, str(args.rally_id))
    if not (df_seg["rally_id"] == target_rally_id).any():
        print(f"[警告] 找不到 rally '{args.rally_id}'；請確認 --rally-id 是否為 Score（如 1_00_01）或正確序號。")
        sys.exit(2)
    df_local = df_local_all[df_local_all["rally_id"] == target_rally_id].copy()
    if df_local.empty:
        print(f"[警告] 對齊後沒有該 rally 的座標資料：{target_rally_id}（檢查 set1 與 RallySeg 是否一致）")
        sys.exit(2)

    # 開影片
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"無法開啟影片：{args.video}")
        sys.exit(1)

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = args.fps if args.fps > 0 else in_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, sample = cap.read()
    if not ok or sample is None:
        print("讀不到影片影格。")
        sys.exit(1)
    if args.resize_width and sample.shape[1] != args.resize_width:
        scale = args.resize_width / sample.shape[1]
        sample = cv2.resize(sample, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    H, W = sample.shape[:2]

    # 輸出
    writer = None
    if args.out_video:
        fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
        writer = cv2.VideoWriter(args.out_video, fourcc, out_fps, (W, H))
        if not writer.isOpened():
            print(f"無法建立輸出影片：{args.out_video}（檢查副檔名與 FourCC）")
            sys.exit(1)

    if not args.no_window:
        cv2.namedWindow(f"Rally {target_rally_id} Visualizer", cv2.WINDOW_NORMAL)

    # 熱圖與面板累計
    heatmap = None
    panel_counts = np.zeros(DEFAULT_PANEL_GRID, dtype=np.int32) if args.panel_counts else None
    show_heatmap = bool(args.heatmap)
    show_panel = bool(args.panel)
    paused = False

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        if not paused:
            ret, frame_img = cap.read()
            if not ret:
                break
            local_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        else:
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame_img = cap.read()
            local_idx = int(pos) - 1

        if frame_img is None:
            break

        if args.resize_width and frame_img.shape[1] != args.resize_width:
            scale = args.resize_width / frame_img.shape[1]
            frame_img = cv2.resize(frame_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if heatmap is None and show_heatmap:
            heatmap = np.zeros((frame_img.shape[0], frame_img.shape[1]), dtype=np.float32)

        rows_now = df_local[df_local["local_frame"] == local_idx]

        # 軌跡
        if args.trail > 0:
            draw_trails(frame_img, df_local, current_local_frame=local_idx, trail_len=args.trail, key_col="label")

        # 點
        draw_points(frame_img, rows_now, radius=6, thickness=-1, show_label=args.show_label)

        # 熱圖
        if show_heatmap and heatmap is not None:
            heatmap_accumulate(heatmap, rows_now, sigma=14, strength=1.0)
            overlay_heatmap(frame_img, heatmap, alpha=0.35)

        # 面板
        if show_panel:
            highlight = None
            if args.panel_mode == "event":
                ev = ""
                if "event" in rows_now.columns and len(rows_now) > 0:
                    ev = next((e for e in rows_now["event"].astype(str).tolist() if e), "")
                highlight = decide_highlight_cell_event(ev, DEFAULT_EVENT_TO_CELL)
            else:  # position 模式
                # 取此幀所有點都計入：用最近一個點或所有點均可；這裡以「所有點」計數，並用最後一個點高亮
                last_cell = None
                for _, r in rows_now.iterrows():
                    cy, cx = int(r["y"]), int(r["x"])
                    rr, cc = _cell_from_xy(cx, cy, frame_img.shape[1], frame_img.shape[0], DEFAULT_PANEL_GRID)
                    last_cell = (rr, cc)
                    if panel_counts is not None:
                        panel_counts[rr, cc] += 1
                highlight = last_cell  # 若此幀沒有點，保持 None

            draw_zone_panel(
                frame_img,
                panel_size=(min(260, frame_img.shape[1] // 4), min(260, frame_img.shape[0] // 4)),
                grid=DEFAULT_PANEL_GRID,
                highlight_cell=highlight,
                counts=panel_counts,
                anchor=args.panel_anchor,
                margin=12,
                label=f"區域指示（{args.panel_mode}）",
            )

        # HUD
        hud = f"Rally {target_rally_id} | local_frame {local_idx}/{total_frames-1}"
        cv2.putText(frame_img, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 255), 2, cv2.LINE_AA)

        # 寫出
        if writer is not None:
            writer.write(frame_img)

        # 顯示
        if not args.no_window:
            cv2.imshow(f"Rally {target_rally_id} Visualizer", frame_img)
            key = cv2.waitKey(1 if not paused else 20) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == 32:
                paused = not paused
            elif key == ord('h'):
                show_heatmap = not show_heatmap
                if not show_heatmap:
                    heatmap = None
            elif key == ord('m'):
                show_panel = not show_panel
            elif key == ord('s'):
                out_path = f"rally_{target_rally_id}_local_{local_idx:06d}.png"
                cv2.imwrite(out_path, frame_img)
                print(f"Saved: {out_path}")
            elif key in (81, ord('a')):
                paused = True
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(local_idx - 1, 0))
            elif key in (83, ord('d')):
                paused = True
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(local_idx + 1, total_frames - 1))

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[完成] 已輸出影片：{args.out_video}")
    if not args.no_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
