#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rally Video Visualizer (v2: 支援輸出影片)
- 在【rally分段影片】上疊加標記與軌跡
- 以 local_frame = set1.frame_num - RallySeg.Start 對齊
- RallySeg.csv 欄位：Score / Start / End（或用參數覆寫）
- 預設以 set1 的 player_location_x / player_location_y 畫點（可用參數覆寫）
- 可即時預覽 + 匯出 mp4；或 pure 批次匯出（--no-window）
- 熱鍵：Space 暫停、←/→ 單步、H/M 熱圖/面板、S 存圖、Q/ESC 離開
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
    "net": (2, 1),
    "smash": (0, 1),
    "drive": (1, 1),
}
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
    """
    讀取 set1（或同等）座標資料。
    標準化欄：frame_main, x, y, label, event, color_bgr, rally_id
    """
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
    rally_id_col: str = "Score",   # 你的 RallySeg.csv 用 Score 當辨識（如 1_00_01）
    start_col: str = "Start",
    end_col: str = "End",
) -> pd.DataFrame:
    """
    讀取 RallySeg.csv，標準化欄：rally_id, start_main, end_main
    """
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
    """
    local_frame = frame_main - start_main（依每筆所屬 rally）
    - 若 positions 含 rally_id → 直接 merge
    - 否則以 frame_main ∈ [start_main, end_main] 反查
    """
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
    """
    支援二種輸入：
      - Score（與 seg_df['rally_id'] 比對），如 '1_00_01'
      - 整數序號（1-based），如 '1' 表示 seg_df 的第 1 列
    """
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


def decide_highlight_cell(event_str: str, event_to_cell: Dict[str, Tuple[int, int]] = DEFAULT_EVENT_TO_CELL) -> Optional[Tuple[int, int]]:
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

    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * cw, r * ch
            x2, y2 = x1 + cw, y1 + ch
            if highlight_cell is not None and (r, c) == highlight_cell:
                cv2.rectangle(panel, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (0, 255, 255), -1)
                cv2.rectangle(panel, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), (0, 0, 0), 2)
            else:
                cv2.rectangle(panel, (x1, y1), (x2, y2), (64, 64, 64), 1)

    cv2.rectangle(panel, (0, 0), (pw, 24), (40, 40, 40), -1)
    cv2.putText(panel, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    frame_img[y0:y0 + ph, x0:x0 + pw] = cv2.addWeighted(panel, 0.92, frame_img[y0:y0 + ph, x0:x0 + pw], 0.08, 0)


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
    ap = argparse.ArgumentParser(description="Rally 分段影片視覺化（以 Start/End 對齊，可輸出影片）")
    ap.add_argument("--video", required=True, help="rally 分段影片路徑（mp4/avi）")
    ap.add_argument("--positions", required=True, help="set1（或同等）座標 CSV（含主影片 frame）")
    ap.add_argument("--rallyseg", default="RallySeg.csv", help="RallySeg.csv（含 Start/End）")
    ap.add_argument("--rally-id", required=True, help="要播放的 rally（可填 Score 如 1_00_01，或整數序號 1 表示第一列）")

    # 欄位對應（可覆寫）
    ap.add_argument("--pos-frame-col", default="", help="positions 的主影片幀欄位（留空自動偵測，set1 預設 frame_num）")
    ap.add_argument("--pos-x-col", default="", help="positions 的 x 欄位（留空自動偵測，set1 預設 player_location_x）")
    ap.add_argument("--pos-y-col", default="", help="positions 的 y 欄位（留空自動偵測，set1 預設 player_location_y）")
    ap.add_argument("--pos-label-col", default="", help="positions 的 label 欄位（可留空）")
    ap.add_argument("--pos-event-col", default="", help="positions 的 event 欄位（可留空）")
    ap.add_argument("--pos-color-col", default="", help="positions 的 color 欄位（可留空）")
    ap.add_argument("--pos-rally-col", default="", help="positions 的 rally_id 欄（若沒有可留空）")

    ap.add_argument("--seg-rally-col", default="Score", help="RallySeg 的 rally 欄位名（你的檔是 Score）")
    ap.add_argument("--seg-start-col", default="Start", help="RallySeg 的主影片起始幀欄位名（預設 Start）")
    ap.add_argument("--seg-end-col", default="End", help="RallySeg 的主影片結束幀欄位名（預設 End）")

    # 顯示/輸出選項
    ap.add_argument("--show-label", action="store_true", help="顯示 label")
    ap.add_argument("--trail", type=int, default=20, help="軌跡往回幾幀")
    ap.add_argument("--panel", action="store_true", help="顯示右下角區域面板")
    ap.add_argument("--panel-anchor", default="br", choices=["br", "tr", "bl", "tl"], help="面板位置")
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

    # 先取第一幀確定尺寸（也讓 VideoWriter 正確初始化）
    ok, sample = cap.read()
    if not ok or sample is None:
        print("讀不到影片影格。")
        sys.exit(1)

    if args.resize_width and sample.shape[1] != args.resize_width:
        scale = args.resize_width / sample.shape[1]
        sample = cv2.resize(sample, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    H, W = sample.shape[:2]

    # 準備輸出
    writer = None
    if args.out_video:
        fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
        writer = cv2.VideoWriter(args.out_video, fourcc, out_fps, (W, H))
        if not writer.isOpened():
            print(f"無法建立輸出影片：{args.out_video}（檢查副檔名與 FourCC）")
            sys.exit(1)

    # 視窗
    win = f"Rally {target_rally_id} Visualizer"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # 熱點圖
    heatmap = None
    show_heatmap = bool(args.heatmap)
    show_panel = bool(args.panel)
    paused = False

    # 已讀一幀，回到開頭
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

        # 縮放
        if args.resize_width and frame_img.shape[1] != args.resize_width:
            scale = args.resize_width / frame_img.shape[1]
            frame_img = cv2.resize(frame_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # 初始化熱圖大小
        if heatmap is None and show_heatmap:
            heatmap = np.zeros((frame_img.shape[0], frame_img.shape[1]), dtype=np.float32)

        # 取此 local_frame 的資料
        rows = df_local[df_local["local_frame"] == local_idx]

        # 畫軌跡（用 local_frame 時間軸）
        if args.trail > 0:
            draw_trails(frame_img, df_local, current_local_frame=local_idx, trail_len=args.trail, key_col="label")

        # 畫點
        draw_points(frame_img, rows, radius=6, thickness=-1, show_label=args.show_label)

        # 熱圖
        if show_heatmap and heatmap is not None:
            heatmap_accumulate(heatmap, rows, sigma=14, strength=1.0)
            overlay_heatmap(frame_img, heatmap, alpha=0.35)

        # 區域面板
        if show_panel:
            event_str = ""
            if "event" in rows.columns and len(rows) > 0:
                event_str = next((e for e in rows["event"].astype(str).tolist() if e), "")
            highlight = decide_highlight_cell(event_str, DEFAULT_EVENT_TO_CELL)
            draw_zone_panel(
                frame_img,
                panel_size=(min(260, frame_img.shape[1] // 4), min(260, frame_img.shape[0] // 4)),
                grid=DEFAULT_PANEL_GRID,
                highlight_cell=highlight,
                anchor=args.panel_anchor,
                margin=12,
                label="區域指示",
            )

        # HUD
        hud = f"Rally {target_rally_id} | local_frame {local_idx}/{total_frames-1}"
        cv2.putText(frame_img, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 255), 2, cv2.LINE_AA)

        # 輸出到影片
        if writer is not None:
            writer.write(frame_img)

        # 顯示
        if not args.no_window:
            cv2.imshow(win, frame_img)
            key = cv2.waitKey(1 if not paused else 20) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == 32:  # Space
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
            elif key == 81 or key == ord('a'):  # ←
                paused = True
                new_idx = max(local_idx - 1, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
            elif key == 83 or key == ord('d'):  # →
                paused = True
                new_idx = min(local_idx + 1, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)

        else:
            # 批次模式不顯示視窗
            pass

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[完成] 已輸出影片：{args.out_video}")
    if not args.no_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
