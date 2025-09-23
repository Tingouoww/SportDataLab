# -*- coding: utf-8 -*-
"""
在影片上：
1) 每位選手只畫「最近一段」箭頭（上一拍→最新拍），兩端點都有點（最新拍點帶 A/B 標籤）。
2) 右上(B)/右下(A) 各有一個五向箭頭面板；每拍依規則亮起單一方向：
   優先序：
     (1) aroundhead==1 → UP
     (2) hit_height==1 & backhand==1 → UL
     (3) hit_height==2 & backhand==1 → DL
     (4) hit_height==2 → DR
     (5) hit_height==1 → UR
"""

import argparse
import numpy as np
import pandas as pd
import cv2
from typing import Dict, Tuple, Optional, List

# ---------- 基礎 I/O ----------

def read_csv_zh(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp950", "big5", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def normalize_player(p) -> str:
    if isinstance(p, str):
        s = p.strip().upper()
        if s.startswith("A"): return "A"
        if s.startswith("B"): return "B"
    try:
        v = int(p)
        if v == 1: return "A"
        if v == 2: return "B"
    except Exception:
        pass
    return str(p)

def load_rallyseg(rallyseg_path: str) -> pd.DataFrame:
    df = read_csv_zh(rallyseg_path)
    lower = {c.lower(): c for c in df.columns}
    if "start" not in lower or "end" not in lower:
        raise ValueError("RallySeg.csv 必須包含 'Start' 與 'End' 欄位")
    df = df.rename(columns={lower["start"]: "Start", lower["end"]: "End"})
    if "rally" in lower:
        df = df.rename(columns={lower["rally"]: "Rally"})
    else:
        df["Rally"] = np.arange(1, len(df)+1, dtype=int)
    for c in ["Rally","Start","End"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["Rally","Start","End"]]

# ---------- 建立 A/B 時間序列（含條件欄位） ----------

def build_series(set1: pd.DataFrame, rally_id: int, start_global: int, debug=False) -> Dict[str, pd.DataFrame]:
    """
    回傳 {'A': dfA, 'B': dfB}；欄位包含：
    local_frame, x, y, aroundhead, backhand, hit_height
    """
    sub = set1[set1["rally"] == rally_id].copy()
    sub["local_frame"] = pd.to_numeric(sub["frame_num"], errors="coerce") - int(start_global)

    need = ["player","player_location_x","player_location_y","local_frame","ball_round",
            "aroundhead","backhand","hit_height"]
    for c in need:
        if c not in sub.columns:
            raise ValueError(f"set1.csv 缺少欄位：{c}")

    sub = sub.dropna(subset=["player_location_x","player_location_y","local_frame"])
    sub = sub[sub["local_frame"] >= 0]
    sub["player_norm"] = sub["player"].apply(normalize_player)

    # 數值化條件欄位
    for c in ["aroundhead","backhand","hit_height"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0).astype(int)

    # 依 local_frame, ball_round 穩定排序
    sub = sub.sort_values(by=["local_frame","ball_round"], kind="mergesort").reset_index(drop=True)

    series: Dict[str, pd.DataFrame] = {}
    for p in ["A","B"]:
        cols = ["local_frame","player_location_x","player_location_y","aroundhead","backhand","hit_height"]
        dfp = sub[sub["player_norm"]==p][cols].copy()
        for col in ["local_frame","player_location_x","player_location_y"]:
            dfp[col] = pd.to_numeric(dfp[col], errors="coerce")
        dfp = dfp.dropna()
        # 同一個 local_frame 僅保留第一筆
        dfp = dfp.drop_duplicates(subset=["local_frame"], keep="first").sort_values("local_frame")
        series[p] = dfp.reset_index(drop=True)
        if debug:
            print(f"[DEBUG] {p} samples: {len(series[p])}")
    return series

# ---------- 幾何輔助 ----------

def to_int_pt(x, y, w, h) -> Optional[Tuple[int,int]]:
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    xi, yi = int(round(x)), int(round(y))
    if 0 <= xi < w and 0 <= yi < h:
        return (xi, yi)
    return None

def put_marker(frame, pt: Optional[Tuple[int,int]], label: Optional[str], color, radius=6, font_scale=0.6):
    if pt is None: return
    h, w = frame.shape[:2]
    x, y = pt
    if x < 0 or y < 0 or x >= w or y >= h: return
    cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    if label:
        cv2.putText(frame, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

# ---------- 面板繪製（五向箭頭） ----------

DIRS = ["UL","UP","UR","DL","DR"]  # 左上、上、右上、左下、右下

def choose_dir(aroundhead: int, hit_height: int, backhand: int) -> Optional[str]:
    """依優先序回傳要亮的方向字串；若皆不符合回傳 None。"""
    if aroundhead == 1:
        return "UP"
    if hit_height == 1 and backhand == 1:
        return "UL"
    if hit_height == 2 and backhand == 1:
        return "DL"
    if hit_height == 2:
        return "DR"
    if hit_height == 1:
        return "UR"
    return None

def _poly_arrow(center, angle_deg, length, width_head, width_shaft) -> np.ndarray:
    """
    產生一支朝向 angle_deg 的箭頭多邊形（以 center 為基準）。
    回傳為 int32 Nx2 頂點陣列（可給 fillPoly 或 polylines）。
    """
    # 箭頭設計：一條主軸 + 三角箭頭；在本函式中用一個近似多邊形
    L  = length
    ws = width_shaft
    wh = width_head
    # 基本形狀（朝上，y- 方向）：從下到上
    pts = np.array([
        [-ws,  L*0.35], [ ws,  L*0.35],  # 桿上緣
        [ ws,  L*0.00], [ wh,  L*0.00],  # 轉入箭頭
        [ 0 , -L*0.45], [-wh, 0.00],     # 箭頭兩側
        [-ws, 0.00]
    ], dtype=float)
    # 旋轉
    th = np.deg2rad(angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]], dtype=float)
    pts = pts @ R.T
    pts[:,0] += center[0]
    pts[:,1] += center[1]
    return pts.astype(np.int32)

def draw_five_arrows_panel(frame, rect, highlight_dir: Optional[str], color_outline=(0,0,255), color_fill=(0,0,255), label=None):
    """
    在 rect=(x,y,w,h) 面板中畫 5 支空心箭頭；若 highlight_dir 指定其一，則將該方向**實心**填色。
    方向配置：
        UL  UP  UR
            .
        DL      DR
    """
    x,y,w,h = rect
    cx = x + w//2
    cy = y + h//2
    # 五個位置（相對中心，一點間距）(每支箭頭頸部中心點)
    r = min(w,h) * 0.34
    centers = {
        "UP": (cx,        int(cy - r*0.85)),
        "UL": (int(cx - r*0.75), int(cy - r*0.25)),
        "UR": (int(cx + r*0.75), int(cy - r*0.25)),
        "DL": (int(cx - r*0.55), int(cy + r*0.65)),
        "DR": (int(cx + r*0.55), int(cy + r*0.65)),
    }
    angles = {"UP": 0, "UL": 315, "UR": 45, "DL": 225, "DR": 135}  # 影像座標系
    L = int(r*0.95); wh = int(L*0.22); ws = int(L*0.10)

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # 外框
    cv2.rectangle(frame, (x,y), (x+w,y+h), (40,40,40), 1)

    # 畫 5 支箭（空心）
    polys = {}
    for d in DIRS:
        poly = _poly_arrow(centers[d], angles[d], L, wh, ws)
        polys[d] = poly
        cv2.polylines(frame, [poly], isClosed=True, color=color_outline, thickness=2, lineType=cv2.LINE_AA)

    # 高亮（填色）
    if highlight_dir in polys:
        cv2.fillPoly(frame, [polys[highlight_dir]], color_fill, lineType=cv2.LINE_AA)

    if label:
        cv2.putText(frame, label, (x+6, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

def load_panel_images(panel_dir: str) -> dict:
    """
    從 panel_dir 載入五張圖：UL.png, UP.png, UR.png, DL.png, DR.png
    回傳 dict，例如: {"UP": np.ndarray or None, ...}
    支援含透明度的 PNG（有 alpha channel 會用作遮罩）。
    """
    import os
    imgs = {}
    for d in DIRS:
        path = os.path.join(panel_dir, f"{d}.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 可能是 RGBA
        imgs[d] = img  # 若不存在或讀不到，值為 None
    return imgs

def _blit_center_fit(dst: np.ndarray, rect, src: np.ndarray, pad_ratio=0.02):
    """
    將 src 圖片按比例縮放後，置中繪製到 dst 的 rect=(x,y,w,h) 範圍內。
    - 若 src 為 RGBA，使用 alpha 當遮罩；否則直接覆蓋（不縮放透明）。
    - pad_ratio: 留白比例避免圖片太貼邊。
    """
    x, y, w, h = rect
    if src is None: 
        return
    # 計算可用區域（留邊距）
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)
    tw = max(1, w - 2*pad_w)
    th = max(1, h - 2*pad_h)

    # 保持長寬比縮放
    sh, sw = src.shape[:2]
    scale = min(tw / sw, th / sh)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)

    # 貼圖位置（置中）
    ox = x + (w - nw)//2
    oy = y + (h - nh)//2

    # RGBA or BGR
    if resized.shape[2] == 4:
        bgr = resized[:, :, :3]
        alpha = resized[:, :, 3]
        # 建 mask（單通道→3通道）
        mask = cv2.merge([alpha, alpha, alpha])
        roi = dst[oy:oy+nh, ox:ox+nw]
        # 先把 ROI 中需要貼的地方清掉，再貼
        inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(roi, inv)
        fg = cv2.bitwise_and(bgr, mask)
        dst[oy:oy+nh, ox:ox+nw] = cv2.add(bg, fg)
    else:
        dst[oy:oy+nh, ox:ox+nw] = resized

def render_panel_image(frame, rect, highlight_dir: Optional[str], panel_imgs: dict,
                       label=None, alpha_bg=0.25, draw_border=True):
    """
    以半透明黑底 + 邊框 +（可選標籤）構成面板；若 highlight_dir 指定並且有對應圖片，就顯示該圖。
    沒有符合條件時，面板只顯示底與邊框（空白）。
    """
    x, y, w, h = rect

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha_bg, frame, 1 - alpha_bg, 0, frame)

    # 邊框
    if draw_border:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (40, 40, 40), 1)

    # 顯示標籤（例如 "A"/"B"）
    if label:
        cv2.putText(frame, label, (x+6, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # 顯示圖片（僅在 highlight_dir 有對應圖時）
    if highlight_dir in panel_imgs and panel_imgs[highlight_dir] is not None:
        _blit_center_fit(frame, rect, panel_imgs[highlight_dir])

# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser(description="Latest segment arrow + per-player 5-direction condition panels.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--set1", required=False, default="set1.csv")
    ap.add_argument("--rallyseg", required=False, default="RallySeg.csv")
    ap.add_argument("--rally_id", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rally_clip", action="store_true")
    ap.add_argument("--radius", type=int, default=6)
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--panel-dir", type=str, required=True,
                help="放五張面板圖的資料夾（檔名必須為 UL.png, UP.png, UR.png, DL.png, DR.png）")
    args = ap.parse_args()

    panel_imgs = load_panel_images(args.panel_dir)

    # 讀 CSV
    set1 = read_csv_zh(args.set1)
    need_cols = ["rally","ball_round","player","frame_num","player_location_x","player_location_y",
                 "aroundhead","backhand","hit_height","end_frame_num"]
    for c in need_cols:
        if c not in set1.columns:
            raise ValueError(f"set1.csv 缺少欄位：{c}")
    # 數值化
    for c in ["rally","ball_round","frame_num","end_frame_num","player_location_x","player_location_y",
              "aroundhead","backhand","hit_height"]:
        set1[c] = pd.to_numeric(set1[c], errors="coerce")

    rseg = load_rallyseg(args.rallyseg)
    row = rseg[rseg["Rally"]==int(args.rally_id)]
    if row.empty:
        if 1 <= args.rally_id <= len(rseg):
            row = rseg.iloc[[args.rally_id-1]]
        else:
            raise ValueError(f"Rally {args.rally_id} 不在 RallySeg 中")
    start_global = int(row["Start"].values[0])
    end_global   = int(row["End"].values[0])
    if end_global <= start_global:
        raise ValueError("RallySeg 的 Start/End 不合理（End <= Start）")

    series = build_series(set1, int(args.rally_id), start_global, debug=args.debug)
    A = series["A"]; B = series["B"]

    # 取 numpy
    def prep(df: pd.DataFrame):
        if df is None or df.empty:
            return (np.array([],dtype=int), np.array([]), np.array([]),
                    np.array([],dtype=int), np.array([],dtype=int), np.array([],dtype=int))
        return (df["local_frame"].to_numpy(dtype=int),
                df["player_location_x"].to_numpy(dtype=float),
                df["player_location_y"].to_numpy(dtype=float),
                df["aroundhead"].to_numpy(dtype=int),
                df["hit_height"].to_numpy(dtype=int),
                df["backhand"].to_numpy(dtype=int))

    A_f, A_x, A_y, A_ar, A_hh, A_bh = prep(A)
    B_f, B_x, B_y, B_ar, B_hh, B_bh = prep(B)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = in_fps if args.fps <= 0 else float(args.fps)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not args.rally_clip:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_global)
        local_len = end_global - start_global + 1
    else:
        local_len = min(end_global - start_global + 1, total)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"無法開啟輸出檔：{args.out}")

    # 顏色（BGR）
    colorA = (0, 0, 0)       # 黑（你上一段的配置）
    colorB = (255, 0, 0)     # 紅（注意：OpenCV BGR，紅色是 (0,0,255)）

    # 面板尺寸與位置（右上 B、右下 A）
    panel_w = max(150, w // 5)
    panel_h = max(150, h // 5)
    margin  = 12
    rect_B  = (w - panel_w - margin, margin, panel_w, panel_h)
    rect_A  = (w - panel_w - margin, h - panel_h - margin, panel_w, panel_h)
    

    if args.debug:
        print(f"[DEBUG] frames={total}, local_len={local_len}, fps_in={in_fps}, fps_out={fps}, size=({w}x{h})")
        print(f"[DEBUG] A shots={len(A_f)}, B shots={len(B_f)}")

    for i in range(local_len):
        ret, frame = cap.read()
        if not ret:
            if args.debug: print(f"[WARN] 提前讀完影片 at local frame {i}")
            break

        # ------ A：最近一段箭頭 + 面板方向 ------
        dirA = None
        if len(A_f) > 0:
            j = np.searchsorted(A_f, i, side="right") - 1

            # 面板條件：使用「當前最新拍 j」的條件欄位
            if j >= 0:
                dirA = choose_dir(A_ar[j] if j < len(A_ar) else 0,
                                  A_hh[j] if j < len(A_hh) else 0,
                                  A_bh[j] if j < len(A_bh) else 0)

            # 最近一段箭頭可視化
            if j == 0:
                p = to_int_pt(A_x[0], A_y[0], w, h)
                put_marker(frame, p, "A", colorA, radius=args.radius)
            elif j >= 1:
                p0 = to_int_pt(A_x[j-1], A_y[j-1], w, h)
                p1 = to_int_pt(A_x[j],   A_y[j],   w, h)
                if p0 and p1:
                    put_marker(frame, p0, None, colorA, radius=args.radius)
                    cv2.arrowedLine(frame, p0, p1, colorA, args.thickness, line_type=cv2.LINE_AA, tipLength=0.15)
                    put_marker(frame, p1, "A", colorA, radius=args.radius)

        # ------ B：最近一段箭頭 + 面板方向 ------
        dirB = None
        if len(B_f) > 0:
            j = np.searchsorted(B_f, i, side="right") - 1
            if j >= 0:
                dirB = choose_dir(B_ar[j] if j < len(B_ar) else 0,
                                  B_hh[j] if j < len(B_hh) else 0,
                                  B_bh[j] if j < len(B_bh) else 0)

            if j == 0:
                p = to_int_pt(B_x[0], B_y[0], w, h)
                put_marker(frame, p, "B", colorB, radius=args.radius)
            elif j >= 1:
                p0 = to_int_pt(B_x[j-1], B_y[j-1], w, h)
                p1 = to_int_pt(B_x[j],   B_y[j],   w, h)
                if p0 and p1:
                    put_marker(frame, p0, None, colorB, radius=args.radius)
                    cv2.arrowedLine(frame, p0, p1, colorB, args.thickness, line_type=cv2.LINE_AA, tipLength=0.15)
                    put_marker(frame, p1, "B", colorB, radius=args.radius)

        # 畫兩個面板（五向箭頭，單一方向高亮）
        render_panel_image(frame, rect_B, dirB, panel_imgs, label="B", alpha_bg=0.25)
        render_panel_image(frame, rect_A, dirA, panel_imgs, label="A", alpha_bg=0.25)

        # 顯示資訊
        cv2.putText(frame, f"Rally {args.rally_id} | Frame {i}/{local_len-1}",
                    (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235,235,235), 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()
