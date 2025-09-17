# -*- coding: utf-8 -*-
"""
每位選手僅顯示「最近一段」箭頭（上一拍→最新拍），並在兩端點都畫上點：
- 只有第一拍：畫該拍的點
- 第 j 拍 (j>=2)：畫 (j-1)->j 的箭頭，並在 p0(上一拍) 與 p1(本拍) 都畫點；p1 顯示 A/B 標籤

用法（你的情境：影片已裁切成單一 rally 段落）：
python annotate_rally_video_latest_arrow.py \
  --video rally1_annot.avi \
  --set1 set1.csv \
  --rallyseg RallySeg.csv \
  --rally_id 1 \
  --rally_clip \
  --out rally1_latestarrow.mp4 \
  --debug
"""

import argparse
import numpy as np
import pandas as pd
import cv2
from typing import Dict, Tuple, Optional

def read_csv_zh(path: str) -> pd.DataFrame:
    """
    讀取csv檔
    encodings : 嘗試不同encode方式
    回傳pandas.DataFrame
    """
    encodings = ["utf-8", "utf-8-sig", "cp950", "big5", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def normalize_player(p) -> str:
    """
    標準化player為A/B
    1. 若為string : 看開頭是A還是B並回傳
    2. 若是數字 : 1 -> A, 2 -> B
    3. 若皆非 : 回傳字串原樣
    (但set1 player欄位僅A,B)
    """
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
    """
    讀取 RallySeg.csv
    標準化欄位名稱Start/End/Rally (but RallySeg.csv doesn't have column 'Rally')
    沒有 'Rally' 自動用行序1,2,3....補上
    取 Rally, Start, End 欄位轉成數字並回傳僅含三欄位的DataFrame
    """
    df = read_csv_zh(rallyseg_path) # 讀取rallyseg檔案(for rally start frame)
    lower = {c.lower(): c for c in df.columns}
    if "start" not in lower or "end" not in lower:
        raise ValueError("RallySeg.csv 必須包含 'Start' 與 'End' 欄位")
    df = df.rename(columns={lower["start"]: "Start", lower["end"]: "End"})
    if "rally" in lower:
        df = df.rename(columns={lower["rally"]: "Rally"})
    else:
        df["Rally"] = np.arange(1, len(df)+1, dtype=int)
    for c in ["Rally", "Start", "End"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["Rally", "Start", "End"]]

def build_series(set1: pd.DataFrame, rally_id: int, start_global: int, debug=False) -> Dict[str, pd.DataFrame]:
    """
    從 set1 中擷取 rally 資料 ["player", "player_location_x", "player_location_y", "local_frame", "ball_round"]
    1. local_frame = frame_num - start
    2. 依據 local_frame, ball_round排序
    回傳 dict{'A':dfA, 'B':dfB}
    """
    sub = set1[set1["rally"] == rally_id].copy()
    if debug:
        print(f"[DEBUG] set1 rows for rally {rally_id}: {len(sub)}")
    sub["local_frame"] = pd.to_numeric(sub["frame_num"], errors="coerce") - int(start_global)

    need = ["player", "player_location_x", "player_location_y", "local_frame", "ball_round"]
    sub = sub.dropna(subset=need) #移除含空缺值資料列
    sub = sub[sub["local_frame"] >= 0]
    sub["player_norm"] = sub["player"].apply(normalize_player)
    sub = sub.sort_values(by=["local_frame", "ball_round"], kind="mergesort").reset_index(drop=True)

    series = {}
    for p in ["A", "B"]:
        dfp = sub[sub["player_norm"] == p][["local_frame", "player_location_x", "player_location_y"]].copy()
        dfp["local_frame"] = pd.to_numeric(dfp["local_frame"], errors="coerce")
        dfp["player_location_x"] = pd.to_numeric(dfp["player_location_x"], errors="coerce")
        dfp["player_location_y"] = pd.to_numeric(dfp["player_location_y"], errors="coerce")
        dfp = dfp.dropna()
        dfp = dfp.drop_duplicates(subset=["local_frame"], keep="first").sort_values("local_frame")
        series[p] = dfp.reset_index(drop=True)
        if debug:
            print(f"[DEBUG] {p} samples: {len(series[p])}")
    return series

def to_int_pt(x, y, w, h) -> Optional[Tuple[int,int]]:
    """
    將 (x, y) 轉成pixel座標
    若超出範圍或不是數字則return None
    """
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    xi, yi = int(round(x)), int(round(y))
    if 0 <= xi < w and 0 <= yi < h:
        return (xi, yi)
    return None

def put_marker(frame, pt: Optional[Tuple[int,int]], label: Optional[str], color, radius=6, font_scale=0.6):
    """
    在 frame 上畫一個實心圓點，並可選擇在旁邊加文字標籤。
    - pt: (x,y) 像素座標
    - label: 'A' 或 'B'；若為 None，則不畫文字
    - color: BGR 顏色
    """
    if pt is None: return
    h, w = frame.shape[:2]
    x, y = pt
    if x < 0 or y < 0 or x >= w or y >= h: return
    cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    if label:
        cv2.putText(frame, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

def main():
    """
    主程式：
    - 讀取 set1 與 RallySeg，挑選指定 rally 的資料
    - 逐 frame 讀取影片
    - 對每位選手，只畫「最近一段」箭頭：
        * 第一拍：只畫一個點
        * 第二拍之後：畫 p0 點、p1 點（帶 A/B 標籤）、以及 p0→p1 的箭頭
    - 將結果寫入輸出影片
    """
    ap = argparse.ArgumentParser(description="Overlay latest-segment arrow (prev→current) with both endpoints marked.")
    ap.add_argument("--video", required=True, help="裁切後的單一 rally 影片，或主影片（若用主影片請勿加 --rally_clip）")
    ap.add_argument("--set1", required=True, help="set1.csv")
    ap.add_argument("--rallyseg", required=True, help="RallySeg.csv（需含 Start/End；無 Rally 欄會用行序）")
    ap.add_argument("--rally_id", type=int, required=True, help="要處理的第幾分（1-based）")
    ap.add_argument("--out", required=True, help="輸出影片路徑（mp4/avi 皆可）")
    ap.add_argument("--rally_clip", action="store_true", help="輸入影片已是該 rally 的片段（從 local frame 0 開始）")
    ap.add_argument("--radius", type=int, default=6, help="標記點半徑")
    ap.add_argument("--thickness", type=int, default=2, help="箭頭線條粗細")
    ap.add_argument("--fps", type=float, default=0.0, help="覆寫輸出 FPS（0=沿用輸入）")
    ap.add_argument("--debug", action="store_true", help="顯示除錯訊息")
    args = ap.parse_args()

    # set1.csv
    set1 = read_csv_zh(args.set1)
    need_cols = ["rally","ball_round","player","frame_num","player_location_x","player_location_y","end_frame_num"]
    for c in need_cols:
        if c not in set1.columns:
            raise ValueError(f"set1.csv 缺少欄位：{c}")
    for c in ["rally","ball_round","frame_num","end_frame_num","player_location_x","player_location_y"]:
        set1[c] = pd.to_numeric(set1[c], errors="coerce")

    # RallySeg.csv
    rseg = load_rallyseg(args.rallyseg)
    row = rseg[rseg["Rally"] == int(args.rally_id)]
    if row.empty:
        if 1 <= args.rally_id <= len(rseg):
            row = rseg.iloc[[args.rally_id - 1]]
        else:
            raise ValueError(f"Rally {args.rally_id} 不在 RallySeg 中")
    start_global = int(row["Start"].values[0])
    end_global   = int(row["End"].values[0])
    if end_global <= start_global:
        raise ValueError("RallySeg 的 Start/End 不合理（End <= Start）")

    series = build_series(set1, int(args.rally_id), start_global, debug=args.debug)
    A = series["A"]; B = series["B"]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = in_fps if args.fps <= 0 else float(args.fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not args.rally_clip:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_global)
        local_len = end_global - start_global + 1
    else:
        local_len = min(end_global - start_global + 1, total)

    def prep(samples: pd.DataFrame):
        if samples is None or samples.empty:
            return np.array([], dtype=int), np.array([]), np.array([])
        return (samples["local_frame"].to_numpy(dtype=int),
                samples["player_location_x"].to_numpy(dtype=float),
                samples["player_location_y"].to_numpy(dtype=float))

    A_frames, A_xs, A_ys = prep(A)
    B_frames, B_xs, B_ys = prep(B)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"無法開啟輸出檔：{args.out}")

    colorA = (0, 0, 0)  # black
    colorB = (255, 0, 0)  # red

    if args.debug:
        print(f"[DEBUG] video frames={total}, local_len={local_len}, fps_in={in_fps}, fps_out={fps}, size=({w}x{h})")
        print(f"[DEBUG] A shots={len(A_frames)}; B shots={len(B_frames)}")

    for i in range(local_len):
        ret, frame = cap.read()
        if not ret:
            if args.debug: print(f"[WARN] 提前讀完影片 at local frame {i}")
            break

        # A：只畫最近一段
        if len(A_frames) > 0:
            jA = np.searchsorted(A_frames, i, side="right") - 1
            if jA == 0:
                pA = to_int_pt(A_xs[0], A_ys[0], w, h)
                put_marker(frame, pA, "A", colorA, radius=args.radius)
            elif jA >= 1:
                p0 = to_int_pt(A_xs[jA-1], A_ys[jA-1], w, h)
                p1 = to_int_pt(A_xs[jA],   A_ys[jA],   w, h)
                if p0 and p1:
                    # 先在尾端點 p0 畫點（不標籤），再畫箭頭 p0->p1，最後在 p1 畫點+標籤
                    put_marker(frame, p0, None, colorA, radius=args.radius)
                    cv2.arrowedLine(frame, p0, p1, colorA, args.thickness, line_type=cv2.LINE_AA, tipLength=0.15)
                    put_marker(frame, p1, "A", colorA, radius=args.radius)

        # B：只畫最近一段
        if len(B_frames) > 0:
            jB = np.searchsorted(B_frames, i, side="right") - 1
            if jB == 0:
                pB = to_int_pt(B_xs[0], B_ys[0], w, h)
                put_marker(frame, pB, "B", colorB, radius=args.radius)
            elif jB >= 1:
                p0 = to_int_pt(B_xs[jB-1], B_ys[jB-1], w, h)
                p1 = to_int_pt(B_xs[jB],   B_ys[jB],   w, h)
                if p0 and p1:
                    put_marker(frame, p0, None, colorB, radius=args.radius)  # 畫尾端點
                    cv2.arrowedLine(frame, p0, p1, colorB, args.thickness, line_type=cv2.LINE_AA, tipLength=0.15)
                    put_marker(frame, p1, "B", colorB, radius=args.radius)

        cv2.putText(frame, f"Rally {args.rally_id} | Frame {i}/{local_len-1}",
                    (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235,235,235), 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    main()
