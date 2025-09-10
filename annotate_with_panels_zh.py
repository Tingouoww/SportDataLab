# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ==========================
# 參數（可由 CLI 覆寫）
# ==========================
DEFAULT_FONT = "C:/Windows/Fonts/msjh.ttc"  # 微軟正黑體
PANEL_W, PANEL_H = 240, 150                  # 單個面板大小
MARGIN = 12                                  # 與畫面邊緣距離

# ==========================
# 工具：中文繪字（避免 cv2.putText 中文亂碼）
# ==========================
def draw_chinese_text(img_bgr, text, xy, font_path=DEFAULT_FONT, font_size=18, color=(255,255,255)):
    if not text:
        return img_bgr
    # OpenCV(BGR) -> PIL(RGB)
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    except Exception:
        font = ImageFont.load_default()
    rgb = (color[2], color[1], color[0])  # BGR -> RGB
    draw.text(xy, text, font=font, fill=rgb)
    # PIL(RGB) -> OpenCV(BGR)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

# ==========================
# 右側 3x2 面板（中文）
# 列：前/中/後（上→下）
# 欄：左/右（左→右）
# lit=(row, col) 表示要亮的格；None 表示不亮
# ==========================
def draw_zone_panel_zh(w=PANEL_W, h=PANEL_H, lit=None, *, font_path=DEFAULT_FONT, font_size=18, title="區域指示"):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 100, 50)  # 綠底

    pad = 12
    title_h = font_size + 6
    inner_w = w - pad*2
    inner_h = h - pad*2 - title_h
    x0 = pad
    y0 = pad + title_h

    # 標題在面板內，避免被裁掉
    img = draw_chinese_text(img, title, (pad, pad), font_path, font_size, (255,255,255))

    # 外框
    cv2.rectangle(img, (x0, y0), (x0+inner_w, y0+inner_h), (220, 220, 220), 2)

    # 3x2 切格
    row_h = inner_h // 3
    col_w = inner_w // 2

    # 高亮
    if lit is not None:
        r, c = lit
        rx = x0 + c*col_w
        ry = y0 + r*row_h
        cv2.rectangle(img, (rx, ry), (rx+col_w, ry+row_h), (40, 200, 255), -1)
        cv2.rectangle(img, (rx, ry), (rx+col_w, ry+row_h), (255, 255, 255), 2)

    # 分隔線
    for r in range(1, 3):
        y = y0 + r*row_h
        cv2.line(img, (x0, y), (x0+inner_w, y), (220,220,220), 1)
    x = x0 + col_w
    cv2.line(img, (x, y0), (x, y0+inner_h), (220,220,220), 1)

    # 中文列標（前/中/後）
    row_labels = ["前", "中", "後"]
    for r, lab in enumerate(row_labels):
        yy = y0 + r*row_h + row_h//2 - font_size//2
        img = draw_chinese_text(img, lab, (x0+4, yy), font_path, font_size, (230,230,230))

    # 中文欄標（左/右）
    col_labels = ["左", "右"]
    for c, lab in enumerate(col_labels):
        xx = x0 + c*col_w + col_w//2 - font_size//2
        img = draw_chinese_text(img, lab, (xx, y0 - font_size - 2), font_path, font_size, (230,230,230))

    return img

# ==========================
# 依座標決定要亮哪一格
# 左右：以 x 的中線分
# 前中後：以 y 的兩條門檻分
# ==========================
def build_zone_thresholds(valid_xy: pd.DataFrame):
    """
    從資料自動估計門檻：
    - 左右：用 x 範圍中點
    - 前/中/後：把 y 幾何範圍等分成 3 段
    回傳 dict: {mid_x, front_max_y, middle_max_y, xmin,xmax,ymin,ymax}
    """
    xmin = float(valid_xy["landing_x"].min())
    xmax = float(valid_xy["landing_x"].max())
    ymin = float(valid_xy["landing_y"].min())
    ymax = float(valid_xy["landing_y"].max())
    mid_x = (xmin + xmax) / 2.0
    front_max_y = ymin + (ymax - ymin) / 3.0
    middle_max_y = ymin + 2.0 * (ymax - ymin) / 3.0
    return {
        "mid_x": mid_x,
        "front_max_y": front_max_y,
        "middle_max_y": middle_max_y,
        "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax
    }

def zone_from_coords(landing_x, landing_y, thres) -> tuple | None:
    """
    thres: 由 build_zone_thresholds 回傳的字典
    回傳 (row, col) 或 None
    """
    if pd.isna(landing_x) or pd.isna(landing_y):
        return None
    try:
        x = float(landing_x); y = float(landing_y)
    except Exception:
        return None

    col = 0 if x < thres["mid_x"] else 1
    if y < thres["front_max_y"]:
        row = 0  # 前
    elif y < thres["middle_max_y"]:
        row = 1  # 中
    else:
        row = 2  # 後
    return (row, col)

# ==========================
# Writer fallback
# ==========================
def open_writer(path: Path, fps: float, size: tuple[int,int]):
    # 先試 mp4v，再退 AVI
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if w.isOpened():
        return w, path
    alt = path.with_suffix(".avi")
    w = cv2.VideoWriter(str(alt), cv2.VideoWriter_fourcc(*"XVID"), fps, size)
    if w.isOpened():
        return w, alt
    return None, None

# ==========================
# 主流程
# ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default= "1_00_01.mp4", help="rally 片段影片（local frame 從 0 開始）")
    ap.add_argument("--out", type=Path, required=True, help="輸出影片路徑（.mp4 或 .avi）")
    ap.add_argument("--set1", type=Path, default=Path("set1.csv"))
    ap.add_argument("--rallyseg", type=Path, default=Path("RallySeg.csv"))
    ap.add_argument("--rally", type=int, required=True, help="rally id (1-based)")
    ap.add_argument("--fps-scale", type=float, default=1.0, help="片段FPS / 資料表FPS（例：30/60=0.5）")
    ap.add_argument("--offset", type=int, default=0, help="幀偏移：片段比資料晚開始為正，早開始為負")
    ap.add_argument("--font", type=str, default=DEFAULT_FONT, help="中文字型檔路徑")
    ap.add_argument("--font-size", type=int, default=18)
    args = ap.parse_args()

    # 讀資料
    df = pd.read_csv(args.set1, encoding="big5")
    rally_seg = pd.read_csv(args.rallyseg)

    # 該 rally 的主影片起點
    seg_start = int(rally_seg.iloc[args.rally - 1]["Start"])

    # 單一 rally 的所有球，依球序
    ev = df[df["rally"] == args.rally].copy()
    ev.sort_values("ball_round", inplace=True)

    # 幀對齊（clip 模式：local = round((global - seg_start)*fps_scale - offset)）
    for c in ["frame_num", "end_frame_num", "landing_x", "landing_y", "player_location_y"]:
        if c in ev.columns:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")

    ev["local_start"] = ((ev["frame_num"] - seg_start) * args.fps_scale - args.offset).round().astype("Int64")
    ev["local_end"]   = ((ev["end_frame_num"] - seg_start) * args.fps_scale - args.offset).round().astype("Int64")

    # 門檻（用整份 set1 的落點座標算，確保左右/前中後穩定）
    all_xy = df.dropna(subset=["landing_x", "landing_y"])[["landing_x", "landing_y"]].copy()
    for c in ["landing_x","landing_y"]:
        all_xy[c] = pd.to_numeric(all_xy[c], errors="coerce")
    thres = build_zone_thresholds(all_xy)

    # 推估上/下選手（小 y 在上；若缺值則預設 B 在上、A 在下）
    who_top, who_bottom = "B", "A"
    try:
        posA = df[df["player"]=="A"]["player_location_y"].dropna().astype(float)
        posB = df[df["player"]=="B"]["player_location_y"].dropna().astype(float)
        if len(posA) and len(posB):
            who_top, who_bottom = ("A","B") if posA.mean() < posB.mean() else ("B","A")
    except Exception:
        pass

    # 開影片
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer, real_out = open_writer(args.out, fps, (W, H))
    if writer is None:
        raise RuntimeError("無法建立輸出影片，請改用 .avi 或安裝適當編碼器")

    # 逐幀處理
    rows = ev.to_dict("records")
    i = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 找出目前幀所屬的球
        active = None
        while i < len(rows) and pd.notna(rows[i]["local_end"]) and rows[i]["local_end"] <= frame_idx:
            i += 1
        if i < len(rows):
            r = rows[i]
            ls = r["local_start"]; le = r["local_end"]
            if pd.notna(ls) and pd.notna(le) and ls <= frame_idx < le:
                active = r

        # 依當前球的落點決定亮燈；分配到上/下兩個面板
        p_top_lit = p_bot_lit = None
        if active is not None:
            lit = zone_from_coords(active.get("landing_x"), active.get("landing_y"), thres)
            hitter = str(active.get("player", ""))
            if hitter == who_top:
                p_top_lit = lit
            elif hitter == who_bottom:
                p_bot_lit = lit

        # 畫面右上/右下兩個面板
        panel_top = draw_zone_panel_zh(PANEL_W, PANEL_H, lit=p_top_lit,
                                       font_path=args.font, font_size=args.font_size, title="上方選手")
        panel_bot = draw_zone_panel_zh(PANEL_W, PANEL_H, lit=p_bot_lit,
                                       font_path=args.font, font_size=args.font_size, title="下方選手")

        ph, pw = panel_top.shape[:2]
        x_right = W - pw - MARGIN
        y_top = MARGIN
        y_bottom = H - ph - MARGIN

        overlay = frame.copy()
        overlay[y_top:y_top+ph, x_right:x_right+pw] = panel_top
        overlay[y_bottom:y_bottom+ph, x_right:x_right+pw] = panel_bot
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        # 右下面板上方顯示目前擊球類型（中文）
        label = str(active.get("type","")) if active else ""
        frame = draw_chinese_text(frame, f"擊球：{label}",
                                  (x_right, y_bottom - args.font_size - 6),
                                  font_path=args.font, font_size=args.font_size, color=(255,255,255))

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[DONE] 輸出：{real_out}")

if __name__ == "__main__":
    main()
