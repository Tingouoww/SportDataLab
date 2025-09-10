import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont  # ← 用來畫中文

# ========= 參數 =========
SET1_CSV = "set1.csv"
RALLYSEG_CSV = "RallySeg.csv"
VIDEO = "1_00_01.mp4"         # 已剪好的 rally 片段（local frame 從 0 開始）
OUT = "out_with_panel_zh.mp4"
RALLY_ID = 1
FPS_SCALE = 1.0                # 片段FPS / 資料表FPS；原60→片段30 => 0.5
OFFSET = 0                     # 幀位移；片段比資料晚開始=>正，早開始=>負
PANEL_SIZE = (240, 170)        # 右下角儀表板尺寸 (w,h)
MARGIN = 12                    # 右下角與邊界的間距
FONT_PATH = "C:/Windows/Fonts/msjh.ttc"   # ← 請改成你電腦上的中文字型檔
FONT_SIZE = 18
# ========================

# 區塊編碼：用 (row, col)，row ∈ {0:前, 1:中, 2:後}，col ∈ {0:左, 1:右}
TYPE_TO_ZONE = {
    # 後場類
    "發長球": (2, 0), "挑球": (2, 0), "高遠球": (2, 0), "後場長球": (2, 0),
    # 前場類
    "放小球": (0, 0), "小球": (0, 0), "勾對角": (0, 1), "網前撲": (0, 0),
    # 中場/平抽擋
    "平抽": (1, 0), "推擋": (1, 1),
    # 殺球（可依喜好調整到前/中）
    "殺球": (1, 0),
}

AREA_TO_ZONE = {
    # 如果有穩定的 landing_area 編碼，在這裡對應，例如：
    # "FL": (0,0), "FR": (0,1), "ML": (1,0), "MR": (1,1), "RL": (2,0), "RR": (2,1)
}

# ===== 中文繪字工具 =====
def draw_chinese_text(img_bgr, text, position, font_path=FONT_PATH,
                      font_size=FONT_SIZE, color=(255,255,255)):
    """
    在 OpenCV BGR 影像上用 Pillow 繪製中文。
    position: (x, y)
    color: BGR
    """
    if not text:
        return img_bgr
    # OpenCV(BGR) -> PIL(RGB)
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    except Exception:
        # 找不到字型就用預設（可能缺字）
        font = ImageFont.load_default()
    # Pillow 用 RGB，將 BGR 顏色轉為 RGB
    rgb = (int(color[2]), int(color[1]), int(color[0]))
    draw.text(position, text, font=font, fill=rgb)
    # PIL(RGB) -> OpenCV(BGR)
    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

def draw_zone_panel_zh(w=240, h=150, lit=None,
                       font_path=FONT_PATH, font_size=FONT_SIZE):
    """
    畫 3x2 迷你球場（中文標示）：
    列：前/中/後（自上而下）
    欄：左/右（自左而右）
    lit=(row,col) 代表要高亮的格子；None 表示不亮。
    回傳 BGR 圖片 (h,w,3)。
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 100, 50)  # 綠底

    pad = 8
    inner_w = w - pad*2
    inner_h = h - pad*2
    x0, y0 = pad, pad

    # 外框
    cv2.rectangle(img, (x0, y0), (x0+inner_w, y0+inner_h), (220, 220, 220), 2)

    # 3 rows, 2 cols
    row_h = inner_h // 3
    col_w = inner_w // 2

    # 高亮
    if lit is not None:
        r, c = lit
        rx = x0 + c*col_w
        ry = y0 + r*row_h
        cv2.rectangle(img, (rx, ry), (rx+col_w, ry+row_h), (40, 200, 255), -1)  # 亮區塊
        cv2.rectangle(img, (rx, ry), (rx+col_w, ry+row_h), (255, 255, 255), 2)  # 白邊

    # 分隔線
    for r in range(1, 3):
        y = y0 + r*row_h
        cv2.line(img, (x0, y), (x0+inner_w, y), (220, 220, 220), 1)
    x = x0 + col_w
    cv2.line(img, (x, y0), (x, y0+inner_h), (220, 220, 220), 1)

    # 中文標註
    # 列標：前/中/後（靠左側置中）
    row_labels = ["前", "中", "後"]
    for r, lab in enumerate(row_labels):
        yy = y0 + r*row_h + row_h//2 - font_size//2
        img = draw_chinese_text(img, lab, (x0+4, yy), font_path, font_size, (230,230,230))

    # 欄標：左/右（放在上邊靠兩欄中央）
    col_labels = ["左", "右"]
    for c, lab in enumerate(col_labels):
        xx = x0 + c*col_w + col_w//2 - font_size//2
        img = draw_chinese_text(img, lab, (xx, y0 - font_size - 2 + 14), font_path, font_size, (230,230,230))

    # 標題（可選）：右下角上方加一行「區域指示」
    title = "區域指示"
    img = draw_chinese_text(img, title, (x0, y0 - font_size - 4), font_path, font_size, (255,255,255))

    return img

def zone_from_row(row):
    """
    給一筆 event row，決定亮哪個格子：
    1) 若 landing_area 有對應，優先用它
    2) 否則看 type（中文）
    3) 都沒有就回傳 None（不亮）
    """
    if row is None:
        return None
    la = row.get("landing_area", None)
    if pd.notna(la) and la in AREA_TO_ZONE:
        return AREA_TO_ZONE[la]

    t = str(row.get("type", "")).strip()
    if t in TYPE_TO_ZONE:
        return TYPE_TO_ZONE[t]

    return None

# 讀資料
df = pd.read_csv(SET1_CSV)
rally_seg = pd.read_csv(RALLYSEG_CSV)

# 對應 RALLY 的 Start（假設 RallySeg 為 1-based 順序）
seg_start = int(rally_seg.iloc[RALLY_ID-1]["Start"])

# 單一 rally 的所有球（依球序排序）
ev = df[df["rally"] == RALLY_ID].copy()
ev.sort_values(["ball_round"], inplace=True)

# 加入 local 區間（支援 fps_scale 與 offset）
ev["local_start"] = ((ev["frame_num"] - seg_start) * FPS_SCALE - OFFSET).round().astype(int)
ev["local_end"]   = ((ev["end_frame_num"] - seg_start) * FPS_SCALE - OFFSET).round().astype(int)

# 影片
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"無法開啟影片：{VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 輸出
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
if not writer.isOpened():
    OUT = Path(OUT).with_suffix(".avi").as_posix()
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"XVID"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("無法建立輸出影片，請改用 .avi")

frame_idx = 0
i = 0
rows = ev.to_dict("records")  # list[dict]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 找出目前幀所屬的球
    active = None
    while i < len(rows) and rows[i]["local_end"] <= frame_idx:
        i += 1
    if i < len(rows):
        r = rows[i]
        if r["local_start"] <= frame_idx < r["local_end"]:
            active = r

    # 右下角儀表板（中文）
    lit = zone_from_row(active)
    panel = draw_zone_panel_zh(PANEL_SIZE[0], PANEL_SIZE[1], lit=lit,
                               font_path=FONT_PATH, font_size=FONT_SIZE)

    # 貼到右下角（半透明）
    ph, pw = panel.shape[:2]
    x1 = W - pw - MARGIN
    y1 = H - ph - MARGIN
    overlay = frame.copy()
    overlay[y1:y1+ph, x1:x1+pw] = panel
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    # 顯示目前擊球類型（中文）
    label = str(active.get("type", "")) if active else ""
    frame = draw_chinese_text(frame, f"擊球：{label}", (x1, y1 - FONT_SIZE - 6),
                              font_path=FONT_PATH, font_size=FONT_SIZE, color=(255,255,255))

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
print(f"輸出完成：{OUT}")

