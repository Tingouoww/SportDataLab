# -*- coding: utf-8 -*-
"""
star_indicator.py
右下角：A 選手五角星指示器（逐 frame 亮起）
右上角：B 選手五角星指示器（逐 frame 亮起）
- 主分類依「正/反手 × 高/低於網」決定 (LH/RH/LD/RD)
- 若 aroundhead==1，除主分類外，"上尖(TOP)" 也一併亮起（可同時兩區）
- 指示器大小：--box-ratio（右上角再自動 *0.85 縮一點）
- 標籤使用 PIL 顯示中文，並自動避免出界
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from PIL import Image, ImageDraw, ImageFont

# -------- CSV 讀取（自動嘗試常見編碼） --------
def safe_read_csv(path: str) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8-sig", "cp950", "big5", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"無法讀取 CSV: {path}，請確認編碼與格式。最後錯誤：{last_err}")

# -------- 文字（支援中文）且自動避免出界 --------
def draw_text_pil_inbounds(
    frame_bgr: np.ndarray,
    text: str,
    org_xy: Tuple[int, int],
    font_path: Optional[str],
    font_size: int,
    fill=(0, 0, 0),
    bg=(255, 255, 255),
    padding=(4, 2),
) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    if not font_path:
        x, y = org_xy
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        cv2.putText(frame_bgr, text, (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        return frame_bgr

    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        x, y = org_xy
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        cv2.putText(frame_bgr, text, (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        return frame_bgr

    bbox = draw.textbbox(org_xy, text, font=font)
    x0, y0, x1, y1 = bbox
    pw, ph = padding

    dx = 0; dy = 0
    if x0 - pw < 0: dx = -(x0 - pw)
    if y0 - ph < 0: dy = -(y0 - ph)
    if x1 + pw > W: dx = min(dx, W - (x1 + pw))
    if y1 + ph > H: dy = min(dy, H - (y1 + ph))

    x0 += dx; y0 += dy; x1 += dx; y1 += dy
    overlay = Image.new('RGBA', (img.width, img.height), (0,0,0,0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([x0 - pw, y0 - ph, x1 + pw, y1 + ph], fill=(*bg, 200))
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)
    draw.text((org_xy[0] + dx, org_xy[1] + dy), text, font=font, fill=fill)
    frame_bgr[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame_bgr

# -------- 五角星元件（可放四角） --------
class StarIndicator:
    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        box_ratio: float = 0.20,
        margin_ratio: float = 0.03,
        label_font: Optional[str] = None,
        label_font_size: int = 18,
        anchor: str = "br",         # "br"右下, "tr"右上, "bl"左下, "tl"左上
        title: Optional[str] = None # 框內上緣標題（如 "A"/"B"）
    ):
        self.fw, self.fh = frame_w, frame_h
        self.label_font = label_font
        self.label_font_size = label_font_size
        self.anchor = anchor
        self.title = title

        box_w = int(frame_w * box_ratio)
        box_h = box_w
        margin = int(frame_w * margin_ratio)

        if anchor == "br":
            self.x0 = frame_w - margin - box_w; self.y0 = frame_h - margin - box_h
        elif anchor == "tr":
            self.x0 = frame_w - margin - box_w; self.y0 = margin
        elif anchor == "bl":
            self.x0 = margin; self.y0 = frame_h - margin - box_h
        else:  # "tl"
            self.x0 = margin; self.y0 = margin

        self.x1 = self.x0 + box_w; self.y1 = self.y0 + box_h

        self.center = np.array([int((self.x0 + self.x1)/2), int((self.y0 + self.y1)/2)])
        R = int(box_w * 0.46); r = int(R * 0.38)

        self.outer = []
        for k in range(5):
            theta = np.deg2rad(-90 + k * 72)
            self.outer.append([int(self.center[0] + R*np.cos(theta)),
                               int(self.center[1] + R*np.sin(theta))])
        self.outer = np.array(self.outer, dtype=np.int32)

        self.inner = []
        for k in range(5):
            theta = np.deg2rad(-90 + 36 + k * 72)
            self.inner.append([int(self.center[0] + r*np.cos(theta)),
                               int(self.center[1] + r*np.sin(theta))])
        self.inner = np.array(self.inner, dtype=np.int32)

        self.regions = []
        for i in range(5):
            p_outer = self.outer[i]
            p_in_l = self.inner[(i - 1) % 5]
            p_in_r = self.inner[(i + 1) % 5]
            self.regions.append(np.array([p_outer, p_in_l, self.center, p_in_r], dtype=np.int32))

        self.bg_color = (255, 255, 255)
        self.line_color = (40, 40, 40)
        self.active_color = (0, 220, 0)

        outs = self.outer
        idx_up = int(np.argmin(outs[:, 1]))
        remain = [i for i in range(5) if i != idx_up]
        remain_sorted = sorted(remain, key=lambda i: (outs[i, 0], outs[i, 1]))
        left_two  = sorted(remain_sorted[:2], key=lambda i: outs[i, 1])
        right_two = sorted(remain_sorted[2:],  key=lambda i: outs[i, 1])

        self.idx_up = idx_up
        self.idx_left = left_two[0]
        self.idx_leftdown = left_two[1]
        self.idx_right = right_two[0]
        self.idx_rightdown = right_two[1]

        self.region_map = {"LH": self.idx_left, "RH": self.idx_right,
                           "RD": self.idx_rightdown, "LD": self.idx_leftdown, "TOP": self.idx_up}

        self.labels_zh = {"LH":"反手高於網","RH":"正手高於網","RD":"正手低於網","LD":"反手低於網","TOP":"繞頭"}
        self.labels_en = {"LH":"Backhand High","RH":"Forehand High","RD":"Forehand Low","LD":"Backhand Low","TOP":"Around-the-Head"}

        self.tip_pos = {}
        offset = int(box_w * 0.06)
        for key, idx in [("LH", self.idx_left), ("RH", self.idx_right),
                         ("RD", self.idx_rightdown), ("LD", self.idx_leftdown),
                         ("TOP", self.idx_up)]:
            ox, oy = self.outer[idx]
            if key == "TOP":
                pos = (ox - offset, oy - int(offset * 1.2))
            elif key == "LH":
                pos = (ox - int(offset * 2.0), oy - int(offset * 0.4))
            elif key == "RH":
                pos = (ox + int(offset * 0.6), oy - int(offset * 0.4))
            elif key == "RD":
                pos = (ox + int(offset * 0.6), oy + int(offset * 0.2))
            else:  # LD
                pos = (ox - int(offset * 2.0), oy + int(offset * 0.2))
            self.tip_pos[key] = pos

    def draw(self, frame: np.ndarray, active_region_indices: Optional[List[int]]) -> np.ndarray:
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.x0, self.y0), (self.x1, self.y1), self.bg_color, thickness=-1)
        cv2.polylines(overlay, [self.outer], True, self.line_color, 2)
        for a, b in [(0,2),(2,4),(4,1),(1,3),(3,0)]:
            cv2.line(overlay, tuple(self.outer[a]), tuple(self.outer[b]), self.line_color, 2)

        if active_region_indices:
            for idx in active_region_indices:
                if 0 <= idx < len(self.regions):
                    cv2.fillPoly(overlay, [self.regions[idx]], self.active_color)

        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if self.title:
            draw_text_pil_inbounds(frame, self.title, (self.x0 + 8, self.y0 + 6),
                                   self.label_font, self.label_font_size, fill=(0,0,0), bg=(255,255,255))
        for key, pos in self.tip_pos.items():
            text = self.labels_zh.get(key) if self.label_font else self.labels_en.get(key)
            draw_text_pil_inbounds(frame, text, pos, self.label_font, self.label_font_size,
                                   fill=(0,0,0), bg=(255,255,255))
        return frame

# -------- 分類：主象限 + (可附加)TOP --------
def classify_main_and_top(row: pd.Series) -> List[str]:
    def to_int_safe(v, default=0):
        try: return int(float(v))
        except Exception: return default
    around = to_int_safe(row.get("aroundhead", 0))
    back   = to_int_safe(row.get("backhand", 0))
    hh     = to_int_safe(row.get("hit_height", 0))

    keys = []
    if back == 1 and hh == 1:   keys.append("LH")
    elif back == 1 and hh == 2: keys.append("LD")
    elif back == 0 and hh == 1: keys.append("RH")
    elif back == 0 and hh == 2: keys.append("RD")
    if around == 1: keys.append("TOP")
    return keys

# -------- RallySeg: 取 Start/End --------
def get_start_end(rallyseg_df: pd.DataFrame, rally_id: int) -> (int, int):
    df = rallyseg_df
    if "rally" in df.columns:
        seg = df[df["rally"].astype(str) == str(rally_id)]
        if seg.empty: raise ValueError(f"RallySeg 中找不到 rally == {rally_id}")
        row = seg.iloc[0]
    else:
        idx = rally_id - 1
        if not (0 <= idx < len(df)): raise ValueError(f"RallySeg 無第 {rally_id} 列（共有 {len(df)} 列）")
        row = df.iloc[idx]
    if "Start" not in row or "End" not in row:
        raise ValueError("RallySeg 應包含 'Start' 與 'End' 欄位")
    return int(row["Start"]), int(row["End"])

# -------- frame → 區域鍵清單 --------
def build_frame_region_map(
    set1_df: pd.DataFrame,
    rallyseg_df: pd.DataFrame,
    rally_id: int,
    focus_player: str,
    filter_server: bool = True
) -> Dict[int, List[str]]:
    start, _ = get_start_end(rallyseg_df, rally_id)
    s = set1_df.copy()
    if "rally" in s.columns:
        s = s[s["rally"].astype(str) == str(rally_id)]
    if focus_player and "player" in s.columns:
        s = s[s["player"].astype(str).str.upper() == focus_player.upper()]
    if filter_server and "server" in s.columns:
        s = s[s["server"].apply(lambda v: str(v).strip() in {"1","2","1.0","2.0"})]

    mapping: Dict[int, List[str]] = {}
    for _, row in s.iterrows():
        cats = classify_main_and_top(row)
        if not cats: continue
        try:
            f0 = int(row["frame_num"]) - start
        except Exception:
            continue
        f1 = row.get("end_frame_num", row.get("frame_num", None))
        if pd.isna(f1): f1 = row.get("frame_num", f0)
        try:
            f1 = int(f1) - start
        except Exception:
            f1 = f0
        if f1 < f0: f0, f1 = f1, f0
        for f in range(max(0, f0), max(0, f1) + 1):
            mapping[f] = cats
    return mapping

# ---------------- 主程式 ----------------
def main():
    ap = argparse.ArgumentParser(description="右下 A、右上 B 的五角星擊球指示器（逐 frame 亮起）")
    ap.add_argument("--video", required=True, help="輸入影片（主影片或該 rally 的切片影片）")
    ap.add_argument("--set1", required=True, help="set1.csv 路徑")
    ap.add_argument("--rallyseg", required=True, help="RallySeg.csv 路徑")
    ap.add_argument("--rally-id", type=int, required=True, help="要可視化的 rally 編號（1-based）")
    ap.add_argument("--out", default="", help="輸出 mp4 路徑；留空則只顯示不存檔")
    ap.add_argument("--show-debug", action="store_true", help="左下角顯示 f 與 A/B 類別")
    # 尺寸與字型
    ap.add_argument("--box-ratio", type=float, default=0.20, help="指示器大小（相對影格寬度），預設 0.20")
    ap.add_argument("--margin-ratio", type=float, default=0.03, help="距邊界比例，預設 0.03")
    ap.add_argument("--label-font", default=None, help="字型檔（如 C:/Windows/Fonts/msjh.ttc）")
    ap.add_argument("--label-font-size", type=int, default=18, help="標籤字級，預設 18")
    args = ap.parse_args()

    set1_df = safe_read_csv(args.set1)
    rallyseg_df = safe_read_csv(args.rallyseg)

    # 右下：A
    frame_map_A = build_frame_region_map(set1_df, rallyseg_df, args.rally_id, focus_player="A", filter_server=True)
    # 右上：B
    frame_map_B = build_frame_region_map(set1_df, rallyseg_df, args.rally_id, focus_player="B", filter_server=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{args.video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    # 右下 A 面板
    indicator_A = StarIndicator(
        w, h,
        box_ratio=args.box_ratio,
        margin_ratio=args.margin_ratio,
        label_font=args.label_font,
        label_font_size=args.label_font_size,
        anchor="br",
        title="A"
    )
    # 右上 B 面板（自動再縮 0.85）
    indicator_B = StarIndicator(
        w, h,
        box_ratio=max(0.12, args.box_ratio * 0.85),
        margin_ratio=args.margin_ratio,
        label_font=args.label_font,
        label_font_size=max(12, int(args.label_font_size * 0.9)),
        anchor="tr",
        title="B"
    )

    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        # A（右下）
        cats_A = frame_map_A.get(fidx, [])
        act_A: List[int] = [indicator_A.region_map[c] for c in cats_A if c in indicator_A.region_map]
        frame = indicator_A.draw(frame, act_A)

        # B（右上）
        cats_B = frame_map_B.get(fidx, [])
        act_B: List[int] = [indicator_B.region_map[c] for c in cats_B if c in indicator_B.region_map]
        frame = indicator_B.draw(frame, act_B)

        if args.show_debug:
            dbg = f"f={fidx}  A={','.join(cats_A) if cats_A else '-'}  B={','.join(cats_B) if cats_B else '-'}"
            cv2.putText(frame, dbg, (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)
        else:
            cv2.imshow("star indicator", frame)
            if cv2.waitKey(int(1000.0 / max(1, fps))) == 27:  # ESC
                break

        fidx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("Done. 輸出：", args.out if args.out else "(未輸出檔案)")

if __name__ == "__main__":
    main()
