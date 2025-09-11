# -*- coding: utf-8 -*-
"""
star_indicator.py
右下角五角星擊球指示器（逐 frame 亮起）+ 五角星角落中文/英文標註。

用法範例：
python star_indicator.py --video 1_00_01.mp4 --set1 set1.csv --rallyseg RallySeg.csv --rally-id 1 --out out_with_star.mp4 --label-font "C:/Windows/Fonts/msjh.ttc" --show-debug
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Optional

# --- PIL 用於顯示中文 ---
from PIL import Image, ImageDraw, ImageFont

def safe_read_csv(path: str) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8-sig", "cp950", "big5", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"無法讀取 CSV: {path}，請確認編碼與格式。最後錯誤：{last_err}")

# ------- 文字繪製（可顯示中文） -------
def draw_text_pil(
    frame_bgr: np.ndarray,
    text: str,
    org_xy: tuple,
    font_path: Optional[str],
    font_size: int = 18,
    fill=(0, 0, 0),
    bg=None
) -> np.ndarray:
    """
    在 OpenCV 影像上畫文字（支援中文）。frame_bgr 會被就地覆蓋。
    org_xy: 左上角座標 (x, y)
    font_path: 字型路徑；若 None 則改用 cv2 的 putText（僅英文安全）
    bg: 背景色 (r,g,b)，若給會先畫一個文字背景框
    """
    if font_path:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            # 字型載入失敗就退回 cv2
            cv2.putText(frame_bgr, text, org_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            return frame_bgr

        # 背景框
        if bg is not None:
            bbox = draw.textbbox(org_xy, text, font=font)
            x0, y0, x1, y1 = bbox
            # 畫半透明背景
            overlay = Image.new('RGBA', (img.width, img.height), (0,0,0,0))
            odraw = ImageDraw.Draw(overlay)
            odraw.rectangle([x0-4, y0-2, x1+4, y1+2], fill=(*bg, 200))
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(img)

        draw.text(org_xy, text, font=font, fill=fill)
        frame_bgr[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        # 後備：cv2（僅英文安全）
        cv2.putText(frame_bgr, text, org_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    return frame_bgr

# ------- 五角星指示器 -------
class StarIndicator:
    def __init__(self, frame_w: int, frame_h: int, box_ratio: float = 0.26, margin_ratio: float = 0.03,
                 label_font: Optional[str] = None):
        self.fw, self.fh = frame_w, frame_h
        self.label_font = label_font

        box_w = int(frame_w * box_ratio)
        box_h = box_w
        margin = int(frame_w * margin_ratio)

        self.x1 = frame_w - margin
        self.y1 = frame_h - margin
        self.x0 = self.x1 - box_w
        self.y0 = self.y1 - box_h

        self.center = np.array([int((self.x0 + self.x1) / 2), int((self.y0 + self.y1) / 2)])
        R = int(box_w * 0.46)
        r = int(R * 0.38)

        # outer[0] 在上方
        self.outer = []
        for k in range(5):
            theta = np.deg2rad(-90 + k * 72)
            self.outer.append([
                int(self.center[0] + R * np.cos(theta)),
                int(self.center[1] + R * np.sin(theta)),
            ])
        self.outer = np.array(self.outer, dtype=np.int32)

        self.inner = []
        for k in range(5):
            theta = np.deg2rad(-90 + 36 + k * 72)
            self.inner.append([
                int(self.center[0] + r * np.cos(theta)),
                int(self.center[1] + r * np.sin(theta)),
            ])
        self.inner = np.array(self.inner, dtype=np.int32)

        # 區域多邊形
        self.regions = []
        for i in range(5):
            p_outer = self.outer[i]
            p_in_l = self.inner[(i - 1) % 5]
            p_in_r = self.inner[(i + 1) % 5]
            poly = np.array([p_outer, p_in_l, self.center, p_in_r], dtype=np.int32)
            self.regions.append(poly)

        # 顏色
        self.bg_color = (255, 255, 255)
        self.line_color = (40, 40, 40)
        self.active_color = (0, 220, 0)

        # 幾何 → 語義索引
        outs = self.outer
        idx_up = int(np.argmin(outs[:, 1]))
        remain = [i for i in range(5) if i != idx_up]
        remain_sorted = sorted(remain, key=lambda i: (outs[i, 0], outs[i, 1]))
        left_two = sorted(remain_sorted[:2], key=lambda i: outs[i, 1])   # y小=較上
        right_two = sorted(remain_sorted[2:], key=lambda i: outs[i, 1])

        self.idx_up = idx_up
        self.idx_left = left_two[0]
        self.idx_leftdown = left_two[1]
        self.idx_right = right_two[0]
        self.idx_rightdown = right_two[1]

        self.region_map = {
            "LH": self.idx_left,       # 反手高於網
            "RH": self.idx_right,      # 正手高於網
            "RD": self.idx_rightdown,  # 正手低於網
            "LD": self.idx_leftdown,   # 反手低於網
            "TOP": self.idx_up         # 繞頭
        }

        # 文字標籤（中文 + 英文後備）
        self.labels_zh = {
            "LH": "反手高於網",
            "RH": "正手高於網",
            "RD": "正手低於網",
            "LD": "反手低於網",
            "TOP": "繞頭"
        }
        self.labels_en = {
            "LH": "Backhand High",
            "RH": "Forehand High",
            "RD": "Forehand Low",
            "LD": "Backhand Low",
            "TOP": "Around-the-Head"
        }

        # 各尖角文字位置（在外尖附近，略偏外以免與圖重疊）
        self.tip_pos = {}
        offset = int(box_w * 0.06)
        for key, idx in [("LH", self.idx_left), ("RH", self.idx_right),
                         ("RD", self.idx_rightdown), ("LD", self.idx_leftdown),
                         ("TOP", self.idx_up)]:
            ox, oy = self.outer[idx]
            # 依相對位置微調
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

    def draw(self, frame: np.ndarray, active_region_idx: Optional[int]) -> np.ndarray:
        overlay = frame.copy()
        # 背景方塊
        cv2.rectangle(overlay, (self.x0, self.y0), (self.x1, self.y1), self.bg_color, thickness=-1)
        # 外圈 & 星形
        cv2.polylines(overlay, [self.outer], True, self.line_color, 2)
        for a, b in [(0, 2), (2, 4), (4, 1), (1, 3), (3, 0)]:
            cv2.line(overlay, tuple(self.outer[a]), tuple(self.outer[b]), self.line_color, 2)
        # 高亮
        if active_region_idx is not None:
            cv2.fillPoly(overlay, [self.regions[active_region_idx]], self.active_color)
        # 疊加
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # 畫標籤
        for key, pos in self.tip_pos.items():
            text = self.labels_zh.get(key) if self.label_font else self.labels_en.get(key)
            # 半透明白底，黑字
            draw_text_pil(frame, text, pos, self.label_font, font_size=18, fill=(0,0,0), bg=(255,255,255))
        return frame

# ------- 分類規則 -------
def classify_region(row: pd.Series) -> Optional[str]:
    def to_int_safe(v, default=0):
        try:
            return int(v)
        except Exception:
            return default
    around = to_int_safe(row.get("aroundhead", 0))
    back = to_int_safe(row.get("backhand", 0))
    hh = to_int_safe(row.get("hit_height", 0))
    if around == 1:
        return "TOP"
    if back == 1 and hh == 1:
        return "LH"
    if back == 1 and hh == 2:
        return "LD"
    if back == 0 and hh == 1:
        return "RH"
    if back == 0 and hh == 2:
        return "RD"
    return None

# ------- RallySeg 讀取 Start/End -------
def get_start_end(rallyseg_df: pd.DataFrame, rally_id: int) -> (int, int):
    df = rallyseg_df
    if "rally" in df.columns:
        seg = df[df["rally"].astype(str) == str(rally_id)]
        if seg.empty:
            raise ValueError(f"RallySeg 中找不到 rally == {rally_id}")
        row = seg.iloc[0]
    else:
        idx = rally_id - 1
        if idx < 0 or idx >= len(df):
            raise ValueError(f"RallySeg 無第 {rally_id} 列（共有 {len(df)} 列）")
        row = df.iloc[idx]
    if "Start" not in row or "End" not in row:
        raise ValueError("RallySeg 應包含 'Start' 與 'End' 欄位")
    return int(row["Start"]), int(row["End"])

# ------- frame → 區域 對照 -------
def build_frame_region_map(
    set1_df: pd.DataFrame,
    rallyseg_df: pd.DataFrame,
    rally_id: int,
    focus_player: str = "A",
    filter_server: bool = True
) -> Dict[int, str]:
    start, _ = get_start_end(rallyseg_df, rally_id)
    s = set1_df.copy()
    if "rally" in s.columns:
        s = s[s["rally"].astype(str) == str(rally_id)]
    if focus_player and focus_player.upper() != "ALL" and "player" in s.columns:
        s = s[s["player"].astype(str).str.upper() == focus_player.upper()]
    if filter_server and "server" in s.columns:
        s = s[s["server"].apply(lambda v: str(v).strip() in {"1", "2", "1.0", "2.0"})]

    mapping: Dict[int, str] = {}
    for _, row in s.iterrows():
        cat = classify_region(row)
        if cat is None:
            continue
        try:
            f0 = int(row["frame_num"]) - start
        except Exception:
            continue
        f1 = row.get("end_frame_num", row.get("frame_num", None))
        if pd.isna(f1):
            f1 = row.get("frame_num", f0)
        try:
            f1 = int(f1) - start
        except Exception:
            f1 = f0
        if f1 < f0:
            f0, f1 = f1, f0
        for f in range(max(0, f0), max(0, f1) + 1):
            mapping[f] = cat
    return mapping

# ------- 主程式 -------
def main():
    ap = argparse.ArgumentParser(description="右下角五角星擊球指示器（逐 frame 亮起 + 角落標註）")
    ap.add_argument("--video", required=True, help="輸入影片（主影片或該 rally 的切片影片）")
    ap.add_argument("--set1", required=True, help="set1.csv 路徑")
    ap.add_argument("--rallyseg", required=True, help="RallySeg.csv 路徑")
    ap.add_argument("--rally-id", type=int, required=True, help="要可視化的 rally 編號（1-based）")
    ap.add_argument("--out", default="", help="輸出 mp4 路徑；留空則只顯示不存檔")
    ap.add_argument("--focus-player", default="A", help="A/B/ALL，預設 A（只亮球員A）")
    ap.add_argument("--no-filter-server", action="store_true", help="不要限制 server in {1,2}")
    ap.add_argument("--show-debug", action="store_true", help="左下角顯示 f 與 cat 以便除錯")
    ap.add_argument("--label-font", default=None, help="字型檔路徑（例如 C:/Windows/Fonts/msjh.ttc；若省略改用英文）")
    args = ap.parse_args()

    set1_df = safe_read_csv(args.set1)
    rallyseg_df = safe_read_csv(args.rallyseg)

    frame_map = build_frame_region_map(
        set1_df,
        rallyseg_df,
        args.rally_id,
        focus_player=args.focus_player,
        filter_server=(not args.no_filter_server),
    )

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

    indicator = StarIndicator(w, h, box_ratio=0.26, margin_ratio=0.03, label_font=args.label_font)

    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cat = frame_map.get(fidx, None)
        active_idx = indicator.region_map.get(cat) if cat is not None else None

        frame = indicator.draw(frame, active_idx)

        if args.show_debug:
            text = f"f={fidx}  cat={cat if cat else '-'}"
            cv2.putText(frame, text, (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)
        else:
            cv2.imshow("star indicator", frame)
            key = cv2.waitKey(int(1000.0 / max(1, fps)))
            if key == 27:  # ESC
                break

        fidx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("Done. 輸出：", args.out if args.out else "(未輸出檔案)")

if __name__ == "__main__":
    main()
