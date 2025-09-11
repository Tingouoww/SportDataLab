# -*- coding: utf-8 -*-
"""
star_indicator.py
右下角五角星擊球指示器（逐 frame 亮起）
- 主分類依「正/反手 × 高/低於網」決定 (LH/RH/LD/RD)
- 若 aroundhead==1，除了主分類外，"上尖(TOP)" 也一併亮起（同時兩區）
- 五角星尺寸可用 --box-ratio 控制（預設 0.20）
- 標籤使用 PIL 顯示中文，並自動避免出界（不會被切掉）

用法範例：
python star_indicator.py --video 1_00_01.mp4 --set1 set1.csv --rallyseg RallySeg.csv --rally-id 1 \
  --out out_with_star.mp4 --label-font "C:/Windows/Fonts/msjh.ttc" --show-debug --box-ratio 0.2
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple

from PIL import Image, ImageDraw, ImageFont

# ----------------------------
# 工具：安全讀 CSV（自動嘗試常見編碼）
# ----------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8-sig", "cp950", "big5", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"無法讀取 CSV: {path}，請確認編碼與格式。最後錯誤：{last_err}")

# ----------------------------
# 文字：PIL 畫中文 + 自動避免出界
# ----------------------------
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
    """
    在 OpenCV 影像上畫文字（支援中文），並確保整段文字不會超出影格。
    org_xy: 文字左上角預期位置
    """
    H, W = frame_bgr.shape[:2]
    if not font_path:
        # 後備（英文安全）
        (x, y) = org_xy
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        cv2.putText(frame_bgr, text, (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        return frame_bgr

    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        # 字型載入失敗就退回 cv2
        (x, y) = org_xy
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        cv2.putText(frame_bgr, text, (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        return frame_bgr

    # 計文字框
    bbox = draw.textbbox(org_xy, text, font=font)
    x0, y0, x1, y1 = bbox
    pw, ph = padding
    # 若超出影格，往內推
    dx = 0
    dy = 0
    if x0 - pw < 0:
        dx = -(x0 - pw)
    if y0 - ph < 0:
        dy = -(y0 - ph)
    if x1 + pw > W:
        dx = min(dx, W - (x1 + pw))  # 負值
    if y1 + ph > H:
        dy = min(dy, H - (y1 + ph))  # 負值

    x0 += dx; y0 += dy; x1 += dx; y1 += dy
    # 畫半透明白底
    overlay = Image.new('RGBA', (img.width, img.height), (0,0,0,0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([x0 - pw, y0 - ph, x1 + pw, y1 + ph], fill=(*bg, 200))
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)

    # 畫字
    draw.text((org_xy[0] + dx, org_xy[1] + dy), text, font=font, fill=fill)
    frame_bgr[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame_bgr

# ----------------------------
# 五角星指示器
# ----------------------------
class StarIndicator:
    def __init__(self, frame_w: int, frame_h: int, box_ratio: float = 0.20, margin_ratio: float = 0.03,
                 label_font: Optional[str] = None, label_font_size: int = 18):
        """
        box_ratio: 指示器方框寬度 / 影格寬度，預設 0.20 (較小)
        margin_ratio: 與右下角邊界距離 / 影格寬度
        """
        self.fw, self.fh = frame_w, frame_h
        self.label_font = label_font
        self.label_font_size = label_font_size

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

        # 標籤（中文/英文）
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

        # 每個尖角的預設文字位置（會再做 in-bounds 調整）
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
        # 背景方塊
        cv2.rectangle(overlay, (self.x0, self.y0), (self.x1, self.y1), self.bg_color, thickness=-1)
        # 外圈 & 星形
        cv2.polylines(overlay, [self.outer], True, self.line_color, 2)
        for a, b in [(0, 2), (2, 4), (4, 1), (1, 3), (3, 0)]:
            cv2.line(overlay, tuple(self.outer[a]), tuple(self.outer[b]), self.line_color, 2)

        # 高亮（支援多區同時亮）
        if active_region_indices:
            for idx in active_region_indices:
                if idx is None or idx < 0 or idx >= len(self.regions):
                    continue
                cv2.fillPoly(overlay, [self.regions[idx]], self.active_color)

        # 疊加
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # 畫標籤（自動避免出界）
        for key, pos in self.tip_pos.items():
            text = self.labels_zh.get(key) if self.label_font else self.labels_en.get(key)
            draw_text_pil_inbounds(
                frame, text, pos, self.label_font, self.label_font_size,
                fill=(0,0,0), bg=(255,255,255)
            )
        return frame

# ----------------------------
# 分類規則
# ----------------------------
def classify_main_and_top(row: pd.Series) -> List[str]:
    """
    回傳一個或兩個區鍵：
      - 主分類（LH/LD/RH/RD）必回
      - 若 aroundhead==1，再加上 "TOP"
    """
    def to_int_safe(v, default=0):
        try:
            return int(float(v))
        except Exception:
            return default

    around = to_int_safe(row.get("aroundhead", 0))
    back = to_int_safe(row.get("backhand", 0))
    hh = to_int_safe(row.get("hit_height", 0))

    keys = []
    # 主分類
    if back == 1 and hh == 1:
        keys.append("LH")
    elif back == 1 and hh == 2:
        keys.append("LD")
    elif back == 0 and hh == 1:
        keys.append("RH")
    elif back == 0 and hh == 2:
        keys.append("RD")
    # 繞頭加註
    if around == 1:
        keys.append("TOP")
    return keys

# ----------------------------
# RallySeg: Start/End
# ----------------------------
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

# ----------------------------
# frame → 區域 對照
# ----------------------------
def build_frame_region_map(
    set1_df: pd.DataFrame,
    rallyseg_df: pd.DataFrame,
    rally_id: int,
    focus_player: str = "A",
    filter_server: bool = True
) -> Dict[int, List[str]]:
    """
    回傳 dict: 相對於該 rally 影片的 frame_idx -> [區鍵...]
      可能是一個（四象限其一），也可能兩個（四象限其一 + TOP）
    """
    start, _ = get_start_end(rallyseg_df, rally_id)
    s = set1_df.copy()
    if "rally" in s.columns:
        s = s[s["rally"].astype(str) == str(rally_id)]
    if focus_player and focus_player.upper() != "ALL" and "player" in s.columns:
        s = s[s["player"].astype(str).str.upper() == focus_player.upper()]
    if filter_server and "server" in s.columns:
        s = s[s["server"].apply(lambda v: str(v).strip() in {"1", "2", "1.0", "2.0"})]

    mapping: Dict[int, List[str]] = {}
    for _, row in s.iterrows():
        cats = classify_main_and_top(row)
        if not cats:
            continue
        # 區間（主影片座標 -> rally 相對）
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
            mapping[f] = cats  # 後寫覆蓋先寫
    return mapping

# ----------------------------
# 主程式
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="右下角五角星擊球指示器（四象限 + 繞頭可同時亮）")
    ap.add_argument("--video", required=True, help="輸入影片（主影片或該 rally 的切片影片）")
    ap.add_argument("--set1", required=True, help="set1.csv 路徑")
    ap.add_argument("--rallyseg", required=True, help="RallySeg.csv 路徑")
    ap.add_argument("--rally-id", type=int, required=True, help="要可視化的 rally 編號（1-based）")
    ap.add_argument("--out", default="", help="輸出 mp4 路徑；留空則只顯示不存檔")
    ap.add_argument("--focus-player", default="A", help="A/B/ALL，預設 A")
    ap.add_argument("--no-filter-server", action="store_true", help="不要限制 server in {1,2}")
    ap.add_argument("--show-debug", action="store_true", help="左下角顯示 f 與 cat 以便除錯")
    # 尺寸控制
    ap.add_argument("--box-ratio", type=float, default=0.20, help="指示器大小（相對影格寬度），預設 0.20")
    ap.add_argument("--margin-ratio", type=float, default=0.03, help="距離右下角邊界比例，預設 0.03")
    # 字型
    ap.add_argument("--label-font", default=None, help="字型檔路徑（例如 C:/Windows/Fonts/msjh.ttc）")
    ap.add_argument("--label-font-size", type=int, default=18, help="標籤字級，預設 18")
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

    indicator = StarIndicator(
        w, h,
        box_ratio=args.box_ratio,
        margin_ratio=args.margin_ratio,
        label_font=args.label_font,
        label_font_size=args.label_font_size
    )

    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cats = frame_map.get(fidx, [])
        active_indices: List[int] = []
        for c in cats:
            idx = indicator.region_map.get(c)
            if idx is not None:
                active_indices.append(idx)

        frame = indicator.draw(frame, active_indices)

        if args.show_debug:
            text = f"f={fidx}  cats={','.join(cats) if cats else '-'}"
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
