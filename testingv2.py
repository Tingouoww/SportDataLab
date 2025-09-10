import cv2
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path

# ========= 你可以調整的參數 =========
SET1_CSV = "set1.csv"
RALLYSEG_CSV = "RallySeg.csv"
VIDEO = "1_00_01.mp4"       # 已剪好的 rally 片段（local frame 從 0 開始）
OUT = "out_with_heatmap.mp4"
RALLY_ID = 1                        # 要顯示的 rally
FPS_SCALE = 1.0                     # 片段FPS / 資料表FPS；若原始60、片段30 → 0.5
OFFSET = 0                          # 幀位移；片段比資料晚開始→正值，早開始→負值
BINS = 40
HEATMAP_SIZE = (400, 400)           # 每張熱圖畫面大小(px)
FONT_PATH = "C:/Windows/Fonts/msjh.ttc"  # 中文字型（Windows: 微軟正黑體）
TITLE_A = "A 選手落點（累積）"
TITLE_B = "B 選手落點（累積）"
# ===================================

mpl.rcParams['axes.unicode_minus'] = False
_font = FontProperties(fname=FONT_PATH)

from matplotlib.patches import Rectangle

import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

# 可放在檔案開頭一次性設定（避免負號變成方塊）
mpl.rcParams['axes.unicode_minus'] = False

def _get_font(font_path: str | None) -> FontProperties | None:
    if not font_path:
        return None
    try:
        return FontProperties(fname=font_path)
    except Exception:
        return None

def _draw_badminton_court(ax, xmin, xmax, ymin, ymax, line_color="white", line_w=2):
    """
    以比例繪製羽球場主要線條，填滿座標框 [xmin,xmax]×[ymin,ymax]
    比例依 BWF：長 13.40 m、寬 6.10 m；含中線、短發球線、雙打長發球線、單打邊線。
    """
    W = xmax - xmin
    H = ymax - ymin
    X = lambda u: xmin + u * W  # u ∈ [0,1]
    Y = lambda v: ymin + v * H  # v ∈ [0,1]

    # 參考比例
    total_len = 13.4
    total_w   = 6.1
    net_y     = 0.5
    short_srv = 1.98 / total_len      # 短發球線距網比例
    dbl_long  = 0.76 / total_len      # 雙打長發球線距底線比例
    center_x  = 0.5
    singles_half = (5.18 / 6.1) / 2.0 # 單打邊線到中線的半寬比例

    # 外框
    ax.add_patch(Rectangle((X(0), Y(0)), W, H, fill=False, edgecolor=line_color, linewidth=line_w))

    # 中線（寬度中央）
    ax.plot([X(center_x), X(center_x)], [Y(0), Y(1)], color=line_color, linewidth=line_w)

    # 短發球線（距網 1.98m）
    ax.plot([X(0), X(1)], [Y(net_y - short_srv), Y(net_y - short_srv)], color=line_color, linewidth=line_w)
    ax.plot([X(0), X(1)], [Y(net_y + short_srv), Y(net_y + short_srv)], color=line_color, linewidth=line_w)

    # 雙打長發球線（距底線 0.76m）
    ax.plot([X(0), X(1)], [Y(1 - dbl_long), Y(1 - dbl_long)], color=line_color, linewidth=line_w)
    ax.plot([X(0), X(1)], [Y(0 + dbl_long), Y(0 + dbl_long)], color=line_color, linewidth=line_w)

    # 單打邊線（5.18m 寬）
    ax.plot([X(0.5 - singles_half), X(0.5 - singles_half)], [Y(0), Y(1)], color=line_color, linewidth=line_w)
    ax.plot([X(0.5 + singles_half), X(0.5 + singles_half)], [Y(0), Y(1)], color=line_color, linewidth=line_w)

def make_heatmap_img(
    xs, ys,
    xmin, xmax, ymin, ymax,
    *,
    bins: int = 40,
    size: tuple[int, int] = (400, 400),
    title: str = "落點熱力圖",
    xlabel: str = "landing_x",
    ylabel: str = "landing_y",
    flip_y: bool = True,
    court_img_path: str | None = None,   # 羽球場底圖（PNG/JPG），可為 None
    use_vector_court: bool = False,      # 不用圖片、以向量畫線呈現球場
    court_facecolor: str = "#0b5d2a",    # 沒有底圖時的底色
    heat_alpha: float = 0.65,            # 熱力圖透明度（0~1）
    cmap: str | None = None,             # colormap 名稱，None 用預設
    font_path: str | None = None,        # 中文字型路徑（Windows: C:/Windows/Fonts/msjh.ttc）
    show_colorbar: bool = True
) -> np.ndarray:
    """
    產生「羽球場底 + 落點熱力圖」的圖像，回傳 OpenCV BGR 格式 (H,W,3)。

    參數重點：
    - xs, ys：落點座標（累積）
    - flip_y=True：讓圖的下方對應場地的下方
    - court_img_path：用圖片當底圖
    - use_vector_court=True：用程式畫線當底圖（沒有圖片時很實用）
    - bins/size/heat_alpha：控制熱力圖解析度、大小與透明度
    - font_path：指定中文字型，標題/座標軸可顯示中文
    """
    # 準備字型
    fp = _get_font(font_path)

    # Figure 尺寸（px → inch）
    w_px, h_px = size
    dpi = 100
    fig_w, fig_h = w_px / dpi, h_px / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # 先畫「球場底」
    if court_img_path:
        try:
            court = mpimg.imread(court_img_path)
            # 用 extent 對齊到你的座標系；origin="lower" 搭配 flip_y 控制方向
            ax.imshow(court, extent=[xmin, xmax, ymin, ymax], origin="lower")
        except Exception:
            # 讀不到圖片就退而用純色底
            ax.set_facecolor(court_facecolor)
    elif use_vector_court:
        ax.set_facecolor(court_facecolor)
        _draw_badminton_court(ax, xmin, xmax, ymin, ymax, line_color="white", line_w=2)
    else:
        ax.set_facecolor(court_facecolor)

    # 畫熱力圖（有資料才畫）
    if len(xs) > 0 and len(ys) > 0:
        H, xedges, yedges = np.histogram2d(xs, ys, bins=bins,
                                           range=[[xmin, xmax], [ymin, ymax]])
        im = ax.imshow(
            H.T,
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            aspect="auto",
            alpha=heat_alpha,
            cmap=cmap
        )
        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax, shrink=0.85)
            if fp:
                cbar.set_label("次數", fontproperties=fp)
            else:
                cbar.set_label("次數")
    else:
        # 空資料時仍顯示軸範圍
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

    # 標題與座標軸
    if fp:
        ax.set_title(title, fontproperties=fp)
        ax.set_xlabel(xlabel, fontproperties=fp)
        ax.set_ylabel(ylabel, fontproperties=fp)
    else:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # 反轉 y 軸，讓場地「下方」顯示在圖的「下方」
    if flip_y:
        ax.set_ylim(ymax, ymin)

    # 輸出成 OpenCV BGR
    fig.canvas.draw()
    img_rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig)
    return cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)


# 1) 讀資料
df = pd.read_csv(SET1_CSV)
rally_seg = pd.read_csv(RALLYSEG_CSV)

# 2) 對應這個 RALLY 的 Start（假設 RallySeg 依序對應 1-based）
seg_start = int(rally_seg.iloc[RALLY_ID-1]["Start"])

# 3) 取該 rally 的所有球，且要有落點座標；分 A / B
base = df[(df["rally"] == RALLY_ID)].dropna(subset=["landing_x", "landing_y"]).copy()

# 轉 local end frame + fps_scale + offset（所有球）
base["local_end"] = ((base["end_frame_num"] - seg_start) * FPS_SCALE - OFFSET).round().astype(int)

# 分 A / B，並各自依 local_end 排序
ev_A = base[base["player"] == "A"].sort_values("local_end").reset_index(drop=True)
ev_B = base[base["player"] == "B"].sort_values("local_end").reset_index(drop=True)

# 4) 固定座標範圍（讓兩張熱圖可比較）
xmin = df["landing_x"].min(); xmax = df["landing_x"].max()
ymin = df["landing_y"].min(); ymax = df["landing_y"].max()

# 5) 開影片
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"無法開啟影片：{VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 畫布：影片 + A 熱圖 + B 熱圖（橫向三欄）
HM_W, HM_H = HEATMAP_SIZE
out_w = W + HM_W + HM_W
out_h = max(H, HM_H)

writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
if not writer.isOpened():
    OUT = Path(OUT).with_suffix(".avi").as_posix()
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"XVID"), fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError("無法建立輸出影片，請安裝對應的編碼器或改用 .avi")

# 6) 逐幀播放 + 動態累積（A 與 B 各自累積）
xs_A, ys_A, xs_B, ys_B = [], [], [], []
ptr_A = 0
ptr_B = 0
last_A = None
last_B = None
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    updated_A = False
    updated_B = False

    # A：把所有 local_end <= 當前幀 的新球加入一次
    while ptr_A < len(ev_A) and ev_A.loc[ptr_A, "local_end"] <= frame_idx:
        xs_A.append(ev_A.loc[ptr_A, "landing_x"])
        ys_A.append(ev_A.loc[ptr_A, "landing_y"])
        ptr_A += 1
        updated_A = True

    # B：把所有 local_end <= 當前幀 的新球加入一次
    while ptr_B < len(ev_B) and ev_B.loc[ptr_B, "local_end"] <= frame_idx:
        xs_B.append(ev_B.loc[ptr_B, "landing_x"])
        ys_B.append(ev_B.loc[ptr_B, "landing_y"])
        ptr_B += 1
        updated_B = True

    # 有新點才重畫各自的熱圖
    if updated_A or last_A is None:
        last_A = make_heatmap_img(xs_A, ys_A, xmin, xmax, ymin, ymax,
                                  bins=BINS, size=HEATMAP_SIZE,
                                  title=TITLE_A, xlabel="X", ylabel="Y", flip_y=True)
    if updated_B or last_B is None:
        last_B = make_heatmap_img(xs_B, ys_B, xmin, xmax, ymin, ymax,
                                  bins=BINS, size=HEATMAP_SIZE,
                                  title=TITLE_B, xlabel="X", ylabel="Y", flip_y=True)

    # 組合畫面：左=影片，中=A 熱圖，右=B 熱圖
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:H, :W] = frame
    # 中間貼 A 熱圖
    canvas[:last_A.shape[0], W:W+HM_W] = last_A
    # 右邊貼 B 熱圖
    canvas[:last_B.shape[0], W+HM_W:W+HM_W*2] = last_B

    # Debug 文字：目前幀、已累積球數、下一個觸發點（A/B 各自顯示）
    next_A = ev_A.loc[ptr_A, "local_end"] if ptr_A < len(ev_A) else -1
    next_B = ev_B.loc[ptr_B, "local_end"] if ptr_B < len(ev_B) else -1
    info = f"Frame={frame_idx}  A_shots={len(xs_A)} (NextEnd={next_A})  B_shots={len(xs_B)} (NextEnd={next_B})"
    cv2.putText(canvas, info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    writer.write(canvas)
    frame_idx += 1

cap.release()
writer.release()
print(f"輸出完成：{OUT}")
