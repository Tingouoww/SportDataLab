import pandas as pd

# 載入資料
df = pd.read_csv("set1.csv")
rally_seg = pd.read_csv("RallySeg.csv")

def get_rally_frames(rally_id: int, ball_round: int, df: pd.DataFrame, rally_seg: pd.DataFrame):
    """
    輸入 rally 編號與該 rally 的第幾球，回傳 (rally_frame, rally_end_frame)
    """
    # 找到該球的資料
    row = df[(df["rally"] == rally_id) & (df["ball_round"] == ball_round)].iloc[0]
    
    # 找到對應的 rally 區間
    seg = rally_seg.iloc[rally_id - 1]  # 假設 rally 編號從 1 開始且順序一致
    
    # 計算相對 frame
    rally_frame = row["frame_num"] - seg["Start"]
    rally_end_frame = row["end_frame_num"] - seg["Start"]
    
    return rally_frame, rally_end_frame

# 範例：取第1個 rally 的第1球
print(get_rally_frames(1, 1, df, rally_seg))
print(rally_seg.iloc[0])


