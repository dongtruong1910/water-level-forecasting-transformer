from pathlib import Path
import os

# --- 1. ĐƯỜNG DẪN (PATHS) ---
PROJECT_ROOT = Path(__file__).parent.parent

# Thư mục Data
PROCESSED_DATA_DIR = PROJECT_ROOT / "data/processed"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

# Trỏ đúng vào file bạn đang có
FINAL_TRAIN_FILE = PROCESSED_DATA_DIR / "du_lieu_huan_luyen_moi.csv"

# Đường dẫn lưu Model và Scaler
MODEL_SAVE_PATH = SAVED_MODELS_DIR / "transformer_best.pth"
SCALER_SAVE_PATH = SAVED_MODELS_DIR / "scaler.pkl"

# --- 2. CẤU HÌNH FEATURES (QUAN TRỌNG) ---
# Chúng ta sẽ dùng tất cả các cột dữ liệu bạn có + cột Mưa tích lũy (tự tạo)
FEATURE_COLUMNS = [
    'MucNuoc_ThucDo',   # Input 1
    'MN_so_voi_QT',     # Input 2 (Mới thêm)
    'W_so_voi_TK',      # Input 3 (Mới thêm)
    'LuuLuongXa',       # Input 4
    'LuongMua',         # Input 5
    'Mua_Tong3Ngay'     # Input 6 (Sẽ được code tự động tạo ra)
]

TARGET_COLUMN = 'MucNuoc_ThucDo'

# --- 3. CẤU HÌNH MODEL (TRANSFORMER) ---
# Giữ nguyên thông số tối ưu cho dữ liệu nhỏ
INPUT_WINDOW_DAYS = 30  # Quá khứ 30 ngày
OUTPUT_WINDOW_DAYS = 7  # Dự báo 7 ngày

D_MODEL = 32            # Kích thước vector ẩn
N_HEADS = 4             # Số đầu Attention
NUM_ENCODER_LAYERS = 1  # Số lớp Encoder
DIM_FEEDFORWARD = 128
DROPOUT_RATE = 0.1

# --- 4. HUẤN LUYỆN ---
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 100
SPLIT_RATIO = 0.8