import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import sys
from tqdm import tqdm

# Import config & modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import src.config as config
from src.models.model import TimeSeriesTransformer
from src.data.data_loader import get_data_loaders


def inverse_transform_target(pred_scaled, actual_scaled, scaler, target_idx):
    """Hàm giải mã (Un-scale) giá trị về đơn vị Mét"""
    # pred_scaled shape: [N, 1] hoặc [N]
    # Tạo ma trận giả để lừa scaler inverse
    n_samples = len(pred_scaled)
    n_features = len(config.FEATURE_COLUMNS)

    dummy_pred = np.zeros((n_samples, n_features))
    dummy_actual = np.zeros((n_samples, n_features))

    dummy_pred[:, target_idx] = pred_scaled.flatten()
    dummy_actual[:, target_idx] = actual_scaled.flatten()

    inv_pred = scaler.inverse_transform(dummy_pred)[:, target_idx]
    inv_actual = scaler.inverse_transform(dummy_actual)[:, target_idx]

    return inv_pred, inv_actual


def evaluate_full_test_set():
    print("--- BẮT ĐẦU ĐÁNH GIÁ TOÀN DIỆN (FULL TEST SET) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data & Scaler
    # Chúng ta cần load lại DataFrame gốc để lấy ngày tháng chuẩn xác
    df_full = pd.read_csv(config.FINAL_TRAIN_FILE, parse_dates=['ThoiGianCapNhat'], index_col='ThoiGianCapNhat')
    split_idx = int(len(df_full) * config.SPLIT_RATIO)
    val_df_orig = df_full.iloc[split_idx:]  # Đây là tập Validation gốc (có ngày tháng)

    # Load Loader & Scaler
    _, val_loader, _ = get_data_loaders()
    scaler = joblib.load(config.SCALER_SAVE_PATH)
    target_idx = config.FEATURE_COLUMNS.index(config.TARGET_COLUMN)

    # 2. Load Model
    # Lấy 1 mẫu để biết số features input
    sample_x, _ = next(iter(val_loader))
    model = TimeSeriesTransformer(
        num_features=sample_x.shape[2],
        d_model=config.D_MODEL,
        nhead=config.N_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT_RATE,
        input_window=config.INPUT_WINDOW_DAYS,
        output_window=config.OUTPUT_WINDOW_DAYS
    ).to(device)

    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        print("✅ Đã load model thành công!")
    else:
        print("❌ Không tìm thấy file model.pth")
        return

    model.eval()

    # 3. Chạy Dự báo trên TOÀN BỘ tập Test (Vòng lặp)
    all_preds = []
    all_actuals = []

    print("📊 Đang chạy dự báo trên toàn bộ tập Test...")
    with torch.no_grad():
        for X_batch, y_batch in tqdm(val_loader):
            X_batch = X_batch.to(device)
            # Output: [Batch, 7, 1]
            y_pred = model(X_batch)

            all_preds.append(y_pred.cpu().numpy())
            all_actuals.append(y_batch.cpu().numpy())

    # Gộp tất cả các batch lại
    # Shape sau khi gộp: [Tổng_số_mẫu, 7, 1]
    all_preds = np.concatenate(all_preds, axis=0)
    all_actuals = np.concatenate(all_actuals, axis=0)

    # 4. Tính toán Metrics (Đánh giá độ chính xác)
    # Để đánh giá tổng quát, ta sẽ so sánh "Dự báo ngày kế tiếp" (Lead time 1)
    # Tức là: Đứng ở hôm nay, dự báo ngày mai (Ngày 1 trong chuỗi 7 ngày)

    # Lấy ngày đầu tiên trong chuỗi dự báo 7 ngày (Day 1 forecast)
    pred_lead1 = all_preds[:, 0, 0]
    actual_lead1 = all_actuals[:, 0, 0]

    # Giải mã về đơn vị Mét
    pred_m, actual_m = inverse_transform_target(pred_lead1, actual_lead1, scaler, target_idx)

    # Tính chỉ số
    rmse = np.sqrt(mean_squared_error(actual_m, pred_m))
    mae = mean_absolute_error(actual_m, pred_m)
    r2 = r2_score(actual_m, pred_m)

    print("\n" + "=" * 40)
    print("KẾT QUẢ ĐÁNH GIÁ (DỰ BÁO NGÀY KẾ TIẾP)")
    print("=" * 40)
    print(f"📉 RMSE (Sai số chuẩn): {rmse:.4f} m")
    print(f"📉 MAE (Sai số tuyệt đối): {mae:.4f} m")
    print(f"📈 R2 Score (Độ phù hợp): {r2:.4f} (Càng gần 1 càng tốt)")
    print("=" * 40)

    # 5. Vẽ biểu đồ "Continuous" (Liên tục theo thời gian)
    # Cần lấy đúng ngày tháng tương ứng.
    # Tập Val loader bắt đầu cắt từ: Input Window.
    # Nên điểm dự báo đầu tiên sẽ tương ứng với ngày thứ (Input_Window) trong tập Val DF

    valid_dates = val_df_orig.index[config.INPUT_WINDOW_DAYS: config.INPUT_WINDOW_DAYS + len(pred_m)]

    plt.figure(figsize=(15, 7))

    # Vẽ đường Thực tế
    plt.plot(valid_dates, actual_m, label='Thực tế (Actual)', color='blue', linewidth=1.5)

    # Vẽ đường Dự báo (Lead 1)
    plt.plot(valid_dates, pred_m, label='Dự báo (Predicted - Lead 1)', color='red', linestyle='--', linewidth=1.5,
             alpha=0.8)

    # Format Trục ngày tháng
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))  # Cách 15 ngày hiện 1 lần
    plt.gcf().autofmt_xdate()  # Xoay chữ cho dễ đọc

    plt.title(f'Dự báo Mực nước hồ trên Tập Kiểm Thử (RMSE: {rmse:.3f}m)')
    plt.ylabel('Mực nước (m)')
    plt.xlabel('Thời gian')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Lưu ảnh
    save_path = config.PROJECT_ROOT / "evaluation_full_test.png"
    plt.savefig(save_path)
    print(f"\n✅ Đã lưu biểu đồ toàn cảnh tại: {save_path}")
    plt.show()


if __name__ == "__main__":
    evaluate_full_test_set()