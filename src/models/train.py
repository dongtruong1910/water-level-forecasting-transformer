import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import time

# Import các module của chúng ta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import src.config as config
from src.models.model import TimeSeriesTransformer
from src.data.data_loader import get_data_loaders


def train_model():
    print("🚀 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MODEL...")

    # 1. Chọn thiết bị (GPU nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Thiết bị sử dụng: {device}")

    # 2. Load Dữ liệu
    train_loader, val_loader, _ = get_data_loaders()

    # Lấy mẫu để xem số chiều input
    X_sample, _ = next(iter(train_loader))
    num_features = X_sample.shape[2]  # Sẽ là 6
    print(f"ℹ️  Số đặc trưng đầu vào (Features): {num_features}")

    # 3. Khởi tạo Model
    model = TimeSeriesTransformer(
        num_features=num_features,
        d_model=config.D_MODEL,
        nhead=config.N_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT_RATE,
        input_window=config.INPUT_WINDOW_DAYS,
        output_window=config.OUTPUT_WINDOW_DAYS
    ).to(device)

    # 4. Cài đặt Loss & Optimizer
    criterion = nn.MSELoss()  # Dùng Mean Squared Error cho bài toán hồi quy
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Learning Rate Scheduler: Giảm LR nếu loss không giảm (giúp hội tụ tốt hơn)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 5. Vòng lặp Training
    best_val_loss = float('inf')

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)  # Forward
            loss = criterion(output, y_batch)  # Tính lỗi
            loss.backward()  # Backward
            optimizer.step()  # Cập nhật trọng số

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Cập nhật LR
        scheduler.step(avg_val_loss)

        # --- LOGGING & SAVING ---
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {epoch_time:.2f}s")

        # Lưu model tốt nhất (Checkpoint)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"   🔥 Đã lưu model tốt nhất (Val Loss giảm từ {best_val_loss:.6f} -> {avg_val_loss:.6f})")

    print("\n✅ HUẤN LUYỆN HOÀN TẤT!")
    print(f"Model đã được lưu tại: {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()