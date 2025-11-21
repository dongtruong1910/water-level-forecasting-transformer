/DuDoanMucNuoc_Transformer/
|
|-- data/
|   |-- raw/
|   |   |-- vuc_mau_data.csv  (Đây là nơi bạn đặt file dữ liệu thô)
|   |-- processed/
|       |-- (Dữ liệu đã xử lý sẽ được lưu ở đây)
|
|-- notebooks/
|   |-- 01_Data_Exploration.ipynb (Nơi bạn khám phá dữ liệu)
|
|-- saved_models/
|   |-- (Các model đã huấn luyện sẽ được lưu ở đây)
|
|-- src/
|   |-- __init__.py
|   |-- config.py           (File chứa các cài đặt và siêu tham số)
|   |-- data_loader.py      (Class để tải, xử lý và tạo cửa sổ trượt)
|   |-- model.py            (Kiến trúc Transformer của bạn)
|   |-- train.py            (Script chính để chạy huấn luyện)
|   |-- predict.py          (Script để chạy dự đoán với model đã lưu)
|   |-- utils.py            (Các hàm tiện ích, ví dụ: save/load model)
|   |-- evaluation.py       (Hàm để đánh giá hiệu suất của model)
|
|-- requirements.txt        (Các thư viện cần thiết)
|-- README.md               (Mô tả dự án của bạn)