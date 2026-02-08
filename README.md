
# Weather Prediction Dashboard

Ứng dụng web hiển thị dự đoán thời tiết sử dụng Flask backend và Chart.js frontend.

## Tính năng

- **Dashboard thời tiết hiện tại**: Hiển thị nhiệt độ, độ ẩm, áp suất, tốc độ gió, lượng mưa và mây che phủ
- **Biểu đồ lịch sử**: Biểu đồ nhiệt độ và độ ẩm 30 ngày gần nhất
- **Dự báo 7 ngày**: Dự báo thời tiết cho 7 ngày tới
- **So sánh mô hình**: So sánh kết quả dự đoán từ Random Forest, XGBoost và LSTM
- **Thống kê tổng quan**: Các chỉ số thống kê về thời tiết
- **Giao diện responsive**: Tương thích với mọi thiết bị

## Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd weather_prediction_web
```

### 2. Tạo môi trường ảo
```bash
python -m venv venv
```

### 3. Kích hoạt môi trường ảo
**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 5. Chuẩn bị dữ liệu
Đặt file `dataset.csv` vào thư mục `data/` hoặc ứng dụng sẽ tạo dữ liệu mẫu.

### 6. Chạy ứng dụng
```bash
python app.py
```

Ứng dụng sẽ chạy tại: http://localhost:5000

## Cấu trúc dự án

```
weather_prediction_web/
├── app.py                 # Flask application chính
├── requirements.txt       # Dependencies
├── README.md             # Hướng dẫn
├── data/                 # Thư mục chứa dữ liệu
│   └── dataset.csv       # File dữ liệu thời tiết
├── templates/            # HTML templates
│   └── index.html        # Trang chủ dashboard
└── static/               # Static files
    ├── css/
    │   └── style.css     # Custom CSS
    └── js/
        └── dashboard.js  # JavaScript cho dashboard
```

## API Endpoints

- `GET /` - Trang chủ dashboard
- `GET /api/current-weather` - Thông tin thời tiết hiện tại
- `GET /api/historical-data` - Dữ liệu lịch sử cho biểu đồ
- `GET /api/forecast-data` - Dữ liệu dự báo 7 ngày
- `GET /api/model-predictions` - Kết quả dự đoán từ các mô hình
- `GET /api/weather-stats` - Thống kê thời tiết

## Công nghệ sử dụng

### Backend
- **Flask**: Web framework
- **Pandas**: Xử lý dữ liệu
- **NumPy**: Tính toán số học
- **Scikit-learn**: Machine learning models
- **XGBoost**: Gradient boosting
- **TensorFlow**: Deep learning (LSTM)

### Frontend
- **Bootstrap 5**: CSS framework
- **Chart.js**: Biểu đồ tương tác
- **Font Awesome**: Icons
- **Vanilla JavaScript**: Tương tác client-side

## Tùy chỉnh

### Thêm mô hình mới
1. Cập nhật hàm `generate_forecast_data()` trong `app.py`
2. Thêm endpoint API mới
3. Cập nhật JavaScript để hiển thị kết quả

### Thay đổi giao diện
1. Chỉnh sửa `templates/index.html` cho layout
2. Cập nhật `static/css/style.css` cho styling
3. Sửa đổi `static/js/dashboard.js` cho tương tác

### Tích hợp dữ liệu thực
1. Thay thế `create_sample_data()` bằng hàm đọc dữ liệu thực
2. Cập nhật các API endpoints để trả về dữ liệu thực
3. Điều chỉnh format dữ liệu cho phù hợp

## Troubleshooting

### Lỗi import
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Lỗi port đã được sử dụng
Thay đổi port trong `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Lỗi dữ liệu
Kiểm tra file `data/dataset.csv` có đúng format không:
- Cột `time`: datetime
- Các cột khác: numeric values

## Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## License


MIT License - xem file LICENSE để biết thêm chi tiết. 
