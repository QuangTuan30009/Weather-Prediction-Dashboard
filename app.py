from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import os
from models.models import WeatherPredictor

app = Flask(__name__)

DATA_FILE = "C:/Users/taqua/OneDrive/Chuyên nghành/Kì 4/DAP391m/Project/clean_repo/data/dataset.csv"

predictor = WeatherPredictor()
predictor.load_models()

def load_data():
    try:
        df = pd.read_csv(DATA_FILE, encoding="latin1")
        df['time'] = pd.to_datetime(df['time'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

# --- HÀM TIỆN ÍCH LẤY FEATURE TRUNG BÌNH THEO NGÀY ---
def get_avg_features_for_date(df, target_date):
    # target_date: datetime.date
    month_day = target_date.strftime('%m-%d')
    df['month_day'] = df['time'].dt.strftime('%m-%d')
    same_days = df[df['month_day'] == month_day]
    if same_days.empty:
        # fallback: lấy dòng cuối cùng
        row = df.iloc[-1]
        return {
            'rain (mm)': row.get('rain (mm)', 0),
            'snowfall (cm)': row.get('snowfall (cm)', 0),
            'snow_depth (m)': row.get('snow_depth (m)', 0),
            'weather_code (wmo code)': row.get('weather_code (wmo code)', 0),
            'is_day ()': 1,
            'relative_humidity_2m (%)': row.get('relative_humidity_2m (%)', 0),
            'dew_point_2m (°C)': row.get('dew_point_2m (°C)', 0),
            'cloud_cover (%)': row.get('cloud_cover (%)', 0),
        }
    else:
        return {
            'rain (mm)': same_days['rain (mm)'].mean() if 'rain (mm)' in same_days else 0,
            'snowfall (cm)': same_days['snowfall (cm)'].mean() if 'snowfall (cm)' in same_days else 0,
            'snow_depth (m)': same_days['snow_depth (m)'].mean() if 'snow_depth (m)' in same_days else 0,
            'weather_code (wmo code)': same_days['weather_code (wmo code)'].mode()[0] if 'weather_code (wmo code)' in same_days and not same_days['weather_code (wmo code)'].isnull().all() else 0,
            'is_day ()': 1,
            'relative_humidity_2m (%)': same_days['relative_humidity_2m (%)'].mean() if 'relative_humidity_2m (%)' in same_days else 0,
            'dew_point_2m (°C)': same_days['dew_point_2m (°C)'].mean() if 'dew_point_2m (°C)' in same_days else 0,
            'cloud_cover (%)': same_days['cloud_cover (%)'].mean() if 'cloud_cover (%)' in same_days else 0,
        }

@app.route('/api/current-weather')
def current_weather():
    date = request.args.get('date')
    df = load_data()
    if df.empty:
        return jsonify({'error': 'No data'}), 404
    if date:
        target_time = pd.to_datetime(date)
        month_day = target_time.strftime('%m-%d')
        df['month_day'] = df['time'].dt.strftime('%m-%d')
        same_days = df[df['month_day'] == month_day]
        if not same_days.empty:
            temp = same_days['temperature_2m (°C)'].mean() if 'temperature_2m (°C)' in same_days else 0
            humidity = same_days['relative_humidity_2m (%)'].mean() if 'relative_humidity_2m (%)' in same_days else 0
            dew_point = same_days['dew_point_2m (°C)'].mean() if 'dew_point_2m (°C)' in same_days else 0
            cloud_cover = same_days['cloud_cover (%)'].mean() if 'cloud_cover (%)' in same_days else 0
            precipitation = same_days['precipitation (mm)'].mean() if 'precipitation (mm)' in same_days else 0
            if 'weather_code (wmo code)' in same_days and not same_days['weather_code (wmo code)'].isnull().all():
                weather_code = int(same_days['weather_code (wmo code)'].mode()[0])
            else:
                weather_code = 0
            current_data = {
                'temperature': round(temp, 1),
                'humidity': round(humidity, 1),
                'dew_point': round(dew_point, 1),
                'cloud_cover': round(cloud_cover, 1),
                'precipitation': round(precipitation, 2),
                'weather_code': weather_code,
                'time': target_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            return jsonify(current_data)
        else:
            latest = df.iloc[-1]
            current_data = {
                'temperature': float(round(latest['temperature_2m (°C)'], 1)),
                'humidity': float(round(latest['relative_humidity_2m (%)'], 1)),
                'dew_point': float(round(latest['dew_point_2m (°C)'], 1)),
                'cloud_cover': float(round(latest['cloud_cover (%)'], 1)),
                'precipitation': float(round(latest['precipitation (mm)'], 2)),
                'weather_code': int(latest['weather_code (wmo code)']),
                'time': target_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            return jsonify(current_data)
    else:
        latest = df.iloc[-1]
        current_data = {
            'temperature': float(round(latest['temperature_2m (°C)'], 1)),
            'humidity': float(round(latest['relative_humidity_2m (%)'], 1)),
            'dew_point': float(round(latest['dew_point_2m (°C)'], 1)),
            'cloud_cover': float(round(latest['cloud_cover (%)'], 1)),
            'precipitation': float(round(latest['precipitation (mm)'], 2)),
            'weather_code': int(latest['weather_code (wmo code)']),
            'time': latest['time'].strftime('%Y-%m-%d %H:%M:%S')
        }
        return jsonify(current_data)

@app.route('/api/historical-data')
def historical_data():
    date = request.args.get('date')
    df = load_data()
    if df.empty:
        return jsonify({'error': 'No data'}), 404
    if date:
        df = df[df['time'].dt.date == pd.to_datetime(date).date()]
        if df.empty:
            return jsonify({'error': 'No data for this date'}), 404
        recent_data = df.copy()
    else:
        end_date = df['time'].max()
        start_date = end_date - pd.Timedelta(days=30)
        recent_data = df[df['time'] >= start_date].copy()
    chart_data = {
        'labels': recent_data['time'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
        'temperature': recent_data['temperature_2m (°C)'].round(1).tolist(),
        'humidity': recent_data['relative_humidity_2m (%)'].round(1).tolist(),
        'precipitation': recent_data['precipitation (mm)'].round(2).tolist()
    }
    return jsonify(chart_data)

@app.route('/api/forecast-data')
def forecast_data():
    date = request.args.get('date')
    df = load_data()
    if df.empty:
        return jsonify({'error': 'No data'}), 404
    if date:
        start_date = pd.to_datetime(date)
        days = [start_date + pd.Timedelta(days=i) for i in range(7)]
        predictions = {'temperature': [], 'precipitation': []}
        future_dates = []
        for d in days:
            avg_features = get_avg_features_for_date(df, d.date())
            features = avg_features.copy()
            features.update({
                'hour': 12,
                'day': d.day,
                'month': d.month,
                'day_of_week': d.weekday()
            })
            X = pd.DataFrame([features])
            X_temp = predictor._prepare_X_for_predict(X.copy(), predictor.temp_features)
            X_precip = predictor._prepare_X_for_predict(X.copy(), predictor.precip_features)
            temp = float(predictor.temp_model.predict(X_temp)[0])
            precipitation = float(predictor.precip_model.predict(X_precip)[0])
            predictions['temperature'].append(temp)
            predictions['precipitation'].append(precipitation)
            future_dates.append(d)
        forecast_data = {
            'dates': [str(d.date()) for d in future_dates],
            'temperature': [round(t, 1) for t in predictions['temperature']],
            'precipitation': [round(p, 2) for p in predictions['precipitation']]
        }
        return jsonify(forecast_data)
    else:
        predictions, future_dates = predictor.predict_future(df, days=7)
        forecast_df = pd.DataFrame({
            'time': future_dates,
            'temperature_2m (°C)': predictions.get('temperature', []),
            'precipitation (mm)': predictions.get('precipitation', [])
        })
        daily_forecast = forecast_df.groupby(forecast_df['time'].dt.date).agg({
            'temperature_2m (°C)': 'mean',
            'precipitation (mm)': 'sum'
        }).reset_index()
        forecast_data = {
            'dates': [str(d) for d in daily_forecast['time']],
            'temperature': daily_forecast['temperature_2m (°C)'].round(1).tolist(),
            'precipitation': daily_forecast['precipitation (mm)'].round(2).tolist(),
        }
        return jsonify(forecast_data)

@app.route('/api/model-predictions')
def model_predictions():
    date = request.args.get('date')
    df = load_data()
    if df.empty:
        return jsonify({'error': 'No data'}), 404
    if date:
        start_date = pd.to_datetime(date)
        days = [start_date + pd.Timedelta(days=i) for i in range(7)]
        predictions = {'temperature': [], 'precipitation': []}
        for d in days:
            avg_features = get_avg_features_for_date(df, d.date())
            features = avg_features.copy()
            features.update({
                'hour': 12,
                'day': d.day,
                'month': d.month,
                'day_of_week': d.weekday()
            })
            X = pd.DataFrame([features])
            X_temp = predictor._prepare_X_for_predict(X.copy(), predictor.temp_features)
            X_precip = predictor._prepare_X_for_predict(X.copy(), predictor.precip_features)
            temp = float(predictor.temp_model.predict(X_temp)[0])
            precipitation = float(predictor.precip_model.predict(X_precip)[0])
            predictions['temperature'].append(temp)
            predictions['precipitation'].append(precipitation)
        dates = [str(d.date()) for d in days]
        model_results = {
            'dates': dates,
            'temperature': [round(t, 1) for t in predictions['temperature']],
            'precipitation': [round(p, 2) for p in predictions['precipitation']]
        }
        return jsonify(model_results)
    else:
        predictions, future_dates = predictor.predict_future(df, days=7)
        dates = [str(d.date()) for d in pd.date_range(start=future_dates[0], periods=7, freq='D')]
        temp_daily = []
        precip_daily = []
        for i in range(7):
            temp_daily.append(np.mean(predictions.get('temperature', [])[i*24:(i+1)*24]))
            precip_daily.append(np.sum(predictions.get('precipitation', [])[i*24:(i+1)*24]))
        model_results = {
            'dates': dates,
            'temperature': np.round(temp_daily, 1).tolist(),
            'precipitation': np.round(precip_daily, 2).tolist(),
        }
        return jsonify(model_results)

@app.route('/api/weather-stats')
def weather_stats():
    date = request.args.get('date')
    df = load_data()
    if df.empty:
        return jsonify({'error': 'No data'}), 404
    if date:
        target_time = pd.to_datetime(date)
        # Lọc đúng ngày trong dataset
        df['date_only'] = df['time'].dt.date
        day_df = df[df['date_only'] == target_time.date()]
        if not day_df.empty:
            stats = {
                'avg_temperature': float(round(day_df['temperature_2m (°C)'].mean(), 1)),
                'max_temperature': float(round(day_df['temperature_2m (°C)'].max(), 1)),
                'min_temperature': float(round(day_df['temperature_2m (°C)'].min(), 1)),
                'avg_humidity': float(round(day_df['relative_humidity_2m (%)'].mean(), 1)),
                'total_precipitation': float(round(day_df['precipitation (mm)'].sum(), 2)),
                'avg_cloud_cover': float(round(day_df['cloud_cover (%)'].mean(), 1))
            }
            return jsonify(stats)
        else:
            # Lấy trung bình các ngày cùng tháng/ngày trong lịch sử
            month_day = target_time.strftime('%m-%d')
            df['month_day'] = df['time'].dt.strftime('%m-%d')
            same_days = df[df['month_day'] == month_day]
            if not same_days.empty:
                stats = {
                    'avg_temperature': float(round(same_days['temperature_2m (°C)'].mean(), 1)),
                    'max_temperature': float(round(same_days['temperature_2m (°C)'].max(), 1)),
                    'min_temperature': float(round(same_days['temperature_2m (°C)'].min(), 1)),
                    'avg_humidity': float(round(same_days['relative_humidity_2m (%)'].mean(), 1)),
                    'total_precipitation': float(round(same_days['precipitation (mm)'].sum(), 2)),
                    'avg_cloud_cover': float(round(same_days['cloud_cover (%)'].mean(), 1))
                }
                return jsonify(stats)
            else:
                # fallback: lấy thống kê toàn bộ dataset
                stats = {
                    'avg_temperature': float(round(df['temperature_2m (°C)'].mean(), 1)),
                    'max_temperature': float(round(df['temperature_2m (°C)'].max(), 1)),
                    'min_temperature': float(round(df['temperature_2m (°C)'].min(), 1)),
                    'avg_humidity': float(round(df['relative_humidity_2m (%)'].mean(), 1)),
                    'total_precipitation': float(round(df['precipitation (mm)'].sum(), 2)),
                    'avg_cloud_cover': float(round(df['cloud_cover (%)'].mean(), 1))
                }
                return jsonify(stats)
    else:
        stats = {
            'avg_temperature': float(round(df['temperature_2m (°C)'].mean(), 1)),
            'max_temperature': float(round(df['temperature_2m (°C)'].max(), 1)),
            'min_temperature': float(round(df['temperature_2m (°C)'].min(), 1)),
            'avg_humidity': float(round(df['relative_humidity_2m (%)'].mean(), 1)),
            'total_precipitation': float(round(df['precipitation (mm)'].sum(), 2)),
            'avg_cloud_cover': float(round(df['cloud_cover (%)'].mean(), 1))
        }
        return jsonify(stats)

@app.route('/api/forecast-5days')
def forecast_5days():
    date_str = request.args.get('date')
    if date_str:
        start_date = pd.to_datetime(date_str).date()
    else:
        start_date = pd.Timestamp.today().date()
    days = [start_date + pd.Timedelta(days=i) for i in range(5)]
    weekday_map = {0: 'Th 2', 1: 'Th 3', 2: 'Th 4', 3: 'Th 5', 4: 'Th 6', 5: 'Th 7', 6: 'CN'}
    labels = [weekday_map[d.weekday()] for d in days]
    df = load_data()
    model_data = []
    actual_data = []
    for i, d in enumerate(days):
        # Dự đoán model
        avg_features = get_avg_features_for_date(df, d)
        features = avg_features.copy()
        features.update({
            'hour': 12,
            'day': d.day,
            'month': d.month,
            'day_of_week': d.weekday()
        })
        X = pd.DataFrame([features])
        X_temp = predictor._prepare_X_for_predict(X.copy(), predictor.temp_features)
        X_precip = predictor._prepare_X_for_predict(X.copy(), predictor.precip_features)
        temp_pred = float(predictor.temp_model.predict(X_temp)[0])
        precip_pred = float(predictor.precip_model.predict(X_precip)[0])
        model_data.append({
            "label": labels[i],
            "date": str(d),
            "temperature": round(temp_pred, 1),
            "precipitation": round(precip_pred, 2)
        })
        # Dữ liệu thực tế
        day_df = df[df['time'].dt.date == d]
        if not day_df.empty:
            temp_actual = day_df['temperature_2m (°C)'].mean()
            precip_actual = day_df['precipitation (mm)'].sum()
            actual_data.append({
                "label": labels[i],
                "date": str(d),
                "temperature": round(temp_actual, 1),
                "precipitation": round(precip_actual, 2)
            })
        else:
            actual_data.append({
                "label": labels[i],
                "date": str(d),
                "temperature": None,
                "precipitation": None
            })
    return jsonify({
        "title": "Dự báo 5 ngày",
        "model_data": model_data,
        "actual_data": actual_data
    })

@app.route('/api/model-comparison')
def model_comparison():
    date = request.args.get('date')
    if not date:
        return jsonify({'error': 'Date parameter required'}), 400
    
    df = load_data()
    if df.empty:
        return jsonify({'error': 'No data available'}), 404
    
    start_date = pd.to_datetime(date)
    days = [start_date + pd.Timedelta(days=i) for i in range(7)]
    prediction_dates = [d.date() for d in days]
    
    predictions = {'temperature': [], 'precipitation': []}
    for d in days:
        avg_features = get_avg_features_for_date(df, d.date())
        features = avg_features.copy()
        features.update({
            'hour': 12,
            'day': d.day,
            'month': d.month,
            'day_of_week': d.weekday()
        })
        X = pd.DataFrame([features])
        X_temp = predictor._prepare_X_for_predict(X.copy(), predictor.temp_features)
        X_precip = predictor._prepare_X_for_predict(X.copy(), predictor.precip_features)
        temp_pred = float(predictor.temp_model.predict(X_temp)[0])
        precip_pred = float(predictor.precip_model.predict(X_precip)[0])
        predictions['temperature'].append(temp_pred)
        predictions['precipitation'].append(precip_pred)
    
    # Chuẩn bị dữ liệu cho bảng dự đoán và tính accuracy cho từng ngày nếu có dữ liệu thực tế
    model_predictions = []
    temp_accuracies = []
    precip_accuracies = []
    avg_errors = []
    model_scores = []
    for i, d in enumerate(days):
        day_df = df[df['time'].dt.date == d.date()]
        if not day_df.empty:
            actual_temp_avg = day_df['temperature_2m (°C)'].mean()
            actual_precip_total = day_df['precipitation (mm)'].sum()
            temp_accuracy = max(0, 100 - abs(predictions['temperature'][i] - actual_temp_avg))
            precip_accuracy = max(0, 100 - abs(predictions['precipitation'][i] - actual_precip_total) * 10)
            avg_error = (abs(predictions['temperature'][i] - actual_temp_avg) + abs(predictions['precipitation'][i] - actual_precip_total)) / 2
            model_score = (temp_accuracy + precip_accuracy) / 2
            accuracy_str = f"{temp_accuracy:.1f}%"
            accuracy_class = 'accuracy-high' if temp_accuracy >= 80 else ('accuracy-medium' if temp_accuracy >= 60 else 'accuracy-low')
            temp_accuracies.append(temp_accuracy)
            precip_accuracies.append(precip_accuracy)
            avg_errors.append(avg_error)
            model_scores.append(model_score)
        else:
            accuracy_str = '-'
            accuracy_class = ''
        model_predictions.append({
            'date': d.strftime('%d/%m/%Y'),
            'temperature': round(predictions['temperature'][i], 1),
            'precipitation': round(predictions['precipitation'][i], 2),
            'accuracy': accuracy_str,
            'accuracy_class': accuracy_class
        })
    # Chuẩn bị dữ liệu cho bảng thực tế (gộp theo ngày, trùng với từng ngày dự đoán)
    actual_table_data = []
    for i, d in enumerate(days):
        day_df = df[df['time'].dt.date == d.date()]
        if not day_df.empty:
            temp_actual = day_df['temperature_2m (°C)'].mean()
            precip_actual = day_df['precipitation (mm)'].sum()
            actual_table_data.append({
                'date': d.strftime('%d/%m/%Y'),
                'temperature': round(temp_actual, 1),
                'precipitation': round(precip_actual, 2)
            })
        else:
            actual_table_data.append({
                'date': d.strftime('%d/%m/%Y'),
                'temperature': '-',
                'precipitation': '-'
            })
    # Tóm tắt: lấy trung bình các ngày có dữ liệu thực tế
    if temp_accuracies:
        summary = {
            'temperature_accuracy': round(np.mean(temp_accuracies), 1),
            'precipitation_accuracy': round(np.mean(precip_accuracies), 1),
            'average_error': round(np.mean(avg_errors), 2),
            'model_score': round(np.mean(model_scores), 1)
        }
    else:
        summary = {
            'temperature_accuracy': '-',
            'precipitation_accuracy': '-',
            'average_error': '-',
            'model_score': '-'
        }
    return jsonify({
        'model_predictions': model_predictions,
        'actual_data': actual_table_data,
        'summary': summary
    })

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)

    app.run(debug=True, host='0.0.0.0', port=5000) 