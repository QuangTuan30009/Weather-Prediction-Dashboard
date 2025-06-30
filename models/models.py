import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class WeatherPredictor:
    def __init__(self):
        self.temp_model = None
        self.precip_model = None
        self.models_dir = 'models'
        self.temp_features = None
        self.precip_features = None
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def prepare_data(self, df):
        # Chỉ sử dụng các cột có sẵn trong dataset thực tế
        feature_cols = [
            'rain (mm)', 'snowfall (cm)', 
            'snow_depth (m)', 'weather_code (wmo code)', 'is_day ()',
            'relative_humidity_2m (%)', 'dew_point_2m (°C)', 'cloud_cover (%)'
        ]
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Tạo features bổ sung từ time
        df['hour'] = df['time'].dt.hour
        df['day'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['day_of_week'] = df['time'].dt.dayofweek
        
        available_features.extend(['hour', 'day', 'month', 'day_of_week'])
        
        X = df[available_features].fillna(0)
        y_temp = df['temperature_2m (°C)'].ffill()
        y_precip = df['precipitation (mm)'].ffill()
        
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_temp_train, y_temp_test = y_temp[:split_idx], y_temp[split_idx:]
        y_precip_train, y_precip_test = y_precip[:split_idx], y_precip[split_idx:]
        
        return X_train, X_test, y_temp_train, y_temp_test, y_precip_train, y_precip_test, available_features

    def train_temperature_model(self, X_train, y_train):
        print("Training Temperature Random Forest...")
        self.temp_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.temp_model.fit(X_train, y_train)
        # Lưu lại feature names
        self.temp_features = X_train.columns.tolist()
        joblib.dump(self.temp_model, f'{self.models_dir}/temperature_model.pkl')
        print("Temperature model trained and saved!")

    def train_precipitation_model(self, X_train, y_train):
        print("Training Precipitation Random Forest...")
        self.precip_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.precip_model.fit(X_train, y_train)
        # Lưu lại feature names
        self.precip_features = X_train.columns.tolist()
        joblib.dump(self.precip_model, f'{self.models_dir}/precipitation_model.pkl')
        print("Precipitation model trained and saved!")

    def evaluate_models(self, X_test, y_temp_test, y_precip_test):
        results = {}
        
        if self.temp_model:
            y_pred_temp = self.temp_model.predict(X_test)
            results['temperature'] = {
                'rmse': np.sqrt(mean_squared_error(y_temp_test, y_pred_temp)),
                'mae': mean_absolute_error(y_temp_test, y_pred_temp),
                'r2': r2_score(y_temp_test, y_pred_temp)
            }
        
        if self.precip_model:
            y_pred_precip = self.precip_model.predict(X_test)
            results['precipitation'] = {
                'rmse': np.sqrt(mean_squared_error(y_precip_test, y_pred_precip)),
                'mae': mean_absolute_error(y_precip_test, y_pred_precip),
                'r2': r2_score(y_precip_test, y_pred_precip)
            }
        
        return results

    def load_models(self):
        try:
            self.temp_model = joblib.load(f'{self.models_dir}/temperature_model.pkl')
            # Lấy feature names từ model
            self.temp_features = list(self.temp_model.feature_names_in_)
            print("Temperature model loaded!")
        except:
            print("Temperature model not found!")
        
        try:
            self.precip_model = joblib.load(f'{self.models_dir}/precipitation_model.pkl')
            self.precip_features = list(self.precip_model.feature_names_in_)
            print("Precipitation model loaded!")
        except:
            print("Precipitation model not found!")

    def _prepare_X_for_predict(self, X, feature_names):
        # Đảm bảo X chỉ có đúng các cột đã train, thêm cột thiếu với giá trị 0
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        # Loại bỏ mọi cột lạ (chỉ giữ đúng feature_names)
        X = X[feature_names]
        return X

    def predict_future(self, df, days=7, start_offset_hours=0):
        # Chỉ sử dụng các cột có sẵn trong dataset thực tế
        feature_cols = [
            'rain (mm)', 'snowfall (cm)', 
            'snow_depth (m)', 'weather_code (wmo code)', 'is_day ()',
            'relative_humidity_2m (%)', 'dew_point_2m (°C)', 'cloud_cover (%)'
        ]
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Tạo features bổ sung từ time
        df['hour'] = df['time'].dt.hour
        df['day'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['day_of_week'] = df['time'].dt.dayofweek
        
        available_features.extend(['hour', 'day', 'month', 'day_of_week'])
        
        latest_data = df[available_features].fillna(0).iloc[-1:]
        
        future_dates = pd.date_range(
            start=df['time'].max() + pd.Timedelta(hours=1 + start_offset_hours),
            periods=days*24,
            freq='H'
        )
        
        future_data = []
        for date in future_dates:
            row = latest_data.copy()
            row['hour'] = date.hour
            row['day'] = date.day
            row['month'] = date.month
            row['day_of_week'] = date.dayofweek
            future_data.append(row.values[0])
        
        future_df = pd.DataFrame(future_data, columns=available_features)
        
        predictions = {}
        
        if self.temp_model:
            X_temp = self._prepare_X_for_predict(future_df.copy(), self.temp_features)
            temp_preds = self.temp_model.predict(X_temp)
            predictions['temperature'] = temp_preds
        
        if self.precip_model:
            X_precip = self._prepare_X_for_predict(future_df.copy(), self.precip_features)
            precip_preds = self.precip_model.predict(X_precip)
            predictions['precipitation'] = precip_preds
        
        return predictions, future_dates

def train_models(data_file="C:/Users/taqua/OneDrive/Chuyên nghành/Kì 4/DAP391m/Project/clean_repo/data/dataset.csv"):
    # Đọc file CSV với low_memory=False để tránh warning
    df = pd.read_csv(data_file, encoding='latin1', low_memory=False)
    
    # Kiểm tra xem có cột 'time' không
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    else:
        # Nếu không có cột time, tạo một cột time giả dựa trên index
        print("Không tìm thấy cột 'time', tạo cột time giả...")
        df['time'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
    
    predictor = WeatherPredictor()
    
    X_train, X_test, y_temp_train, y_temp_test, y_precip_train, y_precip_test, features = predictor.prepare_data(df)
    
    predictor.train_temperature_model(X_train, y_temp_train)
    predictor.train_precipitation_model(X_train, y_precip_train)
    
    results = predictor.evaluate_models(X_test, y_temp_test, y_precip_test)
    
    print("\n=== Kết quả đánh giá mô hình ===")
    for target, metrics in results.items():
        print(f"\n{target.upper()}:")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  R²: {metrics['r2']:.3f}")
    
    return predictor

if __name__ == "__main__":
    predictor = train_models() 