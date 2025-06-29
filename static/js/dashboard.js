// Weather Prediction Dashboard JavaScript

// Global variables for charts
let historicalChart, forecastChart, modelPredictionChart;

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', function() {
    updateCurrentTime();
    loadCurrentWeather();
    loadHistoricalData();
    loadForecastData();
    loadModelPrediction();
    loadWeatherStats();
    loadForecast5Days();
    loadCurrentWeather();
    loadWeatherStats();
    loadModelComparisonDefault();
    
    // Update time every second
    setInterval(updateCurrentTime, 1000);
    
    // Refresh data every 5 minutes
    setInterval(function() {
        loadCurrentWeather();
        loadHistoricalData();
        loadForecastData();
    }, 300000);

    // Thêm sự kiện cho nút Xem thời tiết
    document.getElementById('btn-weather-date').addEventListener('click', function() {
        const date = document.getElementById('weather-date-input').value;
        loadForecast5Days(date);
        loadCurrentWeather(date);
        loadWeatherStats(date);
    });
    // Tải bảng dự báo 5 ngày, thời tiết hiện tại, thống kê tổng quan mặc định (hôm nay)
    loadForecast5Days();
    loadCurrentWeather();
    loadWeatherStats();

    // Thêm sự kiện cho nút So sánh
    const btnComparison = document.getElementById('btn-comparison');
    if (btnComparison) {
        btnComparison.addEventListener('click', function() {
            const date = document.getElementById('comparison-date-input').value;
            loadModelComparison(date);
        });
    }
});

// Update current time
function updateCurrentTime() {
    const now = new Date();
    const timeString = now.toLocaleString('vi-VN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    document.getElementById('current-time').textContent = timeString;
}

// Load current weather data
async function loadCurrentWeather(date) {
    try {
        let url = '/api/current-weather';
        if (date) {
            url += `?date=${date}`;
        }
        const response = await fetch(url);
        const data = await response.json();
        
        const cardsContainer = document.getElementById('current-weather-cards');
        cardsContainer.innerHTML = `
            <div class="col-lg-2 col-md-4 col-sm-6 mb-3">
                <div class="weather-card temperature fade-in">
                    <div class="content">
                        <div class="icon"><i class="fas fa-thermometer-half"></i></div>
                        <div class="value">${data.temperature ?? 'N/A'}°C</div>
                        <div class="label">Nhiệt độ</div>
                    </div>
                </div>
            </div>
            <div class="col-lg-2 col-md-4 col-sm-6 mb-3">
                <div class="weather-card humidity fade-in">
                    <div class="content">
                        <div class="icon"><i class="fas fa-tint"></i></div>
                        <div class="value">${data.humidity ?? 'N/A'}%</div>
                        <div class="label">Độ ẩm</div>
                    </div>
                </div>
            </div>
            <div class="col-lg-2 col-md-4 col-sm-6 mb-3">
                <div class="weather-card pressure fade-in">
                    <div class="content">
                        <div class="icon"><i class="fas fa-thermometer-empty"></i></div>
                        <div class="value">${data.dew_point ?? 'N/A'}°C</div>
                        <div class="label">Điểm sương</div>
                    </div>
                </div>
            </div>
            <div class="col-lg-2 col-md-4 col-sm-6 mb-3">
                <div class="weather-card cloud fade-in">
                    <div class="content">
                        <div class="icon"><i class="fas fa-cloud"></i></div>
                        <div class="value">${data.cloud_cover ?? 'N/A'}%</div>
                        <div class="label">Mây che phủ</div>
                    </div>
                </div>
            </div>
            <div class="col-lg-2 col-md-4 col-sm-6 mb-3">
                <div class="weather-card precipitation fade-in">
                    <div class="content">
                        <div class="icon"><i class="fas fa-cloud-rain"></i></div>
                        <div class="value">${data.precipitation ?? 'N/A'}</div>
                        <div class="label">Lượng mưa (mm)</div>
                    </div>
                </div>
            </div>
            <div class="col-lg-2 col-md-4 col-sm-6 mb-3">
                <div class="weather-card code fade-in">
                    <div class="content">
                        <div class="icon"><i class="fas fa-sun"></i></div>
                        <div class="value">${data.weather_code ?? 'N/A'}</div>
                        <div class="label">Mã thời tiết</div>
                    </div>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Error loading current weather:', error);
    }
}

// Load historical data and create chart
async function loadHistoricalData() {
    try {
        const response = await fetch('/api/historical-data');
        const data = await response.json();
        
        const ctx = document.getElementById('historicalChart').getContext('2d');
        
        if (historicalChart) {
            historicalChart.destroy();
        }
        
        historicalChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [
                    {
                        label: 'Nhiệt độ (°C)',
                        data: data.temperature,
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Độ ẩm (%)',
                        data: data.humidity,
                        borderColor: '#74b9ff',
                        backgroundColor: 'rgba(116, 185, 255, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y1'
                    },
                    {
                        label: 'Lượng mưa (mm)',
                        data: data.precipitation,
                        borderColor: '#00b894',
                        backgroundColor: 'rgba(0, 184, 148, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y2'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Thời gian'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Nhiệt độ (°C)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Độ ẩm (%)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    },
                    y2: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Lượng mưa (mm)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Dữ liệu thời tiết 30 ngày gần nhất'
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading historical data:', error);
    }
}

// Load forecast data and create chart
async function loadForecastData() {
    try {
        const response = await fetch('/api/forecast-data');
        const data = await response.json();
        
        const ctx = document.getElementById('forecastChart').getContext('2d');
        
        if (forecastChart) {
            forecastChart.destroy();
        }
        
        forecastChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.dates,
                datasets: [
                    {
                        label: 'Nhiệt độ dự báo (°C)',
                        data: data.temperature,
                        backgroundColor: 'rgba(255, 107, 107, 0.8)',
                        borderColor: '#ff6b6b',
                        borderWidth: 1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Lượng mưa dự báo (mm)',
                        data: data.precipitation,
                        backgroundColor: 'rgba(0, 184, 148, 0.8)',
                        borderColor: '#00b894',
                        borderWidth: 1,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Ngày'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Nhiệt độ (°C)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Lượng mưa (mm)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Dự báo nhiệt độ và lượng mưa 7 ngày tới (Random Forest)'
                    }
                }
            }
        });
        
        // Create forecast cards
        createForecastCards(data);
        
    } catch (error) {
        console.error('Error loading forecast data:', error);
    }
}

// Create forecast cards
function createForecastCards(data) {
    const cardsContainer = document.getElementById('forecast-cards');
    cardsContainer.innerHTML = '';
    
    data.dates.forEach((date, index) => {
        const card = document.createElement('div');
        card.className = 'col-lg-1 col-md-2 col-sm-3 col-4 mb-3';
        card.innerHTML = `
            <div class="forecast-card fade-in">
                <div class="date">${formatDate(date)}</div>
                <div class="temp">${data.temperature[index]}°C</div>
                <div class="details">
                    <div><i class="fas fa-cloud-rain text-info"></i> ${data.precipitation[index]}mm</div>
                </div>
            </div>
        `;
        cardsContainer.appendChild(card);
    });
}

// Load model prediction and create chart
async function loadModelPrediction() {
    try {
        const response = await fetch('/api/model-predictions');
        const data = await response.json();
        
        const ctx = document.getElementById('modelPredictionChart').getContext('2d');
        
        if (modelPredictionChart) {
            modelPredictionChart.destroy();
        }
        
        modelPredictionChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [
                    {
                        label: 'Nhiệt độ (°C)',
                        data: data.temperature,
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        borderWidth: 3,
                        fill: false,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Lượng mưa (mm)',
                        data: data.precipitation,
                        borderColor: '#00b894',
                        backgroundColor: 'rgba(0, 184, 148, 0.1)',
                        borderWidth: 3,
                        fill: false,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Ngày'
                        }
                    },
                    y: {
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Nhiệt độ (°C)'
                        }
                    },
                    y1: {
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Lượng mưa (mm)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Dự báo chi tiết 7 ngày tới (Random Forest)'
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading model prediction:', error);
    }
}

// Load weather statistics
async function loadWeatherStats(date) {
    try {
        let url = '/api/weather-stats';
        if (date) {
            url += `?date=${date}`;
        }
        const response = await fetch(url);
        const data = await response.json();
        
        const statsContainer = document.getElementById('weather-stats');
        statsContainer.innerHTML = `
            <div class="row">
                <div class="col-6 mb-3">
                    <div class="stats-card">
                        <div class="value">${data.avg_temperature}°C</div>
                        <div class="label">Nhiệt độ TB</div>
                    </div>
                </div>
                <div class="col-6 mb-3">
                    <div class="stats-card">
                        <div class="value">${data.max_temperature}°C</div>
                        <div class="label">Nhiệt độ cao nhất</div>
                    </div>
                </div>
                <div class="col-6 mb-3">
                    <div class="stats-card">
                        <div class="value">${data.min_temperature}°C</div>
                        <div class="label">Nhiệt độ thấp nhất</div>
                    </div>
                </div>
                <div class="col-6 mb-3">
                    <div class="stats-card">
                        <div class="value">${data.avg_humidity}%</div>
                        <div class="label">Độ ẩm TB</div>
                    </div>
                </div>
                <div class="col-6 mb-3">
                    <div class="stats-card">
                        <div class="value">${data.total_precipitation}mm</div>
                        <div class="label">Tổng lượng mưa</div>
                    </div>
                </div>
                <div class="col-6 mb-3">
                    <div class="stats-card">
                        <div class="value">${data.avg_cloud_cover}%</div>
                        <div class="label">Mây che phủ TB</div>
                    </div>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Error loading weather stats:', error);
    }
}

// Utility function to format date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('vi-VN', {
        month: 'short',
        day: 'numeric'
    });
}

// Add smooth scrolling for better UX
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Hàm gọi API dự báo 5 ngày và cập nhật giao diện
async function loadForecast5Days(date) {
    try {
        let url = '/api/forecast-5days';
        if (date) {
            url += `?date=${date}`;
        }
        const response = await fetch(url);
        const result = await response.json();

        // Cập nhật bảng model
        const modelTbody = document.getElementById('forecast5-model-tbody');
        modelTbody.innerHTML = '';
        result.model_data.forEach(item => {
            modelTbody.innerHTML += `
                <tr>
                    <td>${item.label}</td>
                    <td>${item.temperature ?? '-'}</td>
                    <td>${item.precipitation ?? '-'}</td>
                </tr>
            `;
        });
        if (result.model_data.length === 0) {
            modelTbody.innerHTML = '<tr><td colspan="3" class="text-center text-muted">Không có dữ liệu</td></tr>';
        }

        // Cập nhật bảng thực tế
        const actualTbody = document.getElementById('forecast5-actual-tbody');
        actualTbody.innerHTML = '';
        result.actual_data.forEach(item => {
            actualTbody.innerHTML += `
                <tr>
                    <td>${item.label}</td>
                    <td>${item.temperature ?? '-'}</td>
                    <td>${item.precipitation ?? '-'}</td>
                </tr>
            `;
        });
        if (result.actual_data.length === 0) {
            actualTbody.innerHTML = '<tr><td colspan="3" class="text-center text-muted">Không có dữ liệu</td></tr>';
        }
    } catch (error) {
        console.error('Error loading 5-day forecast:', error);
    }
}

// Hàm tự động lấy ngày mới nhất và load so sánh
async function loadModelComparisonDefault() {
    try {
        const response = await fetch('/api/historical-data');
        const data = await response.json();
        if (data.labels && data.labels.length > 0) {
            const lastDateTime = data.labels[data.labels.length - 1];
            const lastDate = lastDateTime.split(' ')[0];
            loadModelComparison(lastDate);
            document.getElementById('comparison-date-input').value = lastDate;
        }
    } catch (error) {
        console.error('Không thể load ngày mặc định cho so sánh:', error);
    }
}

// Hàm gọi API so sánh model vs thực tế và cập nhật bảng, tóm tắt
async function loadModelComparison(date) {
    try {
        if (!date) return;
        const response = await fetch(`/api/model-comparison?date=${date}`);
        const result = await response.json();
        // Cập nhật bảng dự đoán model
        const modelTbody = document.getElementById('model-predictions-tbody');
        modelTbody.innerHTML = '';
        if (result.model_predictions && result.model_predictions.length > 0) {
            result.model_predictions.forEach(row => {
                modelTbody.innerHTML += `
                    <tr>
                        <td>${row.date}</td>
                        <td>${row.temperature}</td>
                        <td>${row.precipitation}</td>
                        <td class="${row.accuracy_class}">${row.accuracy}</td>
                    </tr>
                `;
            });
        } else {
            modelTbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">Không có dữ liệu</td></tr>';
        }
        // Cập nhật bảng thực tế
        const actualTbody = document.getElementById('actual-data-tbody');
        actualTbody.innerHTML = '';
        if (result.actual_data && result.actual_data.length > 0) {
            result.actual_data.forEach(row => {
                actualTbody.innerHTML += `
                    <tr>
                        <td>${row.date}</td>
                        <td>${row.temperature}</td>
                        <td>${row.precipitation}</td>
                    </tr>
                `;
            });
        } else {
            actualTbody.innerHTML = '<tr><td colspan="3" class="text-center text-muted">Không có dữ liệu</td></tr>';
        }
        // Cập nhật tóm tắt so sánh
        if (result.summary) {
            document.getElementById('temp-accuracy').textContent = result.summary.temperature_accuracy + '%';
            document.getElementById('precip-accuracy').textContent = result.summary.precipitation_accuracy + '%';
            document.getElementById('avg-error').textContent = result.summary.average_error;
            document.getElementById('model-score').textContent = result.summary.model_score;
        } else {
            document.getElementById('temp-accuracy').textContent = '-';
            document.getElementById('precip-accuracy').textContent = '-';
            document.getElementById('avg-error').textContent = '-';
            document.getElementById('model-score').textContent = '-';
        }
    } catch (error) {
        // Nếu lỗi (ví dụ không có dữ liệu), reset bảng và tóm tắt
        document.getElementById('model-predictions-tbody').innerHTML = '<tr><td colspan="4" class="text-center text-muted">Không có dữ liệu</td></tr>';
        document.getElementById('actual-data-tbody').innerHTML = '<tr><td colspan="3" class="text-center text-muted">Không có dữ liệu</td></tr>';
        document.getElementById('temp-accuracy').textContent = '-';
        document.getElementById('precip-accuracy').textContent = '-';
        document.getElementById('avg-error').textContent = '-';
        document.getElementById('model-score').textContent = '-';
        console.error('Error loading model comparison:', error);
    }
} 