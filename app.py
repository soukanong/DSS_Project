from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import os
import traceback 

app = Flask(__name__)
app.secret_key = 'dss-diet-secret-key-2024-vietnam'

# === 1. LOAD TÀI NGUYÊN (MODEL & DATABASE) ===
def load_assets():
    try:
        with open('nutrition_model.pkl', 'rb') as f:
            nutrition_model = pickle.load(f)
        
        meal_database = pd.read_csv('train_filtered.csv')
        
        print("✓ Đã load model 'nutrition_model.pkl'")
        print(f"✓ Đã load 'train_filtered.csv' ({meal_database.shape[0]} hồ sơ)")
        return nutrition_model, meal_database
        
    except FileNotFoundError as e:
        print(f"❌ LỖI KHỞI ĐỘNG: Không tìm thấy file 'nutrition_model.pkl' hoặc 'train_filtered.csv'.")
        print("Hãy chạy 'train_model.py' trước để tạo model.")
        print(e)
        return None, None

nutrition_model, meal_database = load_assets()

# === 2. ĐỊNH NGHĨA CHẾ ĐỘ ĂN (DIET RECOMMENDATIONS) ===
DIET_RECOMMENDATIONS = {
    'keto': {
        'name': 'Keto',
        'description': 'Chế độ ăn ít carb, nhiều chất béo lành mạnh...',
        'foods': ['Thịt', 'Cá', 'Trứng', 'Bơ', 'Dầu ô liu', 'Rau xanh'],
        'avoid': ['Gạo', 'Bánh mì', 'Mì', 'Khoai tây', 'Đồ ngọt']
    },
    'vegan': {
        'name': 'Vegan',
        'description': 'Chế độ ăn thuần chay 100%...',
        'foods': ['Rau củ', 'Trái cây', 'Các loại hạt', 'Đậu nành', 'Ngũ cốc nguyên hạt'],
        'avoid': ['Thịt', 'Cá', 'Trứng', 'Sữa', 'Sản phẩm động vật']
    },
    'low_carb': {
        'name': 'Low Carb',
        'description': 'Giảm tinh bột, kiểm soát đường huyết...',
        'foods': ['Trứng', 'Cá', 'Thịt gà', 'Bông cải xanh', 'Dầu dừa', 'Hạt chia'],
        'avoid': ['Cơm trắng', 'Bánh mì', 'Đồ ngọt', 'Nước ngọt có gas']
    },
    'mediterranean': {
        'name': 'Mediterranean',
        'description': 'Giàu omega-3, nhiều rau củ, dầu ô liu và cá...',
        'foods': ['Cá hồi', 'Dầu ô liu', 'Hạt hạnh nhân', 'Rau xanh', 'Trái cây tươi'],
        'avoid': ['Đồ chiên rán', 'Thức ăn nhanh', 'Đồ ngọt công nghiệp']
    },
    'paleo': {
        'name': 'Paleo',
        'description': 'Lấy cảm hứng từ thời kỳ đồ đá...',
        'foods': ['Thịt nạc', 'Cá', 'Trái cây', 'Rau củ', 'Hạt', 'Trứng'],
        'avoid': ['Đường', 'Ngũ cốc tinh chế', 'Sữa', 'Đồ ăn công nghiệp']
    },
    'balanced': {
        'name': 'Balanced',
        'description': 'Cân bằng dinh dưỡng, phù hợp với mọi người...',
        'foods': ['Ngũ cốc nguyên hạt', 'Thịt nạc', 'Cá', 'Rau củ', 'Sữa'],
        'avoid': ['Đồ chiên rán', 'Fast food', 'Thức ăn chế biến sẵn']
    }
}

# === 3. CÁC HÀM HỖ TRỢ (HELPER FUNCTIONS) ===

def calculate_bmi(weight, height):
    if height == 0: return 0
    height_m = height / 100
    return weight / (height_m ** 2)

def get_bmi_category(bmi):
    if bmi < 18.5: return 'Thiếu cân'
    elif 18.5 <= bmi < 25: return 'Bình thường'
    elif 25 <= bmi < 30: return 'Thừa cân'
    elif 30 <= bmi < 35: return 'Béo phì độ I'
    elif 35 <= bmi < 40: return 'Béo phì độ II'
    else: return 'Béo phì độ III'

def recommend_diet(age, gender, weight, height, goal):
    bmi = calculate_bmi(weight, height)
    if goal == 'giam_can' or bmi >= 25: return 'keto'
    elif goal == 'tang_can': return 'paleo'
    elif goal == 'duy_tri':
        if bmi < 18.5: return 'balanced'
        return 'mediterranean' 
    elif age > 45: return 'mediterranean'
    elif gender.lower() == 'nữ' and bmi < 20: return 'vegan'
    else: return 'low_carb'

# === KHÔI PHỤC HÀM GÂY LỖI (ĐỂ KHÔNG BỊ "DATA NOT FOUND") ===
def get_meal_recommendations(df, needs, goal, diet_key):
    if df is None: return []
    target_cal = needs['calories']
    if goal == 'giam_can': target_cal *= 0.85
    elif goal == 'tang_can': target_cal *= 1.15
    
    diet_name = DIET_RECOMMENDATIONS.get(diet_key, {}).get('name', 'Balanced')
    
    if 'Diet_Type' in df.columns:
        filtered_df = df[df['Diet_Type'].str.lower() == diet_name.lower()].copy()
    else:
        print("CẢNH BÁO: Không tìm thấy cột 'Diet_Type'. Bỏ qua lọc theo chế độ ăn.")
        filtered_df = df.copy()

    if filtered_df.empty:
        print(f"Không tìm thấy hồ sơ nào cho '{diet_name}'. Dùng toàn bộ CSDL.")
        filtered_df = df.copy()

    # Tính score dựa trên HỒ SƠ NGƯỜI DÙNG
    filtered_df['score'] = (
        (filtered_df['Calories'] - target_cal).abs() * 1.5 +
        (filtered_df['Proteins'] - needs['proteins']).abs() * 1.0 +
        (filtered_df['Fats'] - needs['fats']).abs() * 1.0 +
        (filtered_df['Carbs'] - needs['carbs']).abs() * 1.0
    )
    
    result_columns = ['Meal', 'Calories', 'Proteins', 'Fats', 'Carbs', 'Diet_Type']
    final_columns = [col for col in result_columns if col in df.columns]
    
    top_5_matches = filtered_df.nsmallest(5, 'score')[final_columns]
    return top_5_matches.to_dict('records')


# === 4. ROUTES (FLASK ENDPOINTS) ===

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    models_available = nutrition_model is not None
    return render_template('predict.html', models_available=models_available)

@app.route('/result')
def result_page():
    if 'analysis_result' not in session:
        return redirect(url_for('predict_page'))
    return render_template('result.html')

@app.route('/api/get_result', methods=['GET'])
def get_result():
    if 'analysis_result' in session:
        return jsonify(session['analysis_result'])
    return jsonify({'error': 'No data found'}), 404

@app.route('/analyze', methods=['POST'])
def analyze():
    if nutrition_model is None:
        return jsonify({'success': False, 'error': 'Hệ thống chưa sẵn sàng, model chưa được load.'}), 500

    try:
        data = request.json
        print("Received data:", data)
        
        # 1. Lấy thông tin
        name = data.get('ho_ten') or data.get('name')
        age = int(data.get('tuoi') or data.get('age'))
        weight = float(data.get('can_nang') or data.get('weight'))
        height = float(data.get('chieu_cao') or data.get('height'))
        goal = data.get('muc_tieu') or data.get('goal')
        gender_from_form = data.get('gioi_tinh') or data.get('gender') 
        activity_from_form = data.get('hoat_dong') or data.get('activity_level')

        # 2. Tự động chọn chế độ ăn
        diet_key = recommend_diet(age, gender_from_form, weight, height, goal)
        print(f"Logic đã chọn chế độ ăn: {diet_key}")

        # 3. Ánh xạ cho Model
        gender_map = {'Nam': 'Male', 'Nữ': 'Female', 'Male': 'Male', 'Female': 'Female'}
        gender_for_model = gender_map.get(gender_from_form, 'Male')
        activity_map = {
            'it_hoat_dong': 1, 'hoat_dong_nhe': 2, 'hoat_dong_trung_binh': 3,
            'hoat_dong_nang': 5, 'hoat_dong_rat_nang': 7
        }
        workout_freq = activity_map.get(activity_from_form, 3)
        exp_level = int(data.get('experience_level', 1)) 

        # 4. Tính toán
        bmi = calculate_bmi(weight, height)
        bmi_category = get_bmi_category(bmi)
        height_m = height / 100

        feature_columns = [
            'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI',
            'Workout_Frequency (days/week)', 'Experience_Level'
        ]
        input_data = pd.DataFrame(
            [[age, gender_for_model, weight, height_m, bmi, workout_freq, exp_level]],
            columns=feature_columns
        )
        
        # 5. GỌI MODEL ML
        predicted_values = nutrition_model.predict(input_data)[0]
        predicted_needs = {
            'calories': round(predicted_values[0]),
            'proteins': round(predicted_values[1]),
            'fats': round(predicted_values[2]),
            'carbs': round(predicted_values[3])
        }

        # 6. GỌI HÀM KHUYẾN NGHỊ (HÀM CŨ GÂY LỖI)
        # NOTE: Giữ lại hàm này để không làm vỡ result.html
        meal_recommendations = get_meal_recommendations(
            meal_database,
            predicted_needs,
            goal,
            diet_key
        )
      
        # 7. Đóng gói kết quả
        diet_info = DIET_RECOMMENDATIONS.get(diet_key, {})

        result = {
            'success': True,
            'user_info': {
                'name': name, 'age': age, 'gender': gender_from_form,
                'weight': weight, 'height': height, 'goal': goal
            },
            'health_metrics': {
                'bmi': round(bmi, 2),
                'bmi_category': bmi_category,
            },
            'predicted_nutrition': predicted_needs,
            'diet_info': diet_info,
            'meal_recommendations': meal_recommendations # <-- KHÔI PHỤC KEY NÀY
        }
        
        session['analysis_result'] = result
        print("Analysis complete. Saved to session.")
        
        return jsonify(result)
        
    except Exception as e:
        print("LỖI TRONG KHI PHÂN TÍCH:", str(e))
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

# === 5. CHẠY APP ===
if __name__ == '__main__':
    if nutrition_model is None or meal_database is None:
        print("="*50)
        print("KHỞI ĐỘNG THẤT BẠI. VUI LÒNG KIỂM TRA LỖI Ở TRÊN.")
        print("="*50)
    else:
        print("="*50)
        print("Hệ thống DSS sẵn sàng tại http://127.0.0.1:5000")
        print("="*50)
        app.run(debug=True, port=5000)