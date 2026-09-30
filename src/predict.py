import os
import joblib
import pandas as pd
import numpy as np

from src.data_processing import engineer_features

def load_inference_artifacts(model_name='Gradient Boosting'):
    """
    Tải mô hình và bộ tiền xử lý đã lưu từ thư mục models/
    """
    model_file_map = {
        'Random Forest'      : 'models/random_forest_model.joblib',
        'Decision Tree'      : 'models/decision_tree_model.joblib',
        'Logistic Regression': 'models/logistic_regression_model.joblib',
        'Gradient Boosting'  : 'models/gradient_boosting_model.joblib'
    }

    if model_name not in model_file_map:
        model_name = 'Gradient Boosting'
        
    model_path = model_file_map[model_name]
    preprocessor_path = 'models/preprocessor.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Chưa tìm thấy model artifacts! Vui lòng khởi chạy train.py trước.")
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    return model, preprocessor

def generate_synthesis_analysis(c):
    """
    Tạo câu phân tích kết hợp tự nhiên lý giải hành vi khách hàng bám sát 8 thuộc tính
    """
    purchases = c.get('total_purchases', 1)
    aov = c.get('avg_order_value', 200000)
    recency = c.get('days_since_last_purchase', 30)
    sat = c.get('satisfaction_score', 3)
    voucher = c.get('used_voucher', 0)
    
    parts = []
    
    # 1. Tần suất mua
    if purchases <= 3:
        parts.append(f"Mua rất ít ({purchases} lần)")
    elif purchases <= 8:
        parts.append(f"Tần suất mua khá ổn ({purchases} lần)")
    else:
        parts.append(f"Tần suất mua cao ({purchases} lần)")
        
    # 2. Giá trị đơn hàng
    if aov <= 300000:
        parts.append(f"giá trị đơn thấp ({aov:,.0f} VNĐ)")
    elif aov <= 1000000:
        parts.append(f"giá trị đơn trung bình ({aov:,.0f} VNĐ)")
    else:
        parts.append(f"giá trị đơn lớn ({aov:,.0f} VNĐ)")
        
    # 3. Recency
    if recency > 90:
        months = round(recency / 30)
        parts.append(f"đã {months} tháng ({recency} ngày) không quay lại")
    elif recency > 30:
        parts.append(f"đã {recency} ngày chưa phát sinh đơn mới")
    else:
        parts.append(f"mới mua dưới 30 ngày ({recency} ngày)")
        
    # 4. Trải nghiệm
    if sat <= 2:
        parts.append(f"và có trải nghiệm không tốt (điểm {sat}/5).")
    elif sat == 3:
        parts.append(f"điểm hài lòng mức trung bình ({sat}/5).")
    else:
        parts.append(f"điểm hài lòng tốt ({sat}/5).")
        
    if voucher == 1 and aov <= 300000:
        parts.append("Đặc biệt chỉ mua khi có mã giảm giá.")
        
    return " ".join(parts)

def predict_single_customer(customer_data, threshold=0.50, model_name='Gradient Boosting'):
    """
    Dự đoán khả năng quay lại cho 1 khách hàng cụ thể.
    """
    model, preprocessor = load_inference_artifacts(model_name)

    df_single = pd.DataFrame([customer_data])
    df_single = engineer_features(df_single)   # ← Áp dụng FE như khi training
    X_trans = preprocessor.transform(df_single)
    
    prob_return = float(model.predict_proba(X_trans)[0, 1])
    prediction = 1 if prob_return >= threshold else 0
    
    prob_return_pct = round(prob_return * 100, 1)
    churn_risk_pct = round((1 - prob_return) * 100, 1)
    
    synthesis = generate_synthesis_analysis(customer_data)
    
    if prediction == 1:
        if prob_return > 0.80:
            risk_level = "Rất Thấp"
            advice = "Khách hàng rất trung thành. Gửi mã ưu đãi Tri ân VIP 15% & đề xuất bộ sưu tập sản phẩm mới."
            result_str = f"Nhãn 1 (Quay lại) — Xác suất: {prob_return_pct}% (nguy cơ rời bỏ chỉ {churn_risk_pct}%)."
        else:
            risk_level = "Trung Bình"
            advice = "Lưu ý: Khách vẫn sẽ quay lại nhưng doanh nghiệp cần gửi voucher giảm giá để kích thích mua hàng nhanh hơn."
            result_str = f"Nhãn 1 (Quay lại) — Xác suất: {prob_return_pct}% (Lưu ý: Khách vẫn sẽ quay lại nhưng doanh nghiệp cần gửi voucher để kích thích mua hàng)."
    else:
        if prob_return < 0.20:
            risk_level = "Rất Cao (Nguy cơ Churn)"
            advice = "CẢNH BÁO: Khách có nguy cơ rời bỏ rất cao! Cần kích hoạt chiến dịch Chăm sóc đặc biệt, tặng quà tri ân + Voucher 25% ngay lập tức."
            result_str = f"Nhãn 0 (Không quay lại) — Xác suất: {prob_return_pct}% (nguy cơ rời bỏ {churn_risk_pct}%)."
        else:
            risk_level = "Cao"
            advice = "Khách đang phai nhạt tương tác. Gửi thông báo đẩy (Push notification) nhắc lịch tái mua và tặng mã hỗ trợ phí ship."
            result_str = f"Nhãn 0 (Không quay lại) — Xác suất: {prob_return_pct}% (nguy cơ rời bỏ {churn_risk_pct}%)."
            
    return {
        'probability_return': prob_return,
        'probability_return_pct': prob_return_pct,
        'churn_risk_pct': churn_risk_pct,
        'prediction_label': prediction,
        'label_formatted': "Nhãn 1 (Quay lại)" if prediction == 1 else "Nhãn 0 (Không quay lại)",
        'status_text': 'Quay lại mua sắm' if prediction == 1 else 'Không quay lại (Churn)',
        'threshold_used': threshold,
        'risk_level': risk_level,
        'synthesis_analysis': synthesis,
        'formatted_result_string': result_str,
        'recommended_action': advice,
        'model_used': model_name
    }

def predict_batch_df(df_input, threshold=0.50, model_name='Gradient Boosting'):
    """
    Dự đoán hàng loạt từ tập dữ liệu DataFrame
    """
    model, preprocessor = load_inference_artifacts(model_name)

    required_cols = [
        'age', 'gender', 'total_purchases', 'avg_order_value',
        'days_since_last_purchase', 'membership_level', 'used_voucher', 'satisfaction_score'
    ]

    missing = [c for c in required_cols if c not in df_input.columns]
    if missing:
        raise ValueError(f"File CSV thiếu các cột bắt buộc: {missing}")

    df_proc = df_input[required_cols].copy()
    df_proc = engineer_features(df_proc)       # ← Áp dụng FE như khi training
    X_trans = preprocessor.transform(df_proc)
    
    probs = model.predict_proba(X_trans)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    result_df = df_input.copy()
    result_df['xac_suat_quay_lai_%'] = np.round(probs * 100, 2)
    result_df['du_doan_quay_lai'] = preds
    result_df['trang_thai'] = np.where(preds == 1, 'Quay lại', 'Nguy cơ rời bỏ (Churn)')
    
    return result_df
