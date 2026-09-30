import os
import sys
import joblib
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# -----------------------------------------------------------------------
# FEATURE ENGINEERING HELPERS
# -----------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo thêm 4 đặc trưng tổng hợp (Interaction Features) giúp mô hình
    học được mối quan hệ phi tuyến giữa các thuộc tính RFM+:

    1. purchase_intensity   = total_purchases / (days_since_last_purchase + 1)
       → Nhịp mua hàng: mua nhiều + gần đây = score cao.

    2. ltv_score            = total_purchases × avg_order_value / 1_000_000
       → Customer Lifetime Value đơn giản hóa (triệu đồng).

    3. satisfaction_recency = satisfaction_score / (days_since_last_purchase + 1) × 30
       → Kết hợp mức hài lòng với độ tươi của lần mua gần nhất.

    4. voucher_low_value    = used_voucher × (avg_order_value ≤ 300.000 VNĐ)
       → Phát hiện nhóm "Voucher Hunter" - chỉ mua khi có mã + AOV thấp.
    """
    df = df.copy()
    df['purchase_intensity'] = df['total_purchases'] / (df['days_since_last_purchase'] + 1)
    df['ltv_score'] = df['total_purchases'] * df['avg_order_value'] / 1_000_000
    df['satisfaction_recency'] = df['satisfaction_score'] / (df['days_since_last_purchase'] + 1) * 30
    df['voucher_low_value'] = ((df['used_voucher'] == 1) & (df['avg_order_value'] <= 300_000)).astype(int)
    return df


def load_and_clean_kaggle_data():
    """
    Tự động tìm kiếm và nạp bộ dữ liệu.
    Hỗ trợ:
    - data/customer_repurchase_data.csv  (CSV đã ánh xạ sẵn — ưu tiên)
    - data/raw/E Commerce Dataset.xlsx   (file gốc Kaggle)
    - /kaggle/input/...                  (môi trường Kaggle Kernel)
    """
    search_paths = [
        'data/customer_repurchase_data.csv',
        'data/raw/E Commerce Dataset.xlsx',
        '/kaggle/input/ecommerce-customer-churn-analysis-and-prediction/E Commerce Dataset.xlsx',
        '../input/ecommerce-customer-churn-analysis-and-prediction/E Commerce Dataset.xlsx',
    ]

    found_path = None
    for p in search_paths:
        if os.path.exists(p):
            found_path = p
            break

    if not found_path:
        raise FileNotFoundError(
            "Không tìm thấy bộ dữ liệu! Vui lòng đặt file 'E Commerce Dataset.xlsx' "
            "tại thư mục 'data/raw/' hoặc cung cấp 'data/customer_repurchase_data.csv'."
        )

    print(f"  -> Đã tìm thấy bộ dữ liệu tại: {found_path}")

    if found_path.endswith('.csv'):
        df = pd.read_csv(found_path)
        required_base = [
            'age', 'gender', 'total_purchases', 'avg_order_value',
            'days_since_last_purchase', 'membership_level',
            'used_voucher', 'satisfaction_score', 'will_return'
        ]
        if all(c in df.columns for c in required_base):
            return df[required_base]
    else:
        df_raw = pd.read_excel(found_path, sheet_name='E Comm')
        df = df_raw.copy()

        num_cols = ['Tenure', 'OrderCount', 'DaySinceLastOrder', 'CouponUsed',
                    'OrderAmountHikeFromlastYear', 'CashbackAmount', 'SatisfactionScore']
        for c in num_cols:
            if c in df.columns:
                df[c] = df[c].fillna(df[c].median())

        df['will_return'] = (df['Churn'] == 0).astype(int)
        df['gender'] = df['Gender'].replace({'Male': 'Nam', 'Female': 'Nữ'}).fillna('Nữ')
        df['age'] = np.clip(
            22 + df['Tenure'] * 0.8 + np.random.normal(0, 1.5, size=len(df)), 18, 70
        ).astype(int)
        df['total_purchases'] = np.clip(df['OrderCount'].astype(int), 1, 60)
        df['avg_order_value'] = np.round(
            np.clip(df['CashbackAmount'] * 3200 + df['OrderAmountHikeFromlastYear'] * 15000,
                    150000, 4500000), -3
        )
        df['days_since_last_purchase'] = np.clip(df['DaySinceLastOrder'].astype(int), 1, 180)

        cashback_q = df['CashbackAmount'].quantile([0.35, 0.70, 0.90])

        def map_membership(cb):
            if cb <= cashback_q[0.35]: return 'Đồng'
            elif cb <= cashback_q[0.70]: return 'Bạc'
            elif cb <= cashback_q[0.90]: return 'Vàng'
            else: return 'Kim Cương'

        df['membership_level'] = df['CashbackAmount'].apply(map_membership)
        df['used_voucher'] = (df['CouponUsed'] > 0).astype(int)
        df['satisfaction_score'] = np.clip(df['SatisfactionScore'].astype(int), 1, 5)

        feature_cols = [
            'age', 'gender', 'total_purchases', 'avg_order_value',
            'days_since_last_purchase', 'membership_level',
            'used_voucher', 'satisfaction_score', 'will_return'
        ]
        df = df[feature_cols]
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/customer_repurchase_data.csv', index=False, encoding='utf-8')

    return df


def prepare_data(test_size=0.2, random_state=42):
    """
    Quy trình Tiền xử lý Dữ liệu Chuẩn ML — Chống Data Leakage:
      1. Nạp & làm sạch dữ liệu gốc
      2. Feature Engineering (thêm 4 đặc trưng tổng hợp)
      3. Phân chia Train/Test TRƯỚC khi fit bất kỳ transformer nào
      4. Fit StandardScaler + OneHotEncoder CHỈ TRÊN TẬP TRAIN
      5. Transform X_test (không fit lại)
      6. Lưu preprocessor, feature_names, raw_feature_cols ra models/
      7. Trả về dict đầy đủ cho pipeline training
    """
    df_base = load_and_clean_kaggle_data()

    # Feature Engineering áp dụng trên toàn bộ tập trước khi split
    df = engineer_features(df_base)

    base_feature_cols = [
        'age', 'gender', 'total_purchases', 'avg_order_value',
        'days_since_last_purchase', 'membership_level',
        'used_voucher', 'satisfaction_score'
    ]
    engineered_cols = ['purchase_intensity', 'ltv_score', 'satisfaction_recency', 'voucher_low_value']
    all_feature_cols = base_feature_cols + engineered_cols
    target_col = 'will_return'

    X = df[all_feature_cols]
    y = df[target_col]

    # STEP 1: Phân chia Train/Test stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    cat_features = ['gender', 'membership_level']
    num_features = [
        'age', 'total_purchases', 'avg_order_value', 'days_since_last_purchase',
        'used_voucher', 'satisfaction_score',
        'purchase_intensity', 'ltv_score', 'satisfaction_recency', 'voucher_low_value'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ],
        remainder='passthrough'
    )

    # STEP 2: Fit CHỈ trên X_train (Anti Data Leakage)
    X_train_trans = preprocessor.fit_transform(X_train)

    # STEP 3: Transform X_test (không fit lại)
    X_test_trans = preprocessor.transform(X_test)

    ohe_categories = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
    feature_names_transformed = list(num_features) + list(ohe_categories)

    # STEP 4: Lưu artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    joblib.dump(feature_names_transformed, 'models/feature_names.joblib')
    joblib.dump(base_feature_cols, 'models/raw_feature_cols.joblib')

    print(f"  Kích thước tập Train sau xử lý: {X_train_trans.shape}")
    print(f"  Kích thước tập Test sau xử lý:  {X_test_trans.shape}")

    return {
        'X_train': X_train_trans,
        'y_train': y_train,
        'X_test': X_test_trans,
        'y_test': y_test,
        'feature_names': feature_names_transformed,
        'preprocessor': preprocessor,
        'raw_df': df_base
    }


if __name__ == '__main__':
    data_dict = prepare_data()
    print("Hoàn tất xử lý dữ liệu!")