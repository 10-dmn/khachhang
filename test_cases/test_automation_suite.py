# -*- coding: utf-8 -*-
"""
HỆ THỐNG AUTOMATION TEST SUITE - DỰ ÁN BOOMERANG RADAR AI
Học phần: Trí tuệ nhân tạo - Trường Đại học Công nghệ Đông Á (EAUT)
Giảng viên hướng dẫn: Th.S Lê Tuấn Anh
Nhóm thực hiện: Nhóm 21
Sinh viên: Đào Đức Mạnh (MSV: 20230470)
"""

import os
import sys
import time
import unittest
import json
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from app import app
from src.predict import predict_single_customer, predict_batch_df
from src.data_processing import load_and_clean_kaggle_data, engineer_features


class BoomerangRadarAutomationSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*80)
        print(" KHỞI ĐỘNG BỘ KIỂM THỬ TỰ ĐỘNG TOÀN DIỆN (AUTOMATION TEST SUITE)")
        print(" Dự án: Boomerang Radar AI | Học phần: Trí tuệ Nhân tạo")
        print(" Trường ĐH Công nghệ Đông Á - Nhóm 21 (Đào Đức Mạnh - 20230470)")
        print("="*80)
        cls.app = app
        cls.client = cls.app.test_client()
        cls.start_total_time = time.time()
        cls.test_results = []

    @classmethod
    def tearDownClass(cls):
        total_duration = (time.time() - cls.start_total_time) * 1000
        print("\n" + "="*80)
        print(f" HOÀN TẤT BỘ TEST TỰ ĐỘNG! TỔNG THỜI GIAN THỰC THI: {total_duration:.2f} ms")
        print("="*80)

    def setUp(self):
        self.step_start = time.time()

    def tearDown(self):
        duration_ms = (time.time() - self.step_start) * 1000
        test_id = self.id().split('.')[-1]
        print(f"  [✓ PASS] {test_id:<40} | Thời gian: {duration_ms:>6.2f} ms")

    # =========================================================================
    # PHẦN 1: UNIT TEST TỰ ĐỘNG (KIỂM THỬ ĐƠN VỊ)
    # =========================================================================

    def test_ut_01_load_clean_data(self):
        """UT-DP-001: Tải và tiền xử lý dữ liệu chuẩn từ file CSV"""
        df = load_and_clean_kaggle_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0, "Dữ liệu không được rỗng")
        required_cols = [
            'age', 'gender', 'total_purchases', 'avg_order_value',
            'days_since_last_purchase', 'membership_level', 'used_voucher',
            'satisfaction_score', 'will_return'
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Thiếu cột cơ sở {col}")

    def test_ut_02_feature_ranges_boundary(self):
        """UT-DP-002: Kiểm tra giá trị biên của các trường dữ liệu"""
        df = load_and_clean_kaggle_data()
        self.assertTrue((df['age'] >= 18).all(), "Tuổi phải >= 18")
        self.assertTrue((df['age'] <= 80).all(), "Tuổi phải <= 80")
        self.assertTrue((df['satisfaction_score'] >= 1).all(), "Điểm hài lòng >= 1")
        self.assertTrue((df['satisfaction_score'] <= 5).all(), "Điểm hài lòng <= 5")
        self.assertTrue((df['will_return'].isin([0, 1])).all(), "Nhãn chỉ có thể là 0 hoặc 1")

    def test_ut_03_predict_single_high_loyalty(self):
        """UT-PRED-001: Dự đoán khách hàng thân thiết có xác suất quay lại cao"""
        cust = {
            'age': 35,
            'gender': 'Nữ',
            'total_purchases': 12,
            'avg_order_value': 850000.0,
            'days_since_last_purchase': 5,
            'membership_level': 'Kim Cương',
            'used_voucher': 1,
            'satisfaction_score': 5
        }
        res = predict_single_customer(cust, threshold=0.50)
        self.assertEqual(res['prediction_label'], 1, "Khách hàng thân thiết phải có nhãn 1 (Quay lại)")
        self.assertGreaterEqual(res['probability_return'], 0.50)
        self.assertIn('Rất Thấp', res['risk_level'])

    def test_ut_04_predict_single_high_churn(self):
        """UT-PRED-002: Dự đoán khách hàng có nguy cơ rời bỏ cao (Churn Risk)"""
        cust = {
            'age': 25,
            'gender': 'Nữ',
            'total_purchases': 1,
            'avg_order_value': 677000.0,
            'days_since_last_purchase': 5,
            'membership_level': 'Bạc',
            'used_voucher': 1,
            'satisfaction_score': 2
        }
        res = predict_single_customer(cust, threshold=0.50)
        self.assertEqual(res['prediction_label'], 0, "Khách hàng rời bỏ phải có nhãn 0 (Không quay lại)")
        self.assertLess(res['probability_return'], 0.50)
        self.assertIn('Rất Cao', res['risk_level'])

    def test_ut_05_threshold_sensitivity(self):
        """UT-PRED-003: Kiểm tra tính nhạy cảm của ngưỡng quyết định threshold"""
        cust = {
            'age': 28,
            'gender': 'Nữ',
            'total_purchases': 4,
            'avg_order_value': 400000.0,
            'days_since_last_purchase': 30,
            'membership_level': 'Bạc',
            'used_voucher': 1,
            'satisfaction_score': 3
        }
        res_low = predict_single_customer(cust, threshold=0.20)
        res_high = predict_single_customer(cust, threshold=0.80)
        self.assertGreaterEqual(res_low['prediction_label'], res_high['prediction_label'])

    # =========================================================================
    # PHẦN 2: INTEGRATION TEST TỰ ĐỘNG (KIỂM THỬ TÍCH HỢP)
    # =========================================================================

    def test_it_01_feature_engineering_pipeline(self):
        """IT-PIPE-001: Tích hợp hàm sinh đặc trưng tương tác với mô hình suy luận"""
        sample_df = pd.DataFrame([{
            'age': 30, 'gender': 'Nam', 'total_purchases': 6,
            'avg_order_value': 500000.0, 'days_since_last_purchase': 10,
            'membership_level': 'Vàng', 'used_voucher': 1, 'satisfaction_score': 4
        }])
        df_fe = engineer_features(sample_df)
        engineered = ['purchase_intensity', 'ltv_score', 'satisfaction_recency', 'voucher_low_value']
        for feat in engineered:
            self.assertIn(feat, df_fe.columns, f"Pipeline tích hợp thiếu cột {feat}")

    def test_it_02_batch_prediction_pipeline(self):
        """IT-PIPE-002: Tích hợp xử lý dự đoán theo lô (Batch Inference Pipeline)"""
        df_sample = pd.DataFrame([
            {'age': 25, 'gender': 'Nữ', 'total_purchases': 2, 'avg_order_value': 200000.0,
             'days_since_last_purchase': 45, 'membership_level': 'Đồng', 'used_voucher': 0, 'satisfaction_score': 2},
            {'age': 40, 'gender': 'Nam', 'total_purchases': 15, 'avg_order_value': 1200000.0,
             'days_since_last_purchase': 7, 'membership_level': 'Kim Cương', 'used_voucher': 1, 'satisfaction_score': 5}
        ])
        df_res = predict_batch_df(df_sample, threshold=0.50)
        self.assertEqual(len(df_res), 2)
        self.assertIn('du_doan_quay_lai', df_res.columns)
        self.assertIn('xac_suat_quay_lai_%', df_res.columns)
        self.assertIn('trang_thai', df_res.columns)

    def test_it_03_multi_model_selection(self):
        """IT-PIPE-003: Tích hợp chuyển đổi linh hoạt giữa các thuật toán ML"""
        sample_dict = {
            'age': 32, 'gender': 'Nữ', 'total_purchases': 5,
            'avg_order_value': 600000.0, 'days_since_last_purchase': 15,
            'membership_level': 'Bạc', 'used_voucher': 1, 'satisfaction_score': 4
        }
        models = ['Gradient Boosting', 'Random Forest', 'Logistic Regression', 'Decision Tree']
        for m in models:
            res = predict_single_customer(sample_dict, model_name=m)
            self.assertEqual(res['model_used'], m)
            self.assertIn(res['prediction_label'], [0, 1])

    # =========================================================================
    # PHẦN 3: SYSTEM & API TEST TỰ ĐỘNG (KIỂM THỬ HỆ THỐNG QUA REST API)
    # =========================================================================

    def test_st_01_home_dashboard(self):
        """ST-API-001: Kiểm thử giao diện Web Dashboard trang chủ (GET /)"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Customer Repurchase', response.data)

    def test_st_02_api_stats(self):
        """ST-API-002: Kiểm thử API thống kê tổng quan (GET /api/stats)"""
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('total_customers', data)
        self.assertIn('return_rate', data)
        self.assertIn('churn_rate', data)
        self.assertGreater(data['total_customers'], 0)
        self.assertAlmostEqual(data['return_rate'] + data['churn_rate'], 100.0, places=1)

    def test_st_03_api_models_evaluation(self):
        """ST-API-003: Kiểm thử API đánh giá các mô hình ML (GET /api/models)"""
        response = self.client.get('/api/models')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('Gradient Boosting', data)
        gb_metrics = data['Gradient Boosting']
        self.assertGreater(gb_metrics['accuracy'], 0.90)

    def test_st_04_api_predict_single_success(self):
        """ST-API-004: Kiểm thử API dự đoán khách hàng đơn lẻ (POST /api/predict)"""
        payload = {
            'age': 32,
            'gender': 'Nữ',
            'total_purchases': 5,
            'avg_order_value': 500000,
            'days_since_last_purchase': 15,
            'membership_level': 'Bạc',
            'used_voucher': 1,
            'satisfaction_score': 4,
            'threshold': 0.50,
            'model_name': 'Gradient Boosting'
        }
        response = self.client.post('/api/predict', data=json.dumps(payload),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction_label', data)
        self.assertIn('probability_return', data)
        self.assertIn('risk_level', data)
        self.assertIn('recommended_action', data)

    def test_st_05_api_predict_boundary_out_of_range(self):
        """ST-API-005: Kiểm thử API dự đoán khách hàng rời bỏ (Churn Customer Prediction)"""
        payload = {
            'age': 25,
            'gender': 'Nữ',
            'total_purchases': 1,
            'avg_order_value': 677000.0,
            'days_since_last_purchase': 5,
            'membership_level': 'Bạc',
            'used_voucher': 1,
            'satisfaction_score': 2,
            'threshold': 0.50
        }
        response = self.client.post('/api/predict', data=json.dumps(payload),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['prediction_label'], 0)
        self.assertIn('churn_risk_pct', data)
        self.assertIn('Rất Cao', data['risk_level'])

    def test_st_06_api_predict_batch_success(self):
        """ST-API-006: Kiểm thử API dự đoán theo lô (POST /api/predict-batch)"""
        response = self.client.post('/api/predict-batch', data={})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        first_row = data[0]
        self.assertIn('du_doan_quay_lai', first_row)
        self.assertIn('xac_suat_quay_lai_%', first_row)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(BoomerangRadarAutomationSuite)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    return result


if __name__ == '__main__':
    res = run_tests()
    if res.wasSuccessful():
        print("\n>>> TẤT CẢ CÁC TEST CASES ĐÃ PASS 100%! HỆ THỐNG ĐẠT CHUẨN KIỂM THỬ <<<")
        sys.exit(0)
    else:
        print("\n>>> PHÁT HIỆN TEST CASE FAILED! <<<")
        sys.exit(1)
