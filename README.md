# 🪃 Boomerang Radar AI (Customer Repurchase & Churn Prediction System)

> **Hệ thống Trí tuệ Nhân tạo dự đoán và phân loại hành vi khách hàng dựa trên 8 chỉ số RFM+ và mô hình Gradient Boosting (Accuracy 94.23%, F1 96.56%).**  
> **Học phần:** Trí tuệ Nhân tạo — Trường Đại học Công nghệ Đông Á (EAUT)  
> **Giảng viên hướng dẫn:** Th.S Lê Tuấn Anh  
> **Sinh viên thực hiện:** Đào Đức Mạnh (MSV: 20230470 — Lớp: DCCNTT14.10.1 — Nhóm 21)

---

## 🌟 Giới thiệu dự án
**Boomerang Radar AI** hoạt động như một "đài radar thông minh" dành cho các doanh nghiệp bán lẻ và thương mại điện tử:
* **Quét và định vị:** Liên tục phân tích hành vi của khách hàng thông qua tần suất, giá trị đơn hàng, độ tươi mới giao dịch và trải nghiệm dịch vụ.
* **Cảnh báo sớm:** Phân cấp rủi ro rời bỏ thành 4 cấp độ (*Rất thấp*, *Thấp*, *Trung bình*, *Rất cao*).
* **Đề xuất hành động tự động:** Đưa ra các gợi ý marketing/CSKH tức thì để kích hoạt khách hàng quay lại ("hiệu ứng Boomerang").
* **Hiệu năng thời gian thực:** Phản hồi API suy luận dưới 50ms (thực tế đo lường: 18.5ms).

---

## 📄 Hồ sơ Báo cáo Bài tập lớn (BTL) & Slide thuyết trình

* 📘 **Bản báo cáo BTL hoàn chỉnh (Word):** [`BAO_CAO_BTL_TRI_TUE_NHAN_TAO_NHOM_21.docx`](BAO_CAO_BTL_TRI_TUE_NHAN_TAO_NHOM_21.docx) *(Đúng mẫu EAUT, Times New Roman 13pt, A4 lề 3-2-2-2cm, mục lục, bảng biểu và hình ảnh chuẩn)*
* 📕 **Bản báo cáo nộp chính thức (PDF):** [`BAO_CAO_BTL_TRI_TUE_NHAN_TAO_NHOM_21.pdf`](BAO_CAO_BTL_TRI_TUE_NHAN_TAO_NHOM_21.pdf) *(Xuất trực tiếp từ Word, sẵn sàng nộp Elearning)*
* 📊 **Slide báo cáo thuyết trình (PowerPoint 16:9):** [`SLIDE_BAO_CAO_BTL_TRI_TUE_NHAN_TAO_NHOM_21.pptx`](SLIDE_BAO_CAO_BTL_TRI_TUE_NHAN_TAO_NHOM_21.pptx) *(12 slide thiết kế hiện đại, bố cục học thuật)*
* 📝 **Phiếu đánh giá mức độ hoàn thành CV (BM.T07):** [`BM.T07_Phieu_danh_gia_Nhom_21.docx`](BM.T07_Phieu_danh_gia_Nhom_21.docx) *(Mẫu chính thức của trường, đánh giá 100% hoàn thành)*

---

## 📊 Kết quả thực nghiệm và Đánh giá mô hình

| Mô hình | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Xếp hạng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 68.47% | 0.9395 | 0.6635 | 0.7777 | 0.7906 | Baseline |
| **Decision Tree** | 74.96% | 0.9348 | 0.7511 | 0.8329 | 0.8089 | Khá |
| **Random Forest** | 91.56% | 0.9367 | 0.9637 | 0.9500 | 0.9575 | Tốt |
| **Gradient Boosting** | **94.23%** | **0.9570** | **0.9744** | **0.9656** | **0.9695** | **★ Champion** |

---

## 🚀 Cài đặt và Khởi chạy ứng dụng

### 1. Cài đặt môi trường
```powershell
# Kích hoạt môi trường ảo (nếu có)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Khởi chạy Web Dashboard
```powershell
python app.py
```
Mở trình duyệt và truy cập địa chỉ: **`http://localhost:5000`**

### 3. Thực thi Bộ kiểm thử tự động (Automation Test Suite)
```powershell
python test_cases/test_automation_suite.py
```
*(Thực thi 14 ca kiểm thử tự động trong ~1.05 giây, kết quả Đạt 100%)*

---

## 📁 Cấu trúc thư mục dự án
* `app.py`: Web server Flask và REST API endpoints (`/api/predict`, `/api/predict-batch`, `/api/models`, `/api/stats`).
* `src/data_processing.py`: Module nạp dữ liệu, chống rò rỉ dữ liệu (Anti-Data Leakage) và sinh 4 đặc trưng tương tác.
* `src/train.py`: Pipeline huấn luyện 4 mô hình học máy và đánh giá 5 chỉ số đo lường.
* `src/predict.py`: Bộ máy suy luận nhị phân, phân tầng rủi ro Churn và sinh gợi ý Marketing.
* `templates/index.html`: Giao diện Dashboard 4 thẻ điều hướng tương tác thời gian thực.
* `models/`: Chứa các mô hình đã được huấn luyện sẵn định dạng `.joblib`.
* `data/`: Bộ dữ liệu thô `customer_repurchase_data.csv`.
* `test_cases/`: Bộ kiểm thử tự động hóa và 3 tệp Excel thiết kế ca test chi tiết.
