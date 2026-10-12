# 🏦 Bankacılık Kredi Risk ve Temerrüt Tahmin Sistemi (End-to-End ML & FastAPI)

Bu proje, müşterilerin demografik ve finansal verilerini kullanarak kredi temerrüt (batık/default) olasılığını tahmin eden uçtan uca bir Veri Bilimi bitirme projesidir.

## 📌 Proje Adımları ve Mimari
1. **Veri Temizliği & Ön İşleme (Data Leakage Önlemli):**
   - Veri seti (`credit_risk_dataset.csv`) önce **Train (%80)** ve **Test (%20)** olarak ayrılmış, aykırı değer (outlier) temizliği eğitim seti üzerinde uygulanmıştır.
2. **Scikit-Learn Pipeline & ColumnTransformer:**
   - **Eksik Veriler:** `person_emp_length` ve `loan_int_rate` sütunlarındaki boş değerler `SimpleImputer(strategy='median')` ile dolduruldu.
   - **Çarpıklık Dönüşümü:** Sağa çarpık sayısal değişkenlere **Yeo-Johnson (`PowerTransformer`)** uygulandı.
   - **Kategorik Kodlama:** Kredi derecesi (`loan_grade`) için `OrdinalEncoder`, diğer kategorik değişkenler için `OneHotEncoder` kullanıldı.
3. **Modelleme & Eşik Değeri (Threshold) Optimizasyonu:**
   - Sınıf dengesizliği `scale_pos_weight` ile dengelenerek **XGBoost** ve **LightGBM** modelleri eğitildi.
   - Precision-Recall eğrisi üzerinden F1-Skorunu maksimize eden özel karar eşiği (**%67.8**) belirlendi.
4. **Canlıya Alma (FastAPI & HTML/CSS):**
   - Eğitilen tek parça Pipeline `.joblib` olarak kaydedildi ve **FastAPI** ile web servisine dönüştürüldü.

## 🚀 Kurulum ve Çalıştırma

Gerekli kütüphaneleri yükleyin:
```

FastAPI sunucusunu başlatın:
```bash
python main.py
```
Tarayıcınızda `[http://127.0.0.1:8000](http://127.0.0.1:8000)` adresine giderek kredi risk karar destek arayüzünü kullanabilirsiniz.

## 🖥️ Uygulama Ekran Görüntüleri
<img width="1123" height="880" alt="Ekran görüntüsü 2026-10-12 181539" src="https://github.com/user-attachments/assets/8e9e9eba-9c6d-462d-a041-ec9433d34a3b" />
<img width="1125" height="882" alt="image" src="https://github.com/user-attachments/assets/7a4076e0-0ad1-41d7-b05f-4e94fc002653" />
