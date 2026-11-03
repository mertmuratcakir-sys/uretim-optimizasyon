# Üretim Ayar Optimizasyonu

Operatör ayarlarını, ölçülen sonuçlardan öğrenerek hedeflere göre öneren Streamlit uygulaması.

- **5 model yarışır:** Gaussian Process, Random Forest, XGBoost, Lojistik Regresyon, SVM
- **Hedef bazlı optimizasyon:** nicel çıktılarda hedef aralık, nitel çıktılarda hedef kategori (Optuna TPE)
- **Etki analizi:** girdi önemi, duyarlılık, korelasyon
- **Google Sheets:** veriler, model durumu ve etki analizi tabloya yazılır

| Dosya | İçerik |
|---|---|
| `app.py` | Streamlit arayüzü |
| `engine.py` | Modeller, optimizasyon, etki analizi |
| `plots.py` | Grafikler ve sonuç tabloları |
| `store.py` | Google Sheets / yerel depolama |

Kurulum: `KURULUM.md`
