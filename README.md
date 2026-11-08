# Üretim Ayar Optimizasyonu

Operatör ayarlarını, ölçülen sonuçlardan öğrenerek hedeflere göre öneren Streamlit uygulaması.

- **5 model yarışır:** Gaussian Process, Random Forest, XGBoost, Lojistik Regresyon, SVM
- **Hedef bazlı optimizasyon:** nicel çıktılarda hedef aralık, nitel çıktılarda hedef kategori (Optuna TPE)
- **Etki analizi:** girdi önemi, duyarlılık, korelasyon
- **Google Sheets:** veriler, model durumu ve etki analizi tabloya yazılır
- **Çoklu çalışma:** her çalışma aynı dosyada kendi sayfalarıyla; uygulamadan oluşturulur ve seçilir

| Dosya | İçerik |
|---|---|
| `app.py` | Streamlit arayüzü |
| `engine.py` | Modeller, optimizasyon, etki analizi |
| `plots.py` | Grafikler ve sonuç tabloları |
| `store.py` | Google Sheets / yerel depolama |
| `calismalar.py` | Birden fazla çalışmanın sayfa yönetimi |
| `doe_tasarim.py` | DOE tasarımları (faktöriyel, PB, CCD, BBD, DSD, Taguchi, karışım, D-optimal) |
| `doe_analiz.py` | Minitab tarzı DOE analizi, Taguchi S/N, istenebilirlik optimizasyonu |
| `doe_plots.py` | DOE grafikleri |
| `doe_ui.py` | 🧪 DOE sekmesi ve yan panel asistanı |
| `doe_tani.py` | Mevcut kayıtlarda tasarım tanıma ve otomatik analiz |

Kurulum: `KURULUM.md`
