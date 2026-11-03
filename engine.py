"""
engine.py — Hesap motoru
=========================
Yapı okuma, veri hazırlama, 5 model AutoML (GP, RF, XGBoost, Lojistik, SVM),
tahmin, hedef bazlı skor, Optuna optimizasyonu, model güveni ve etki analizi.

Arayüzden ve depolamadan bağımsızdır; tek başına test edilebilir.
"""
from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")

try:  # XGBoost opsiyonel — yoksa diğer modeller yarışır
    import xgboost as xgb
    XGB_VAR = True
except Exception:  # pragma: no cover
    XGB_VAR = False

try:  # Optuna opsiyonel — yoksa akıllı rastgele aramaya düşer
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_VAR = True
except Exception:  # pragma: no cover
    OPTUNA_VAR = False

MIN_KAYIT = 5  # Çapraz doğrulama için gereken en az kayıt (çıktı başına)

YAPI_SUTUNLAR = ["Tür", "İsim", "Tip", "Min", "Max", "Kategoriler",
                 "Hedef_Min", "Hedef_Max", "Hedef_Kategori"]

MODEL_EMOJI = {"GaussianProcess": "🟢", "RandomForest": "🟠", "XGBoost": "🟣",
               "Lojistik": "🔵", "SVM": "🟡", "Sabit": "⚪"}
MODEL_RENK = {"GaussianProcess": "#1a9e5a", "RandomForest": "#e8710a",
              "XGBoost": "#8e44ad", "Lojistik": "#2a6fdb", "SVM": "#c9a400",
              "Sabit": "#888888"}


# ════════════════════════════════════════════════════════════════
# Yardımcılar
# ════════════════════════════════════════════════════════════════
def sayi(v):
    """'12,5' / '1.234,5' / 12.5 / '' → float veya None."""
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return None if (isinstance(v, float) and np.isnan(v)) else float(v)
    s = str(v).strip().replace(" ", "")
    if s == "":
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def hucre(v):
    """Sheets'e yazılabilir sade Python değeri."""
    if v is None:
        return ""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return "" if np.isnan(f) else round(f, 6)
    return v


def kategoriler(s):
    return [k.strip() for k in str(s or "").split(",") if k.strip()]


# ════════════════════════════════════════════════════════════════
# Yapı
# ════════════════════════════════════════════════════════════════
def yapi_oku(rows: list[list]) -> dict:
    """Yapilanma sekmesindeki satırları yapı sözlüğüne çevirir."""
    yapi = {"girdiler": [], "ciktilar": []}
    if not rows or len(rows) < 2:
        return yapi
    baslik = [str(h).strip() for h in rows[0]]
    for r in rows[1:]:
        d = {baslik[i]: (r[i] if i < len(r) else "") for i in range(len(baslik))}
        tur = str(d.get("Tür", "")).strip().upper()
        isim = str(d.get("İsim", "")).strip()
        tip = str(d.get("Tip", "")).strip().lower()
        if not isim:
            continue
        tip = "nitel" if tip.startswith("nitel") else "nicel"
        if tur in ("GİRDİ", "GIRDI"):
            g = {"isim": isim, "tip": tip,
                 "min": sayi(d.get("Min")), "max": sayi(d.get("Max")),
                 "kategoriler": kategoriler(d.get("Kategoriler")) if tip == "nitel" else None}
            if tip == "nicel":
                g["min"] = 0.0 if g["min"] is None else g["min"]
                g["max"] = 100.0 if g["max"] is None else g["max"]
                if g["max"] < g["min"]:
                    g["min"], g["max"] = g["max"], g["min"]
            yapi["girdiler"].append(g)
        elif tur in ("ÇIKTI", "CIKTI", "ÇİKTI"):
            c = {"isim": isim, "tip": tip,
                 "hedef_min": sayi(d.get("Hedef_Min")),
                 "hedef_max": sayi(d.get("Hedef_Max")),
                 "hedef_kat": str(d.get("Hedef_Kategori", "") or "").strip() or None,
                 "kategoriler": kategoriler(d.get("Kategoriler")) if tip == "nitel" else None}
            if (c["hedef_min"] is not None and c["hedef_max"] is not None
                    and c["hedef_max"] < c["hedef_min"]):
                c["hedef_min"], c["hedef_max"] = c["hedef_max"], c["hedef_min"]
            yapi["ciktilar"].append(c)
    return yapi


def yapi_satirlari(yapi: dict) -> list[list]:
    rows = [YAPI_SUTUNLAR]
    for g in yapi["girdiler"]:
        rows.append(["GİRDİ", g["isim"], g["tip"],
                     hucre(g.get("min")) if g["tip"] == "nicel" else "",
                     hucre(g.get("max")) if g["tip"] == "nicel" else "",
                     ",".join(g.get("kategoriler") or []), "", "", ""])
    for c in yapi["ciktilar"]:
        rows.append(["ÇIKTI", c["isim"], c["tip"], "", "",
                     ",".join(c.get("kategoriler") or []),
                     hucre(c.get("hedef_min")), hucre(c.get("hedef_max")),
                     c.get("hedef_kat") or ""])
    return rows


def yapi_kontrol(yapi: dict) -> list[str]:
    """Yapıdaki hataları/uyarıları döndürür."""
    hatalar = []
    if not yapi["girdiler"]:
        hatalar.append("En az bir GİRDİ tanımlayın.")
    if not yapi["ciktilar"]:
        hatalar.append("En az bir ÇIKTI tanımlayın.")
    isimler = [x["isim"] for x in yapi["girdiler"] + yapi["ciktilar"]]
    tekrar = {i for i in isimler if isimler.count(i) > 1}
    if tekrar:
        hatalar.append(f"Aynı isim birden fazla kez kullanılmış: {', '.join(sorted(tekrar))}")
    for g in yapi["girdiler"]:
        if g["tip"] == "nitel" and not g["kategoriler"]:
            hatalar.append(f"'{g['isim']}' nitel girdi ama Kategoriler boş.")
    for c in yapi["ciktilar"]:
        if c["tip"] == "nicel" and (c["hedef_min"] is None or c["hedef_max"] is None):
            hatalar.append(f"'{c['isim']}' için Hedef_Min / Hedef_Max girin.")
        if c["tip"] == "nitel":
            if not c["kategoriler"]:
                hatalar.append(f"'{c['isim']}' nitel çıktı ama Kategoriler boş.")
            elif c["hedef_kat"] and c["hedef_kat"] not in c["kategoriler"]:
                hatalar.append(f"'{c['isim']}' hedef kategorisi '{c['hedef_kat']}' "
                               f"kategori listesinde yok.")
            elif not c["hedef_kat"]:
                hatalar.append(f"'{c['isim']}' için Hedef_Kategori seçin.")
    return hatalar


# ════════════════════════════════════════════════════════════════
# Veri
# ════════════════════════════════════════════════════════════════
def veri_oku(rows: list[list], yapi: dict) -> pd.DataFrame:
    """Veriler sekmesi → DataFrame. Nicel sütunlar float, nitel sütunlar str/None."""
    sutunlar = [x["isim"] for x in yapi["girdiler"] + yapi["ciktilar"]]
    if not rows or len(rows) < 2:
        return pd.DataFrame(columns=sutunlar)
    baslik = [str(h).strip() for h in rows[0]]
    kayitlar = []
    for r in rows[1:]:
        d = {baslik[i]: (r[i] if i < len(r) else "") for i in range(len(baslik))}
        if all(str(d.get(s, "")).strip() == "" for s in sutunlar):
            continue
        kayitlar.append(d)
    df = pd.DataFrame(kayitlar)
    for s in sutunlar:
        if s not in df.columns:
            df[s] = None
    tipler = {x["isim"]: x["tip"] for x in yapi["girdiler"] + yapi["ciktilar"]}
    for s in sutunlar:
        if tipler[s] == "nicel":
            df[s] = [sayi(v) for v in df[s]]
            df[s] = pd.to_numeric(df[s], errors="coerce")
        else:
            df[s] = [(str(v).strip() if str(v).strip() != "" else None)
                     if v is not None else None for v in df[s]]
    return df[sutunlar].reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# Kodlayıcı — nicel: standartlaştırma, nitel: one-hot
# ════════════════════════════════════════════════════════════════
class Kodlayici:
    def fit(self, df: pd.DataFrame, yapi: dict):
        self.girdiler = yapi["girdiler"]
        self.ort, self.std, self.kat, self.gruplar = {}, {}, {}, {}
        sutun = 0
        for g in self.girdiler:
            if g["tip"] == "nicel":
                v = pd.to_numeric(df[g["isim"]], errors="coerce").dropna()
                self.ort[g["isim"]] = float(v.mean()) if len(v) else 0.0
                s = float(v.std()) if len(v) > 1 else 0.0
                self.std[g["isim"]] = s if s > 1e-9 else 1.0
                self.gruplar[g["isim"]] = [sutun]
                sutun += 1
            else:
                kat = list(g.get("kategoriler") or [])
                for k in df[g["isim"]].dropna().unique():
                    if k not in kat:
                        kat.append(k)
                self.kat[g["isim"]] = kat
                self.gruplar[g["isim"]] = list(range(sutun, sutun + len(kat)))
                sutun += len(kat)
        self.boyut = sutun
        return self

    def donustur(self, kayitlar) -> np.ndarray:
        """kayitlar: DataFrame veya dict listesi."""
        if isinstance(kayitlar, pd.DataFrame):
            kayitlar = kayitlar.to_dict("records")
        X = np.zeros((len(kayitlar), self.boyut))
        for i, r in enumerate(kayitlar):
            for g in self.girdiler:
                kol = self.gruplar[g["isim"]]
                if g["tip"] == "nicel":
                    v = sayi(r.get(g["isim"]))
                    v = self.ort[g["isim"]] if v is None else v
                    X[i, kol[0]] = (v - self.ort[g["isim"]]) / self.std[g["isim"]]
                else:
                    v = r.get(g["isim"])
                    if v in self.kat[g["isim"]]:
                        X[i, kol[self.kat[g["isim"]].index(v)]] = 1.0
        return X

    def referans(self, df: pd.DataFrame) -> dict:
        """Duyarlılık analizi için referans nokta: nicel ortalama, nitel en sık."""
        ref = {}
        for g in self.girdiler:
            if g["tip"] == "nicel":
                ref[g["isim"]] = self.ort[g["isim"]]
            else:
                mod = df[g["isim"]].dropna().mode()
                ref[g["isim"]] = mod.iloc[0] if len(mod) else self.kat[g["isim"]][0]
        return ref


# ════════════════════════════════════════════════════════════════
# Model adayları
# ════════════════════════════════════════════════════════════════
def _regresyon_adaylari():
    adaylar = {
        "GaussianProcess": lambda: GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(),
            n_restarts_optimizer=2, normalize_y=True, random_state=42),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=1, random_state=42, n_jobs=-1),
        "SVM": lambda: SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale"),
    }
    if XGB_VAR:
        adaylar["XGBoost"] = lambda: xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
            random_state=42, verbosity=0, n_jobs=-1)
    return adaylar


def _siniflandirma_adaylari(binary: bool, n_sinif: int):
    adaylar = {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1),
        "SVM": lambda: SVC(kernel="rbf", C=10, gamma="scale",
                           probability=True, random_state=42),
    }
    if XGB_VAR:
        # Sınıf sayısını XGBoost kendisi algılar (ikili / çok sınıflı)
        adaylar["XGBoost"] = lambda: xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
            random_state=42, verbosity=0, n_jobs=-1)
    if binary:
        adaylar["Lojistik"] = lambda: LogisticRegression(max_iter=2000, C=1.0)
    return adaylar


class SabitSiniflandirici:
    """Veride tek sınıf varsa kullanılır."""
    def __init__(self, n_sinif=1):
        self.classes_ = np.arange(n_sinif)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])

    def predict_proba(self, X):
        return np.ones((len(X), 1))


# ════════════════════════════════════════════════════════════════
# AutoML eğitimi
# ════════════════════════════════════════════════════════════════
def egit(yapi: dict, df: pd.DataFrame, log=print) -> dict | None:
    girdi_isim = [g["isim"] for g in yapi["girdiler"]]
    # Tüm girdileri dolu satırlar
    tam = df.dropna(subset=girdi_isim).reset_index(drop=True)
    if len(tam) < MIN_KAYIT:
        log(f"⚠️ Girdileri tam dolu kayıt sayısı {len(tam)}. "
            f"Model için en az {MIN_KAYIT} gerekli.")
        return None

    kod = Kodlayici().fit(tam, yapi)
    X_tum = kod.donustur(tam)
    modeller, yaris = {}, {}

    for c in yapi["ciktilar"]:
        i = c["isim"]
        maske = tam[i].notna().values
        n = int(maske.sum())
        if n < MIN_KAYIT:
            log(f"  ⏭ {i}: {n} kayıt var, en az {MIN_KAYIT} gerekli — atlandı.")
            continue
        X = X_tum[maske]
        skorlar = {}

        if c["tip"] == "nicel":
            y = tam.loc[maske, i].astype(float).values
            cv = KFold(n_splits=min(5, n), shuffle=True, random_state=42)
            for ad, yap in _regresyon_adaylari().items():
                try:
                    oof = cross_val_predict(yap(), X, y, cv=cv)
                    skorlar[ad] = float(r2_score(y, oof))
                except Exception as e:
                    log(f"     {ad} çalışmadı: {str(e)[:80]}")
            if not skorlar:
                continue
            kaz = max(skorlar, key=skorlar.get)
            m = _regresyon_adaylari()[kaz]().fit(X, y)
            modeller[i] = {"tip": "nicel", "model": m, "kazanan": kaz,
                           "maske": maske, "y": y}
            yaris[i] = {"tip": "Nicel", "metrik": "R²", "skorlar": skorlar,
                        "kazanan": kaz, "n": n}
            log(f"  {i} [Nicel, {n} kayıt]: " + "  ".join(
                f"{MODEL_EMOJI[k]}{k[:3]}={v:+.2f}" for k, v in skorlar.items())
                + f"  → 🏆 {kaz}")
        else:
            y_raw = tam.loc[maske, i].astype(str).values
            siniflar = sorted(set(y_raw))
            y = np.array([siniflar.index(v) for v in y_raw])
            n_sinif = len(siniflar)
            if n_sinif == 1:
                m = SabitSiniflandirici().fit(X, y)
                modeller[i] = {"tip": "nitel", "model": m, "kazanan": "Sabit",
                               "siniflar": siniflar, "maske": maske, "y": y}
                yaris[i] = {"tip": "Nitel (tek sınıf)", "metrik": "Doğruluk",
                            "skorlar": {}, "kazanan": "Sabit", "n": n}
                log(f"  {i}: veride tek sınıf ('{siniflar[0]}') var — "
                    f"farklı sonuçlar girildikçe öğrenecek.")
                continue
            binary = n_sinif == 2
            min_sinif = int(np.bincount(y).min())
            if min_sinif >= 2:
                cv = StratifiedKFold(n_splits=min(5, min_sinif), shuffle=True,
                                     random_state=42)
            else:
                cv = KFold(n_splits=min(3, n), shuffle=True, random_state=42)
            for ad, yap in _siniflandirma_adaylari(binary, n_sinif).items():
                try:
                    oof = cross_val_predict(yap(), X, y, cv=cv)
                    skorlar[ad] = float(accuracy_score(y, oof))
                except Exception as e:
                    log(f"     {ad} çalışmadı: {str(e)[:80]}")
            if not skorlar:
                continue
            # Eşitlikte basit modeli tercih et
            oncelik = ["Lojistik", "RandomForest", "XGBoost", "SVM"]
            kaz = max(skorlar, key=lambda k: (skorlar[k], -oncelik.index(k)))
            try:
                m = _siniflandirma_adaylari(binary, n_sinif)[kaz]().fit(X, y)
            except Exception:
                kaz = "RandomForest"
                m = _siniflandirma_adaylari(binary, n_sinif)[kaz]().fit(X, y)
            modeller[i] = {"tip": "nitel", "model": m, "kazanan": kaz,
                           "siniflar": siniflar, "maske": maske, "y": y}
            yaris[i] = {"tip": "Nitel (ikili)" if binary else f"Nitel ({n_sinif} sınıf)",
                        "metrik": "Doğruluk", "skorlar": skorlar,
                        "kazanan": kaz, "n": n}
            log(f"  {i} [{yaris[i]['tip']}, {n} kayıt]: " + "  ".join(
                f"{MODEL_EMOJI[k]}{k[:3]}={v:.2f}" for k, v in skorlar.items())
                + f"  → 🏆 {kaz}")

    if not modeller:
        log("⚠️ Hiçbir çıktı için model kurulamadı.")
        return None
    return {"kod": kod, "modeller": modeller, "yaris": yaris, "X": X_tum,
            "df": tam, "n": len(tam),
            "zaman": datetime.now().strftime("%d.%m.%Y %H:%M")}


# ════════════════════════════════════════════════════════════════
# Tahmin & skor
# ════════════════════════════════════════════════════════════════
def tahmin_toplu(model: dict, kayitlar: list[dict]) -> list[dict]:
    X = model["kod"].donustur(kayitlar)
    sonuclar = [dict() for _ in range(len(kayitlar))]
    for i, m in model["modeller"].items():
        if m["tip"] == "nicel":
            if m["kazanan"] == "GaussianProcess":
                p, s = m["model"].predict(X, return_std=True)
            else:
                p, s = m["model"].predict(X), [None] * len(X)
            for k in range(len(X)):
                sonuclar[k][i] = {"tip": "nicel", "deger": float(p[k]),
                                  "std": None if s[k] is None else float(s[k]),
                                  "model": m["kazanan"]}
        else:
            prob = m["model"].predict_proba(X)
            sinif_idx = list(getattr(m["model"], "classes_", range(prob.shape[1])))
            for k in range(len(X)):
                olas = {s: 0.0 for s in m["siniflar"]}
                for j, ci in enumerate(sinif_idx):
                    olas[m["siniflar"][int(ci)]] = float(prob[k, j])
                en = max(olas, key=olas.get)
                sonuclar[k][i] = {"tip": "nitel", "deger": en, "olas": olas,
                                  "model": m["kazanan"]}
    return sonuclar


def tahmin(model: dict, girdi: dict) -> dict:
    return tahmin_toplu(model, [girdi])[0]


def skor(yapi: dict, sonuc: dict) -> float:
    """
    Hedef bazlı skor (yüksek = iyi). Her çıktı en fazla 100 puan.
    Nicel: hedef aralık içinde 90–100 (merkeze yakın daha yüksek),
           dışında 90'dan başlayıp uzaklıkla hızla düşer.
    Nitel: hedef kategorinin olasılığı × 100.
    """
    if not sonuc:
        return 0.0
    toplam, adet = 0.0, 0
    for c in yapi["ciktilar"]:
        r = sonuc.get(c["isim"])
        if r is None:
            continue
        if c["tip"] == "nicel":
            lo, hi = c.get("hedef_min"), c.get("hedef_max")
            if lo is None or hi is None:
                continue
            v = r["deger"]
            yari = max((hi - lo) / 2, 1e-6)
            orta = (lo + hi) / 2
            if lo <= v <= hi:
                p = 100 - 10 * abs(v - orta) / yari
            else:
                d = min(abs(v - lo), abs(v - hi)) / yari
                p = 90 - 60 * d
            toplam += p
        else:
            hk = c.get("hedef_kat")
            if not hk:
                continue
            toplam += 100 * r.get("olas", {}).get(hk, 0.0)
        adet += 1
    return toplam / adet if adet else 0.0


def hedef_durum(c: dict, r: dict) -> bool | None:
    if r is None:
        return None
    if c["tip"] == "nicel":
        lo, hi = c.get("hedef_min"), c.get("hedef_max")
        if lo is None or hi is None:
            return None
        return lo <= r["deger"] <= hi
    return r["deger"] == c.get("hedef_kat")


# ════════════════════════════════════════════════════════════════
# Optimizasyon
# ════════════════════════════════════════════════════════════════
def optimize(model: dict, yapi: dict, n_deneme: int = 150,
             sabit: dict | None = None, tohum: int | None = None):
    """
    Hedef skorunu maksimize eden girdi ayarını arar.
    Optuna (TPE) varsa onu, yoksa toplu rastgele aramayı kullanır.
    sabit: {'girdi_ismi': değer} — bu girdiler değiştirilmez.
    Dönüş: (en_iyi_girdi, tahmin, skor, gecmis_skorlar, onem, yontem)
    """
    sabit = sabit or {}
    serbest = [g for g in yapi["girdiler"] if g["isim"] not in sabit]

    def aday_kur(trial=None, rng=None):
        a = dict(sabit)
        for g in serbest:
            if g["tip"] == "nicel":
                a[g["isim"]] = (trial.suggest_float(g["isim"], g["min"], g["max"])
                                if trial else float(rng.uniform(g["min"], g["max"])))
            else:
                kat = g["kategoriler"] or model["kod"].kat[g["isim"]]
                a[g["isim"]] = (trial.suggest_categorical(g["isim"], kat)
                                if trial else kat[int(rng.integers(len(kat)))])
        return a

    if not serbest:
        t = tahmin(model, dict(sabit))
        return dict(sabit), t, skor(yapi, t), [skor(yapi, t)], {}, "Sabit"

    if OPTUNA_VAR:
        def amac(trial):
            return skor(yapi, tahmin(model, aday_kur(trial=trial)))
        calisma = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=min(20, max(5, n_deneme // 5)), seed=tohum))
        calisma.optimize(amac, n_trials=n_deneme, show_progress_bar=False)
        en_iyi = dict(sabit)
        en_iyi.update(calisma.best_params)
        gecmis = [t.value for t in calisma.trials if t.value is not None]
        try:
            onem = optuna.importance.get_param_importances(calisma) if len(serbest) > 1 else {}
        except Exception:
            onem = {}
        yontem = "Optuna TPE"
    else:
        rng = np.random.default_rng(tohum)
        adaylar = [aday_kur(rng=rng) for _ in range(n_deneme * 5)]
        tahminler = tahmin_toplu(model, adaylar)
        gecmis = [skor(yapi, t) for t in tahminler]
        en_iyi = adaylar[int(np.argmax(gecmis))]
        onem = {}
        yontem = "Rastgele arama"

    # Nicel değerleri okunur hale getir
    for g in yapi["girdiler"]:
        if g["tip"] == "nicel" and g["isim"] not in sabit:
            aralik = g["max"] - g["min"]
            basamak = 3 if aralik < 1 else 2 if aralik < 10 else 1
            en_iyi[g["isim"]] = round(float(en_iyi[g["isim"]]), basamak)
    t = tahmin(model, en_iyi)
    return en_iyi, t, skor(yapi, t), gecmis, onem, yontem


# ════════════════════════════════════════════════════════════════
# Model güveni
# ════════════════════════════════════════════════════════════════
def guven(model: dict, yapi: dict) -> dict:
    n = model["n"]
    if n < 10:
        durum = "🔴 Yetersiz veri"
    elif n < 30:
        durum = "🟡 Öğreniyor"
    elif n < 60:
        durum = "🟠 Gelişiyor"
    else:
        durum = "🟢 Güvenilir"
    detay = {}
    for i, y in model["yaris"].items():
        en = y["skorlar"].get(y["kazanan"]) if y["skorlar"] else None
        if en is None:
            yorum = "—"
        elif y["metrik"] == "R²":
            yorum = ("🟢 Çok iyi" if en > 0.9 else "🟢 İyi" if en > 0.7
                     else "🟡 Orta" if en > 0.5 else "🔴 Zayıf")
        else:
            yorum = ("🟢 İyi" if en > 0.85 else "🟡 Orta" if en > 0.65 else "🔴 Zayıf")
        detay[i] = {"skor": en, "yorum": yorum}
    skorlar = [d["skor"] for d in detay.values() if d["skor"] is not None]
    return {"n": n, "durum": durum, "detay": detay,
            "ortalama": float(np.mean(skorlar)) if skorlar else None,
            "zaman": model["zaman"]}


def durum_satirlari(model: dict, yapi: dict, gv: dict) -> list[list]:
    adlar = ["GaussianProcess", "RandomForest", "XGBoost", "Lojistik", "SVM"]
    rows = [["MODEL DURUMU — 5 Model AutoML"],
            ["Son güncelleme", gv["zaman"]],
            ["Eğitimde kullanılan kayıt", gv["n"]],
            ["Durum", gv["durum"]],
            [],
            ["HEDEFLER"],
            ["Çıktı", "Tip", "Hedef"]]
    for c in yapi["ciktilar"]:
        hedef = (f"{c['hedef_min']} – {c['hedef_max']}" if c["tip"] == "nicel"
                 else f"'{c.get('hedef_kat')}'")
        rows.append([c["isim"], c["tip"], hedef])
    rows += [[],
             ["YARIŞ SONUÇLARI (çapraz doğrulama — görmediği veride başarı)"],
             ["Çıktı", "Tip", "Metrik"] + adlar + ["Kazanan", "Değerlendirme", "Kayıt"]]
    for i, y in model["yaris"].items():
        rows.append([i, y["tip"], y["metrik"]]
                    + [round(y["skorlar"][a], 3) if a in y["skorlar"] else "—" for a in adlar]
                    + [f"{MODEL_EMOJI.get(y['kazanan'], '')} {y['kazanan']}",
                       gv["detay"][i]["yorum"], y["n"]])
    rows += [[],
             ["NASIL OKUNUR?"],
             ["R² (nicel)", "1 = kusursuz, 0.7 üstü iyi, 0.5 altı zayıf"],
             ["Doğruluk (nitel)", "Doğru tahmin edilen oran, 0.85 üstü iyi"],
             ["Not", "Skorlar modelin daha önce görmediği kayıtlar üzerinden ölçülür"]]
    return rows


# ════════════════════════════════════════════════════════════════
# Etki analizi
# ════════════════════════════════════════════════════════════════
def _model_skoru(m, X, y):
    if m["tip"] == "nicel":
        return r2_score(y, m["model"].predict(X))
    return accuracy_score(y, m["model"].predict(X))


def onem_hesapla(model: dict, tekrar: int = 8, tohum: int = 42) -> dict:
    """
    Girdi grubu bazlı permütasyon önemi: bir girdinin değerleri karıştırılınca
    kazanan modelin başarısı ne kadar düşüyor? (Nitel girdinin tüm one-hot
    sütunları birlikte karıştırılır.) Değerler % pay olarak döner.
    """
    rng = np.random.default_rng(tohum)
    kod = model["kod"]
    sonuc = {}
    for i, m in model["modeller"].items():
        if m["kazanan"] == "Sabit":
            continue
        X = model["X"][m["maske"]]
        y = m["y"]
        taban = _model_skoru(m, X, y)
        dusus = {}
        for g in kod.girdiler:
            kol = kod.gruplar[g["isim"]]
            d = []
            for _ in range(tekrar):
                Xp = X.copy()
                Xp[:, kol] = X[rng.permutation(len(X))][:, kol]
                d.append(taban - _model_skoru(m, Xp, y))
            dusus[g["isim"]] = max(0.0, float(np.mean(d)))
        toplam = sum(dusus.values())
        sonuc[i] = {k: (100 * v / toplam if toplam > 0 else 0.0)
                    for k, v in sorted(dusus.items(), key=lambda kv: -kv[1])}
    return sonuc


def duyarlilik_hesapla(model: dict, yapi: dict, n_nokta: int = 30) -> dict:
    """
    Her nicel girdi min→max taranır, diğerleri referans noktada sabit tutulur.
    Nicel çıktıda tahmin değeri, nitel çıktıda hedef kategori olasılığı izlenir.
    """
    ref = model["kod"].referans(model["df"])
    sonuc = {}
    for c in yapi["ciktilar"]:
        if c["isim"] not in model["modeller"]:
            continue
        sonuc[c["isim"]] = {}
        for g in yapi["girdiler"]:
            if g["tip"] != "nicel":
                continue
            xs = np.linspace(g["min"], g["max"], n_nokta)
            adaylar = []
            for x in xs:
                a = dict(ref)
                a[g["isim"]] = float(x)
                adaylar.append(a)
            tl = tahmin_toplu(model, adaylar)
            r = [t[c["isim"]] for t in tl]
            if c["tip"] == "nicel":
                ys = [t["deger"] for t in r]
                sd = [t["std"] for t in r]
            else:
                hk = c.get("hedef_kat")
                ys = [t["olas"].get(hk, 0.0) for t in r]
                sd = [None] * len(r)
            sonuc[c["isim"]][g["isim"]] = {"x": xs.tolist(), "y": ys, "std": sd}
    return sonuc


def korelasyon_hesapla(model: dict, yapi: dict) -> pd.DataFrame | None:
    """
    Nicel girdiler × çıktılar arası Spearman korelasyonu.
    Nitel çıktılar 'hedef kategoride mi? (1/0)' olarak alınır.
    """
    df = model["df"]
    g_is = [g["isim"] for g in yapi["girdiler"] if g["tip"] == "nicel"]
    tablo = pd.DataFrame(index=g_is)
    for c in yapi["ciktilar"]:
        if c["tip"] == "nicel":
            hedef = df[c["isim"]]
            ad = c["isim"]
        else:
            if not c.get("hedef_kat"):
                continue
            hedef = df[c["isim"]].map(
                lambda v, h=c["hedef_kat"]: np.nan if v is None else float(v == h))
            ad = f"{c['isim']} = {c['hedef_kat']}"
        deger = []
        for g in g_is:
            ikili = pd.concat([df[g], hedef], axis=1).dropna()
            if len(ikili) < 4 or ikili.iloc[:, 0].nunique() < 2 or ikili.iloc[:, 1].nunique() < 2:
                deger.append(np.nan)
            else:
                deger.append(ikili.iloc[:, 0].corr(ikili.iloc[:, 1], method="spearman"))
        tablo[ad] = deger
    if tablo.empty or tablo.shape[1] == 0:
        return None
    return tablo


def etki_satirlari(onem: dict, duy: dict, kor, yapi: dict, model: dict) -> list[list]:
    rows = [["ETKİ ANALİZİ"],
            ["Son güncelleme", datetime.now().strftime("%d.%m.%Y %H:%M")],
            [],
            ["1. GİRDİ ÖNEMİ — çıktıyı en çok hangi girdi belirliyor? (toplam %100)"],
            ["Çıktı", "Kazanan model", "Sıra", "Girdi", "Pay %", "Yorum"]]
    for i, d in onem.items():
        for s, (g, p) in enumerate(d.items(), 1):
            yorum = "⭐ Belirleyici" if p >= 30 else "✅ Etkili" if p >= 10 else "○ Zayıf etki"
            rows.append([i if s == 1 else "",
                         model["modeller"][i]["kazanan"] if s == 1 else "",
                         s, g, round(p, 1), yorum])
    rows += [[], ["2. DUYARLILIK — girdi artınca ne oluyor? (diğer girdiler ortalamada)"],
             ["Çıktı", "Girdi", "Yön", "En iyi girdi değeri", "Çıktı aralığı"]]
    for c in yapi["ciktilar"]:
        if c["isim"] not in duy:
            continue
        for g, d in duy[c["isim"]].items():
            x, y = np.array(d["x"]), np.array(d["y"])
            fark = y[-1] - y[0]
            olcek = max(abs(y).max(), 1e-9)
            yon = ("↑ Artırıyor" if fark > 0.03 * olcek else
                   "↓ Azaltıyor" if fark < -0.03 * olcek else "→ Belirgin etki yok")
            if c["tip"] == "nicel":
                sk = [skor({"ciktilar": [c]}, {c["isim"]: {"deger": v}}) for v in y]
                en = x[int(np.argmax(sk))]
                aralik = f"{y.min():.2f} – {y.max():.2f}"
            else:
                en = x[int(np.argmax(y))]
                aralik = f"'{c['hedef_kat']}' olasılığı %{100*y.min():.0f} – %{100*y.max():.0f}"
            rows.append([c["isim"], g, yon, round(float(en), 3), aralik])
    if kor is not None:
        rows += [[], ["3. KORELASYON (Spearman) — +1 birlikte artar, −1 ters yönde"],
                 ["Girdi"] + list(kor.columns)]
        for g, r in kor.iterrows():
            rows.append([g] + ["—" if pd.isna(v) else round(float(v), 2) for v in r.values])
        rows.append(["Not: Nitel çıktılar 'hedef kategoride mi (1/0)' olarak hesaplanır."])
    return rows
