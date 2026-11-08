"""
doe_analiz.py — DOE analizi (Minitab çıktı düzeninde)
======================================================
- Kodlanmış katsayılar (Etki, Katsayı, SE, T, P, VIF)
- Model özeti (S, R², R²(düz), R²(tah))
- Varyans analizi (Adj SS; Model / Bloklar / Doğrusal / Kare / 2'li Etkileşim /
  Eğrilik / Hata / Uyum eksikliği / Saf hata / Toplam)
- Kodlanmamış birimlerde regresyon denklemi
- Karışık (aliased) terimler, Lenth PSE (doymuş tasarımlar)
- Hiyerarşik geri eleme, olağandışı gözlemler
- Taguchi S/N analizi, karışım (Scheffé) modelleri
- Çok yanıtlı istenebilirlik (desirability) optimizasyonu
- Tasarım uygunluk mesajları
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

import doe_tasarim as T

KTS = "__CtPt__"
BLK = "__Blok__"

GRUP_ADLARI = {1: "Doğrusal", 2: "2'li Etkileşim", 3: "3'lü Etkileşim", 4: "4'lü Etkileşim",
               5: "5'li Etkileşim"}


# ════════════════════════════════════════════════════════════════
# Faktör kodlaması
# ════════════════════════════════════════════════════════════════
def _aile(meta):
    return T.AILE.get(meta["tip"], "yuzey")


def _kategorik(meta, f):
    return f["tip"] == "nitel" or _aile(meta) in ("genel", "taguchi")


def _seviyeler(meta, f):
    if f["tip"] == "nitel":
        return list(f["seviyeler"])
    return list(f.get("seviyeler_sayi") or [])


def _ayni(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return str(a) == str(b)


def _faktor(meta, ad):
    return next(f for f in meta["faktorler"] if f["isim"] == ad)


def _kat_sutunlari(meta, f, deger):
    """Etki kodlaması: 2 seviye → ±1; m seviye → m-1 sütun (1, 0, -1)."""
    L = _seviyeler(meta, f)
    m = len(L)
    idx = next((i for i, l in enumerate(L) if _ayni(deger, l)), None)
    if idx is None:
        return None
    if m == 2:
        return [-1.0 if idx == 0 else 1.0]
    out = []
    for i in range(m - 1):
        out.append(1.0 if idx == i else (-1.0 if idx == m - 1 else 0.0))
    return out


def seviye_yaz(v):
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{float(v):g}"
    return str(v)


def _kat_etiketleri(meta, f):
    L = [seviye_yaz(v) for v in _seviyeler(meta, f)]
    if len(L) == 2 and f["tip"] == "nitel" and _aile(meta) == "faktoriyel":
        return [f["isim"]]
    if len(L) == 2:
        return [f"{f['isim']} {L[1]}"]
    return [f"{f['isim']} {L[i]}" for i in range(len(L) - 1)]


def _karisim_z(meta, a):
    """Gerçek miktarlar → sözde bileşenler (toplamı tam 1 olacak şekilde normalize)."""
    z = {}
    for f in meta["faktorler"]:
        x = _sayi(a.get(f["isim"]))
        if x is None:
            return {}
        z[f["isim"]] = max(0.0, (x - f.get("kar_alt", 0.0)) / meta["kalan"])
    t = sum(z.values())
    return {k: v / t for k, v in z.items()} if t > 0 else {}


def _kodla(meta, f, deger):
    if deger is None or deger == "":
        return None
    try:
        x = float(deger)
    except (TypeError, ValueError):
        return None
    if meta["tip"] == "karisim":
        return (x - f.get("kar_alt", 0.0)) / meta["kalan"]
    return (x - f["merkez"]) / f["yari"]


# ════════════════════════════════════════════════════════════════
# Terimler
# ════════════════════════════════════════════════════════════════
def terim_adi(t):
    if t == (KTS,):
        return "Ct Pt"
    if t == (BLK,):
        return "Bloklar"
    return "*".join(t)


def _derece(t):
    return len(t)


def _kare_mi(t):
    return len(t) != len(set(t))


def varsayilan_terimler(meta, model=None, derece=2, merkez_var=False, blok_var=False):
    fs = meta["faktorler"]
    ad = [f["isim"] for f in fs]
    surekli = [f["isim"] for f in fs if not _kategorik(meta, f)]
    aile = _aile(meta)
    terim = []
    if blok_var:
        terim.append((BLK,))
    if aile == "karisim":
        model = model or "kuadratik"
        terim += [(a,) for a in ad]
        if model in ("kuadratik", "ozel_kubik"):
            terim += list(itertools.combinations(ad, 2))
        if model == "ozel_kubik":
            terim += list(itertools.combinations(ad, 3))
        return terim
    if aile in ("faktoriyel", "genel", "taguchi") and model is None:
        for d in range(1, min(derece, len(ad)) + 1):
            terim += list(itertools.combinations(ad, d))
        if merkez_var and aile == "faktoriyel":
            terim.append((KTS,))
        return terim
    model = model or "kuadratik"
    terim += [(a,) for a in ad]
    if model in ("etkilesim", "kuadratik"):
        terim += list(itertools.combinations(ad, 2))
    if model in ("kare", "kuadratik"):
        terim += [(a, a) for a in surekli]
    return sorted(terim, key=lambda t: (t[0] != BLK, _derece(t), _kare_mi(t)))


# ════════════════════════════════════════════════════════════════
# Model matrisi
# ════════════════════════════════════════════════════════════════
def model_matrisi(meta, ayarlar, terimler, pt=None, blok=None, bloklar=None, kesisim=None):
    """
    ayarlar: [{faktör: değer}] (kodlanmamış). pt/blok: satır başına listeler.
    Dönüş: X, sütun bilgileri, terim → sütun indisleri
    """
    if kesisim is None:
        kesisim = meta["tip"] != "karisim"
    n = len(ayarlar)
    zler = [_karisim_z(meta, a) for a in ayarlar] if meta["tip"] == "karisim" else None
    sut, bilgi, harita = [], [], {}
    if kesisim:
        sut.append(np.ones(n))
        bilgi.append({"etiket": "Sabit", "terim": None, "parca": []})
    for t in terimler:
        idx = []
        if t == (KTS,):
            sut.append(np.array([1.0 if (pt[i] == 0) else 0.0 for i in range(n)]))
            bilgi.append({"etiket": "Ct Pt", "terim": t, "parca": [("ct",)]})
            idx.append(len(sut) - 1)
        elif t == (BLK,):
            L = bloklar or sorted({b for b in blok})
            for j in range(len(L) - 1):
                sut.append(np.array([1.0 if blok[i] == L[j] else (-1.0 if blok[i] == L[-1] else 0.0)
                                     for i in range(n)]))
                bilgi.append({"etiket": f"Bloklar {L[j]}", "terim": t, "parca": [("blok", j)]})
                idx.append(len(sut) - 1)
        else:
            # Her faktör için sütun listesi (sürekli: 1 sütun; kategorik: m-1)
            parcalar = []
            for ad in t:
                f = _faktor(meta, ad)
                if _kategorik(meta, f):
                    kod = [_kat_sutunlari(meta, f, a.get(ad)) for a in ayarlar]
                    m = len(_kat_etiketleri(meta, f))
                    parcalar.append([(("kat", ad, j), np.array([k[j] if k else np.nan for k in kod]),
                                      _kat_etiketleri(meta, f)[j]) for j in range(m)])
                elif meta["tip"] == "karisim":
                    v = np.array([z.get(ad, np.nan) for z in zler])
                    parcalar.append([(("c", ad), v, ad)])
                else:
                    v = np.array([np.nan if _kodla(meta, f, a.get(ad)) is None else _kodla(meta, f, a.get(ad))
                                  for a in ayarlar])
                    parcalar.append([(("c", ad), v, ad)])
            for komb in itertools.product(*parcalar):
                v = np.ones(n)
                for _, x, _ in komb:
                    v = v * x
                sut.append(v)
                etiket = "*".join(e for _, _, e in komb)
                bilgi.append({"etiket": etiket, "terim": t, "parca": [k for k, _, _ in komb]})
                idx.append(len(sut) - 1)
        harita[t] = idx
    X = np.column_stack(sut) if sut else np.zeros((n, 0))
    return X, bilgi, harita


# ════════════════════════════════════════════════════════════════
# Veri hazırlama
# ════════════════════════════════════════════════════════════════
def _sayi(v):
    if v is None:
        return None
    if isinstance(v, (int, float, np.floating, np.integer)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    s = str(v).strip().replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def calisma_oku(rows, meta, yanitlar):
    """DOE sayfası satırları → [{'StdOrder','RunOrder','pt','blok','deg','y','Veri_ID','_satir'}]."""
    if not rows:
        return []
    bas = [str(h).strip() for h in rows[0]]
    ix = {h: i for i, h in enumerate(bas)}
    out = []
    for k, r in enumerate(rows[1:], start=2):
        g = lambda h: (r[ix[h]] if h in ix and ix[h] < len(r) else "")
        deg = {}
        for f in meta["faktorler"]:
            v = g(f["isim"])
            deg[f["isim"]] = _sayi(v) if f["tip"] == "nicel" else (str(v).strip() or None)
        if all(v is None for v in deg.values()):
            continue
        pt_ad = meta.get("pt_sutun")
        pt = _sayi(g(pt_ad)) if pt_ad else 1
        out.append({"StdOrder": int(_sayi(g("StdOrder")) or (k - 1)),
                    "RunOrder": int(_sayi(g("RunOrder")) or (k - 1)),
                    "pt": int(pt) if pt is not None else 1,
                    "blok": int(_sayi(g("Blocks")) or 1),
                    "deg": deg,
                    "y": {c: g(c) for c in yanitlar},
                    "Veri_ID": g("Veri_ID"), "_satir": k})
    return out


def ilerleme(satirlar, yanit_sutunlari):
    tam = sum(1 for s in satirlar
              if all(str(s["y"].get(c, "")).strip() != "" for c in yanit_sutunlari))
    kismi = sum(1 for s in satirlar
                if any(str(s["y"].get(c, "")).strip() != "" for c in yanit_sutunlari))
    return tam, kismi, len(satirlar)


# ════════════════════════════════════════════════════════════════
# Analiz
# ════════════════════════════════════════════════════════════════
def _sse(X, y):
    if X.shape[1] == 0:
        return float(np.sum(y ** 2)), 0
    b, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return float(e @ e), int(rank)


def _sabit_uzayda(X):
    if X.shape[1] == 0:
        return False
    one = np.ones(X.shape[0])
    b, *_ = np.linalg.lstsq(X, one, rcond=None)
    return np.allclose(X @ b, one, atol=1e-8)


def _karisik_ile(X, bilgi, tut, j):
    """j sütununun birebir aynı (±) olduğu önceki terim; yoksa 'doğrusal bağımlı'."""
    xj = X[:, j]
    for k in tut:
        if bilgi[k]["terim"] is None:
            continue
        xk = X[:, k]
        if np.allclose(xj, xk) or np.allclose(xj, -xk):
            return bilgi[k]["etiket"]
    return "önceki terimlerin bileşimi"


def analiz(meta, satirlar, yanit, terimler, alfa=0.05):
    """Tek yanıt için Minitab tarzı analiz. satirlar: calisma_oku çıktısı (veya eşdeğeri)."""
    veri = [(s, _sayi(s["y"].get(yanit))) for s in satirlar]
    veri = [(s, y) for s, y in veri if y is not None
            and all(v is not None for v in s["deg"].values())]
    if len(veri) < 3:
        raise ValueError(f"'{yanit}' için en az 3 sonuç gerekli (şu an {len(veri)}).")
    sat = [s for s, _ in veri]
    y = np.array([v for _, v in veri])
    pt = [s["pt"] for s in sat]
    blok = [s["blok"] for s in sat]
    bloklar = sorted(set(blok))
    terimler = [t for t in terimler if not (t == (BLK,) and len(bloklar) < 2)]
    terimler = [t for t in terimler if not (t == (KTS,) and (0 not in pt or all(p == 0 for p in pt)))]
    X_tum, bilgi_tum, harita = model_matrisi(meta, [s["deg"] for s in sat], terimler, pt, blok, bloklar)
    gecerli = ~np.isnan(X_tum).any(axis=1)
    if not gecerli.all():
        X_tum, y = X_tum[gecerli], y[gecerli]
        sat = [s for s, g in zip(sat, gecerli) if g]
    n = len(y)

    # Sıralı karışıklık (alias) kontrolü: bağımlı sütunları çıkar
    tut, karisik = [], []
    kesisim = meta["tip"] != "karisim"
    if kesisim:
        tut.append(0)
    for t in terimler:
        for j in harita[t]:
            aday = tut + [j]
            if np.linalg.matrix_rank(X_tum[:, aday], tol=1e-8) > len(tut):
                tut.append(j)
            else:
                karisik.append((bilgi_tum[j]["etiket"], _karisik_ile(X_tum, bilgi_tum, tut, j)))
    X = X_tum[:, tut]
    bilgi = [bilgi_tum[j] for j in tut]
    p = X.shape[1]
    if p > n:
        raise ValueError("Model, veriden fazla terim içeriyor; terim sayısını azaltın.")
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    yhat = X @ beta
    e = y - yhat
    sse = float(e @ e)
    dfE = n - p
    ybar = y.mean()
    sst = float(np.sum((y - ybar) ** 2))
    sabit = kesisim or _sabit_uzayda(X)
    df_model = p - 1 if sabit else p
    ssr = sst - sse if sabit else float(np.sum(yhat ** 2))
    mse = sse / dfE if dfE > 0 else None
    s = math.sqrt(mse) if mse is not None else None
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    tcrit = stats.t.ppf(1 - alfa / 2, dfE) if dfE > 0 else None

    # ── Katsayılar ─────────────────────────────────────────────
    aile = _aile(meta)
    etki_var = aile == "faktoriyel"
    katsayi = []
    lenth = None
    for j, b in enumerate(beta):
        ib = bilgi[j]
        se = math.sqrt(mse * XtX_inv[j, j]) if mse is not None else None
        tv = b / se if se else None
        pv = 2 * stats.t.sf(abs(tv), dfE) if tv is not None else None
        vif = None
        if ib["terim"] is not None and kesisim:
            xc = X[:, j] - X[:, j].mean()
            vif = float(XtX_inv[j, j] * (xc @ xc))
        etki = 2 * b if (etki_var and ib["terim"] not in (None, (KTS,), (BLK,))) else None
        if aile == "karisim" and ib["terim"] is not None and len(ib["terim"]) == 1:
            tv, pv, vif = None, None, None  # Minitab: doğrusal bileşenler için test yok
        katsayi.append({"Terim": ib["etiket"], "Etki": etki, "Katsayı": float(b), "SE Katsayı": se,
                        "T": tv, "P": pv, "VIF": vif, "_terim": ib["terim"]})
    if dfE == 0 and etki_var:
        etkiler = np.array([k["Etki"] for k in katsayi if k["Etki"] is not None])
        if len(etkiler) >= 3:
            s0 = 1.5 * np.median(np.abs(etkiler))
            kucuk = np.abs(etkiler)[np.abs(etkiler) < 2.5 * s0]
            pse = 1.5 * np.median(kucuk) if len(kucuk) else s0
            m = len(etkiler)
            me = stats.t.ppf(0.975, m / 3) * pse
            lenth = {"PSE": float(pse), "ME": float(me), "t_kritik": float(stats.t.ppf(0.975, m / 3))}
            for k in katsayi:
                if k["Etki"] is not None and pse > 0:
                    k["T"] = k["Etki"] / pse  # Lenth t

    # ── ANOVA ───────────────────────────────────────────────────
    def grup_adi(t):
        if t == (BLK,):
            return "Bloklar"
        if t == (KTS,):
            return "Eğrilik"
        if aile == "karisim":
            return {1: "Doğrusal", 2: "Kuadratik", 3: "Özel kübik"}[len(t)]
        if _kare_mi(t):
            return "Kare"
        return GRUP_ADLARI.get(len(t), f"{len(t)}'li Etkileşim")

    terim_sut = {}
    for j, ib in enumerate(bilgi):
        if ib["terim"] is not None:
            terim_sut.setdefault(ib["terim"], []).append(j)
    sira = ["Bloklar", "Doğrusal", "Kare", "Kuadratik", "Özel kübik"] + \
           [GRUP_ADLARI[d] for d in range(2, 6)] + ["Eğrilik"]
    gruplar = {}
    for t in terim_sut:
        gruplar.setdefault(grup_adi(t), []).append(t)

    def satir(ad, df, ss, girinti=0):
        ms = ss / df if df else None
        F = ms / mse if (ms is not None and mse) else None
        P = stats.f.sf(F, df, dfE) if F is not None else None
        return {"Kaynak": ("  " * girinti) + ad, "SD": df, "Adj SS": ss, "Adj MS": ms,
                "F": F, "P": P}

    anova = [satir("Model", df_model, ssr)]
    terim_p = {}
    sirali_gruplar = sorted(gruplar, key=lambda x: sira.index(x) if x in sira else 99)
    if aile == "karisim":
        # Ardışık SS (doğrusal Scheffé modeli sabit terimi içerir)
        once, sse_once = [], sst
        for g in sirali_gruplar:
            ek = [j for t in gruplar[g] for j in terim_sut[t]]
            sse_ile = _sse(X[:, once + ek], y)[0]
            df_g = len(ek) - (1 if g == "Doğrusal" else 0)
            r = satir(g, df_g, sse_once - sse_ile, 1)
            r["Kaynak"] = r["Kaynak"]
            anova.append(r)
            for t in gruplar[g]:
                terim_p[t] = None
            once, sse_once = once + ek, sse_ile
        # Doğrusal olmayan terimler için katsayı p-değerleri kullanılır
        for k in katsayi:
            if k["_terim"] is not None and len(k["_terim"]) > 1:
                terim_p[k["_terim"]] = k["P"]
    else:
        for g in sirali_gruplar:
            ts = gruplar[g]
            grup_sut = [j for t in ts for j in terim_sut[t]]
            kalan = [j for j in range(p) if j not in grup_sut]
            anova.append(satir(g, len(grup_sut), _sse(X[:, kalan], y)[0] - sse, 1))
            if len(ts) == 1:
                terim_p[ts[0]] = anova[-1]["P"]
                continue
            for t in ts:
                kal = [j for j in range(p) if j not in terim_sut[t]]
                r = satir(terim_adi(t), len(terim_sut[t]), _sse(X[:, kal], y)[0] - sse, 2)
                terim_p[t] = r["P"]
                anova.append(r)
    anova.append({"Kaynak": "Hata", "SD": dfE, "Adj SS": sse, "Adj MS": mse, "F": None, "P": None})
    # Uyum eksikliği / saf hata
    anahtar = [tuple([str(s["blok"])] + [str(s["deg"][f["isim"]]) for f in meta["faktorler"]])
               for s in sat]
    gruplar_y = {}
    for a, yy in zip(anahtar, y):
        gruplar_y.setdefault(a, []).append(yy)
    sspe = sum(float(np.sum((np.array(v) - np.mean(v)) ** 2)) for v in gruplar_y.values())
    dfpe = n - len(gruplar_y)
    lof = None
    if dfpe > 0 and dfE - dfpe > 0:
        sslof, dflof = sse - sspe, dfE - dfpe
        F = (sslof / dflof) / (sspe / dfpe) if sspe > 0 else None
        P = stats.f.sf(F, dflof, dfpe) if F is not None else None
        lof = {"F": F, "P": P}
        anova.append({"Kaynak": "  Uyum eksikliği", "SD": dflof, "Adj SS": sslof,
                      "Adj MS": sslof / dflof, "F": F, "P": P})
        anova.append({"Kaynak": "  Saf hata", "SD": dfpe, "Adj SS": sspe,
                      "Adj MS": sspe / dfpe, "F": None, "P": None})
    anova.append({"Kaynak": "Toplam", "SD": n - 1 if sabit else n, "Adj SS": sst if sabit
                  else float(np.sum(y ** 2)), "Adj MS": None, "F": None, "P": None})

    # ── Model özeti ────────────────────────────────────────────
    r2 = ssr / sst if sst > 0 else None
    r2adj = 1 - (sse / dfE) / (sst / (n - 1)) if (dfE > 0 and sst > 0) else None
    if np.all(h < 1 - 1e-8) and sst > 0:
        press = float(np.sum((e / (1 - h)) ** 2))
        r2pred = 1 - press / sst
    else:
        press, r2pred = None, None
    ozet = {"S": s, "R-kare": r2, "R-kare(düz)": r2adj, "R-kare(tah)": r2pred, "PRESS": press,
            "n": n, "p": p, "SD_hata": dfE}

    # ── Olağandışı gözlemler ───────────────────────────────────
    olagandisi = []
    if s:
        for i in range(n):
            if h[i] >= 1 - 1e-9:
                continue
            r = e[i] / (s * math.sqrt(1 - h[i]))
            isaret = ("R" if abs(r) > 2 else "") + (" X" if h[i] > 3 * p / n else "")
            if isaret.strip():
                olagandisi.append({"Gözlem (RunOrder)": sat[i]["RunOrder"], yanit: float(y[i]),
                                   "Uyum": float(yhat[i]), "Artık": float(e[i]),
                                   "Std Artık": float(r), "": isaret.strip()})
    std_artik = (e / (s * np.sqrt(np.clip(1 - h, 1e-12, None)))) if s else np.zeros(n)

    sonuc = {"meta": meta, "yanit": yanit, "terimler": list(terim_sut.keys()),
             "tut": tut, "terimler_tum": terimler, "bloklar": bloklar,
             "beta": beta, "XtX_inv": XtX_inv, "mse": mse, "dfE": dfE, "alfa": alfa,
             "katsayi": katsayi, "anova": anova, "ozet": ozet, "karisik": karisik,
             "lenth": lenth, "olagandisi": olagandisi, "terim_p": terim_p, "lof": lof,
             "y": y, "yhat": yhat, "artik": e, "std_artik": std_artik,
             "runorder": [s_["RunOrder"] for s_ in sat], "satirlar": sat, "tcrit": tcrit}
    sonuc["denklem"] = denklem(sonuc)
    sonuc["mesajlar"] = yorumla(sonuc)
    return sonuc


def geri_eleme(meta, satirlar, yanit, terimler, alfa=0.05, adim_alfa=0.10):
    """Hiyerarşiye uyan geri eleme (Minitab 'Backward elimination', α-çıkar = 0.10)."""
    terimler = list(terimler)
    cikan = []
    s = analiz(meta, satirlar, yanit, terimler, alfa)
    if s["dfE"] == 0 and s["lenth"]:
        # Doymuş tasarım: Lenth'e göre önemsiz terimleri (hiyerarşiyi koruyarak) çıkar
        tk = s["lenth"]["t_kritik"]
        onemli = {k["_terim"] for k in s["katsayi"] if k["_terim"] not in (None, (KTS,), (BLK,))
                  and k["T"] is not None and abs(k["T"]) > tk}
        gerekli = set(onemli)
        for t in onemli:
            for t2 in s["terimler"]:
                if _icerir(t, t2):
                    gerekli.add(t2)
        for t in list(terimler):
            if t not in ((KTS,), (BLK,)) and t not in gerekli:
                terimler.remove(t)
                cikan.append(terim_adi(t))
    while True:
        s = analiz(meta, satirlar, yanit, terimler, alfa)
        if s["dfE"] <= 0:
            break
        aday = []
        for t, pv in s["terim_p"].items():
            if t in ((BLK,),) or pv is None:
                continue
            gerekli = any(t2 != t and t2 not in ((KTS,), (BLK,)) and _icerir(t2, t)
                          for t2 in s["terimler"])
            if not gerekli and pv > adim_alfa:
                aday.append((pv, t))
        if not aday:
            break
        _, t = max(aday)
        terimler.remove(t)
        cikan.append(terim_adi(t))
        if len([t for t in terimler if t not in ((BLK,), (KTS,))]) <= 1:
            break
    s = analiz(meta, satirlar, yanit, terimler, alfa)
    s["cikarilan"] = cikan
    return s


def ileri_secim(meta, satirlar, yanit, temel, adaylar, alfa=0.05, giris_alfa=0.05):
    """Temel terimlerle başlar; aday terimleri (kare/etkileşim) en küçük p'ye göre ekler.
    Kesin eleme tasarımları (DSD) için Minitab'daki adımsal yaklaşıma benzer."""
    terimler = list(temel)
    eklenen = []
    while True:
        en = None
        for t in adaylar:
            if t in terimler:
                continue
            if not all((x,) in terimler for x in set(t)):
                continue  # hiyerarşi
            try:
                s = analiz(meta, satirlar, yanit, terimler + [t], alfa)
            except ValueError:
                continue
            if s["dfE"] < 3:
                continue
            pv = s["terim_p"].get(t)
            if pv is not None and pv < giris_alfa and (en is None or pv < en[0]):
                en = (pv, t)
        if en is None:
            break
        terimler.append(en[1])
        eklenen.append(terim_adi(en[1]))
    terimler = sorted(terimler, key=lambda t: (t[0] != BLK, t == (KTS,), _derece(t), _kare_mi(t)))
    s = analiz(meta, satirlar, yanit, terimler, alfa)
    s["eklenen"] = eklenen
    return s


def _icerir(buyuk, kucuk):
    if KTS in buyuk or BLK in buyuk or KTS in kucuk or BLK in kucuk:
        return False
    b = list(buyuk)
    for x in kucuk:
        if x in b:
            b.remove(x)
        else:
            return False
    return len(buyuk) > len(kucuk)


# ════════════════════════════════════════════════════════════════
# Tahmin
# ════════════════════════════════════════════════════════════════
def tahmin(sonuc, ayarlar, ct=0):
    """ayarlar: [{faktör: kodlanmamış değer}] → (tahmin, SE uyum) dizileri."""
    meta = sonuc["meta"]
    n = len(ayarlar)
    bloklar = sonuc["bloklar"] or [1]
    X, bilgi, _ = model_matrisi(meta, ayarlar, sonuc["terimler_tum"], pt=[1 - ct] * n,
                                blok=[bloklar[0]] * n, bloklar=bloklar)
    for j, ib in enumerate(bilgi):  # bloklar üzerinden ortalama: blok sütunları 0
        if ib["terim"] == (BLK,):
            X[:, j] = 0.0
    X = X[:, sonuc["tut"]]
    yhat = X @ sonuc["beta"]
    se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", X, sonuc["XtX_inv"], X), 0, None) *
                 (sonuc["mse"] or 0))
    return yhat, se


# ════════════════════════════════════════════════════════════════
# Kodlanmamış denklem
# ════════════════════════════════════════════════════════════════
def _bicim(v):
    return f"{v:.5g}"


def denklem(sonuc):
    meta = sonuc["meta"]
    tut = sonuc["tut"]
    beta = sonuc["beta"]
    satir0 = sonuc["satirlar"][0]["deg"]
    _, bilgi_tum, _ = model_matrisi(meta, [satir0], sonuc["terimler_tum"], pt=[1], blok=[1],
                                    bloklar=sonuc["bloklar"])
    bilgi = [bilgi_tum[j] for j in tut]
    yan = sonuc["yanit"]
    if meta["tip"] == "karisim":
        terimler = []
        for b, ib in zip(beta, bilgi):
            terimler.append(f"{_bicim(b)} {ib['etiket']}")
        return [f"{yan} = " + " + ".join(terimler).replace("+ -", "- ")], \
            "Sözde bileşenler (pseudo-component) cinsinden"
    sira_f = {f["isim"]: i for i, f in enumerate(meta["faktorler"])}
    nitel = [f for f in meta["faktorler"] if _kategorik(meta, f)]
    kombs = list(itertools.product(*[_seviyeler(meta, f) for f in nitel])) if nitel else [()]
    not_ = ""
    if len(kombs) > 12:
        kombs = kombs[:12]
        not_ = "İlk 12 kategori birleşimi gösteriliyor."
    denklemler = []
    for komb in kombs:
        ayar = {f["isim"]: v for f, v in zip(nitel, komb)}
        poli = {}
        ct = None
        for b, ib in zip(beta, bilgi):
            if ib["terim"] == (BLK,):
                continue
            if ib["terim"] == (KTS,):
                ct = b
                continue
            terim = {(): float(b)}
            for par in ib["parca"]:
                if par[0] == "kat":
                    f = _faktor(meta, par[1])
                    kod = _kat_sutunlari(meta, f, ayar[par[1]])
                    terim = {k: v * kod[par[2]] for k, v in terim.items()}
                elif par[0] == "c":
                    f = _faktor(meta, par[1])
                    a, c = 1 / f["yari"], -f["merkez"] / f["yari"]
                    yeni = {}
                    for mono, v in terim.items():
                        m2 = tuple(sorted(mono + (par[1],), key=sira_f.get))
                        yeni[m2] = yeni.get(m2, 0) + v * a
                        yeni[mono] = yeni.get(mono, 0) + v * c
                    terim = yeni
            for k, v in terim.items():
                poli[k] = poli.get(k, 0) + v
        sabit = poli.pop((), 0.0)
        parcalar = [_bicim(sabit)]
        for mono in sorted(poli, key=lambda m: (len(m), _kare_mi(m), [sira_f[x] for x in m])):
            v = poli[mono]
            if abs(v) < 1e-12:
                continue
            parcalar.append(f"{'+' if v >= 0 else '-'} {_bicim(abs(v))} {'*'.join(mono)}")
        if ct is not None:
            parcalar.append(f"{'+' if ct >= 0 else '-'} {_bicim(abs(ct))} Ct Pt")
        on = ", ".join(f"{k} = {seviye_yaz(v)}" for k, v in ayar.items())
        denklemler.append((f"[{on}]  " if on else "") + f"{yan} = " + " ".join(parcalar))
    if sonuc["bloklar"] and len(sonuc["bloklar"]) > 1:
        not_ = (not_ + " " if not_ else "") + "Denklem bloklar üzerinden ortalamadır."
    return denklemler, not_


# ════════════════════════════════════════════════════════════════
# Otomatik yorumlar (bilgi mesajları)
# ════════════════════════════════════════════════════════════════
def yorumla(s):
    m = []
    o = s["ozet"]
    alfa = s["alfa"]
    if s["dfE"] == 0:
        if s["lenth"]:
            m.append(("info", "Doymuş tasarım: hata serbestlik derecesi 0. Anlamlılık Lenth "
                              f"yöntemiyle değerlendirildi (PSE = {s['lenth']['PSE']:.4g}, "
                              f"ME = {s['lenth']['ME']:.4g}). Önemsiz terimleri çıkararak "
                              "p-değerleri elde edebilirsiniz."))
        else:
            m.append(("warning", "Hata serbestlik derecesi 0 — p-değerleri hesaplanamıyor. "
                                 "Terim sayısını azaltın veya tekrar/merkez nokta ekleyin."))
    anlamli = [terim_adi(t) for t, pv in s["terim_p"].items()
               if pv is not None and pv < alfa and t != (BLK,)]
    if s["lenth"]:
        anlamli = [k["Terim"] for k in s["katsayi"] if k["Etki"] is not None and k["T"] is not None
                   and abs(k["T"]) > s["lenth"]["t_kritik"]]
    if anlamli:
        m.append(("success", f"α = {alfa} düzeyinde anlamlı terimler: {', '.join(anlamli)}"))
    elif s["dfE"] > 0:
        m.append(("warning", f"α = {alfa} düzeyinde anlamlı terim yok. Faktör aralıklarını "
                             "genişletmeyi veya tekrar eklemeyi düşünün."))
    ct = s["terim_p"].get((KTS,))
    if ct is not None:
        if ct < alfa:
            m.append(("warning", f"Eğrilik anlamlı (P = {ct:.3f}): yanıt doğrusal değil. "
                                 "Merkez noktadaki tahminler yanıltıcı olabilir — yanıt yüzeyi "
                                 "tasarımına (CCD / Box-Behnken) geçmeyi düşünün."))
        else:
            m.append(("info", f"Eğrilik anlamlı değil (P = {ct:.3f}): doğrusal model "
                              "deney bölgesinde yeterli görünüyor."))
    if s["lof"] and s["lof"]["P"] is not None:
        if s["lof"]["P"] < alfa:
            m.append(("warning", f"Uyum eksikliği anlamlı (P = {s['lof']['P']:.3f}): model "
                                 "verideki yapıyı tam açıklamıyor; kare/etkileşim terimi ekleyin."))
        else:
            m.append(("info", f"Uyum eksikliği anlamlı değil (P = {s['lof']['P']:.3f})."))
    if o["R-kare(tah)"] is not None and o["R-kare"] is not None:
        if o["R-kare"] - o["R-kare(tah)"] > 0.2:
            m.append(("warning", f"R²(tah) = {100 * o['R-kare(tah)']:.1f}% ile R² = "
                                 f"{100 * o['R-kare']:.1f}% arasındaki fark büyük — model "
                                 "aşırı uyum gösteriyor olabilir."))
    if s["olagandisi"]:
        m.append(("info", f"{len(s['olagandisi'])} olağandışı gözlem var (R: büyük artık, "
                          "X: yüksek kaldıraç) — aşağıdaki tabloya bakın."))
    if s["karisik"]:
        m.append(("info", f"{len(s['karisik'])} terim başka terimlerle karışık (aliased) "
                          "olduğu için modelden çıkarıldı."))
    vif = [k for k in s["katsayi"] if k["VIF"] is not None and k["VIF"] > 10]
    if vif:
        m.append(("warning", "Yüksek VIF (>10): " + ", ".join(k["Terim"] for k in vif) +
                             " — terimler birbiriyle ilişkili, katsayılar güvenilmez olabilir."))
    return m


# ════════════════════════════════════════════════════════════════
# Taguchi
# ════════════════════════════════════════════════════════════════
SN_ADLARI = {"buyuk": "Büyük daha iyi", "kucuk": "Küçük daha iyi",
             "nominal": "Nominal en iyi"}


def taguchi_analiz(meta, satirlar, cikti, hedef="buyuk"):
    r = meta.get("tekrar_sutun", 1)
    sutun = [f"{cikti}_{i}" for i in range(1, r + 1)] if r > 1 else [cikti]
    kayit = []
    for s in satirlar:
        v = np.array([x for x in (_sayi(s["y"].get(c)) for c in sutun) if x is not None])
        if len(v) == 0:
            continue
        if hedef == "buyuk":
            if np.any(v == 0):
                continue
            sn = -10 * math.log10(np.mean(1 / v ** 2))
        elif hedef == "kucuk":
            sn = -10 * math.log10(max(np.mean(v ** 2), 1e-300))
        else:
            if len(v) < 2 or v.std(ddof=1) == 0:
                continue
            sn = 10 * math.log10(v.mean() ** 2 / v.var(ddof=1))
        kayit.append({"deg": s["deg"], "SN": sn, "Ort": float(v.mean()), "n": len(v)})
    if len(kayit) < 3:
        if hedef == "nominal" and r < 2:
            raise ValueError("'Nominal en iyi' S/N oranı her deneyde en az 2 tekrar ölçümü ister. "
                             "Tasarımı tekrar sütunuyla oluşturun ya da başka bir hedef seçin.")
        raise ValueError(f"Taguchi analizi için yeterli sonuç yok ({len(kayit)} deney).")
    tablolar = {}
    en_iyi = {}
    for olcut in ("SN", "Ort"):
        genel = np.mean([k[olcut] for k in kayit])
        satir = {}
        delta = {}
        for f in meta["faktorler"]:
            L = _seviyeler(meta, f)
            ort = []
            for l in L:
                v = [k[olcut] for k in kayit if _ayni(k["deg"][f["isim"]], l)]
                ort.append(float(np.mean(v)) if v else None)
            satir[f["isim"]] = ort
            vv = [x for x in ort if x is not None]
            delta[f["isim"]] = max(vv) - min(vv) if vv else 0
        sira = {a: i + 1 for i, (a, _) in enumerate(sorted(delta.items(), key=lambda x: -x[1]))}
        tablolar[olcut] = {"genel": genel, "seviye": satir, "delta": delta, "sira": sira}
    for f in meta["faktorler"]:
        L = _seviyeler(meta, f)
        sn = tablolar["SN"]["seviye"][f["isim"]]
        j = int(np.nanargmax([x if x is not None else -np.inf for x in sn]))
        en_iyi[f["isim"]] = L[j]
    tahmin_ = {}
    for olcut in ("SN", "Ort"):
        g = tablolar[olcut]["genel"]
        v = g
        for f in meta["faktorler"]:
            L = _seviyeler(meta, f)
            j = next(i for i, l in enumerate(L) if _ayni(l, en_iyi[f["isim"]]))
            v += tablolar[olcut]["seviye"][f["isim"]][j] - g
        tahmin_[olcut] = v
    return {"kayit": kayit, "tablo": tablolar, "en_iyi": en_iyi, "tahmin": tahmin_,
            "hedef": hedef, "cikti": cikti, "meta": meta}


def taguchi_yanit_tablosu(ta, olcut="SN"):
    meta = ta["meta"]
    t = ta["tablo"][olcut]
    maks = max(len(_seviyeler(meta, f)) for f in meta["faktorler"])
    rows = []
    for i in range(maks):
        r = {"Seviye": i + 1}
        for f in meta["faktorler"]:
            v = t["seviye"][f["isim"]]
            r[f["isim"]] = round(v[i], 4) if i < len(v) and v[i] is not None else None
        rows.append(r)
    rows.append({"Seviye": "Delta", **{f["isim"]: round(t["delta"][f["isim"]], 4)
                                       for f in meta["faktorler"]}})
    rows.append({"Seviye": "Sıra", **{f["isim"]: t["sira"][f["isim"]] for f in meta["faktorler"]}})
    return rows


# ════════════════════════════════════════════════════════════════
# İstenebilirlik (desirability) optimizasyonu
# ════════════════════════════════════════════════════════════════
def _d(y, lo, hi, ymin, ymax):
    if lo is None and hi is None:
        return np.ones_like(y)
    d = np.ones_like(y, dtype=float)
    if lo is not None:
        alt = min(ymin, lo - max(hi - lo if hi is not None else 0, abs(lo) * 0.1, 1e-9))
        alt = min(alt, lo - 1e-9)
        d = np.where(y < lo, np.clip((y - alt) / (lo - alt), 0, 1), d)
    if hi is not None:
        ust = max(ymax, hi + max(hi - lo if lo is not None else 0, abs(hi) * 0.1, 1e-9))
        ust = max(ust, hi + 1e-9)
        d = np.where(y > hi, np.clip((ust - y) / (ust - hi), 0, 1), d)
    return d


def _ornekle(meta, n, rng, sabit=None):
    sabit = sabit or {}
    fs = meta["faktorler"]
    aile = _aile(meta)
    out = [dict() for _ in range(n)]
    if meta["tip"] == "karisim":
        z = rng.dirichlet(np.ones(len(fs)), n)
        for i in range(n):
            for f, zz in zip(fs, z[i]):
                out[i][f["isim"]] = f.get("kar_alt", 0) + meta["kalan"] * zz
        return out
    a = meta.get("alfa", 1.0) if meta["tip"] == "ccd" else 1.0
    for f in fs:
        if f["isim"] in sabit:
            for o in out:
                o[f["isim"]] = sabit[f["isim"]]
        elif _kategorik(meta, f):
            L = _seviyeler(meta, f)
            ix = rng.integers(0, len(L), n)
            for o, i in zip(out, ix):
                o[f["isim"]] = L[i]
        else:
            c = rng.uniform(-a, a, n)
            for o, cc in zip(out, c):
                o[f["isim"]] = f["merkez"] + cc * f["yari"]
    return out


def optimize(sonuclar, hedefler, n=4000, tohum=1, sabit=None):
    """
    sonuclar: {yanıt: analiz sonucu} (aynı tasarım)
    hedefler: {yanıt: (hedef_min, hedef_max)}
    """
    ilk = next(iter(sonuclar.values()))
    meta = ilk["meta"]
    rng = np.random.default_rng(tohum)
    aday = _ornekle(meta, n, rng, sabit)

    def D(ayarlar):
        toplam = np.ones(len(ayarlar))
        ayri = {}
        for yan, s in sonuclar.items():
            yh, _ = tahmin(s, ayarlar)
            lo, hi = hedefler[yan]
            d = _d(yh, lo, hi, float(s["y"].min()), float(s["y"].max()))
            ayri[yan] = (yh, d)
            toplam = toplam * np.maximum(d, 1e-12)
        return toplam ** (1 / len(sonuclar)), ayri

    Dv, _ = D(aday)
    # Yerel iyileştirme (sürekli faktörlerde küçülen adımlarla)
    sira = np.argsort(-Dv)[:8]
    en_ayar, en_D = aday[sira[0]], Dv[sira[0]]
    a = meta.get("alfa", 1.0) if meta["tip"] == "ccd" else 1.0
    surekli = [f for f in meta["faktorler"] if not _kategorik(meta, f)
               and f["isim"] not in (sabit or {})]
    if meta["tip"] != "karisim" and surekli:
        for i0 in sira:
            mevcut, mD = dict(aday[i0]), Dv[i0]
            adim = 0.25
            for _ in range(12):
                komsu = []
                for _ in range(60):
                    yeni = dict(mevcut)
                    for f in surekli:
                        c = (yeni[f["isim"]] - f["merkez"]) / f["yari"]
                        c = float(np.clip(c + rng.normal(0, adim), -a, a))
                        yeni[f["isim"]] = f["merkez"] + c * f["yari"]
                    komsu.append(yeni)
                kd, _ = D(komsu)
                j = int(np.argmax(kd))
                if kd[j] > mD:
                    mevcut, mD = komsu[j], kd[j]
                else:
                    adim *= 0.6
            if mD > en_D:
                en_ayar, en_D = mevcut, mD
    # Sonuçlar
    for f in meta["faktorler"]:
        if not _kategorik(meta, f):
            en_ayar[f["isim"]] = T._yuvarla(en_ayar[f["isim"]], f)
    Dson, ayri = D([en_ayar])
    tablo = []
    for yan, s in sonuclar.items():
        yh, se = tahmin(s, [en_ayar])
        t = s["tcrit"]
        satir = {"Yanıt": yan, "Uyum": float(yh[0]),
                 "SE Uyum": float(se[0]) if s["mse"] else None,
                 "Hedef": f"{hedefler[yan][0]} – {hedefler[yan][1]}",
                 "İstenebilirlik": float(ayri[yan][1][0])}
        if t and s["mse"]:
            satir["%95 GA"] = f"({yh[0] - t * se[0]:.4g}; {yh[0] + t * se[0]:.4g})"
            pi = math.sqrt(se[0] ** 2 + s["mse"])
            satir["%95 TA"] = f"({yh[0] - t * pi:.4g}; {yh[0] + t * pi:.4g})"
        tablo.append(satir)
    return {"ayar": en_ayar, "D": float(Dson[0]), "tablo": tablo}


# ════════════════════════════════════════════════════════════════
# Uygunluk (hangi tasarım çalışabilir?) ve öneri
# ════════════════════════════════════════════════════════════════
def uygunluk(faktorler):
    """
    faktorler: [{'isim','tip','n_kat'}] (seçili DOE faktörleri)
    Dönüş: (satırlar, öneri metni)
    """
    k = len(faktorler)
    kc = sum(1 for f in faktorler if f["tip"] == "nicel")
    kn = k - kc
    cok_kat = [f["isim"] for f in faktorler if f["tip"] == "nitel" and f.get("n_kat", 2) > 2]
    nitel_komb = int(np.prod([max(1, f.get("n_kat", 2)) for f in faktorler if f["tip"] == "nitel"])) \
        if kn else 1
    rows = []

    def ekle(kod, durum, deney, aciklama):
        rows.append({"kod": kod, "Tasarım": T.TASARIM_ADLARI[kod], "Durum": durum,
                     "Deney": deney, "Açıklama": aciklama})

    if k == 0:
        return [], "Önce faktör seçin."
    ek2 = (f" '{', '.join(cok_kat)}' için 2 kategori seçmeniz gerekir." if cok_kat else "")
    # 2 seviyeli tam
    if k < 2:
        ekle("faktoriyel2", "❌", "—", "En az 2 faktör gerekir.")
    elif k <= 5:
        ekle("faktoriyel2", "✅", 2 ** k, "Tüm ana etkiler ve etkileşimler tahmin edilir." + ek2)
    elif k <= 10:
        ekle("faktoriyel2", "⚠️", 2 ** k, "Deney sayısı yüksek — kesirli faktöriyel veya "
                                          "eleme tasarımı daha verimli." + ek2)
    else:
        ekle("faktoriyel2", "❌", 2 ** k, "Çok fazla deney; en fazla 10 faktör.")
    # Kesirli
    sec = T.kesirli_secenekler(k) if 3 <= k <= 11 else []
    if sec:
        metin = "; ".join(f"{s['n']} deney (Res {T.roma(s['cozunurluk'])})" for s in sec)
        en = max(sec, key=lambda s: s["cozunurluk"] - s["n"] / 1000)
        ekle("kesirli", "✅" if k >= 4 else "⚠️", min(s["n"] for s in sec),
             f"Seçenekler: {metin}. " + ("Res V ve üstü: ana etkiler ve 2'li etkileşimler temiz. "
                                         if any(s["cozunurluk"] >= 5 for s in sec) else
                                         "Res IV: ana etkiler temiz, 2'li etkileşimler karışık. ")
             + ek2)
    else:
        ekle("kesirli", "❌", "—", "3–11 faktör gerekir.")
    # PB
    pbn = T.pb_deney_sayisi(k)
    if k >= 2 and pbn:
        ekle("pb", "✅" if k >= 5 else "⚠️", pbn[0],
             ("Çok faktörlü eleme için en ekonomik tasarım; sadece ana etkiler." if k >= 5 else
              "Az faktör için tam faktöriyel daha bilgilendirici.") + ek2)
    else:
        ekle("pb", "❌", "—", "2–23 faktör gerekir.")
    # Genel
    seviye = [3 if f["tip"] == "nicel" else f.get("n_kat", 2) for f in faktorler]
    gn = int(np.prod(seviye))
    ekle("genel", "✅" if gn <= 81 else ("⚠️" if gn <= 1000 else "❌"), gn,
         "Her faktör istediğiniz sayıda seviyede (nicel için 3 varsayıldı); çok kategorili "
         "nitel faktörler için uygundur." + (" Deney sayısı yüksek." if gn > 81 else ""))
    # CCD
    if kc >= 2 and kc <= 7:
        nf = 2 ** kc if kc < 5 else 2 ** (kc - 1)
        nn = (nf + 2 * kc + (T.CCD_MERKEZ_YARIM if kc >= 5 else T.CCD_MERKEZ).get(kc, 6)) * nitel_komb
        ekle("ccd", "✅" if kc <= 5 else "⚠️", nn,
             "Eğrilik (kare terimleri) ve etkileşimler; optimizasyon için standart tasarım. "
             "CCC eksenel noktaları aralık dışına taşar — makine sınırı varsa CCI/CCF."
             + (f" Nitel faktörlerin {nitel_komb} birleşimi için tekrarlanır." if kn else ""))
    else:
        ekle("ccd", "❌", "—", "2–7 nicel faktör gerekir.")
    # BBD
    if 3 <= kc <= 7:
        npt = len(T.BBD_GRUPLAR[kc]) * 2 ** len(T.BBD_GRUPLAR[kc][0])
        ekle("bbd", "✅", (npt + T.BBD_MERKEZ[kc]) * nitel_komb,
             "3 seviyeli; köşe noktaları (aşırı koşullar) denenmez — güvenli işletme için uygun.")
    else:
        ekle("bbd", "❌", "—", "3–7 nicel faktör gerekir.")
    # DSD
    if kc >= 3 and kn == 0:
        ekle("dsd", "✅", T.dsd_deney_sayisi(kc),
             "Az deneyle eleme + eğrilik tespiti; ana etkiler 2'li etkileşimlerden bağımsız.")
    elif kc >= 3:
        ekle("dsd", "⚠️", T.dsd_deney_sayisi(kc),
             "Sadece nicel faktörlerle kurulur; nitel faktörleri sabit tutmanız gerekir.")
    else:
        ekle("dsd", "❌", "—", "En az 3 nicel faktör gerekir.")
    # Taguchi
    n2 = sum(1 for f in faktorler if f["tip"] == "nitel" and f.get("n_kat", 2) == 2) + kc
    n3 = sum(1 for f in faktorler if f["tip"] == "nitel" and f.get("n_kat", 2) == 3)
    t2 = T.taguchi_oner(n2, n3)
    t3 = T.taguchi_oner(n2 - kc, n3 + kc) if kc else []
    oneriler = sorted(set(t2[:1] + t3[:1]), key=lambda a: T.TAGUCHI_DIZILER[a][0])
    if oneriler and all(f.get("n_kat", 2) <= 3 for f in faktorler if f["tip"] == "nitel"):
        ekle("taguchi", "✅", " / ".join(f"{a} ({T.TAGUCHI_DIZILER[a][0]})" for a in oneriler),
             "Gürbüz (robust) tasarım: sinyal/gürültü oranı. Nicel faktörlere 2 veya 3 seviye "
             "verilebilir.")
    else:
        ekle("taguchi", "❌", "—", "Faktörlerin 2 veya 3 seviyeli olması ve uygun dizi bulunması gerekir.")
    # Karışım
    if kn == 0 and 2 <= kc <= 10:
        ekle("karisim", "⚠️", 2 ** kc - 1 + kc,
             "Sadece girdiler bir karışımın bileşen oranlarıysa (toplamı sabit) anlamlıdır — "
             "örn. alaşım/toz/kimyasal oranları. Proses parametreleri için kullanmayın.")
    else:
        ekle("karisim", "❌", "—", "2–10 nicel bileşen gerekir; nitel faktör olamaz.")
    # D-optimal
    sd = [1 if f["tip"] == "nicel" else max(1, f.get("n_kat", 2) - 1) for f in faktorler]
    p = 1 + sum(sd) + sum(a * b for a, b in itertools.combinations(sd, 2)) + kc
    ekle("doptimal", "✅", f"≥ {p} (kuadratik)",
         "Deney bütçeniz sınırlıysa veya standart tasarım uymuyorsa; deney sayısını siz seçersiniz.")

    if k >= 6:
        oneri = (f"{k} faktör var: önce **eleme** yapın (Plackett-Burman, Res IV kesirli "
                 f"faktöriyel{' veya DSD' if kc >= 3 and kn == 0 else ''}), etkili 2–4 faktörü "
                 "bulduktan sonra **yanıt yüzeyi** (CCD / Box-Behnken) ile optimum ayarı belirleyin.")
    elif kc >= 2:
        oneri = (f"{k} faktör var: eğrilik beklemiyorsanız **2 seviyeli tam faktöriyel + merkez "
                 "noktaları**, optimum arıyorsanız doğrudan **CCD veya Box-Behnken** uygundur.")
    else:
        oneri = "Faktörler çoğunlukla nitel: **genel tam faktöriyel** en uygun seçim."
    return rows, oneri


def veri_uygun_mu(df, yapi):
    """Veriler sayfasındaki kayıtlar tasarlanmış deney düzeninde mi?"""
    nicel = [g for g in yapi["girdiler"] if g["tip"] == "nicel" and g["isim"] in df.columns]
    if not nicel or len(df) < 4:
        return None
    seviye = [df[g["isim"]].dropna().nunique() for g in nicel]
    if all(2 <= s <= 5 for s in seviye):
        return (f"Veriler sayfasındaki {len(df)} kayıtta nicel girdiler 2–5 sabit seviyede "
                f"tekrarlanıyor (tasarlanmış deney düzeni). DOE sekmesinde 'Veriler sayfası' "
                f"kaynağıyla Minitab tarzı analiz yapılabilir.")
    return None
