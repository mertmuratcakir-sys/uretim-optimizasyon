"""
doe_tani.py — Mevcut Veriler kayıtlarında deney tasarımı tanıma + otomatik analiz
==================================================================================
Yeni deney yapmadan, kayıtlı girdi/çıktı değerlerinin hangi DOE düzenine uyduğunu
bulur (2 seviyeli tam/kesirli faktöriyel, Plackett-Burman, genel faktöriyel, CCD,
Box-Behnken, Taguchi ortogonal dizisi, karışım) ve seçilen tasarımla tüm nicel
çıktıları Minitab tarzı analiz edip hedeflere göre optimize eder.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np

import doe_analiz as A
import doe_tasarim as T

MAKS_SEVIYE = 5


# ════════════════════════════════════════════════════════════════
# Seviye yakalama
# ════════════════════════════════════════════════════════════════
def _yuvarla(v, aralik):
    bas = 4 if aralik < 1 else 3 if aralik < 10 else 2
    return round(float(v), bas)


def _guzel(grup, tol, aralik):
    """Grubu temsil eden en 'yuvarlak' değer (ör. 29.96, 30.05 → 30)."""
    med = float(np.median(grup))
    lo, hi = min(grup) - tol * 0.25, max(grup) + tol * 0.25
    if len(grup) > 1 or tol > 1e-9:
        j = math.ceil(math.log10(max(2 * tol, 1e-12)))
        for _ in range(8):
            st_ = 10.0 ** j
            v = round(med / st_) * st_
            if lo <= v <= hi:
                return _yuvarla(v, aralik) if abs(v) > 1e-12 else 0.0
            j -= 1
    return _yuvarla(med, aralik)


def _seviyele(degerler):
    """Yakın değerleri aynı seviyeye toplar. Dönüş: (seviyeler, değer→seviye eşlemesi)."""
    v = np.array(sorted(set(degerler)))
    aralik = float(v.max() - v.min()) if len(v) else 0.0
    tol = max(aralik * 0.02, 1e-9)
    gruplar = [[v[0]]]
    for x in v[1:]:
        if x - gruplar[-1][-1] > tol:
            gruplar.append([x])
        else:
            gruplar[-1].append(x)
    seviye = [_guzel(g, tol, aralik) for g in gruplar]
    harita = {}
    for g, s in zip(gruplar, seviye):
        for x in g:
            harita[float(x)] = s
    return seviye, harita


def _hazirla(df, yapi):
    """Faktörleri (değişen girdiler), sabitleri ve seviyelenmiş satırları çıkarır."""
    girdiler = [g for g in yapi["girdiler"] if g["isim"] in df.columns]
    d = df.dropna(subset=[g["isim"] for g in girdiler]).reset_index(drop=True)
    faktorler, sabitler = [], {}
    for g in girdiler:
        s = d[g["isim"]]
        if g["tip"] == "nicel":
            sev, harita = _seviyele([float(x) for x in s])
            if len(sev) == 1:
                sabitler[g["isim"]] = sev[0]
                continue
            faktorler.append({"g": g, "seviye": sev, "harita": harita})
        else:
            sev = sorted({str(x) for x in s})
            if g.get("kategoriler"):
                sev = [k for k in g["kategoriler"] if k in sev] + [k for k in sev if k not in g["kategoriler"]]
            if len(sev) == 1:
                sabitler[g["isim"]] = sev[0]
                continue
            faktorler.append({"g": g, "seviye": sev, "harita": None})
    satir = []
    for _, r in d.iterrows():
        deg = {}
        for f in faktorler:
            ad = f["g"]["isim"]
            deg[ad] = f["harita"][float(r[ad])] if f["harita"] else str(r[ad])
        ham = {f["g"]["isim"]: (float(r[f["g"]["isim"]]) if f["harita"] else str(r[f["g"]["isim"]]))
               for f in faktorler}
        satir.append({"deg": deg, "ham": ham,
                      "y": {c["isim"]: r.get(c["isim"]) for c in yapi["ciktilar"]}})
    return faktorler, sabitler, satir


def _idx(f, v):
    return next(i for i, l in enumerate(f["seviye"]) if A._ayni(l, v))


# ════════════════════════════════════════════════════════════════
# Meta / satır üretimi (doe_tasarim ile aynı yapı)
# ════════════════════════════════════════════════════════════════
def _faktor_meta(f, dusuk=None, yuksek=None, sayi_seviye=False):
    g = f["g"]
    if g["tip"] == "nitel":
        fm = T.faktor(g["isim"], "nitel", seviyeler=f["seviye"])
    else:
        lo = f["seviye"][0] if dusuk is None else dusuk
        hi = f["seviye"][-1] if yuksek is None else yuksek
        fm = T.faktor(g["isim"], "nicel", lo, hi)
        fm["sinir_min"], fm["sinir_max"] = g.get("min"), g.get("max")
        if sayi_seviye:
            fm["seviyeler_sayi"] = list(f["seviye"])
    return fm


def _meta(tip, ad, fms, sabitler, yapi, **ek):
    meta = {"tip": tip, "ad": ad, "faktorler": T._harfle(fms), "sabitler": dict(sabitler),
            "ciktilar": [c["isim"] for c in yapi["ciktilar"]], "blok_sayisi": 1,
            "pt_sutun": ek.pop("pt_sutun", "PtType"), "kaynak": "Veriler sayfası (otomatik tanıma)"}
    meta.update(ek)
    return meta


def _satirlar(satir, pt_fn=lambda s: 1):
    out = []
    for i, s in enumerate(satir):
        out.append({"StdOrder": i + 1, "RunOrder": i + 1, "pt": pt_fn(s), "blok": 1,
                    "deg": dict(s["deg"]), "y": dict(s["y"])})
    return out


# ════════════════════════════════════════════════════════════════
# Tanıma kuralları
# ════════════════════════════════════════════════════════════════
def _ortogonal(M):
    """Kodlu ±1 matrisinde sütunlar dengeli ve ikişer ortogonal mi?"""
    M = np.asarray(M, float)
    if np.any(np.abs(M.sum(0)) > 1e-9):
        return False
    G = M.T @ M
    return np.allclose(G - np.diag(np.diag(G)), 0)


def _dengeli_dizi(kombs, faktorler):
    """Her faktör ve her faktör çifti için seviye kombinasyonları eşit sıklıkta mı?"""
    n = len(kombs)
    for j, f in enumerate(faktorler):
        c = Counter(k[j] for k in kombs)
        if len(c) != len(f["seviye"]) or len(set(c.values())) != 1:
            return False
    for a, b in itertools.combinations(range(len(faktorler)), 2):
        c = Counter((k[a], k[b]) for k in kombs)
        beklenen = len(faktorler[a]["seviye"]) * len(faktorler[b]["seviye"])
        if len(c) != beklenen or len(set(c.values())) != 1:
            return False
    return n > 0


def _iki_seviyeli(faktorler, satir, sabitler, yapi):
    """2 seviyeli tam / kesirli faktöriyel, Plackett-Burman (merkez noktalı olabilir)."""
    nicel = [f for f in faktorler if f["g"]["tip"] == "nicel"]
    for f in faktorler:
        n = len(f["seviye"])
        if f["g"]["tip"] == "nitel" and n != 2:
            return None, "nitel faktör 2'den fazla kategori içeriyor"
        if f["g"]["tip"] == "nicel" and n not in (2, 3):
            return None, f"'{f['g']['isim']}' {n} seviyede"
    merkez_var = any(len(f["seviye"]) == 3 for f in nicel)

    def tur(s):
        orta = [len(f["seviye"]) == 3 and _idx(f, s["deg"][f["g"]["isim"]]) == 1 for f in nicel]
        if all(orta) and merkez_var:
            return 0
        if any(orta):
            return None
        return 1

    if merkez_var and any(len(f["seviye"]) == 2 for f in nicel):
        return None, "bazı nicel faktörler 3 seviyeli, bazıları 2 seviyeli"
    turler = [tur(s) for s in satir]
    if any(t is None for t in turler):
        return None, "orta seviye sadece merkez noktalarında olmalı"
    kose = [s for s, t in zip(satir, turler) if t == 1]
    kombs = sorted({tuple(0 if _idx(f, s["deg"][f["g"]["isim"]]) == 0 else 1 for f in faktorler)
                    for s in kose})
    k, n = len(faktorler), len(kombs)
    M = np.array([[2 * x - 1 for x in kmb] for kmb in kombs])
    merkez_n = turler.count(0)
    fms = [_faktor_meta(f) for f in faktorler]
    pt = lambda s: tur(s)
    if n == 2 ** k:
        tip, ad = "faktoriyel2", f"2 seviyeli tam faktöriyel (2^{k})"
    elif n >= k + 1 and _ortogonal(M):
        if n & (n - 1) == 0:
            p = int(round(k - math.log2(n)))
            tip, ad = "kesirli", f"2 seviyeli kesirli faktöriyel (2^({k}-{p}))"
        else:
            tip, ad = "pb", f"Plackett-Burman ({n} deney)"
    else:
        eksik = 2 ** k - n
        return None, (f"köşe kombinasyonları ne tam ne de ortogonal kesir "
                      f"({n}/{2 ** k} kombinasyon; {eksik} eksik)")
    meta = _meta(tip, ad, fms, sabitler, yapi, pt_sutun="CenterPt", merkez=merkez_n)
    if tip == "kesirli":
        meta["karisiklik"] = _karisiklik(M, [f["harf"] for f in fms])
    tekrar = len(kose) / n
    ozet = (f"{n} farklı köşe" + (f" × ~{tekrar:.1f} tekrar" if tekrar > 1.01 else "")
            + (f" + {merkez_n} merkez noktası" if merkez_n else ""))
    return (meta, _satirlar(satir, pt), ozet), None


def _karisiklik(M, harfler, derece=3):
    """Kodlu matristen karışıklık (alias) listesi: aynı sütunu veren etkileşimler."""
    n, k = M.shape
    sutun = {}
    for d in range(1, min(derece, k) + 1):
        for komb in itertools.combinations(range(k), d):
            v = tuple(np.prod(M[:, komb], axis=1).astype(int))
            sutun.setdefault(v, []).append("".join(harfler[i] for i in komb))
            neg = tuple(-x for x in v)
            if neg in sutun and neg != v:
                sutun[neg].append("-" + "".join(harfler[i] for i in komb))
    out = []
    for adlar in sutun.values():
        if len(adlar) > 1 and len(adlar[0].lstrip("-")) <= 2:
            out.append(" + ".join(adlar))
    return out[:60]


def _ccd(faktorler, satir, sabitler, yapi):
    if any(f["g"]["tip"] == "nitel" for f in faktorler):
        return None, "nitel faktör var"
    k = len(faktorler)
    if k < 2:
        return None, "en az 2 faktör gerekir"
    sev = {len(f["seviye"]) for f in faktorler}
    if sev not in ({5}, {3}):
        return None, "her faktör 5 seviyede (CCC/CCI) veya 3 seviyede (CCF) olmalı"
    bes = sev == {5}

    def sinif(s):
        ix = [_idx(f, s["deg"][f["g"]["isim"]]) for f in faktorler]
        if bes:
            if all(i == 2 for i in ix):
                return 0
            if all(i in (1, 3) for i in ix):
                return 1
            if sum(i in (0, 4) for i in ix) == 1 and sum(i == 2 for i in ix) == k - 1:
                return -1
            return None
        if all(i == 1 for i in ix):
            return 0
        if all(i in (0, 2) for i in ix):
            return 1
        if sum(i != 1 for i in ix) == 1:
            return -1
        return None

    sn = [sinif(s) for s in satir]
    if any(x is None for x in sn):
        return None, "bazı kayıtlar köşe / eksenel / merkez noktası değil"
    kose = {tuple(_idx(f, s["deg"][f["g"]["isim"]]) for f in faktorler)
            for s, x in zip(satir, sn) if x == 1}
    eks = {tuple(_idx(f, s["deg"][f["g"]["isim"]]) for f in faktorler)
           for s, x in zip(satir, sn) if x == -1}
    if len(eks) != 2 * k or len(kose) not in (2 ** k, 2 ** (k - 1)) or sn.count(0) < 1:
        return None, (f"{len(kose)} köşe, {len(eks)}/{2 * k} eksenel, {sn.count(0)} merkez — "
                      "CCD yapısı tamamlanmamış")
    if bes:
        fms = [_faktor_meta(f, f["seviye"][1], f["seviye"][3]) for f in faktorler]
        alfa = np.mean([(f["seviye"][4] - f["seviye"][2]) / (f["seviye"][3] - f["seviye"][2])
                        for f in faktorler])
        cesit = f"α = {alfa:.3g}"
    else:
        fms = [_faktor_meta(f) for f in faktorler]
        cesit = "yüzey merkezli (CCF)"
    meta = _meta("ccd", f"Merkezi kompozit — {cesit}", fms, sabitler, yapi)
    pt = lambda s: sinif(s)
    return (meta, _satirlar(satir, pt),
            f"{len(kose)} köşe + {len(eks)} eksenel + {sn.count(0)} merkez noktası"), None


def _bbd(faktorler, satir, sabitler, yapi):
    if any(f["g"]["tip"] == "nitel" for f in faktorler):
        return None, "nitel faktör var"
    k = len(faktorler)
    if k < 3 or any(len(f["seviye"]) != 3 for f in faktorler):
        return None, "en az 3 faktör, her biri 3 seviyede olmalı"
    m_say = set()

    def sinif(s):
        ix = [_idx(f, s["deg"][f["g"]["isim"]]) for f in faktorler]
        m = sum(i != 1 for i in ix)
        if m == 0:
            return 0
        m_say.add(m)
        return 2

    sn = [sinif(s) for s in satir]
    if sn.count(0) < 1 or len(m_say) != 1 or next(iter(m_say)) >= k:
        return None, "kenar orta noktaları + merkez yapısı yok (köşe noktaları var)"
    kenar = {tuple(_idx(f, s["deg"][f["g"]["isim"]]) for f in faktorler)
             for s, x in zip(satir, sn) if x == 2}
    beklenen = {3: 12, 4: 24, 5: 40}.get(k)
    if beklenen and len(kenar) != beklenen:
        return None, f"{len(kenar)}/{beklenen} kenar noktası"
    meta = _meta("bbd", "Box-Behnken", [_faktor_meta(f) for f in faktorler], sabitler, yapi)
    return (meta, _satirlar(satir, sinif), f"{len(kenar)} kenar + {sn.count(0)} merkez noktası"), None


def _genel(faktorler, satir, sabitler, yapi):
    if any(len(f["seviye"]) > MAKS_SEVIYE for f in faktorler):
        return None, f"bir faktör {MAKS_SEVIYE}'ten fazla seviyede"
    if all(len(f["seviye"]) == 2 for f in faktorler):
        return None, "tüm faktörler 2 seviyeli (2 seviyeli faktöriyel olarak incelenir)"
    kombs = {tuple(_idx(f, s["deg"][f["g"]["isim"]]) for f in faktorler) for s in satir}
    tam = int(np.prod([len(f["seviye"]) for f in faktorler]))
    if len(kombs) != tam:
        return None, f"{len(kombs)}/{tam} seviye kombinasyonu var"
    fms = [_faktor_meta(f, sayi_seviye=True) for f in faktorler]
    meta = _meta("genel", "Genel tam faktöriyel", fms, sabitler, yapi)
    tekrar = len(satir) / tam
    return (meta, _satirlar(satir),
            " × ".join(str(len(f["seviye"])) for f in faktorler) + f" = {tam} kombinasyon"
            + (f", ~{tekrar:.1f} tekrar" if tekrar > 1.01 else "")), None


def _taguchi(faktorler, satir, sabitler, yapi):
    if any(len(f["seviye"]) not in (2, 3) for f in faktorler):
        return None, "her faktör 2 veya 3 seviyeli olmalı"
    gruplar = {}
    for s in satir:
        key = tuple(_idx(f, s["deg"][f["g"]["isim"]]) for f in faktorler)
        gruplar.setdefault(key, []).append(s)
    kombs = list(gruplar)
    if not _dengeli_dizi(kombs, faktorler):
        return None, "kombinasyonlar ortogonal dizi oluşturmuyor (dengesiz)"
    n = len(kombs)
    n2 = sum(len(f["seviye"]) == 2 for f in faktorler)
    n3 = len(faktorler) - n2
    dizi = next((a for a, (N, m2, m3) in T.TAGUCHI_DIZILER.items()
                 if N == n and n2 <= m2 and n3 <= m3), f"OA{n}")
    r = min(len(v) for v in gruplar.values())
    fms = [_faktor_meta(f, sayi_seviye=True) for f in faktorler]
    if r >= 2:
        meta = _meta("taguchi", f"Taguchi {dizi} (tekrarlı)", fms, sabitler, yapi,
                     pt_sutun=None, dizi=dizi, tekrar_sutun=r)
        sat = []
        for i, (key, grup) in enumerate(gruplar.items()):
            y = {}
            for c in yapi["ciktilar"]:
                for j in range(r):
                    y[f"{c['isim']}_{j + 1}"] = grup[j]["y"].get(c["isim"])
                y[c["isim"]] = grup[0]["y"].get(c["isim"])
            sat.append({"StdOrder": i + 1, "RunOrder": i + 1, "pt": 1, "blok": 1,
                        "deg": dict(grup[0]["deg"]), "y": y})
        return (meta, _satirlar(satir), f"{n} deneylik ortogonal dizi, her deneyde {r} tekrar "
                "ölçüm (S/N hesaplanabilir)", sat), None
    else:
        meta = _meta("taguchi", f"Taguchi {dizi}", fms, sabitler, yapi,
                     pt_sutun=None, dizi=dizi, tekrar_sutun=1)
        sat = _satirlar(satir)
    return (meta, sat, f"{n} deneylik ortogonal dizi"), None


def _karisim(faktorler, satir, sabitler, yapi):
    nicel = [f for f in faktorler if f["g"]["tip"] == "nicel"]
    if len(nicel) != len(faktorler) or len(nicel) < 3:
        return None, "en az 3 nicel bileşen gerekir"
    toplam = [sum(float(s["deg"][f["g"]["isim"]]) for f in nicel) for s in satir]
    ort = float(np.mean(toplam))
    if ort <= 0 or np.std(toplam) > 0.005 * abs(ort):
        return None, "girdilerin toplamı sabit değil (karışım değil)"
    fms = []
    for f in nicel:
        fm = _faktor_meta(f)
        fm["kar_alt"] = 0.0
        fms.append(fm)
    meta = _meta("karisim", "Karışım (kayıtlı oranlar)", fms, sabitler, yapi, toplam=ort, kalan=ort)
    return (meta, _satirlar(satir), f"bileşen toplamı ≈ {ort:g}"), None


def _ozel(faktorler, satir, sabitler, yapi):
    fms = []
    for f in faktorler:
        if f["g"]["tip"] == "nicel":
            v = [s["ham"][f["g"]["isim"]] for s in satir]
            fms.append(_faktor_meta(f, min(v), max(v)))
        else:
            fms.append(_faktor_meta(f))
    nicel3 = sum(1 for f in faktorler if f["g"]["tip"] == "nicel" and len(f["seviye"]) >= 3)
    meta = _meta("ozel", "Regresyon (özel tasarım)", fms, sabitler, yapi)
    sat = [{"StdOrder": i + 1, "RunOrder": i + 1, "pt": 1, "blok": 1, "deg": dict(s["ham"]),
            "y": dict(s["y"])} for i, s in enumerate(satir)]
    modeller = (["kuadratik"] if nicel3 else []) + ["etkilesim", "dogrusal"]
    meta["oto_model"] = "dogrusal"
    for m in modeller:
        meta["oto_model"] = m
        if _parametre(meta, sat, A.varsayilan_terimler(meta, m)) <= len(sat) - 2:
            break
    adlar = {"kuadratik": "tam kuadratik", "etkilesim": "doğrusal + etkileşim", "dogrusal": "doğrusal"}
    meta["ad"] = f"Regresyon (özel tasarım, {adlar[meta['oto_model']]} model)"
    return (meta, sat,
            f"{len({tuple(s['deg'].values()) for s in satir})} farklı ayar noktası"), None


KURALLAR = [("faktoriyel", _iki_seviyeli), ("ccd", _ccd), ("bbd", _bbd), ("genel", _genel),
            ("taguchi", _taguchi), ("karisim", _karisim), ("ozel", _ozel)]


# ════════════════════════════════════════════════════════════════
# Çıktı kontrolü + ana fonksiyon
# ════════════════════════════════════════════════════════════════
def _terimler(meta, sat):
    aile = A._aile(meta)
    k = len(meta["faktorler"])
    merkez = any(s["pt"] == 0 for s in sat) and not all(s["pt"] == 0 for s in sat)
    if aile == "faktoriyel":
        return A.varsayilan_terimler(meta, None, 1 if meta["tip"] == "pb" else min(2, k), merkez)
    if aile in ("genel", "taguchi"):
        return A.varsayilan_terimler(meta, None, 1 if aile == "taguchi" else min(2, k))
    if aile == "karisim":
        return A.varsayilan_terimler(meta, "kuadratik")
    return A.varsayilan_terimler(meta, meta.get("oto_model", "kuadratik"))


def _parametre(meta, sat, ter):
    X, _, _ = A.model_matrisi(meta, [s["deg"] for s in sat], ter, pt=[s["pt"] for s in sat],
                              blok=[1] * len(sat), bloklar=[1])
    X = X[~np.isnan(X).any(axis=1)]
    return int(np.linalg.matrix_rank(X)) if len(X) else X.shape[1]


def _cikti_kontrol(meta, sat, yapi):
    """Her nicel çıktı için analiz edilebilir mi? [(isim, n_sonuç, gerekli, uygun)]"""
    ter = _terimler(meta, sat)
    p = _parametre(meta, sat, ter)
    out = []
    for c in yapi["ciktilar"]:
        if c["tip"] != "nicel":
            continue
        n = sum(1 for s in sat if A._sayi(s["y"].get(c["isim"])) is not None)
        out.append({"isim": c["isim"], "n": n, "p": p, "uygun": n > p or
                    (n == p and A._aile(meta) == "faktoriyel")})
    return out


def tani(df, yapi):
    """
    Dönüş: {'adaylar': [{'kod','ad','ozet','meta','satirlar','ciktilar','uygun'}],
            'reddedilen': [(kod, neden)], 'sabitler', 'nitel_ciktilar', 'n'}
    """
    sonuc = {"adaylar": [], "reddedilen": [], "sabitler": {}, "n": 0,
             "nitel_ciktilar": [c["isim"] for c in yapi["ciktilar"] if c["tip"] == "nitel"]}
    if df is None or len(df) < 4 or not yapi["girdiler"]:
        return sonuc
    faktorler, sabitler, satir = _hazirla(df, yapi)
    sonuc["sabitler"], sonuc["n"] = sabitler, len(satir)
    if not faktorler or len(satir) < 4:
        return sonuc
    for kod, fn in KURALLAR:
        try:
            r, neden = fn(faktorler, satir, sabitler, yapi)
        except Exception as e:  # tanıma hiçbir zaman uygulamayı durdurmamalı
            r, neden = None, f"{type(e).__name__}: {e}"
        if r is None:
            sonuc["reddedilen"].append((kod, neden))
            continue
        meta, sat, ozet = r[:3]
        sn_sat = r[3] if len(r) > 3 else None
        try:
            cik = _cikti_kontrol(meta, sat, yapi)
        except Exception:
            cik = []
        uygun = any(c["uygun"] for c in cik)
        sonuc["adaylar"].append({"kod": kod, "ad": meta["ad"], "ozet": ozet, "meta": meta,
                                 "satirlar": sat, "sn_satirlar": sn_sat or sat,
                                 "ciktilar": cik, "uygun": uygun})
    return sonuc


def ozet_mesaj(t):
    uy = [a for a in t["adaylar"] if a["uygun"] and a["kod"] != "ozel"]
    if uy:
        adlar = " / ".join(a["ad"] for a in uy)
        return (f"Kayıtlı {t['n']} deney **{adlar}** düzenine uyuyor — yeni çıktı girmeden "
                "otomatik DOE analizi uygulanabilir.")
    oz = next((a for a in t["adaylar"] if a["kod"] == "ozel" and a["uygun"]), None)
    if oz:
        return (f"Kayıtlı {t['n']} deney standart bir DOE düzeninde değil; **regresyon (özel "
                "tasarım)** ile otomatik analiz uygulanabilir.")
    return None


# ════════════════════════════════════════════════════════════════
# Otomatik uygulama: tüm nicel çıktıları analiz et + optimize et
# ════════════════════════════════════════════════════════════════
def uygula(aday, yapi, alfa=0.05):
    meta, sat = aday["meta"], aday["satirlar"]
    aile = A._aile(meta)
    ter = _terimler(meta, sat)
    sonuclar, hatalar, taguchi = {}, [], {}
    for c in aday["ciktilar"]:
        if not c["uygun"]:
            hatalar.append(f"'{c['isim']}': {c['n']} sonuç, model {c['p']} katsayı — atlandı.")
            continue
        try:
            if aile in ("faktoriyel", "yuzey", "genel", "karisim") or meta["tip"] == "ozel":
                s = A.geri_eleme(meta, sat, c["isim"], ter, alfa)
            else:
                s = A.analiz(meta, sat, c["isim"], ter, alfa)
            s["kaynak"] = "oto"
            sonuclar[c["isim"]] = s
        except Exception as e:
            hatalar.append(f"'{c['isim']}': {e}")
        if aile == "taguchi":
            hedef_tipi = _sn_turu(meta, sat, c["isim"], yapi)
            try:
                taguchi[c["isim"]] = A.taguchi_analiz(meta, aday["sn_satirlar"], c["isim"], hedef_tipi)
            except Exception as e:
                hatalar.append(f"'{c['isim']}' S/N: {e}")
    hedefler = {}
    for y in sonuclar:
        c = next(x for x in yapi["ciktilar"] if x["isim"] == y)
        if c.get("hedef_min") is not None and c.get("hedef_max") is not None:
            hedefler[y] = (c["hedef_min"], c["hedef_max"])
    opt = None
    if hedefler:
        try:
            opt = A.optimize({y: sonuclar[y] for y in hedefler}, hedefler)
        except Exception as e:
            hatalar.append(f"Optimizasyon: {e}")
    elif sonuclar:
        hatalar.append("Nicel çıktılar için ⚙️ Yapı'da Hedef_Min / Hedef_Max tanımlı değil — "
                       "optimizasyon atlandı.")
    return {"aday": {k: aday[k] for k in ("kod", "ad", "ozet")}, "meta": meta, "sonuclar": sonuclar,
            "taguchi": taguchi, "opt": opt, "hedefler": hedefler, "hatalar": hatalar}


def _sn_turu(meta, sat, cikti, yapi):
    """S/N türü: tekrar ölçüm varsa 'nominal en iyi'; yoksa hedef aralığı verinin
    ortalamasının üstündeyse 'büyük daha iyi', altındaysa 'küçük daha iyi'."""
    if meta.get("tekrar_sutun", 1) >= 2:
        return "nominal"
    c = next(x for x in yapi["ciktilar"] if x["isim"] == cikti)
    lo, hi = c.get("hedef_min"), c.get("hedef_max")
    v = [A._sayi(s["y"].get(cikti)) for s in sat]
    v = [x for x in v if x is not None]
    if lo is None or hi is None or not v:
        return "buyuk"
    return "buyuk" if (lo + hi) / 2 >= float(np.mean(v)) else "kucuk"
