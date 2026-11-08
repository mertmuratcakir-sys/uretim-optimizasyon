"""
doe_ui.py — 🧪 DOE sekmesi (Streamlit)
=======================================
Alt sekmeler: Uygunluk · Tasarım · Deneyler · Analiz · Optimizasyon
Hiçbir hesap kendiliğinden çalışmaz; her adım bir butonla başlatılır.
"""
from __future__ import annotations

import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import doe_analiz as A
import doe_plots as DP
import doe_tani as DT
import doe_tasarim as T


# ════════════════════════════════════════════════════════════════
# Kayıt / yükleme
# ════════════════════════════════════════════════════════════════
def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray, set, frozenset, tuple)):
        return list(o)
    return str(o)


def yukle(depo):
    """Çalışmanın DOE planını okur: {'meta','rows'} veya None."""
    try:
        bilgi = depo.oku("DOE_Bilgi")
        if len(bilgi) < 2 or not bilgi[1] or not str(bilgi[1][0]).startswith("{"):
            return None
        meta = json.loads(bilgi[1][0])
        return {"meta": meta, "rows": depo.oku("DOE")}
    except Exception:
        return None


def kaydet(depo, meta, satirlar):
    rows = T.calisma_sayfasi(meta, satirlar, meta["ciktilar"])
    depo.yaz("DOE", rows)
    bilgi = [["DOE tasarım bilgisi — A2 hücresi uygulama tarafından kullanılır, düzenlemeyin."],
             [json.dumps(meta, default=_json, ensure_ascii=False)], [],
             ["TASARIM ÖZETİ"]] + [[k, str(v)] for k, v in meta.get("ozet", [])]
    if meta.get("karisiklik"):
        bilgi += [[], ["KARIŞIKLIK YAPISI (Alias Structure)"]] + [[a] for a in meta["karisiklik"]]
    if meta.get("sabitler"):
        bilgi += [[], ["SABİT TUTULAN GİRDİLER"]] + [[k, str(v)] for k, v in meta["sabitler"].items()]
    depo.yaz("DOE_Bilgi", bilgi)
    return {"meta": json.loads(json.dumps(meta, default=_json)), "rows": rows}


def plan_satirlari(doe):
    meta = doe["meta"]
    return A.calisma_oku(doe["rows"], meta, T.yanit_sutunlari(meta, meta["ciktilar"]))


# ════════════════════════════════════════════════════════════════
# Bilgi mesajları (sayfanın üstünde)
# ════════════════════════════════════════════════════════════════
def banner(st, P, YAPI):
    doe = P.get("doe")
    if doe:
        meta = doe["meta"]
        sat = plan_satirlari(doe)
        nicel = [c["isim"] for c in YAPI["ciktilar"] if c["tip"] == "nicel"
                 and c["isim"] in meta["ciktilar"]]
        if meta["tip"] == "taguchi":
            sut = T.yanit_sutunlari(meta, nicel[:1]) if nicel else []
        else:
            sut = nicel[:1]
        if not sut:
            return
        tam, kismi, n = A.ilerleme(sat, sut)
        if tam == n and n:
            st.success(f"🧪 DOE planındaki {n} deneyin tamamı sonuçlandı — **🧪 DOE → Analiz** "
                       "sekmesinden analiz edebilirsiniz.")
        elif tam >= max(3, min_deney(meta, sat)):
            st.info(f"🧪 DOE: {tam}/{n} deney sonuçlandı — analiz şimdiden çalıştırılabilir; "
                    "eksik deneyler tamamlandıkça sonuçlar güçlenir.")
        else:
            st.caption(f"🧪 DOE: {tam}/{n} deney sonuçlandı.")
    t = tani_al(P, YAPI)
    if t and DT.ozet_mesaj(t) and not st.session_state.get("doe_oto"):
        st.info("🧪 " + DT.ozet_mesaj(t) + " Yan paneldeki **🧪 DOE asistanı**ndan "
                "tek tuşla uygulayabilirsiniz.")


def min_deney(meta, sat):
    """Varsayılan modelin parametre sayısı + 1."""
    try:
        aile = A._aile(meta)
        if aile == "taguchi":
            return 3
        model = {"faktoriyel": None, "genel": None}.get(aile, "dogrusal")
        ter = A.varsayilan_terimler(meta, model, derece=1)
        X, _, _ = A.model_matrisi(meta, [s["deg"] for s in sat[:1]], ter, pt=[1], blok=[1])
        return X.shape[1] + 1
    except Exception:
        return 3


# ════════════════════════════════════════════════════════════════
# Yardımcılar
# ════════════════════════════════════════════════════════════════
def _f(v, p=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "*"
    if p:
        return f"{v:.3f}"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.4g}" if abs(v) < 1e5 else f"{v:.6g}"


def _tablo(rows, sutunlar, p_sutun=("P",), tam=()):
    out = []
    for r in rows:
        out.append({c: (r.get(c) if c in tam or isinstance(r.get(c), str)
                        else _f(r.get(c), c in p_sutun)) for c in sutunlar})
    return pd.DataFrame(out)


def ciz(st, fig):
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)


def _mesajlar(st, mesajlar):
    for tur, m in mesajlar:
        getattr(st, tur if tur in ("info", "warning", "success", "error") else "info")(m)


def ozel_meta(YAPI):
    fs = []
    for g in YAPI["girdiler"]:
        if g["tip"] == "nicel":
            f = T.faktor(g["isim"], "nicel", g["min"], g["max"])
        else:
            f = T.faktor(g["isim"], "nitel", seviyeler=g["kategoriler"] or [])
        fs.append(f)
    meta = {"tip": "ozel", "ad": "Özel tasarım (Veriler sayfası)", "faktorler": T._harfle(fs),
            "pt_sutun": None, "ciktilar": [c["isim"] for c in YAPI["ciktilar"]], "sabitler": {},
            "blok_sayisi": 1}
    return meta


def ozel_satirlar(df, YAPI):
    out = []
    for i, (_, r) in enumerate(df.iterrows()):
        deg = {}
        for g in YAPI["girdiler"]:
            v = r.get(g["isim"])
            if g["tip"] == "nicel":
                deg[g["isim"]] = None if v is None or pd.isna(v) else float(v)
            else:
                deg[g["isim"]] = None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
        y = {c["isim"]: (None if r.get(c["isim"]) is None or
                         (isinstance(r.get(c["isim"]), float) and pd.isna(r.get(c["isim"])))
                         else r.get(c["isim"])) for c in YAPI["ciktilar"]}
        out.append({"StdOrder": i + 1, "RunOrder": i + 1, "pt": 1, "blok": 1, "deg": deg, "y": y})
    return out


# ════════════════════════════════════════════════════════════════
# Ana sekme
# ════════════════════════════════════════════════════════════════
def goster(st, ctx):
    P, DEPO, YAPI = ctx["P"], ctx["DEPO"], ctx["YAPI"]
    if DEPO is None or not YAPI["girdiler"]:
        st.info("Önce bir çalışma ve yapısını tanımlayın.")
        return
    if not st.session_state.get("doe_acik"):
        st.markdown(
            "**Deney tasarımı (DOE) modülü** — Minitab'daki gibi tasarım oluşturur (çalışma "
            "sayfası Sheets'e yazılır), sonuçları analiz eder ve optimum ayarı bulur. "
            "Siz açmadıkça hiçbir hesap çalışmaz.")
        doe = P.get("doe")
        if doe:
            st.caption(f"Bu çalışmada kayıtlı plan: **{doe['meta']['ad']}**")
        if st.button("🧪 DOE modülünü aç", type="primary"):
            st.session_state["doe_acik"] = True
            st.rerun()
        return
    if st.button("Kapat", key="doe_kapat"):
        st.session_state["doe_acik"] = False
        st.rerun()
    m = st.session_state.pop("doe_mesaj", None)
    if m:
        (st.error if m.startswith("❌") else st.success)(m)
    ctx = {**ctx, "sec": _secili_faktorler(st, YAPI)}
    s0, s1, s2, s3, s4, s5 = st.tabs(["🤖 Otomatik", "Uygunluk", "Tasarım", "Deneyler",
                                      "Analiz", "Optimizasyon"])
    with s0:
        _otomatik(st, ctx)
    with s1:
        _uygunluk(st, ctx)
    with s2:
        _tasarim(st, ctx)
    with s3:
        _deneyler(st, ctx)
    with s4:
        _analiz(st, ctx)
    with s5:
        _optimizasyon(st, ctx)


# ── 1. Uygunluk ─────────────────────────────────────────────────
def _secili_faktorler(st, YAPI, anahtar="doe_fsec"):
    adlar = [g["isim"] for g in YAPI["girdiler"]]
    if st.session_state.get(anahtar) is None or \
            any(a not in adlar for a in st.session_state[anahtar]):
        st.session_state[anahtar] = adlar
    return st.multiselect("DOE faktörleri (seçilmeyenler deney boyunca sabit tutulur)",
                          adlar, key=anahtar)


def _uygunluk(st, ctx):
    YAPI, P = ctx["YAPI"], ctx["P"]
    sec = ctx["sec"]
    fs = [{"isim": g["isim"], "tip": g["tip"], "n_kat": len(g["kategoriler"] or [])}
          for g in YAPI["girdiler"] if g["isim"] in sec]
    if not fs:
        st.info("En az bir faktör seçin.")
        return
    rows, oneri = A.uygunluk(fs)
    st.markdown(f"**Öneri:** {oneri}")
    st.dataframe(pd.DataFrame([{k: r[k] for k in ("Tasarım", "Durum", "Deney", "Açıklama")}
                               for r in rows]).astype(str),
                 hide_index=True, use_container_width=True)
    st.caption("✅ uygun · ⚠️ kullanılabilir ama dikkat · ❌ bu faktör yapısıyla kurulamaz")
    t = tani_al(P, YAPI)
    m = DT.ozet_mesaj(t) if t else None
    if m:
        st.info("Mevcut kayıtlar: " + m + " → **🤖 Otomatik** sekmesi.")


# ── 2. Tasarım ──────────────────────────────────────────────────
def _faktor_tablosu(st, YAPI, sec, tip):
    anahtar = f"doe_ftab_{tip}_{'_'.join(sec)}"
    satir = []
    for g in YAPI["girdiler"]:
        if g["isim"] not in sec:
            continue
        if g["tip"] == "nicel":
            lo, hi = (0, 1) if tip == "karisim" else (g["min"], g["max"])
            satir.append({"Faktör": g["isim"], "Tip": "nicel", "Düşük": f"{lo:g}",
                          "Yüksek": f"{hi:g}", "Seviyeler (nitel)": "",
                          "Seviye sayısı": "3" if tip == "genel" else ("2" if tip == "taguchi" else "")})
        else:
            satir.append({"Faktör": g["isim"], "Tip": "nitel", "Düşük": "", "Yüksek": "",
                          "Seviyeler (nitel)": ",".join(g["kategoriler"] or []), "Seviye sayısı": ""})
    df = pd.DataFrame(satir)
    yardim = {"faktoriyel2": "2 seviyeli tasarımlarda nitel faktörler için tam 2 kategori yazın.",
              "kesirli": "2 seviyeli tasarımlarda nitel faktörler için tam 2 kategori yazın.",
              "pb": "2 seviyeli tasarımlarda nitel faktörler için tam 2 kategori yazın.",
              "genel": "Nicel faktörler için 'Seviye sayısı' (2–5) belirleyin.",
              "taguchi": "Nicel faktörler için 'Seviye sayısı' 2 veya 3; nitel için 2–3 kategori.",
              "ccd": "CCI'da Düşük/Yüksek eksenel noktalardır; CCC/CCF'de küp noktalarıdır.",
              "karisim": "Karışım: faktörler bileşen oranlarıdır, toplamları sabittir. Düşük = "
                         "bileşenin alt sınırı (alt sınırların toplamı karışım toplamından küçük olmalı)."}.get(tip, "")
    if yardim:
        st.caption(yardim)
    return st.data_editor(df, key=anahtar, hide_index=True, use_container_width=True,
                          disabled=["Faktör", "Tip"])


def _faktorler_kur(YAPI, tablo):
    fs = []
    for _, r in tablo.iterrows():
        g = next(x for x in YAPI["girdiler"] if x["isim"] == r["Faktör"])
        if g["tip"] == "nicel":
            lo, hi = A._sayi(r["Düşük"]), A._sayi(r["Yüksek"])
            if lo is None or hi is None or lo == hi:
                raise ValueError(f"'{g['isim']}' için geçerli Düşük/Yüksek girin.")
            if lo > hi:
                lo, hi = hi, lo
            f = T.faktor(g["isim"], "nicel", lo, hi)
            f["sinir_min"], f["sinir_max"] = g["min"], g["max"]
            n = A._sayi(r.get("Seviye sayısı"))
            if n:
                f["seviye_sayisi"] = int(n)
                f["taguchi_seviye"] = int(n)
        else:
            sev = [s.strip() for s in str(r["Seviyeler (nitel)"]).split(",") if s.strip()]
            if len(sev) < 2:
                raise ValueError(f"'{g['isim']}' için en az 2 kategori yazın.")
            f = T.faktor(g["isim"], "nitel", seviyeler=sev)
        fs.append(f)
    return fs


def _sabitler(st, YAPI, sec):
    diger = [g for g in YAPI["girdiler"] if g["isim"] not in sec]
    if not diger:
        return {}
    st.markdown("**Sabit tutulan girdiler**")
    df = pd.DataFrame([{"Girdi": g["isim"],
                        "Sabit değer": (f"{(g['min'] + g['max']) / 2:g}" if g["tip"] == "nicel"
                                        else (g["kategoriler"] or [""])[0])} for g in diger])
    t = st.data_editor(df, key=f"doe_sabit_{'_'.join(sec)}", hide_index=True,
                       use_container_width=True, disabled=["Girdi"])
    out = {}
    for _, r in t.iterrows():
        g = next(x for x in YAPI["girdiler"] if x["isim"] == r["Girdi"])
        out[g["isim"]] = A._sayi(r["Sabit değer"]) if g["tip"] == "nicel" else str(r["Sabit değer"])
    return out


def _tasarim(st, ctx):
    YAPI, P, DEPO = ctx["YAPI"], ctx["P"], ctx["DEPO"]
    sec = ctx["sec"]
    if not sec:
        st.info("En az bir faktör seçin.")
        return
    fs_kisa = [{"isim": g["isim"], "tip": g["tip"], "n_kat": len(g["kategoriler"] or [])}
               for g in YAPI["girdiler"] if g["isim"] in sec]
    uyg = {r["kod"]: r for r in A.uygunluk(fs_kisa)[0]}
    kodlar = list(T.TASARIM_ADLARI)
    tip = st.selectbox("Tasarım türü", kodlar, key="doe_tip",
                       format_func=lambda k: f"{uyg[k]['Durum']} {T.TASARIM_ADLARI[k]}  "
                                             f"({uyg[k]['Deney']} deney)")
    durum = uyg[tip]
    (st.error if durum["Durum"] == "❌" else st.warning if durum["Durum"] == "⚠️" else st.info)(
        durum["Açıklama"])
    tablo = _faktor_tablosu(st, YAPI, sec, tip)
    sabit = _sabitler(st, YAPI, sec)
    k = len(sec)
    kc = sum(1 for g in YAPI["girdiler"] if g["isim"] in sec and g["tip"] == "nicel")
    sec_ = {}
    st.markdown("**Seçenekler**")
    c1, c2 = st.columns(2)
    if tip in ("faktoriyel2", "kesirli", "pb"):
        sec_["merkez"] = int(c1.number_input("Merkez nokta (blok başına)", 0, 20, 0, key="doe_mz"))
    if tip == "kesirli":
        ops = T.kesirli_secenekler(k) if 3 <= k <= 11 else []
        if ops:
            o = c2.selectbox("Kesir", ops, format_func=lambda d: d["ad"], key="doe_kes")
            sec_["p"] = o["p"]
    if tip == "pb":
        ops = T.pb_deney_sayisi(k)
        if ops:
            sec_["n"] = c2.selectbox("Deney sayısı", ops, key="doe_pbn")
    if tip == "ccd":
        sec_["cesit"] = c1.radio("CCD türü", ["CCC", "CCI", "CCF"], index=1, key="doe_ccdt",
                                 help="Minitab varsayılanı CCC'dir; ancak CCC eksenel noktaları "
                                      "Düşük/Yüksek aralığının dışına taşar. Aralık makine sınırıysa "
                                      "CCI (içsel) veya CCF seçin.",
                                 format_func=lambda x: {"CCC": "Çevrel (CCC) — dönebilir",
                                                        "CCI": "İçsel (CCI) — sınır içinde",
                                                        "CCF": "Yüzey merkezli (CCF) — 3 seviye"}[x])
        sec_["yarim"] = c2.checkbox("Küp için yarım kesir (5+ faktör)", value=kc >= 5,
                                    key="doe_ccdy", disabled=kc < 5)
        vars_ = (T.CCD_MERKEZ_YARIM if sec_["yarim"] and kc >= 5 else T.CCD_MERKEZ).get(kc, 6)
        sec_["merkez"] = int(c2.number_input("Merkez nokta", 1, 30, vars_, key=f"doe_ccdm{kc}"))
    if tip == "bbd":
        sec_["merkez"] = int(c1.number_input("Merkez nokta", 1, 20, T.BBD_MERKEZ.get(kc, 3),
                                             key=f"doe_bbm{kc}"))
    if tip == "dsd":
        sec_["ek_merkez"] = int(c1.number_input("Ek merkez nokta", 0, 10, 0, key="doe_dsdm"))
    if tip == "taguchi":
        sec_["tekrar_sutun"] = int(c1.number_input(
            "Tekrar ölçümü (gürültü koşulu) sayısı", 1, 6, 1, key="doe_tgr",
            help="'Nominal en iyi' S/N oranı için en az 2 gerekir. Her deney satırında "
                 "Y_1, Y_2, ... sütunları açılır."))
        try:
            tf = _faktorler_kur(YAPI, tablo)
            sev = [int(f.get("taguchi_seviye") or 2) if f["tip"] == "nicel" else len(f["seviyeler"])
                   for f in tf]
            ops = T.taguchi_oner(sev.count(2), sev.count(3))
        except ValueError:
            ops = []
        if ops:
            sec_["dizi"] = c2.selectbox("Ortogonal dizi", ops, key="doe_tgd")
    if tip == "karisim":
        cesit = c1.radio("Tür", ["centroid", "lattice"], key="doe_kc",
                         format_func=lambda x: {"centroid": "Simplex centroid",
                                                "lattice": "Simplex lattice"}[x])
        sec_["cesit"] = cesit
        if cesit == "lattice":
            sec_["derece"] = int(c2.selectbox("Lattice derecesi", [2, 3, 4], key="doe_kd"))
        sec_["eksenel"] = c2.checkbox("Eksenel noktalar ekle", True, key="doe_ke")
        sec_["merkez_ekle"] = c2.checkbox("Merkez (centroid) ekle", True, key="doe_km")
        sec_["toplam"] = float(c1.number_input("Karışım toplamı", 0.0001, 1e6, 1.0, key="doe_kt"))
    if tip == "doptimal":
        model = c1.radio("Model", ["dogrusal", "etkilesim", "kuadratik"], index=2, key="doe_dm",
                         format_func=lambda x: {"dogrusal": "Doğrusal", "etkilesim": "Doğrusal + etkileşim",
                                                "kuadratik": "Tam kuadratik"}[x])
        sec_["model"] = model
        sec_["n"] = int(c2.number_input("Deney sayısı", 2, 200, 20, key="doe_dn"))
        sec_["merkez"] = int(c2.number_input("Ek merkez nokta", 0, 10, 0, key="doe_dmz"))
    if tip != "taguchi":
        sec_["tekrar"] = int(c1.number_input("Tekrar (replicate)", 1, 10, 1, key="doe_tkr"))
        if sec_["tekrar"] > 1:
            sec_["tekrar_blok"] = c2.checkbox("Her tekrarı ayrı blok yap", False, key="doe_tkb")
        sec_["rastgele"] = c2.checkbox("Deney sırasını rastgeleleştir", True, key="doe_rnd")
    tohum = int(c1.number_input("Rastgele tohum (0 = otomatik)", 0, 99999, 0, key="doe_seed"))
    sec_["tohum"] = tohum or None

    if st.button("🧪 Tasarımı oluştur (önizleme)", type="primary", key="doe_olustur"):
        try:
            fs = _faktorler_kur(YAPI, tablo)
            meta, sat = T.olustur(tip, fs, **sec_)
            meta["ciktilar"] = [c["isim"] for c in YAPI["ciktilar"]]
            meta["sabitler"] = sabit
            meta["olusturma"] = datetime.now().strftime("%d.%m.%Y %H:%M")
            st.session_state["doe_onizleme"] = (meta, sat)
        except Exception as e:
            st.session_state.pop("doe_onizleme", None)
            st.error(f"❌ {e}")
    on = st.session_state.get("doe_onizleme")
    if not on:
        return
    meta, sat = on
    st.markdown(f"#### Tasarım özeti — {meta['ad']}")
    st.dataframe(pd.DataFrame([{"Özellik": a, "Değer": str(b)} for a, b in meta["ozet"]]),
                 hide_index=True, use_container_width=True)
    for u in meta.get("uyarilar", []):
        st.warning(u)
    if meta.get("karisiklik"):
        with st.expander("Karışıklık yapısı (Alias Structure)"):
            st.code("\n".join(meta["karisiklik"][:60]), language=None)
    rows = T.calisma_sayfasi(meta, sat, meta["ciktilar"])
    st.markdown("**Çalışma sayfası (Minitab düzeni, deney sırasına göre)**")
    st.dataframe(pd.DataFrame(rows[1:], columns=rows[0]).astype(str), hide_index=True,
                 use_container_width=True, height=min(420, 38 + 35 * len(rows)))
    mevcut = P.get("doe")
    onay = True
    if mevcut:
        sat_m = plan_satirlari(mevcut)
        dolu = A.ilerleme(sat_m, T.yanit_sutunlari(mevcut["meta"], mevcut["meta"]["ciktilar"]))[1]
        st.warning(f"Bu çalışmada zaten bir DOE planı var ({mevcut['meta']['ad']}, "
                   f"{dolu} deneyde sonuç girilmiş). Kaydederseniz eski plan silinir.")
        onay = st.checkbox("Eski planı silmeyi onaylıyorum", key="doe_onay")
    if st.button("💾 Çalışma sayfasına kaydet (Sheets)", disabled=not onay, key="doe_kaydet"):
        P["doe"] = kaydet(DEPO, meta, sat)
        st.session_state.pop("doe_onizleme", None)
        st.session_state.pop("doe_sonuc", None)
        st.session_state["doe_mesaj"] = (f"✅ {meta['ad']} — {meta['deney']} deney "
                                         f"'{DEPO.sekme['DOE']}' sayfasına yazıldı. Deneyleri "
                                         "RunOrder sırasıyla yapıp sonuçları girin.")
        st.rerun()


# ── 3. Deneyler ─────────────────────────────────────────────────
def _deneyler(st, ctx):
    P, DEPO, YAPI = ctx["P"], ctx["DEPO"], ctx["YAPI"]
    doe = P.get("doe")
    if not doe:
        st.info("Henüz DOE planı yok. **Tasarım** sekmesinden oluşturun.")
        return
    meta, rows = doe["meta"], doe["rows"]
    yan = T.yanit_sutunlari(meta, meta["ciktilar"])
    sat = plan_satirlari(doe)
    nicel = [c["isim"] for c in YAPI["ciktilar"] if c["tip"] == "nicel" and c["isim"] in meta["ciktilar"]]
    tam, kismi, n = A.ilerleme(sat, T.yanit_sutunlari(meta, nicel) or yan)
    st.markdown(f"**{meta['ad']}** · oluşturma {meta.get('olusturma', '')}")
    st.progress(tam / n if n else 0.0, text=f"{tam}/{n} deney tamamlandı")
    link = DEPO.url("DOE")
    st.caption("Sonuçları buraya ya da Sheets'teki "
               + (f"[{DEPO.sekme['DOE']}]({link})" if link else f"**{DEPO.sekme['DOE']}**")
               + " sayfasına girebilirsiniz. Deneyleri **RunOrder** sırasıyla yapın.")
    if meta.get("sabitler"):
        st.caption("Sabit: " + ", ".join(f"{k} = {v}" for k, v in meta["sabitler"].items()))
    bas = rows[0]
    gov = [[str(x) for x in (list(r) + [""] * len(bas))[:len(bas)]] for r in rows[1:]]
    df = pd.DataFrame(gov, columns=bas)
    cfg = {}
    for c in YAPI["ciktilar"]:
        if c["tip"] == "nitel":
            for col in [y for y in yan if y == c["isim"] or y.startswith(c["isim"] + "_")]:
                if col in df.columns:
                    cfg[col] = st.column_config.SelectboxColumn(options=[""] + (c["kategoriler"] or []))
    duz = st.data_editor(df, key=f"doe_den_{len(rows)}_{meta.get('olusturma', '')}",
                         hide_index=True, use_container_width=True,
                         disabled=[c for c in df.columns if c not in yan], column_config=cfg)
    c1, c2 = st.columns(2)
    if c1.button("💾 Sonuçları kaydet", type="primary", key="doe_snc"):
        yeni = [bas] + duz.fillna("").astype(str).values.tolist()
        DEPO.yaz("DOE", yeni)
        doe["rows"] = yeni
        st.session_state["doe_mesaj"] = "✅ Sonuçlar DOE sayfasına kaydedildi."
        st.rerun()
    egit = c2.checkbox("Aktarımdan sonra ML modelini yeniden eğit", True, key="doe_ml")
    if c2.button("➡ Tamamlananları Veriler sayfasına aktar", key="doe_aktar",
                 help="Ana modeller (🎯 Öneri) DOE sonuçlarından da öğrensin"):
        st.session_state["doe_mesaj"] = aktar(ctx, egit)
        st.rerun()


def aktar(ctx, egit):
    P, DEPO, YAPI = ctx["P"], ctx["DEPO"], ctx["YAPI"]
    doe = P["doe"]
    meta, rows = doe["meta"], doe["rows"]
    bas = rows[0]
    ix = {h: i for i, h in enumerate(bas)}
    if "Veri_ID" not in ix:
        return "❌ DOE sayfasında Veri_ID sütunu yok."
    r_sut = meta.get("tekrar_sutun", 1) if meta["tip"] == "taguchi" else 1
    vrows = DEPO.oku("Veriler")
    vbas = [str(h).strip() for h in vrows[0]] if vrows else []
    gerekli = ["ID", "Tarih"] + [x["isim"] for x in YAPI["girdiler"] + YAPI["ciktilar"]]
    if not vbas:
        vbas = gerekli
        DEPO.yaz("Veriler", [vbas])
        vrows = [vbas]
    elif any(h not in vbas for h in gerekli):
        vbas = vbas + [h for h in gerekli if h not in vbas]
        DEPO.baslik_yaz("Veriler", vbas)
    idler = [A._sayi(r[0]) for r in vrows[1:] if r]
    sonraki = int(max([i for i in idler if i is not None], default=0)) + 1
    yeni, sayac = [], 0
    zaman = datetime.now().strftime("%d.%m.%Y %H:%M")
    for k, r in enumerate(rows[1:], start=1):
        r = list(r) + [""] * (len(bas) - len(r))
        if str(r[ix["Veri_ID"]]).strip():
            continue
        tekrarlar = range(1, r_sut + 1) if r_sut > 1 else [None]
        idler_bu = []
        for t in tekrarlar:
            kayit = {}
            dolu = False
            for c in meta["ciktilar"]:
                col = f"{c}_{t}" if t else c
                v = r[ix[col]] if col in ix else ""
                if str(v).strip():
                    dolu = True
                kayit[c] = v
            if not dolu:
                continue
            for f in meta["faktorler"]:
                kayit[f["isim"]] = r[ix[f["isim"]]]
            kayit.update({kk: vv for kk, vv in meta.get("sabitler", {}).items()})
            kayit["ID"], kayit["Tarih"] = sonraki, zaman
            idler_bu.append(str(sonraki))
            sonraki += 1
            yeni.append([kayit.get(h, "") if kayit.get(h) is not None else "" for h in vbas])
        if idler_bu:
            rows[k] = r[:ix["Veri_ID"]] + [",".join(idler_bu)] + r[ix["Veri_ID"] + 1:]
            sayac += 1
    if not yeni:
        return "Aktarılacak yeni sonuç yok (sonucu girilmiş ve henüz aktarılmamış deney bulunamadı)."
    DEPO.ekle_coklu("Veriler", yeni)
    DEPO.yaz("DOE", rows)
    doe["rows"] = rows
    m = f"✅ {sayac} deneyden {len(yeni)} kayıt Veriler sayfasına aktarıldı."
    if egit:
        ctx["egit"]()
        m += " ML modeli yeniden eğitildi."
    else:
        ctx["yukle"]()
    return m


# ── 4. Analiz ───────────────────────────────────────────────────
def _kaynak(st, ctx):
    P, YAPI = ctx["P"], ctx["YAPI"]
    ops = []
    if P.get("doe"):
        ops.append("plan")
    if P.get("df") is not None and len(P["df"]) >= 3:
        ops.append("veri")
    if not ops:
        return None, None, None
    k = st.radio("Veri kaynağı", ops, horizontal=True, key="doe_kaynak",
                 format_func=lambda x: {"plan": "DOE planı", "veri": "Veriler sayfası (özel tasarım)"}[x])
    if k == "plan":
        doe = P["doe"]
        return k, doe["meta"], plan_satirlari(doe)
    t = tani_al(P, YAPI)
    uy = [a for a in (t or {}).get("adaylar", []) if a["uygun"]]
    if uy:
        st.caption(f"Veriler sayfası **{uy[0]['ad']}** olarak tanındı ({uy[0]['ozet']}).")
        return k, uy[0]["meta"], uy[0]["satirlar"]
    return k, ozel_meta(YAPI), ozel_satirlar(P["df"], YAPI)


def _aday_terimler(meta, sat):
    aile = A._aile(meta)
    k = len(meta["faktorler"])
    merkez = any(s["pt"] == 0 for s in sat) and not all(s["pt"] == 0 for s in sat)
    blok = len({s["blok"] for s in sat}) > 1
    if aile == "faktoriyel":
        return A.varsayilan_terimler(meta, None, min(k, 4), merkez, blok)
    if aile == "genel":
        return A.varsayilan_terimler(meta, None, min(k, 3), blok_var=blok)
    if aile == "karisim":
        return A.varsayilan_terimler(meta, "ozel_kubik", blok_var=blok)
    return A.varsayilan_terimler(meta, "kuadratik", blok_var=blok)


def _analiz(st, ctx):
    YAPI = ctx["YAPI"]
    kaynak, meta, sat = _kaynak(st, ctx)
    if kaynak is None:
        st.info("Analiz için DOE planı veya Veriler sayfasında en az 3 kayıt gerekir.")
        return
    aile = A._aile(meta)
    nicel_c = [c for c in YAPI["ciktilar"] if c["tip"] == "nicel" and c["isim"] in meta["ciktilar"]]
    if not nicel_c:
        st.warning("Analiz edilecek nicel çıktı yok. Nitel çıktılar DOE regresyon analizine alınmaz; "
                   "bunlar için 🎯 Öneri sekmesindeki sınıflandırma modelleri kullanılır.")
        return
    if aile == "taguchi":
        return _taguchi(st, meta, sat, nicel_c)
    yanit = st.selectbox("Yanıt", [c["isim"] for c in nicel_c], key="doe_yanit")
    tam = sum(1 for s in sat if A._sayi(s["y"].get(yanit)) is not None)
    if tam < 3:
        st.warning(f"Henüz analiz için yeterli sonuç yok: '{yanit}' için {tam} sonuç var.")
        return

    k = len(meta["faktorler"])
    merkez = any(s["pt"] == 0 for s in sat) and not all(s["pt"] == 0 for s in sat)
    blok = len({s["blok"] for s in sat}) > 1
    c1, c2 = st.columns(2)
    if aile in ("faktoriyel", "genel"):
        vd = 1 if meta["tip"] in ("pb",) else min(2, k)
        derece = int(c1.number_input("En yüksek etkileşim derecesi", 1, min(k, 4 if aile == "faktoriyel" else 3),
                                     vd, key=f"doe_der_{kaynak}"))
        terimler = A.varsayilan_terimler(meta, None, derece, merkez and aile == "faktoriyel", blok)
    elif aile == "karisim":
        model = c1.radio("Model", ["dogrusal", "kuadratik", "ozel_kubik"], index=1, key="doe_kmod",
                         format_func=lambda x: {"dogrusal": "Doğrusal", "kuadratik": "Kuadratik",
                                                "ozel_kubik": "Özel kübik"}[x])
        terimler = A.varsayilan_terimler(meta, model, blok_var=blok)
    else:
        vm = "dogrusal" if meta["tip"] == "dsd" else "kuadratik"
        ops = ["dogrusal", "etkilesim", "kare", "kuadratik"]
        model = c1.radio("Model", ops, index=ops.index(vm), key=f"doe_ymod_{meta['tip']}",
                         format_func=lambda x: {"dogrusal": "Doğrusal", "etkilesim": "Doğrusal + etkileşim",
                                                "kare": "Doğrusal + kare", "kuadratik": "Tam kuadratik"}[x])
        terimler = A.varsayilan_terimler(meta, model, blok_var=blok)
    yontem_ops = ["yok", "geri"] + (["ileri"] if aile in ("yuzey", "faktoriyel") else [])
    yontem = c2.radio("Terim seçimi", yontem_ops, key="doe_yontem",
                      index=yontem_ops.index("ileri") if meta["tip"] == "dsd" else 0,
                      format_func=lambda x: {"yok": "Tüm terimler", "geri": "Geri eleme (α çıkar 0.10)",
                                             "ileri": "İleri seçim (α gir 0.05)"}[x])
    alfa = float(c2.number_input("Anlamlılık düzeyi α", 0.01, 0.20, 0.05, 0.01, key="doe_alfa"))
    adaylar = _aday_terimler(meta, sat)
    with st.expander("Terimleri elle seç (Minitab 'Terms')"):
        adlar = [A.terim_adi(t) for t in adaylar]
        harita = dict(zip(adlar, adaylar))
        secili = st.multiselect("Modeldeki terimler", adlar,
                                default=[A.terim_adi(t) for t in terimler if A.terim_adi(t) in harita],
                                key=f"doe_ter_{kaynak}_{meta['tip']}_{yanit}_{len(terimler)}")
        terimler = [harita[a] for a in secili] or terimler
    p = _parametre(meta, sat, terimler)
    if tam >= len(sat) and kaynak == "plan" and tam > p:
        st.success(f"Analiz yapılabilir: planın {tam} deneyinin tamamında '{yanit}' sonucu var "
                   f"(model {p} katsayı).")
    elif tam > p:
        st.info(f"Analiz yapılabilir: {tam}/{len(sat)} kayıtta '{yanit}' sonucu var; seçili model "
                f"{p} katsayı içeriyor (hata için {tam - p} serbestlik derecesi kalır).")
    elif tam == p:
        st.warning(f"{tam} sonuç = {p} katsayı: model doymuş, p-değerleri Lenth yöntemiyle "
                   "yaklaşık hesaplanır. Daha basit model seçin veya deneyleri tamamlayın.")
    else:
        st.warning(f"{tam} sonuç seçili modelin {p} katsayısı için yetersiz — karışan terimler "
                   "otomatik çıkarılır. Daha basit model (ör. Doğrusal) seçin veya deneyleri tamamlayın.")
    if st.button("📈 Analiz et", type="primary", key="doe_analiz"):
        try:
            if yontem == "geri":
                s = A.geri_eleme(meta, sat, yanit, terimler, alfa)
            elif yontem == "ileri":
                temel = [t for t in terimler if len(t) == 1]
                aday = [t for t in terimler if len(t) > 1] or \
                       [t for t in adaylar if len(t) > 1 and t not in (( A.KTS,), (A.BLK,))]
                s = A.ileri_secim(meta, sat, yanit, temel, aday, alfa)
            else:
                s = A.analiz(meta, sat, yanit, terimler, alfa)
            s["kaynak"] = kaynak
            st.session_state.setdefault("doe_sonuc", {})[yanit] = s
        except Exception as e:
            st.error(f"❌ {e}")
    s = (st.session_state.get("doe_sonuc") or {}).get(yanit)
    if s and s.get("kaynak") == kaynak:
        _sonuc_goster(st, s, YAPI)


def _parametre(meta, sat, terimler):
    try:
        X, _, _ = A.model_matrisi(meta, [x["deg"] for x in sat], terimler,
                                  pt=[x["pt"] for x in sat], blok=[x["blok"] for x in sat],
                                  bloklar=sorted({x["blok"] for x in sat}))
        return X.shape[1]
    except Exception:
        return len(terimler) + 1


def _sonuc_goster(st, s, YAPI, on=""):
    meta = s["meta"]
    aile = A._aile(meta)
    st.markdown(f"### {s['yanit']} — {meta['ad']}")
    if s.get("cikarilan"):
        st.caption("Geri elemede çıkarılan terimler: " + ", ".join(s["cikarilan"]))
    if s.get("eklenen") is not None:
        st.caption("İleri seçimde eklenen terimler: " + (", ".join(s["eklenen"]) or "yok"))
    _mesajlar(st, s["mesajlar"])
    st.markdown("**Kodlanmış Katsayılar**")
    sut = ["Terim"] + (["Etki"] if aile == "faktoriyel" else []) + \
          ["Katsayı", "SE Katsayı", "T", "P", "VIF"]
    st.dataframe(_tablo(s["katsayi"], sut), hide_index=True, use_container_width=True)
    if s["lenth"]:
        st.caption(f"Doymuş model: T sütunu Lenth t değeridir (PSE = {s['lenth']['PSE']:.4g}, "
                   f"kritik t = {s['lenth']['t_kritik']:.3f}).")
    o = s["ozet"]
    st.markdown("**Model Özeti**")
    st.dataframe(pd.DataFrame([{"S": _f(o["S"]),
                                "R-kare": "*" if o["R-kare"] is None else f"{100 * o['R-kare']:.2f}%",
                                "R-kare(düz)": "*" if o["R-kare(düz)"] is None else f"{100 * o['R-kare(düz)']:.2f}%",
                                "R-kare(tah)": "*" if o["R-kare(tah)"] is None else f"{100 * o['R-kare(tah)']:.2f}%"}]),
                 hide_index=True, use_container_width=True)
    st.markdown("**Varyans Analizi**" + (" (karışım: ardışık SS)" if aile == "karisim" else ""))
    st.dataframe(_tablo(s["anova"], ["Kaynak", "SD", "Adj SS", "Adj MS", "F", "P"]),
                 hide_index=True, use_container_width=True)
    denk, not_ = s["denklem"]
    st.markdown("**Kodlanmamış Birimlerde Regresyon Denklemi**")
    st.code("\n".join(denk), language=None)
    if not_:
        st.caption(not_)
    if s["karisik"]:
        with st.expander(f"Karışık (aliased) terimler — {len(s['karisik'])}"):
            st.dataframe(pd.DataFrame([{"Terim": a, "Karışık olduğu": b} for a, b in s["karisik"]]),
                         hide_index=True, use_container_width=True)
    if meta.get("karisiklik"):
        with st.expander("Tasarımın karışıklık yapısı"):
            st.code("\n".join(meta["karisiklik"][:60]), language=None)
    if s["olagandisi"]:
        st.markdown("**Olağandışı Gözlemler için Uyum ve Tanılar**")
        st.dataframe(pd.DataFrame(s["olagandisi"]).round(4).astype(str), hide_index=True,
                     use_container_width=True)
        st.caption("R: büyük standart artık (|r| > 2) · X: yüksek kaldıraç")
    with st.expander("📊 Etki grafikleri (Pareto, normal)", expanded=aile == "faktoriyel"):
        ciz(st, DP.pareto(s))
        ciz(st, DP.normal_etki(s))
    with st.expander("📈 Ana etkiler ve etkileşimler"):
        ciz(st, DP.ana_etki(meta, s["satirlar"], s["yanit"]))
        ciz(st, DP.etkilesim(meta, s["satirlar"], s["yanit"]))
    with st.expander("🔍 Artık grafikleri (4'ü 1 arada)"):
        ciz(st, DP.artik4(s))
    surekli = [f for f in meta["faktorler"] if not A._kategorik(meta, f)]
    if meta["tip"] == "karisim" and len(meta["faktorler"]) == 3:
        with st.expander("🔺 Karışım kontur grafiği"):
            ciz(st, DP.karisim_ucgen(s))
    elif len(surekli) >= 2 and meta["tip"] != "karisim":
        with st.expander("🗺 Kontur ve yüzey grafikleri"):
            adlar = [f["isim"] for f in surekli]
            c1, c2 = st.columns(2)
            x = c1.selectbox("X ekseni", adlar, index=0, key=f"{on}doe_kx")
            y = c2.selectbox("Y ekseni", [a for a in adlar if a != x], index=0, key=f"{on}doe_ky")
            sabit = {}
            for f in meta["faktorler"]:
                if f["isim"] in (x, y):
                    continue
                if A._kategorik(meta, f):
                    L = A._seviyeler(meta, f)
                    sabit[f["isim"]] = st.selectbox(f"{f['isim']} (sabit)", L, key=f"{on}doe_ks_{f['isim']}",
                                                    format_func=A.seviye_yaz)
                else:
                    sabit[f["isim"]] = f["merkez"]
            hedef = next(((c.get("hedef_min"), c.get("hedef_max")) for c in YAPI["ciktilar"]
                          if c["isim"] == s["yanit"]), None)
            ciz(st, DP.kontur(s, x, y, sabit, hedef))
            ciz(st, DP.yuzey(s, x, y, sabit))


def _taguchi(st, meta, sat, nicel_c):
    c1, c2 = st.columns(2)
    yanit = c1.selectbox("Yanıt", [c["isim"] for c in nicel_c], key="doe_tg_y")
    hedef = c2.radio("S/N oranı", ["buyuk", "kucuk", "nominal"], key="doe_tg_h",
                     format_func=lambda x: A.SN_ADLARI[x])
    sut = T.yanit_sutunlari(meta, [yanit])
    tam = sum(1 for s in sat if any(A._sayi(s["y"].get(c)) is not None for c in sut))
    if tam == len(sat):
        st.success(f"Analiz yapılabilir: {meta.get('dizi', '')} dizisinin {tam} deneyinin tamamında sonuç var.")
    elif tam >= 3:
        st.info(f"Analiz yapılabilir ama {len(sat) - tam} deney eksik — Taguchi yanıt tabloları "
                "dengesiz olur.")
    else:
        st.warning(f"Henüz yeterli sonuç yok ({tam}/{len(sat)}).")
        return
    if hedef == "nominal" and meta.get("tekrar_sutun", 1) < 2:
        st.warning("'Nominal en iyi' her deneyde en az 2 tekrar ölçümü ister; bu plan tek ölçümlü.")
    if st.button("📈 Taguchi analizi", type="primary", key="doe_tg_b"):
        try:
            st.session_state["doe_taguchi"] = A.taguchi_analiz(meta, sat, yanit, hedef)
        except Exception as e:
            st.error(f"❌ {e}")
    ta = st.session_state.get("doe_taguchi")
    if not ta or ta["cikti"] != yanit or ta["hedef"] != hedef:
        return
    st.markdown(f"**S/N Oranları için Yanıt Tablosu** — {A.SN_ADLARI[hedef]}")
    st.dataframe(pd.DataFrame(A.taguchi_yanit_tablosu(ta, "SN")).astype(str), hide_index=True,
                 use_container_width=True)
    st.markdown("**Ortalamalar için Yanıt Tablosu**")
    st.dataframe(pd.DataFrame(A.taguchi_yanit_tablosu(ta, "Ort")).astype(str), hide_index=True,
                 use_container_width=True)
    eniyi = ", ".join(f"{k} = {A.seviye_yaz(v)}" for k, v in ta["en_iyi"].items())
    st.success(f"S/N'yi en büyükleyen seviyeler: {eniyi} → tahmini S/N {ta['tahmin']['SN']:.3f}, "
               f"tahmini ortalama {ta['tahmin']['Ort']:.4g}")
    st.session_state["doe_tg_oneri"] = {**{k: (float(v) if isinstance(v, (int, float)) else v)
                                           for k, v in ta["en_iyi"].items()},
                                        **meta.get("sabitler", {})}
    for fig in DP.taguchi_grafik(ta):
        ciz(st, fig)
    # S/N için doğrusal model (ANOVA)
    sn_sat = [{"StdOrder": i + 1, "RunOrder": i + 1, "pt": 1, "blok": 1, "deg": k["deg"],
               "y": {"S/N": k["SN"]}} for i, k in enumerate(ta["kayit"])]
    try:
        s = A.analiz(meta, sn_sat, "S/N", A.varsayilan_terimler(meta, None, 1))
        if s["dfE"] > 0:
            st.markdown("**S/N için Varyans Analizi (doğrusal model)**")
            st.dataframe(_tablo(s["anova"], ["Kaynak", "SD", "Adj SS", "Adj MS", "F", "P"]),
                         hide_index=True, use_container_width=True)
        else:
            st.caption("S/N için ANOVA: hata serbestlik derecesi kalmadığından p-değeri hesaplanamaz "
                       "(dizi doymuş). Delta/Sıra sütunları etki büyüklüğünü gösterir.")
    except Exception:
        pass
    if st.button("➡ Bu seviyeleri 🎯 Öneri sekmesine aktar", key="doe_tg_akt"):
        st.session_state["uygula"] = st.session_state["doe_tg_oneri"]
        st.session_state["doe_mesaj"] = "✅ Taguchi seviyeleri 🎯 Öneri sekmesine aktarıldı."
        st.rerun()


# ── 5. Optimizasyon ─────────────────────────────────────────────
def _optimizasyon(st, ctx):
    YAPI = ctx["YAPI"]
    sonuc = st.session_state.get("doe_sonuc") or {}
    if not sonuc:
        st.info("Önce **Analiz** sekmesinde en az bir yanıtı analiz edin. Analiz edilen her yanıt, "
                "hedefleriyle birlikte burada birlikte optimize edilir (Minitab Response Optimizer).")
        return
    kaynaklar = {s["kaynak"] for s in sonuc.values()}
    kaynak = st.radio("Model kaynağı", sorted(kaynaklar), horizontal=True, key="doe_opk",
                      format_func=lambda x: {"plan": "DOE planı", "veri": "Veriler sayfası"}[x]) \
        if len(kaynaklar) > 1 else next(iter(kaynaklar))
    secilen = {y: s for y, s in sonuc.items() if s["kaynak"] == kaynak}
    st.markdown("**Hedefler** (⚙️ Yapı'daki Hedef_Min / Hedef_Max; burada değiştirebilirsiniz)")
    hedef = {}
    for y in secilen:
        c = next((x for x in YAPI["ciktilar"] if x["isim"] == y), {})
        c1, c2, c3 = st.columns([2, 1, 1])
        dahil = c1.checkbox(y, True, key=f"doe_oph_{y}")
        lo = c2.number_input("Alt", value=float(c.get("hedef_min") if c.get("hedef_min") is not None
                                                  else secilen[y]["y"].min()), key=f"doe_opl_{y}")
        hi = c3.number_input("Üst", value=float(c.get("hedef_max") if c.get("hedef_max") is not None
                                                  else secilen[y]["y"].max()), key=f"doe_opu_{y}")
        if dahil:
            hedef[y] = (min(lo, hi), max(lo, hi))
    if not hedef:
        st.info("En az bir yanıt seçin.")
        return
    if st.button("🎯 Yanıtları birlikte optimize et", type="primary", key="doe_opt"):
        try:
            st.session_state["doe_opt"] = A.optimize({y: secilen[y] for y in hedef}, hedef)
        except Exception as e:
            st.error(f"❌ {e}")
    o = st.session_state.get("doe_opt")
    if not o:
        return
    meta = next(iter(secilen.values()))["meta"]
    (st.success if o["D"] > 0.9 else st.info if o["D"] > 0.5 else st.warning)(
        f"Birleşik istenebilirlik D = {o['D']:.3f}"
        + (" — tüm hedefler sağlanıyor." if o["D"] > 0.999 else
           " — hedeflerin bir kısmı tam sağlanamıyor; hedefleri veya deney bölgesini gözden geçirin."
           if o["D"] < 0.9 else ""))
    st.markdown("**Önerilen ayar**")
    st.dataframe(pd.DataFrame([{"Faktör": k, "Değer": A.seviye_yaz(v)} for k, v in o["ayar"].items()]
                              + [{"Faktör": f"{k} (sabit)", "Değer": str(v)}
                                 for k, v in meta.get("sabitler", {}).items()]),
                 hide_index=True, use_container_width=True)
    st.markdown("**Çoklu Yanıt Tahmini**")
    st.dataframe(_tablo(o["tablo"], ["Yanıt", "Uyum", "SE Uyum", "%95 GA", "%95 TA", "Hedef",
                                     "İstenebilirlik"]), hide_index=True, use_container_width=True)
    st.caption("GA: güven aralığı (ortalama yanıt) · TA: tahmin aralığı (tek bir yeni deney). "
               "Arama deney bölgesiyle sınırlıdır.")
    if st.button("➡ Bu ayarı 🎯 Öneri sekmesine aktar", key="doe_opt_akt"):
        st.session_state["uygula"] = {**{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                         for k, v in o["ayar"].items()}, **meta.get("sabitler", {})}
        st.session_state["doe_mesaj"] = ("✅ Ayar 🎯 Öneri sekmesine aktarıldı — orada 5 ML modeliyle "
                                         "de tahmin edebilirsiniz.")
        st.rerun()


# ════════════════════════════════════════════════════════════════
# 🤖 Otomatik DOE: mevcut kayıtlardan tasarım tanıma ve uygulama
# ════════════════════════════════════════════════════════════════
def tani_al(P, YAPI):
    """Veriler sayfası değişmedikçe tanıma sonucunu önbellekten verir."""
    df = P.get("df")
    if df is None or not len(df) or not YAPI["girdiler"]:
        return None
    anahtar = (len(P.get("veri_rows") or []), hash(str(P.get("veri_rows"))),
               hash(str(YAPI)))
    onb = P.get("doe_tani")
    if onb and onb[0] == anahtar:
        return onb[1]
    try:
        t = DT.tani(df, YAPI)
    except Exception:
        t = None
    P["doe_tani"] = (anahtar, t)
    return t


def _oto_calistir(st, YAPI, aday):
    with st.spinner(f"{aday['ad']} uygulanıyor..."):
        st.session_state["doe_oto"] = DT.uygula(aday, YAPI)
    st.session_state["doe_acik"] = True
    st.session_state["doe_mesaj"] = (f"✅ {aday['ad']} kayıtlı verilere uygulandı — sonuçlar "
                                     "**🧪 DOE → 🤖 Otomatik** sekmesinde.")
    st.rerun()


def _aday_satiri(st, a, anahtar, YAPI):
    cik = ", ".join(f"{c['isim']} ({c['n']} sonuç)" for c in a["ciktilar"] if c["uygun"])
    yetersiz = [c["isim"] for c in a["ciktilar"] if not c["uygun"]]
    st.markdown(f"**{a['ad']}**  \n{a['ozet']}")
    st.caption(("Analiz edilecek çıktılar: " + cik) if cik else "Analiz için yeterli sonuç yok.")
    if yetersiz:
        st.caption("Yetersiz sonuç: " + ", ".join(yetersiz))
    if a["uygun"] and st.button("▶ Otomatik uygula", key=f"{anahtar}_{a['kod']}",
                                type="primary" if a.get("onerilen") else "secondary",
                                use_container_width=True):
        _oto_calistir(st, YAPI, a)


def _uygun_adaylar(t):
    uy = [a for a in t["adaylar"] if a["uygun"]]
    if uy:
        uy[0]["onerilen"] = True
    return uy


def kenar(st, P, YAPI):
    """Yan panel: DOE asistanı."""
    t = tani_al(P, YAPI)
    with st.sidebar:
        st.markdown("### 🧪 DOE asistanı")
        if not t or not t["adaylar"]:
            n = 0 if P.get("df") is None else len(P["df"])
            st.caption(f"Veriler sayfasında {n} kayıt var. En az 4 eksiksiz kayıt olunca "
                       "kayıtların hangi deney tasarımına uyduğu burada gösterilir.")
            return
        uy = _uygun_adaylar(t)
        m = DT.ozet_mesaj(t)
        if m:
            st.success(m)
        else:
            st.warning("Kayıtlar henüz otomatik DOE analizi için yeterli değil.")
        if t["sabitler"]:
            st.caption("Sabit kalan girdiler: " + ", ".join(f"{k} = {A.seviye_yaz(v)}"
                                                            for k, v in t["sabitler"].items()))
        for a in uy:
            st.divider()
            _aday_satiri(st, a, "doe_k", YAPI)
        if t["nitel_ciktilar"]:
            st.caption("Nitel çıktılar (" + ", ".join(t["nitel_ciktilar"]) + ") DOE regresyonuna "
                       "alınmaz; 🎯 Öneri modelleri bunları kullanır.")
        with st.expander("Uymayan tasarımlar ve nedenleri"):
            for kod, neden in t["reddedilen"]:
                st.caption(f"✗ {T.TASARIM_ADLARI.get(kod, kod) if kod != 'faktoriyel' else '2 seviyeli faktöriyel / PB'}: {neden}")


def _minitab_csv(sonuc):
    meta = sonuc["meta"]
    fis = [f["isim"] for f in meta["faktorler"]]
    yan = list(meta["ciktilar"])
    sat = None
    for s in sonuc["sonuclar"].values():
        sat = s["satirlar"]
        break
    sat = sat or []
    pt_ad = meta.get("pt_sutun") or "PtType"
    rows = [["StdOrder", "RunOrder", pt_ad, "Blocks"] + fis + yan]
    for s in sat:
        rows.append([s["StdOrder"], s["RunOrder"], s["pt"], s["blok"]]
                    + [A.seviye_yaz(s["deg"][f]) for f in fis]
                    + ["" if s["y"].get(c) is None or (isinstance(s["y"].get(c), float)
                                                        and np.isnan(s["y"].get(c)))
                       else s["y"].get(c) for c in yan])
    import csv
    import io
    b = io.StringIO()
    csv.writer(b).writerows(rows)
    return b.getvalue().encode("utf-8-sig")


def _otomatik(st, ctx):
    P, YAPI = ctx["P"], ctx["YAPI"]
    t = tani_al(P, YAPI)
    st.caption("Veriler sayfasındaki mevcut kayıtlar incelenir; yeni deney veya sonuç girmeniz "
               "gerekmez. Kayıtlar bir deney tasarımına uyuyorsa o tasarımla analiz edilir ve "
               "⚙️ Yapı'daki hedeflere göre optimum ayar bulunur.")
    if t and t["adaylar"]:
        m = DT.ozet_mesaj(t)
        (st.success if m else st.warning)(m or "Kayıtlar henüz otomatik analiz için yeterli değil.")
        uy = _uygun_adaylar(t)
        if uy:
            kol = st.columns(min(3, len(uy)))
            for i, a in enumerate(uy):
                with kol[i % len(kol)]:
                    _aday_satiri(st, a, "doe_o", YAPI)
    else:
        st.info("Veriler sayfasında otomatik analiz için en az 4 eksiksiz kayıt gerekir.")
    o = st.session_state.get("doe_oto")
    if not o:
        return
    st.divider()
    meta = o["meta"]
    st.markdown(f"## {o['aday']['ad']}")
    st.caption(f"{o['aday']['ozet']} · kaynak: Veriler sayfası"
               + (" · sabit: " + ", ".join(f"{k} = {A.seviye_yaz(v)}" for k, v in meta["sabitler"].items())
                  if meta.get("sabitler") else ""))
    for h in o["hatalar"]:
        st.warning(h)
    st.download_button("⬇ Minitab çalışma sayfası (CSV)", _minitab_csv(o),
                       file_name=f"DOE_{meta['tip']}.csv", mime="text/csv", key="doe_oto_csv")
    opt = o.get("opt")
    if opt:
        st.markdown("### 🎯 Optimum ayar (Response Optimizer)")
        (st.success if opt["D"] > 0.9 else st.info if opt["D"] > 0.5 else st.warning)(
            f"Birleşik istenebilirlik D = {opt['D']:.3f}"
            + (" — tüm hedefler sağlanıyor." if opt["D"] > 0.999 else ""))
        st.dataframe(pd.DataFrame([{"Faktör": k, "Değer": A.seviye_yaz(v)} for k, v in opt["ayar"].items()]
                                  + [{"Faktör": f"{k} (sabit)", "Değer": A.seviye_yaz(v)}
                                     for k, v in meta.get("sabitler", {}).items()]),
                     hide_index=True, use_container_width=True)
        st.dataframe(_tablo(opt["tablo"], ["Yanıt", "Uyum", "SE Uyum", "%95 GA", "%95 TA", "Hedef",
                                           "İstenebilirlik"]), hide_index=True, use_container_width=True)
        if st.button("➡ Bu ayarı 🎯 Öneri sekmesine aktar", key="doe_oto_akt", type="primary"):
            st.session_state["uygula"] = {**{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                             for k, v in opt["ayar"].items()}, **meta.get("sabitler", {})}
            st.session_state["doe_mesaj"] = "✅ Ayar 🎯 Öneri sekmesine aktarıldı."
            st.rerun()
    for y, ta in o.get("taguchi", {}).items():
        st.markdown(f"### Taguchi — {y} · {A.SN_ADLARI[ta['hedef']]}")
        st.dataframe(pd.DataFrame(A.taguchi_yanit_tablosu(ta, "SN")).astype(str), hide_index=True,
                     use_container_width=True)
        st.caption("S/N için yanıt tablosu · ortalamalar:")
        st.dataframe(pd.DataFrame(A.taguchi_yanit_tablosu(ta, "Ort")).astype(str), hide_index=True,
                     use_container_width=True)
        st.info("S/N'yi en büyükleyen seviyeler: "
                + ", ".join(f"{k} = {A.seviye_yaz(v)}" for k, v in ta["en_iyi"].items())
                + f" → tahmini ortalama {ta['tahmin']['Ort']:.4g}")
        for fig in DP.taguchi_grafik(ta):
            ciz(st, fig)
    if o["sonuclar"]:
        st.markdown("### Analiz sonuçları")
        adlar = list(o["sonuclar"])
        sekmeler = st.tabs(adlar) if len(adlar) > 1 else [st.container()]
        for ad, sek in zip(adlar, sekmeler):
            with sek:
                _sonuc_goster(st, o["sonuclar"][ad], YAPI, on=f"oto_{ad}_")
