"""
Üretim Ayar Optimizasyonu — Streamlit arayüzü (Streamlit Community Cloud)
=========================================================================
Gizli bilgiler (App → Settings → Secrets):
  GOOGLE_CREDENTIALS  Service Account JSON anahtarının tamamı (''' ''' içinde)
  SHEET_ID            Google Sheets dosya kimliği
  APP_PASSWORD        Uygulamaya giriş şifresi (önerilir)
"""
import hmac
import json
import os
import threading
import time
import traceback
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="Üretim Ayar Optimizasyonu", page_icon="⚙️",
                   layout="centered")


def _sirlari_aktar():
    """Streamlit secrets → ortam değişkenleri (store.py bunları okur)."""
    try:
        for k in ("GOOGLE_CREDENTIALS", "SHEET_ID", "APP_PASSWORD"):
            if k in st.secrets and not os.environ.get(k):
                v = st.secrets[k]
                os.environ[k] = v if isinstance(v, str) else json.dumps(dict(v))
    except Exception:
        pass  # secrets dosyası yoksa yerel deneme modunda devam


_sirlari_aktar()
os.environ.setdefault("TZ", "Europe/Istanbul")
try:
    time.tzset()
except AttributeError:
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import engine as E
import plots as G
from store import depo_ac


# ════════════════════════════════════════════════════════════════
# Giriş
# ════════════════════════════════════════════════════════════════
def giris_kontrol():
    sifre = os.environ.get("APP_PASSWORD", "").strip()
    if not sifre or st.session_state.get("giris_ok"):
        return True
    st.title("⚙️ Üretim Ayar Optimizasyonu")
    with st.form("giris"):
        p = st.text_input("Şifre", type="password")
        if st.form_submit_button("Giriş", type="primary"):
            if hmac.compare_digest(p.encode(), sifre.encode()):
                st.session_state["giris_ok"] = True
                st.rerun()
            else:
                st.error("Şifre yanlış.")
    return False


if not giris_kontrol():
    st.stop()


# ════════════════════════════════════════════════════════════════
# Paylaşılan durum (tüm kullanıcılar için tek model)
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def paylasilan():
    depo, uyari = depo_ac()
    return {"depo": depo, "uyari": uyari, "kilit": threading.Lock(),
            "yapi": None, "df": None, "model": None, "guven": None,
            "etki": None, "log": ""}


P = paylasilan()
DEPO = P["depo"]

ORNEK_YAPI = [
    E.YAPI_SUTUNLAR,
    ["GİRDİ", "Akim_kA", "nicel", 8, 16, "", "", "", ""],
    ["GİRDİ", "Sure_cycle", "nicel", 5, 20, "", "", "", ""],
    ["GİRDİ", "Kuvvet_kN", "nicel", 2, 6, "", "", "", ""],
    ["GİRDİ", "Malzeme", "nitel", "", "", "DC01,DP600", "", "", ""],
    ["ÇIKTI", "Cekme_kN", "nicel", "", "", "", 9, 12, ""],
    ["ÇIKTI", "Cekirdek_mm", "nicel", "", "", "", 5, 6.5, ""],
    ["ÇIKTI", "Kalite", "nitel", "", "", "Ret,Uygun", "", "", "Uygun"],
]


def ornek_veri(n=40, tohum=7):
    rng = np.random.default_rng(tohum)
    rows = [["ID", "Tarih", "Akim_kA", "Sure_cycle", "Kuvvet_kN", "Malzeme",
             "Cekme_kN", "Cekirdek_mm", "Kalite"]]
    for k in range(n):
        a, s, f = rng.uniform(8, 16), rng.uniform(5, 20), rng.uniform(2, 6)
        m = rng.choice(["DC01", "DP600"])
        cek = (0.9 * a + 0.12 * s - 0.6 * abs(f - 4) - (0.8 if m == "DP600" else 0)
               + rng.normal(0, .4) - 1.5 - 0.48 * max(0, a - 13.5) ** 2)
        ck = 0.35 * a + 0.05 * s - 0.2 * abs(f - 4) + rng.normal(0, .15) + 0.4
        kal = "Uygun" if (8.5 < cek < 12.5 and ck > 4.8) else "Ret"
        rows.append([k + 1, "örnek", round(a, 2), round(s, 1), round(f, 2), m,
                     round(cek, 2), round(ck, 2), kal])
    return rows


def yukle():
    rows = DEPO.oku("Yapilanma")
    if (not rows or len(rows) < 2) and DEPO.tur.startswith("Yerel"):
        DEPO.yaz("Yapilanma", ORNEK_YAPI)
        DEPO.yaz("Veriler", ornek_veri())
        rows = ORNEK_YAPI
    P["yapi"] = E.yapi_oku(rows)
    P["df"] = E.veri_oku(DEPO.oku("Veriler"), P["yapi"])


def egit_hepsi():
    """Veriyi çek → modelleri yarıştır → güven + etki analizini tabloya yaz."""
    with P["kilit"]:
        log = []
        try:
            yukle()
            yapi, df = P["yapi"], P["df"]
            if not yapi["girdiler"] or not yapi["ciktilar"]:
                P["model"] = P["guven"] = P["etki"] = None
                P["log"] = "Yapı tanımlı değil."
                return
            log.append(f"📥 {len(df)} kayıt okundu ({DEPO.tur}).")
            log += [f"⚠️ {h}" for h in E.yapi_kontrol(yapi)]
            t0 = time.time()
            model = E.egit(yapi, df, log=log.append)
            P["model"] = model
            if model is None:
                P["guven"] = P["etki"] = None
                return
            P["guven"] = E.guven(model, yapi)
            try:
                DEPO.yaz("Model_Durumu", E.durum_satirlari(model, yapi, P["guven"]))
            except Exception as e:
                log.append(f"⚠️ Model_Durumu yazılamadı: {e}")
            try:
                onem = E.onem_hesapla(model)
                duy = E.duyarlilik_hesapla(model, yapi)
                kor = E.korelasyon_hesapla(model, yapi)
                P["etki"] = {"onem": onem, "duy": duy, "kor": kor}
                DEPO.yaz("Etki_Analizi", E.etki_satirlari(onem, duy, kor, yapi, model))
            except Exception as e:
                P["etki"] = None
                log.append(f"⚠️ Etki analizi: {e}")
            log.append(f"⏱ {time.time() - t0:.1f} sn")
        except Exception as e:
            log.append(f"❌ {type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}")
        finally:
            P["log"] = "\n".join(log)


def model_hazir():
    if P["model"] is None:
        with st.spinner("Model eğitiliyor..."):
            egit_hepsi()
    return P["model"]


def ciz(fig):
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)


def bildir(anahtar):
    """Yeniden yüklemeden önce kaydedilen mesajı göster."""
    m = st.session_state.pop(anahtar, None)
    if m:
        (st.error if m.startswith("❌") else st.success)(m)


# ════════════════════════════════════════════════════════════════
# İlk yükleme
# ════════════════════════════════════════════════════════════════
if P["yapi"] is None:
    with st.spinner("Veriler okunuyor ve modeller yarışıyor..."):
        try:
            yukle()
            if len(P["df"]) >= E.MIN_KAYIT:
                egit_hepsi()
        except Exception as e:
            P["log"] = f"❌ {type(e).__name__}: {e}"
            P["yapi"] = {"girdiler": [], "ciktilar": []}

YAPI = P["yapi"] or {"girdiler": [], "ciktilar": []}

# ════════════════════════════════════════════════════════════════
# Başlık
# ════════════════════════════════════════════════════════════════
st.title("⚙️ Üretim Ayar Optimizasyonu")
st.caption("5 model yarışır (GP · RF · XGBoost · Lojistik · SVM), "
           "Optuna hedeflere en yakın ayarı bulur.")
if P["uyari"]:
    st.warning(P["uyari"])
gv = P["guven"]
if gv:
    ort = f" · ortalama skor {gv['ortalama']:.2f}" if gv["ortalama"] is not None else ""
    st.markdown(f"**{DEPO.tur}** · Model: {gv['durum']} · {gv['n']} kayıt "
                f"({gv['zaman']}){ort}")
else:
    n = 0 if P["df"] is None else len(P["df"])
    st.markdown(f"**{DEPO.tur}** · Model henüz eğitilmedi — {n} kayıt var, "
                f"en az {E.MIN_KAYIT} tam kayıt gerekli.")

sek_oneri, sek_veri, sek_model, sek_etki, sek_yapi = st.tabs(
    ["🎯 Ayar Önerisi", "📝 Veri Girişi", "📊 Model", "🔍 Etki", "⚙️ Yapı"])

# ════════════════════════════════════════════════════════════════
# 1. Ayar önerisi
# ════════════════════════════════════════════════════════════════
with sek_oneri:
    if not YAPI["girdiler"]:
        st.info("Önce **⚙️ Yapı** sekmesinden girdi ve çıktıları tanımlayın.")
    else:
        # Optimizasyon sonucu bir önceki çalışmada bulunduysa giriş kutularına aktar
        bekleyen = st.session_state.pop("uygula", None)
        if bekleyen:
            for k, v in bekleyen.items():
                st.session_state[f"t_{k}"] = float(v) if isinstance(v, (int, float)) else v

        st.write("Mevcut ayarları girip **Tahmin et**'e basın, ya da "
                 "**En iyi ayarı bul** ile hedeflere en yakın ayarı arayın.")
        with st.form("tahmin_form"):
            girdi = {}
            kolonlar = st.columns(2)
            for k, g in enumerate(YAPI["girdiler"]):
                anahtar = f"t_{g['isim']}"
                with kolonlar[k % 2]:
                    if g["tip"] == "nicel":
                        if not isinstance(st.session_state.get(anahtar), float):
                            st.session_state[anahtar] = float(round((g["min"] + g["max"]) / 2, 3))
                        girdi[g["isim"]] = st.number_input(
                            f"{g['isim']}  ({g['min']:g} – {g['max']:g})", key=anahtar)
                    else:
                        kat = g["kategoriler"] or ["-"]
                        if st.session_state.get(anahtar) not in kat:
                            st.session_state[anahtar] = kat[0]
                        girdi[g["isim"]] = st.selectbox(g["isim"], kat, key=anahtar)
            with st.expander("Optimizasyon ayarları"):
                sabit_liste = st.multiselect(
                    "Sabit tutulacak girdiler (yukarıdaki değerleri kullanılır)",
                    [g["isim"] for g in YAPI["girdiler"]],
                    default=[g["isim"] for g in YAPI["girdiler"] if g["tip"] == "nitel"])
                n_deneme = st.slider("Deneme sayısı (fazlası daha iyi, daha yavaş)",
                                     30, 400, 150, 10)
            c1, c2 = st.columns(2)
            b_tahmin = c1.form_submit_button("▶ Tahmin et", use_container_width=True)
            b_opt = c2.form_submit_button("⬡ En iyi ayarı bul", type="primary",
                                          use_container_width=True)

        if b_tahmin or b_opt:
            m = model_hazir()
            if m is None:
                st.warning("Model yok — yeterli veri girildiğinde çalışır.")
            elif b_tahmin:
                t = E.tahmin(m, girdi)
                st.session_state["sonuc"] = ("Tahmin", None, t, E.skor(YAPI, t), None)
            else:
                sabit = {k: girdi[k] for k in sabit_liste if girdi.get(k) not in (None, "")}
                with st.spinner("En iyi ayar aranıyor..."):
                    en, t, p, gecmis, onem, yontem = E.optimize(m, YAPI, int(n_deneme),
                                                                 sabit=sabit)
                st.session_state["sonuc"] = (
                    f"Önerilen ayar ({yontem}, {len(gecmis)} deneme)", en, t, p,
                    (gecmis, onem))
                st.session_state["uygula"] = en
                st.rerun()

        sonuc = st.session_state.get("sonuc")
        if sonuc:
            baslik, en, t, p, opt = sonuc
            st.markdown(G.sonuc_md(YAPI, baslik, en, t, p))
            st.caption("Tahminler modelin öğrendiği veriye dayanır; daha önce denenmemiş "
                       "bölgelerde belirsizlik artar. Öneriyi uygulayıp sonucu kaydetmek "
                       "modeli o bölgede de öğretir.")
            if opt:
                with st.expander("Optimizasyon grafiği"):
                    ciz(G.opt_grafik(*opt))
            if st.button("📤 Sonucu 'Tahminler' sekmesine kaydet"):
                girdi_kayit = en if en is not None else girdi
                bas = (["Tarih", "Tür"] + [g["isim"] for g in YAPI["girdiler"]]
                       + [c["isim"] for c in YAPI["ciktilar"]] + ["Skor", "Tüm hedefler"])
                mevcut = DEPO.oku("Tahminler")
                if not mevcut:
                    DEPO.yaz("Tahminler", [bas])
                elif [str(h) for h in mevcut[0]] != bas:
                    DEPO.ekle("Tahminler", [])
                    DEPO.ekle("Tahminler", bas)
                satir = [datetime.now().strftime("%d.%m.%Y %H:%M"),
                         "Optimizasyon" if en is not None else "Tahmin"]
                satir += [E.hucre(girdi_kayit.get(g["isim"])) for g in YAPI["girdiler"]]
                tumu = True
                for c in YAPI["ciktilar"]:
                    r = t.get(c["isim"])
                    if r is None:
                        satir.append("")
                        tumu = False
                        continue
                    tumu &= bool(E.hedef_durum(c, r))
                    if c["tip"] == "nicel":
                        satir.append(round(r["deger"], 4))
                    else:
                        satir.append(f"{r['deger']} "
                                     f"(%{100 * r['olas'].get(c.get('hedef_kat'), 0):.0f})")
                satir += [round(p, 1), "Evet" if tumu else "Hayır"]
                DEPO.ekle("Tahminler", satir)
                st.success("Tahminler sekmesine kaydedildi.")

# ════════════════════════════════════════════════════════════════
# 2. Veri girişi
# ════════════════════════════════════════════════════════════════
def son_kayitlar(n=15):
    rows = DEPO.oku("Veriler")
    if not rows or len(rows) < 2:
        return pd.DataFrame(columns=rows[0] if rows else ["ID"])
    bas = rows[0]
    gov = [list(r) + [""] * (len(bas) - len(r)) for r in rows[1:]][-n:][::-1]
    return pd.DataFrame([r[:len(bas)] for r in gov], columns=bas)


def veri_kaydet(d, yeniden_egit):
    isimler = [x["isim"] for x in YAPI["girdiler"] + YAPI["ciktilar"]]
    eksik = [g["isim"] for g in YAPI["girdiler"] if d.get(g["isim"]) in (None, "")]
    if eksik:
        return f"❌ Girdiler eksik: {', '.join(eksik)}"
    if all(d.get(c["isim"]) in (None, "") for c in YAPI["ciktilar"]):
        return "❌ En az bir ölçüm sonucu (çıktı) girin."
    rows = DEPO.oku("Veriler")
    bas = [str(h).strip() for h in rows[0]] if rows else []
    gerekli = ["ID", "Tarih"] + isimler
    if not bas:
        bas = gerekli
        DEPO.yaz("Veriler", [bas])
        rows = [bas]
    elif any(h not in bas for h in gerekli):
        bas = bas + [h for h in gerekli if h not in bas]
        DEPO.baslik_yaz("Veriler", bas)
    idler = [E.sayi(r[0]) for r in rows[1:] if r]
    yeni_id = int(max([i for i in idler if i is not None], default=0)) + 1
    d = dict(d, ID=yeni_id, Tarih=datetime.now().strftime("%d.%m.%Y %H:%M"))
    DEPO.ekle("Veriler", [E.hucre(d.get(h)) if d.get(h) is not None else "" for h in bas])
    mesaj = f"Kayıt #{yeni_id} eklendi."
    if yeniden_egit:
        egit_hepsi()
        mesaj += " Model yeniden eğitildi."
    else:
        P["df"] = E.veri_oku(DEPO.oku("Veriler"), YAPI)
    return "✅ " + mesaj


with sek_veri:
    bildir("veri_mesaj")
    if not YAPI["girdiler"]:
        st.info("Önce **⚙️ Yapı** sekmesinden girdi ve çıktıları tanımlayın.")
    else:
        st.write("Denemeden sonra uyguladığınız ayarları ve ölçtüğünüz sonuçları girin. "
                 "Google Sheets'teki **Veriler** sekmesine doğrudan da yazabilirsiniz.")
        with st.form("veri_form", clear_on_submit=True):
            d = {}
            st.markdown("**Ayarlar (girdiler)**")
            kol = st.columns(2)
            for k, g in enumerate(YAPI["girdiler"]):
                with kol[k % 2]:
                    if g["tip"] == "nicel":
                        d[g["isim"]] = st.number_input(g["isim"], value=None,
                                                       key=f"v_{g['isim']}")
                    else:
                        d[g["isim"]] = st.selectbox(g["isim"], g["kategoriler"] or [],
                                                    index=None, placeholder="Seçin",
                                                    key=f"v_{g['isim']}")
            st.markdown("**Ölçüm sonuçları (çıktılar)**")
            kol = st.columns(2)
            for k, c in enumerate(YAPI["ciktilar"]):
                with kol[k % 2]:
                    if c["tip"] == "nicel":
                        d[c["isim"]] = st.number_input(c["isim"], value=None,
                                                       key=f"v_{c['isim']}")
                    else:
                        d[c["isim"]] = st.selectbox(c["isim"], c["kategoriler"] or [],
                                                    index=None, placeholder="Seçin",
                                                    key=f"v_{c['isim']}")
            yeniden = st.checkbox("Kaydettikten sonra modeli yeniden eğit", value=True)
            kaydet = st.form_submit_button("💾 Kaydet", type="primary",
                                           use_container_width=True)
        if kaydet:
            with st.spinner("Kaydediliyor..."):
                st.session_state["veri_mesaj"] = veri_kaydet(d, yeniden)
            st.rerun()
    st.markdown("**Son kayıtlar**")
    st.dataframe(son_kayitlar(), hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# 3. Model
# ════════════════════════════════════════════════════════════════
with sek_model:
    bildir("model_mesaj")
    c1, c2 = st.columns([3, 2])
    if c1.button("🔄 Verileri çek ve modelleri yeniden yarıştır", type="primary",
                 use_container_width=True):
        with st.spinner("Modeller yarışıyor..."):
            egit_hepsi()
        st.session_state["model_mesaj"] = ("✅ Model güncellendi." if P["model"]
                                           else "❌ Model kurulamadı — günlüğe bakın.")
        st.rerun()
    if c2.button("🔌 Bağlantıyı yenile", use_container_width=True,
                 help="Sheets ayarlarını değiştirdiyseniz bağlantıyı baştan kurar"):
        st.cache_resource.clear()
        st.rerun()
    if P["model"]:
        st.dataframe(G.yaris_tablosu(P["model"], P["guven"]), hide_index=True,
                     use_container_width=True)
        ciz(G.yaris_grafik(P["model"]))
    st.caption("R²: 1 kusursuz, 0.7 üstü iyi. Doğruluk: doğru tahmin oranı. Skorlar "
               "modelin daha önce görmediği kayıtlar üzerinden (çapraz doğrulama) ölçülür.")
    with st.expander("Eğitim günlüğü"):
        st.code(P["log"] or "—", language=None)

# ════════════════════════════════════════════════════════════════
# 4. Etki analizi
# ════════════════════════════════════════════════════════════════
with sek_etki:
    etki = P["etki"]
    if not etki:
        st.info("Etki analizi model eğitildikten sonra görünür.")
    else:
        st.markdown("**Girdi önemi** — bir girdinin değerleri karıştırıldığında modelin "
                    "başarısı ne kadar düşüyor? Pay ne kadar büyükse o girdi sonucu o kadar "
                    "belirliyor.")
        ciz(G.onem_grafik(etki, P["model"]))
        secenek = list((etki.get("duy") or {}).keys())
        if secenek and any(etki["duy"][s] for s in secenek):
            st.markdown("**Duyarlılık** — tek bir girdi değişirken çıktı nasıl değişiyor?")
            secim = st.selectbox("Çıktı", secenek)
            ciz(G.duy_grafik(etki, YAPI, secim))
        else:
            st.caption("Duyarlılık grafiği için en az bir nicel girdi gerekir.")
        if etki.get("kor") is not None:
            st.markdown("**Korelasyon** — +1 birlikte artar, −1 ters yönde değişir. "
                        "Nitel çıktılar 'hedef kategoride mi?' olarak alınır.")
            ciz(G.kor_grafik(etki))

# ════════════════════════════════════════════════════════════════
# 5. Yapı
# ════════════════════════════════════════════════════════════════
def yapi_tablosu():
    rows = DEPO.oku("Yapilanma")
    if not rows or len(rows) < 2:
        rows = ORNEK_YAPI
    n = len(E.YAPI_SUTUNLAR)
    gov = [[str(x) for x in (list(r) + [""] * n)[:n]] for r in rows[1:]]
    return pd.DataFrame(gov, columns=E.YAPI_SUTUNLAR)


def yapi_kaydet(tablo, sifirla):
    gov = tablo.fillna("").astype(str).values.tolist()
    gov = [r for r in gov if any(x.strip() for x in r)]
    yapi = E.yapi_oku([E.YAPI_SUTUNLAR] + gov)
    hatalar = E.yapi_kontrol(yapi)
    ciddi = [h for h in hatalar if "Hedef" not in h]
    if ciddi:
        return "❌ Kaydedilmedi:\n\n" + "\n".join(f"- {h}" for h in hatalar)
    DEPO.yaz("Yapilanma", E.yapi_satirlari(yapi))
    yeni = ["ID", "Tarih"] + [x["isim"] for x in yapi["girdiler"] + yapi["ciktilar"]]
    mesaj = ["✅ Yapı kaydedildi."]
    if sifirla:
        DEPO.yaz("Veriler", [yeni])
        DEPO.yaz("Tahminler", [])
        mesaj.append("Veriler ve Tahminler sekmeleri sıfırlandı.")
    else:
        mevcut = DEPO.oku("Veriler")
        if not mevcut:
            DEPO.yaz("Veriler", [yeni])
        else:
            bas = [str(h).strip() for h in mevcut[0]]
            eksik = [h for h in yeni if h not in bas]
            if eksik:
                DEPO.baslik_yaz("Veriler", bas + eksik)
                mesaj.append(f"Veriler sekmesine yeni sütun eklendi: {', '.join(eksik)}. "
                             f"Eski kayıtlarda bu sütunlar boş kalır.")
    mesaj += [f"⚠️ {h}" for h in hatalar]
    P["model"] = P["guven"] = P["etki"] = None
    egit_hepsi()
    st.session_state.pop("sonuc", None)
    return "\n\n".join(mesaj)


with sek_yapi:
    bildir("yapi_mesaj")
    st.markdown(
        "Buradan veya Sheets'teki **Yapilanma** sekmesinden düzenleyebilirsiniz.\n"
        "- **Tür:** GİRDİ / ÇIKTI · **Tip:** nicel / nitel\n"
        "- Nicel girdi: **Min–Max** (optimizasyon bu aralıkta arar)\n"
        "- Nitel: **Kategoriler** virgülle, örn. `DC01,DP600`\n"
        "- Hedef: nicel çıktıda **Hedef_Min–Hedef_Max**, nitel çıktıda **Hedef_Kategori**")
    if "yapi_df" not in st.session_state:
        st.session_state["yapi_df"] = yapi_tablosu()
    duzen = st.data_editor(
        st.session_state["yapi_df"], key="yapi_editor", num_rows="dynamic",
        hide_index=True, use_container_width=True,
        column_config={
            "Tür": st.column_config.SelectboxColumn(options=["GİRDİ", "ÇIKTI"], required=True),
            "Tip": st.column_config.SelectboxColumn(options=["nicel", "nitel"], required=True),
        })
    sifirla = st.checkbox("Veriler sekmesini sıfırla (TÜM kayıtlar silinir)")
    c1, c2 = st.columns(2)
    if c1.button("💾 Yapıyı kaydet", type="primary", use_container_width=True):
        with st.spinner("Kaydediliyor ve model yeniden eğitiliyor..."):
            try:
                st.session_state["yapi_mesaj"] = yapi_kaydet(duzen, sifirla)
            except Exception as e:
                st.session_state["yapi_mesaj"] = f"❌ {type(e).__name__}: {e}"
        if not st.session_state["yapi_mesaj"].startswith("❌"):
            st.session_state.pop("yapi_df", None)
            st.session_state.pop("yapi_editor", None)
        st.rerun()
    if c2.button("↩ Sheets'ten yeniden yükle", use_container_width=True):
        yukle()
        st.session_state.pop("yapi_df", None)
        st.session_state.pop("yapi_editor", None)
        st.rerun()
