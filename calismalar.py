"""
calismalar.py — Birden fazla çalışmayı aynı Google Sheets dosyasında yönetir
=============================================================================
'Calismalar' sayfası dizin görevi görür. Her çalışmanın kendi 5 sayfası vardır:
  <Ad>_Yapi · <Ad>_Veriler · <Ad>_Tahmin · <Ad>_Model · <Ad>_Etki
DOE tasarımı oluşturulursa ayrıca <Ad>_DOE (Minitab çalışma sayfası) ve
<Ad>_DOE_Bilgi (tasarım bilgisi) sayfaları açılır.
Sayfa adları dizinde açıkça tutulur; böylece eski tek-çalışmalı düzen
(Yapilanma, Veriler, ...) de bir çalışma olarak kullanılabilir.
"""
from __future__ import annotations

import re
from datetime import datetime

import engine as E

DIZIN = "Calismalar"
TEMEL = ["Yapilanma", "Veriler", "Tahminler", "Model_Durumu", "Etki_Analizi"]
DOE = ["DOE", "DOE_Bilgi"]  # sadece DOE tasarımı oluşturulunca açılır
MANTIKSAL = TEMEL + DOE
DIZIN_SUTUN = ["Ad", "Aciklama", "Olusturma"] + MANTIKSAL
EK = {"Yapilanma": "Yapi", "Veriler": "Veriler", "Tahminler": "Tahmin",
      "Model_Durumu": "Model", "Etki_Analizi": "Etki", "DOE": "DOE", "DOE_Bilgi": "DOE_Bilgi"}
ESKI_AD = "Ilk_calisma"


class Calisma:
    """Bir çalışmanın sayfalarına mantıksal adla erişim (depo ile aynı arayüz)."""

    def __init__(self, depo, kayit: dict):
        self.depo = depo
        self.ad = kayit["Ad"]
        self.aciklama = kayit.get("Aciklama", "")
        self.sekme = {k: kayit[k] for k in MANTIKSAL}
        self.tur = depo.tur
        self.baslik = depo.baslik

    def oku(self, k):
        return self.depo.oku(self.sekme[k])

    def yaz(self, k, rows):
        self.depo.yaz(self.sekme[k], rows)

    def baslik_yaz(self, k, baslik):
        self.depo.baslik_yaz(self.sekme[k], baslik)

    def ekle(self, k, satir):
        self.depo.ekle(self.sekme[k], satir)

    def ekle_coklu(self, k, satirlar):
        self.depo.ekle_coklu(self.sekme[k], satirlar)

    def url(self, k="Veriler"):
        return self.depo.url(self.sekme[k])


def ad_kontrol(ad: str, mevcut: list[str], sekmeler: list[str]) -> str | None:
    """Geçersizse hata mesajı, geçerliyse None."""
    ad = (ad or "").strip()
    if len(ad) < 2 or len(ad) > 30:
        return "Çalışma adı 2–30 karakter olmalı."
    if not re.fullmatch(r"[\w\- ]+", ad):
        return "Sadece harf, rakam, boşluk, '-' ve '_' kullanın."
    if ad.lower() == DIZIN.lower():
        return f"'{DIZIN}' adı ayrılmış."
    if ad.lower() in [m.lower() for m in mevcut]:
        return f"'{ad}' adında bir çalışma zaten var."
    cakisan = [f"{ad}_{e}" for e in EK.values() if f"{ad}_{e}" in sekmeler]
    if cakisan:
        return f"Dosyada bu adla sayfa zaten var: {', '.join(cakisan)}"
    return None


def oku(depo) -> list[dict]:
    """Dizini okur. Dizin yoksa eski düzeni (Yapilanma/Veriler...) içeri aktarır."""
    rows = depo.oku(DIZIN)
    kayitlar = []
    if rows and len(rows) >= 1:
        bas = [str(h).strip() for h in rows[0]]
        for r in rows[1:]:
            d = {bas[i]: (str(r[i]).strip() if i < len(r) else "") for i in range(len(bas))}
            if not d.get("Ad"):
                continue
            for k in MANTIKSAL:
                d.setdefault(k, "")
                if not d[k]:
                    d[k] = f"{d['Ad']}_{EK[k]}"
            kayitlar.append(d)
    if kayitlar:
        return kayitlar
    # Eski tek-çalışmalı düzen var mı?
    sekmeler = depo.sekmeler()
    if "Yapilanma" in sekmeler and len(depo.oku("Yapilanma")) >= 2:
        eski = {"Ad": ESKI_AD, "Aciklama": "Önceki sürümden aktarıldı",
                "Olusturma": datetime.now().strftime("%d.%m.%Y")}
        eski.update({k: k for k in TEMEL})
        eski.update({k: f"{ESKI_AD}_{EK[k]}" for k in DOE})
        yaz(depo, [eski])
        return [eski]
    return []


def yaz(depo, kayitlar: list[dict]):
    depo.yaz(DIZIN, [DIZIN_SUTUN] + [[k.get(s, "") for s in DIZIN_SUTUN] for k in kayitlar])


BOS_YAPI = [
    E.YAPI_SUTUNLAR,
    ["GİRDİ", "Girdi_1", "nicel", 0, 100, "", "", "", ""],
    ["ÇIKTI", "Cikti_1", "nicel", "", "", "", 40, 60, ""],
]


def olustur(depo, kayitlar, ad, aciklama="", yapi_rows=None, veri_rows=None) -> dict:
    """Yeni çalışmanın sayfalarını açar ve dizine ekler."""
    ad = ad.strip()
    hata = ad_kontrol(ad, [k["Ad"] for k in kayitlar], depo.sekmeler())
    if hata:
        raise ValueError(hata)
    yeni = {"Ad": ad, "Aciklama": (aciklama or "").strip(),
            "Olusturma": datetime.now().strftime("%d.%m.%Y %H:%M")}
    yeni.update({k: f"{ad}_{EK[k]}" for k in MANTIKSAL})
    yapi_rows = yapi_rows or BOS_YAPI
    yapi = E.yapi_oku(yapi_rows)
    baslik = ["ID", "Tarih"] + [x["isim"] for x in yapi["girdiler"] + yapi["ciktilar"]]
    depo.yaz(yeni["Yapilanma"], E.yapi_satirlari(yapi))
    depo.yaz(yeni["Veriler"], veri_rows or [baslik])
    for k in ("Tahminler", "Model_Durumu", "Etki_Analizi"):
        depo.yaz(yeni[k], [])  # DOE sayfaları ilk tasarım oluşturulduğunda açılır
    kayitlar = kayitlar + [yeni]
    yaz(depo, kayitlar)
    return yeni


def sil(depo, kayitlar, ad) -> list[dict]:
    hedef = next((k for k in kayitlar if k["Ad"] == ad), None)
    if hedef is None:
        return kayitlar
    for k in MANTIKSAL:
        depo.sekme_sil(hedef[k])
    kalan = [k for k in kayitlar if k["Ad"] != ad]
    yaz(depo, kalan)
    return kalan
