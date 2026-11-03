"""
store.py — Depolama katmanı
============================
GOOGLE_CREDENTIALS ortam değişkeni (Service Account JSON) varsa Google Sheets,
yoksa yerel CSV dosyaları kullanılır (deneme amaçlı — Hugging Face'te kalıcı değildir).
"""
from __future__ import annotations

import csv
import json
import os

SEKMELER = ["Yapilanma", "Veriler", "Tahminler", "Model_Durumu", "Etki_Analizi"]


class SheetsDepo:
    tur = "Google Sheets"

    def __init__(self, kimlik_json: str, sheet_id: str):
        import gspread
        from google.oauth2.service_account import Credentials

        bilgi = json.loads(kimlik_json)
        kimlik = Credentials.from_service_account_info(
            bilgi, scopes=["https://www.googleapis.com/auth/spreadsheets",
                           "https://www.googleapis.com/auth/drive"])
        self.servis_hesabi = bilgi.get("client_email", "")
        self.sh = gspread.authorize(kimlik).open_by_key(sheet_id)
        self.baslik = self.sh.title
        self._ws = {}

    def _sekme(self, ad):
        if ad not in self._ws:
            mevcut = {w.title: w for w in self.sh.worksheets()}
            self._ws[ad] = mevcut.get(ad) or self.sh.add_worksheet(title=ad, rows=1000, cols=40)
        return self._ws[ad]

    def oku(self, ad) -> list[list]:
        return self._sekme(ad).get_all_values()

    def yaz(self, ad, satirlar: list[list]):
        ws = self._sekme(ad)
        ws.clear()
        if satirlar:
            ws.update(range_name="A1", values=satirlar, value_input_option="USER_ENTERED")

    def baslik_yaz(self, ad, baslik: list):
        self._sekme(ad).update(range_name="A1", values=[baslik],
                               value_input_option="USER_ENTERED")

    def ekle(self, ad, satir: list):
        self._sekme(ad).append_row(satir, value_input_option="USER_ENTERED")


class YerelDepo:
    tur = "Yerel dosya (deneme modu)"

    def __init__(self, klasor="veri"):
        self.klasor = klasor
        self.baslik = "Yerel"
        self.servis_hesabi = ""
        os.makedirs(klasor, exist_ok=True)

    def _yol(self, ad):
        return os.path.join(self.klasor, f"{ad}.csv")

    def oku(self, ad):
        if not os.path.exists(self._yol(ad)):
            return []
        with open(self._yol(ad), encoding="utf-8", newline="") as f:
            return [r for r in csv.reader(f)]

    def yaz(self, ad, satirlar):
        with open(self._yol(ad), "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(satirlar)

    def baslik_yaz(self, ad, baslik):
        rows = self.oku(ad)
        if rows:
            rows[0] = baslik
        else:
            rows = [baslik]
        self.yaz(ad, rows)

    def ekle(self, ad, satir):
        rows = self.oku(ad)
        rows.append(satir)
        self.yaz(ad, rows)


def depo_ac():
    """(depo, uyarı_mesajı) döner."""
    kimlik = os.environ.get("GOOGLE_CREDENTIALS", "").strip()
    sheet_id = os.environ.get("SHEET_ID", "").strip()
    if kimlik and sheet_id:
        try:
            return SheetsDepo(kimlik, sheet_id), ""
        except Exception as e:
            return YerelDepo(), (f"Google Sheets'e bağlanılamadı ({type(e).__name__}: "
                                 f"{str(e)[:150]}). Yerel deneme modunda çalışıyor.")
    eksik = [ad for ad, v in [("GOOGLE_CREDENTIALS", kimlik), ("SHEET_ID", sheet_id)] if not v]
    return YerelDepo(), (f"{' ve '.join(eksik)} tanımlı değil — yerel deneme modunda "
                         f"çalışıyor. Veriler kalıcı değildir.")
