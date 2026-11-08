"""
store.py — Depolama katmanı
============================
GOOGLE_CREDENTIALS (Service Account JSON) ve SHEET_ID tanımlıysa Google Sheets,
yoksa yerel CSV dosyaları kullanılır (deneme amaçlı — bulutta kalıcı değildir).

Her iki depo da sekme (sayfa) adıyla çalışır: oku / yaz / baslik_yaz / ekle /
sekmeler / olustur / sekme_sil / url.
"""
from __future__ import annotations

import csv
import json
import os


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
        self.sheet_id = sheet_id
        self.sh = gspread.authorize(kimlik).open_by_key(sheet_id)
        self.baslik = self.sh.title
        self._ws = {}

    def _yenile(self):
        self._ws = {w.title: w for w in self.sh.worksheets()}

    def _sekme(self, ad, olustur=True):
        if ad not in self._ws:
            self._yenile()
        if ad not in self._ws and olustur:
            self._ws[ad] = self.sh.add_worksheet(title=ad, rows=300, cols=30)
        return self._ws.get(ad)

    def sekmeler(self) -> list[str]:
        self._yenile()
        return list(self._ws)

    def olustur(self, ad):
        self._sekme(ad)

    def sekme_sil(self, ad):
        ws = self._sekme(ad, olustur=False)
        if ws is not None:
            self.sh.del_worksheet(ws)
            self._ws.pop(ad, None)

    def url(self, ad=None):
        base = f"https://docs.google.com/spreadsheets/d/{self.sheet_id}/edit"
        ws = self._sekme(ad, olustur=False) if ad else None
        return f"{base}#gid={ws.id}" if ws is not None else base

    def oku(self, ad) -> list[list]:
        ws = self._sekme(ad, olustur=False)
        return ws.get_all_values() if ws is not None else []

    def yaz(self, ad, satirlar: list[list]):
        ws = self._sekme(ad)
        ws.clear()
        if satirlar:
            genislik = max((len(r) for r in satirlar), default=1)
            if ws.col_count < genislik or ws.row_count < len(satirlar):
                ws.resize(rows=max(ws.row_count, len(satirlar) + 50),
                          cols=max(ws.col_count, genislik))
            ws.update(range_name="A1", values=satirlar, value_input_option="USER_ENTERED")

    def baslik_yaz(self, ad, baslik: list):
        ws = self._sekme(ad)
        if ws.col_count < len(baslik):
            ws.resize(cols=len(baslik))
        ws.update(range_name="A1", values=[baslik], value_input_option="USER_ENTERED")

    def ekle(self, ad, satir: list):
        self._sekme(ad).append_row(satir, value_input_option="USER_ENTERED")

    def ekle_coklu(self, ad, satirlar: list[list]):
        if satirlar:
            self._sekme(ad).append_rows(satirlar, value_input_option="USER_ENTERED")


class YerelDepo:
    tur = "Yerel dosya (deneme modu)"

    def __init__(self, klasor="veri"):
        self.klasor = klasor
        self.baslik = "Yerel"
        self.servis_hesabi = ""
        os.makedirs(klasor, exist_ok=True)

    def _yol(self, ad):
        return os.path.join(self.klasor, f"{ad}.csv")

    def sekmeler(self):
        return sorted(f[:-4] for f in os.listdir(self.klasor) if f.endswith(".csv"))

    def olustur(self, ad):
        if not os.path.exists(self._yol(ad)):
            self.yaz(ad, [])

    def sekme_sil(self, ad):
        if os.path.exists(self._yol(ad)):
            os.remove(self._yol(ad))

    def url(self, ad=None):
        return None

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

    def ekle_coklu(self, ad, satirlar):
        rows = self.oku(ad)
        self.yaz(ad, rows + [list(s) for s in satirlar])


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
