"""
doe_tasarim.py — Deney tasarımı (DOE) üreticileri, Minitab çalışma sayfası düzeninde
=====================================================================================
Tasarımlar:
  faktoriyel2   2 seviyeli tam faktöriyel
  kesirli       2 seviyeli kesirli faktöriyel (2^(k-p), Minitab/Montgomery üreteçleri)
  pb            Plackett-Burman eleme tasarımı
  genel         Genel tam faktöriyel (çok seviyeli, nitel faktörler dahil)
  ccd           Merkezi kompozit (CCC çevrel, CCI içsel, CCF yüzey merkezli)
  bbd           Box-Behnken
  dsd           Kesin eleme tasarımı (Definitive Screening)
  taguchi       Taguchi ortogonal dizileri (L4, L8, L9, L12, L16, L18, L27)
  karisim       Karışım (simplex lattice / simplex centroid, eksenel ve merkez ekleme)
  doptimal      D-optimal (aday noktalardan seçim)

Her tasarım bir 'meta' sözlüğü ve satır listesi döndürür. Satırlarda faktörler
kodlanmamış (gerçek) değerlerle tutulur — Minitab'ın varsayılan görünümü gibi.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

HARFLER = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # Minitab 'I' harfini atlar

TASARIM_ADLARI = {
    "faktoriyel2": "2 seviyeli tam faktöriyel",
    "kesirli": "2 seviyeli kesirli faktöriyel",
    "pb": "Plackett-Burman",
    "genel": "Genel tam faktöriyel",
    "ccd": "Merkezi kompozit (CCD)",
    "bbd": "Box-Behnken",
    "dsd": "Kesin eleme (DSD)",
    "taguchi": "Taguchi",
    "karisim": "Karışım",
    "doptimal": "D-optimal",
}

# Analiz ailesi: hangi analiz türüyle incelenir
AILE = {"faktoriyel2": "faktoriyel", "kesirli": "faktoriyel", "pb": "faktoriyel",
        "genel": "genel", "ccd": "yuzey", "bbd": "yuzey", "dsd": "yuzey",
        "doptimal": "yuzey", "taguchi": "taguchi", "karisim": "karisim",
        "ozel": "yuzey"}


# ════════════════════════════════════════════════════════════════
# Kesirli faktöriyel üreteçleri (Montgomery Tablo 8.14 — Minitab varsayılanları)
# anahtar: (k, p) → üreteç listesi ("D=ABC" gibi)
# ════════════════════════════════════════════════════════════════
KESIRLI_URETECLER = {
    (3, 1): ["C=AB"],
    (4, 1): ["D=ABC"],
    (5, 1): ["E=ABCD"],
    (5, 2): ["D=AB", "E=AC"],
    (6, 1): ["F=ABCDE"],
    (6, 2): ["E=ABC", "F=BCD"],
    (6, 3): ["D=AB", "E=AC", "F=BC"],
    (7, 1): ["G=ABCDEF"],
    (7, 2): ["F=ABCD", "G=ABDE"],
    (7, 3): ["E=ABC", "F=BCD", "G=ACD"],
    (7, 4): ["D=AB", "E=AC", "F=BC", "G=ABC"],
    (8, 2): ["G=ABCD", "H=ABEF"],
    (8, 3): ["F=ABC", "G=ABD", "H=BCDE"],
    (8, 4): ["E=BCD", "F=ACD", "G=ABC", "H=ABD"],
    (9, 2): ["H=ACDFG", "J=BCEFG"],
    (9, 3): ["G=ABCD", "H=ACEF", "J=CDEF"],
    (9, 4): ["F=BCDE", "G=ACDE", "H=ABDE", "J=ABCE"],
    (9, 5): ["E=ABC", "F=BCD", "G=ACD", "H=ABD", "J=ABCD"],
    (10, 3): ["H=ABCG", "J=BCDE", "K=ACDF"],
    (10, 4): ["G=BCDF", "H=ACDF", "J=ABDE", "K=ABCE"],
    (10, 5): ["F=ABCD", "G=ABCE", "H=ABDE", "J=ACDE", "K=BCDE"],
    (10, 6): ["E=ABC", "F=BCD", "G=ACD", "H=ABD", "J=ABCD", "K=AB"],
    (11, 4): ["H=ABCG", "J=BCDE", "K=ACDF", "L=ABCDEFG"],
    (11, 5): ["G=CDE", "H=ABCD", "J=ABF", "K=BDEF", "L=ADEF"],
    (11, 6): ["F=ABC", "G=BCD", "H=CDE", "J=ACD", "K=ADE", "L=BDE"],
    (11, 7): ["E=ABC", "F=BCD", "G=ACD", "H=ABD", "J=ABCD", "K=AB", "L=AC"],
}

# Plackett-Burman döngüsel üreteç satırları (Plackett & Burman, 1946)
PB_URETEC = {
    12: "++-+++---+-",
    20: "++--++++-+-+----++-",
    24: "+++++-+-++--++--+-+----",
}

# Minitab varsayılan merkez nokta sayıları
CCD_MERKEZ = {2: 5, 3: 6, 4: 7, 5: 10, 6: 15, 7: 21}
CCD_MERKEZ_YARIM = {5: 6, 6: 9, 7: 14}
BBD_MERKEZ = {3: 3, 4: 3, 5: 6, 6: 6, 7: 6}
BBD_GRUPLAR = {
    3: [(0, 1), (0, 2), (1, 2)],
    4: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    5: [(i, j) for i in range(5) for j in range(i + 1, 5)],
    6: [(0, 1, 3), (1, 2, 4), (2, 3, 5), (0, 3, 4), (1, 4, 5), (0, 2, 5)],
    7: [(3, 4, 5), (0, 5, 6), (1, 4, 6), (0, 1, 3), (2, 3, 6), (1, 2, 5), (0, 2, 4)],
}

L18 = [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 2, 2, 2, 2, 2, 2], [1, 1, 3, 3, 3, 3, 3, 3],
       [1, 2, 1, 1, 2, 2, 3, 3], [1, 2, 2, 2, 3, 3, 1, 1], [1, 2, 3, 3, 1, 1, 2, 2],
       [1, 3, 1, 2, 1, 3, 2, 3], [1, 3, 2, 3, 2, 1, 3, 1], [1, 3, 3, 1, 3, 2, 1, 2],
       [2, 1, 1, 3, 3, 2, 2, 1], [2, 1, 2, 1, 1, 3, 3, 2], [2, 1, 3, 2, 2, 1, 1, 3],
       [2, 2, 1, 2, 3, 1, 3, 2], [2, 2, 2, 3, 1, 2, 1, 3], [2, 2, 3, 1, 2, 3, 2, 1],
       [2, 3, 1, 3, 2, 3, 1, 2], [2, 3, 2, 1, 3, 1, 2, 3], [2, 3, 3, 2, 1, 2, 3, 1]]

TAGUCHI_DIZILER = {  # ad: (deney sayısı, en fazla 2 seviyeli, en fazla 3 seviyeli)
    "L4": (4, 3, 0), "L8": (8, 7, 0), "L12": (12, 11, 0), "L16": (16, 15, 0),
    "L9": (9, 0, 4), "L27": (27, 0, 13), "L18": (18, 1, 7),
}


# ════════════════════════════════════════════════════════════════
# Temel matrisler
# ════════════════════════════════════════════════════════════════
def tam2(k: int) -> np.ndarray:
    """2^k tam faktöriyel, Yates (standart) sırası: A en hızlı değişir."""
    n = 2 ** k
    return np.array([[1 if (r >> j) & 1 else -1 for j in range(k)] for r in range(n)],
                    dtype=float)


def kelime(s: str) -> frozenset:
    return frozenset(s)


def kesirli_matris(k: int, p: int):
    """2^(k-p) tasarımı, tanım bağıntısı ve çözünürlük."""
    uretec = KESIRLI_URETECLER[(k, p)]
    taban = tam2(k - p)
    harf = HARFLER[:k]
    sut = {harf[j]: taban[:, j] for j in range(k - p)}
    kelimeler = []
    for g in uretec:
        sol, sag = g.split("=")
        v = np.ones(len(taban))
        for h in sag:
            v = v * sut[h]
        sut[sol] = v
        kelimeler.append(kelime(sol + sag))
    D = np.column_stack([sut[h] for h in harf])
    grup = set()
    for r in range(1, p + 1):
        for alt in itertools.combinations(kelimeler, r):
            w = frozenset()
            for x in alt:
                w = w ^ x
            grup.add(w)
    cozunurluk = min(len(w) for w in grup)
    return D, uretec, sorted(grup, key=lambda w: (len(w), sorted(w))), cozunurluk


def roma(n: int) -> str:
    return {3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}.get(n, str(n))


def kesirli_secenekler(k: int) -> list[dict]:
    """k faktör için kullanılabilir kesirli tasarımlar."""
    out = []
    for (kk, p), _ in sorted(KESIRLI_URETECLER.items()):
        if kk != k:
            continue
        _, _, _, res = kesirli_matris(k, p)
        out.append({"p": p, "n": 2 ** (k - p), "cozunurluk": res,
                    "ad": f"1/{2 ** p} kesir — {2 ** (k - p)} deney, çözünürlük {roma(res)}"})
    return sorted(out, key=lambda d: d["n"])


def karisiklik_yapisi(grup, k: int, derece: int = 3) -> list[str]:
    """Minitab 'Alias Structure' — ana etkiler ve 2'li etkileşimler için."""
    harf = HARFLER[:k]
    satirlar = ["I + " + " + ".join("".join(sorted(w, key=harf.index)) for w in grup)]
    goruldu = set()
    for d in (1, 2):
        for e in itertools.combinations(harf, d):
            e = frozenset(e)
            if e in goruldu:
                continue
            esler = [e ^ w for w in grup]
            esler = [a for a in esler if len(a) <= derece]
            tum = [e] + sorted(esler, key=lambda a: (len(a), sorted(a, key=harf.index)))
            for a in tum:
                goruldu.add(a)
            satirlar.append(" + ".join("".join(sorted(a, key=harf.index)) for a in tum))
    return satirlar


def hadamard_sylvester(n: int) -> np.ndarray:
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def pb_matris(n: int) -> np.ndarray:
    """N deneyli Plackett-Burman matrisi (N-1 sütun)."""
    if n in (8, 16, 32):
        H = hadamard_sylvester(n)
        return -H[:, 1:]  # ilk satır tümü -1 olacak şekilde
    g = [1 if c == "+" else -1 for c in PB_URETEC[n]]
    satir = []
    for i in range(n - 1):
        satir.append(g[-i:] + g[:-i] if i else g[:])
    satir.append([-1] * (n - 1))
    return np.array(satir, dtype=float)


def pb_deney_sayisi(k: int) -> list[int]:
    return [n for n in (8, 12, 16, 20, 24) if n - 1 >= k]


# ── Konferans matrisi (DSD için) ─────────────────────────────────
def _legendre(a: int, q: int) -> int:
    a %= q
    if a == 0:
        return 0
    return 1 if pow(a, (q - 1) // 2, q) == 1 else -1


def _asal(n):
    return n > 1 and all(n % d for d in range(2, int(n ** .5) + 1))


def _gf9_jacobsthal():
    # GF(9) = GF(3)[i], i^2 = -1
    el = [(a, b) for a in range(3) for b in range(3)]
    def carp(x, y):
        return ((x[0] * y[0] - x[1] * y[1]) % 3, (x[0] * y[1] + x[1] * y[0]) % 3)
    kareler = {carp(x, x) for x in el if x != (0, 0)}
    def chi(z):
        return 0 if z == (0, 0) else (1 if z in kareler else -1)
    return np.array([[chi(((x[0] - y[0]) % 3, (x[1] - y[1]) % 3)) for y in el] for x in el])


def konferans(n: int) -> np.ndarray | None:
    """n mertebeli konferans matrisi (Paley). Yoksa None."""
    if n == 2:
        return np.array([[0, 1], [1, 0]], dtype=float)
    q = n - 1
    if q == 9:
        Q = _gf9_jacobsthal()
        dort = 1
    elif _asal(q):
        Q = np.array([[_legendre(i - j, q) for j in range(q)] for i in range(q)])
        dort = q % 4
    else:
        return None
    C = np.zeros((n, n))
    C[0, 1:] = 1
    C[1:, 0] = 1 if dort == 1 else -1
    C[1:, 1:] = Q
    return C


def dsd_matris(m: int):
    """m sürekli faktör için DSD: [C; -C; 0]. Kullanılan mertebe de döner."""
    n = m if m % 2 == 0 else m + 1
    while konferans(n) is None:
        n += 2
    C = konferans(n)[:, :m]
    return np.vstack([C, -C, np.zeros((1, m))]), n


def dsd_deney_sayisi(m: int) -> int:
    return 2 * dsd_matris(m)[1] + 1


# ── Taguchi ──────────────────────────────────────────────────────
def taguchi_dizi(ad: str) -> np.ndarray:
    """Seviyeleri 1, 2, (3) olan ortogonal dizi."""
    if ad in ("L4", "L8", "L16"):
        b = {"L4": 2, "L8": 3, "L16": 4}[ad]
        n = 2 ** b
        satirlar = []
        for r in range(n):
            bit = [(r >> (b - 1 - i)) & 1 for i in range(b)]  # A en yavaş
            satir = []
            for j in range(1, n):
                s = sum(bit[i] for i in range(b) if (j >> i) & 1) % 2
                satir.append(s + 1)
            satirlar.append(satir)
        return np.array(satirlar)
    if ad == "L12":
        return ((pb_matris(12) + 1) / 2 + 1).astype(int)
    if ad in ("L9", "L27"):
        b = 2 if ad == "L9" else 3
        vekt = []
        for v in itertools.product(range(3), repeat=b):
            if any(v) and v[next(i for i in range(b) if v[i])] == 1:
                vekt.append(v)
        vekt.sort(key=lambda v: (sum(1 for x in v if x), [-x for x in v]))
        if ad == "L9":
            vekt = [(1, 0), (0, 1), (1, 1), (2, 1)]
        satirlar = []
        for r in range(3 ** b):
            taban = [(r // 3 ** (b - 1 - i)) % 3 for i in range(b)]
            satirlar.append([sum(c * t for c, t in zip(v, taban)) % 3 + 1 for v in vekt])
        return np.array(satirlar)
    if ad == "L18":
        return np.array(L18)
    raise ValueError(ad)


def taguchi_oner(n2: int, n3: int) -> list[str]:
    """2 ve 3 seviyeli faktör sayısına uyan diziler (küçükten büyüğe)."""
    uygun = []
    for ad, (n, m2, m3) in sorted(TAGUCHI_DIZILER.items(), key=lambda x: x[1][0]):
        if n3 == 0 and n2 <= m2 and m3 == 0:
            uygun.append(ad)
        elif n2 == 0 and n3 <= m3 and m2 == 0:
            uygun.append(ad)
        elif ad == "L18" and n2 <= 1 and n3 <= 7 and n2 + n3 >= 1:
            uygun.append(ad)
    return uygun


# ════════════════════════════════════════════════════════════════
# Faktör tanımı ve kodlama
# ════════════════════════════════════════════════════════════════
def faktor(isim, tip, dusuk=None, yuksek=None, seviyeler=None, harf=None):
    """
    tip='nicel' → dusuk/yuksek (kodlu -1/+1 karşılıkları)
    tip='nitel' → seviyeler (metin listesi)
    """
    return {"isim": isim, "tip": tip, "dusuk": dusuk, "yuksek": yuksek,
            "seviyeler": list(seviyeler) if seviyeler else None, "harf": harf,
            "merkez": None if tip != "nicel" else (dusuk + yuksek) / 2,
            "yari": None if tip != "nicel" else (yuksek - dusuk) / 2}


def _yuvarla(v, f):
    aralik = abs(f["yuksek"] - f["dusuk"]) if f["tip"] == "nicel" else 1
    bas = 4 if aralik < 1 else 3 if aralik < 10 else 2
    return round(float(v), bas)


def _cozkod(c, f):
    return _yuvarla(f["merkez"] + c * f["yari"], f)


def _harfle(faktorler):
    for i, f in enumerate(faktorler):
        f["harf"] = HARFLER[i]
    return faktorler


def _nitel_kombinasyonlar(faktorler):
    nitel = [f for f in faktorler if f["tip"] == "nitel"]
    if not nitel:
        return [{}]
    return [dict(zip([f["isim"] for f in nitel], k))
            for k in itertools.product(*[f["seviyeler"] for f in nitel])]


# ════════════════════════════════════════════════════════════════
# Ortak son işlem: tekrar, blok, rastgeleleştirme, Minitab sütunları
# ════════════════════════════════════════════════════════════════
def _sonlandir(meta, temel, tekrar=1, tekrar_blok=False, rastgele=True, tohum=None):
    """
    temel: [{'pt': int, 'blok': int, 'deg': {faktör: değer}}]  (standart sıra)
    """
    satirlar = []
    std = 0
    for t in range(tekrar):
        for s in temel:
            std += 1
            satirlar.append({"StdOrder": std, "pt": s["pt"],
                             "blok": (t + 1) if tekrar_blok else s.get("blok", 1),
                             "deg": dict(s["deg"])})
    if tohum is None:
        tohum = int(np.random.default_rng().integers(1, 99999))
    rng = np.random.default_rng(tohum)
    if rastgele:
        bloklar = sorted({s["blok"] for s in satirlar})
        sira = []
        for b in bloklar:
            idx = [i for i, s in enumerate(satirlar) if s["blok"] == b]
            sira += list(rng.permutation(idx))
    else:
        sira = list(range(len(satirlar)))
    for run, i in enumerate(sira, 1):
        satirlar[i]["RunOrder"] = run
    meta.update({"tekrar": tekrar, "tekrar_blok": tekrar_blok, "rastgele": rastgele,
                 "tohum": tohum, "deney": len(satirlar),
                 "blok_sayisi": len({s["blok"] for s in satirlar})})
    return meta, satirlar


def _ozet_ortak(meta, ek):
    s = [("Tasarım", meta["ad"]), ("Faktör sayısı", len(meta["faktorler"])),
         ("Deney sayısı", meta["deney"]), ("Tekrar", meta["tekrar"]),
         ("Blok", meta["blok_sayisi"])]
    s += ek
    s += [("Rastgeleleştirme", f"Evet (tohum {meta['tohum']})" if meta["rastgele"] else "Hayır")]
    meta["ozet"] = s
    return meta


# ════════════════════════════════════════════════════════════════
# Tasarım üreticileri
# ════════════════════════════════════════════════════════════════
def _iki_seviye(D, faktorler, merkez=0, pt_ad="CenterPt"):
    """±1 kodlu matris → temel satırlar (nitel faktörler 2 seviyeli)."""
    temel = []
    for r in D:
        deg = {}
        for c, f in zip(r, faktorler):
            if f["tip"] == "nicel":
                deg[f["isim"]] = _cozkod(c, f)
            else:
                deg[f["isim"]] = f["seviyeler"][0 if c < 0 else 1]
        temel.append({"pt": 1, "deg": deg})
    nicel = [f for f in faktorler if f["tip"] == "nicel"]
    if merkez and nicel:
        for komb in _nitel_kombinasyonlar(faktorler):
            for _ in range(merkez):
                deg = {f["isim"]: _cozkod(0, f) for f in nicel}
                deg.update(komb)
                temel.append({"pt": 0, "deg": deg})
    return temel


def _iki_seviye_kontrol(faktorler):
    for f in faktorler:
        if f["tip"] == "nitel" and len(f["seviyeler"]) != 2:
            raise ValueError(f"'{f['isim']}' için tam olarak 2 kategori seçin "
                             f"(2 seviyeli tasarım).")


def faktoriyel2(faktorler, merkez=0, **ort):
    _iki_seviye_kontrol(faktorler)
    k = len(faktorler)
    if not 2 <= k <= 10:
        raise ValueError("2 seviyeli tam faktöriyel için 2–10 faktör seçin.")
    meta = {"tip": "faktoriyel2", "ad": f"2 seviyeli tam faktöriyel (2^{k})",
            "faktorler": _harfle(faktorler), "pt_sutun": "CenterPt", "merkez": merkez,
            "cozunurluk": "Tam", "karisiklik": []}
    meta, satirlar = _sonlandir(meta, _iki_seviye(tam2(k), faktorler, merkez), **ort)
    return _ozet_ortak(meta, [("Temel tasarım", f"{k}, {2 ** k}"),
                              ("Merkez nokta (blok başına)", merkez),
                              ("Çözünürlük", "Tam")]), satirlar


def kesirli(faktorler, p=1, merkez=0, **ort):
    _iki_seviye_kontrol(faktorler)
    k = len(faktorler)
    if (k, p) not in KESIRLI_URETECLER:
        raise ValueError(f"{k} faktör için 1/{2 ** p} kesir tanımlı değil.")
    D, uretec, grup, res = kesirli_matris(k, p)
    meta = {"tip": "kesirli", "ad": f"2 seviyeli kesirli faktöriyel (2^({k}-{p}))",
            "faktorler": _harfle(faktorler), "pt_sutun": "CenterPt", "merkez": merkez,
            "cozunurluk": res, "uretecler": uretec,
            "karisiklik": karisiklik_yapisi(grup, k), "p": p}
    meta, satirlar = _sonlandir(meta, _iki_seviye(D, faktorler, merkez), **ort)
    return _ozet_ortak(meta, [("Temel tasarım", f"{k}, {2 ** (k - p)}"),
                              ("Kesir", f"1/{2 ** p}"),
                              ("Çözünürlük", roma(res)),
                              ("Üreteçler", ", ".join(uretec)),
                              ("Merkez nokta (blok başına)", merkez)]), satirlar


def pb(faktorler, n=None, merkez=0, **ort):
    _iki_seviye_kontrol(faktorler)
    k = len(faktorler)
    secenek = pb_deney_sayisi(k)
    if not secenek:
        raise ValueError("Plackett-Burman en fazla 23 faktörle kullanılabilir.")
    n = n if n in secenek else secenek[0]
    D = pb_matris(n)[:, :k]
    meta = {"tip": "pb", "ad": f"Plackett-Burman ({n} deney)",
            "faktorler": _harfle(faktorler), "pt_sutun": "CenterPt", "merkez": merkez,
            "cozunurluk": 3, "karisiklik": ["Ana etkiler 2'li etkileşimlerle kısmen karışıktır "
                                            "(çözünürlük III). Sadece ana etkiler tahmin edilir."]}
    meta, satirlar = _sonlandir(meta, _iki_seviye(D, faktorler, merkez), **ort)
    return _ozet_ortak(meta, [("Temel deney sayısı", n), ("Çözünürlük", "III"),
                              ("Merkez nokta (blok başına)", merkez)]), satirlar


def genel(faktorler, nicel_seviye=3, **ort):
    """Nicel faktörler eşit aralıklı 'nicel_seviye' seviyede; nitel faktörler kategorileriyle."""
    seviyeler = []
    for f in faktorler:
        if f["tip"] == "nicel":
            s = int(f.get("seviye_sayisi") or nicel_seviye)
            f["seviyeler_sayi"] = [_yuvarla(v, f) for v in np.linspace(f["dusuk"], f["yuksek"], s)]
            seviyeler.append(f["seviyeler_sayi"])
        else:
            seviyeler.append(f["seviyeler"])
    toplam = int(np.prod([len(s) for s in seviyeler]))
    if toplam > 1000:
        raise ValueError(f"Tasarım {toplam} deney içeriyor — seviye sayısını azaltın.")
    temel = []
    for komb in itertools.product(*seviyeler):  # A en yavaş (Minitab genel faktöriyel)
        temel.append({"pt": 1, "deg": {f["isim"]: v for f, v in zip(faktorler, komb)}})
    meta = {"tip": "genel", "ad": "Genel tam faktöriyel",
            "faktorler": _harfle(faktorler), "pt_sutun": "PtType"}
    meta, satirlar = _sonlandir(meta, temel, **ort)
    return _ozet_ortak(meta, [("Seviyeler", " × ".join(str(len(s)) for s in seviyeler))]), satirlar


def ccd(faktorler, cesit="CCC", merkez=None, yarim=False, **ort):
    nicel = [f for f in faktorler if f["tip"] == "nicel"]
    k = len(nicel)
    if k < 2:
        raise ValueError("Merkezi kompozit tasarım en az 2 nicel faktör gerektirir.")
    if k > 7:
        raise ValueError("Merkezi kompozit tasarım en fazla 7 nicel faktörle kullanılabilir.")
    if yarim and k >= 5:
        kup, _, grup, _ = kesirli_matris(k, 1)
    else:
        kup, yarim = tam2(k), False
    nf = len(kup)
    alfa = nf ** 0.25
    if merkez is None:
        merkez = (CCD_MERKEZ_YARIM if yarim else CCD_MERKEZ).get(k, 6)
    # Kodlama: küp noktaları ±1. CCI'da girilen düşük/yüksek eksenel noktalardır.
    for f in nicel:
        if cesit == "CCI":
            f["yari"] = (f["yuksek"] - f["dusuk"]) / 2 / alfa
        else:
            f["yari"] = (f["yuksek"] - f["dusuk"]) / 2
    a = 1.0 if cesit == "CCF" else alfa
    eksen = []
    for j in range(k):
        for s in (-1, 1):
            v = np.zeros(k)
            v[j] = s * a
            eksen.append(v)
    temel = []
    for komb in _nitel_kombinasyonlar(faktorler):
        for pt, M in ((1, kup), (-1, np.array(eksen)), (0, np.zeros((merkez, k)))):
            for r in M:
                deg = {f["isim"]: _cozkod(c, f) for c, f in zip(r, nicel)}
                deg.update(komb)
                temel.append({"pt": pt, "deg": deg})
    ad = {"CCC": "çevrel (CCC)", "CCI": "içsel (CCI)", "CCF": "yüzey merkezli (CCF)"}[cesit]
    meta = {"tip": "ccd", "ad": f"Merkezi kompozit — {ad}", "faktorler": _harfle(faktorler),
            "pt_sutun": "PtType", "alfa": a, "cesit": cesit, "merkez": merkez, "yarim": yarim}
    uyari = []
    if cesit == "CCC":
        for f in nicel:
            alt, ust = _cozkod(-a, f), _cozkod(a, f)
            if f.get("sinir_min") is not None and (alt < f["sinir_min"] or ust > f["sinir_max"]):
                uyari.append(f"{f['isim']}: eksenel noktalar {alt:g} ve {ust:g} — tanımlı aralığın "
                             f"({f['sinir_min']:g}–{f['sinir_max']:g}) dışında. Makine bu değerlere "
                             f"çıkamıyorsa CCI veya CCF seçin.")
    meta["uyarilar"] = uyari
    meta, satirlar = _sonlandir(meta, temel, **ort)
    return _ozet_ortak(meta, [("Küp noktaları", nf), ("Eksenel noktalar", 2 * k),
                              ("Merkez noktalar", merkez), ("Alfa", round(a, 4)),
                              ("Küp", "Yarım kesir" if yarim else "Tam")]), satirlar


def bbd(faktorler, merkez=None, **ort):
    nicel = [f for f in faktorler if f["tip"] == "nicel"]
    k = len(nicel)
    if k not in BBD_GRUPLAR:
        raise ValueError("Box-Behnken 3–7 nicel faktörle kullanılabilir.")
    merkez = BBD_MERKEZ[k] if merkez is None else merkez
    noktalar = []
    for grup in BBD_GRUPLAR[k]:
        for isaret in itertools.product((-1, 1), repeat=len(grup)):
            v = np.zeros(k)
            for g, s in zip(grup, isaret):
                v[g] = s
            noktalar.append(v)
    temel = []
    for komb in _nitel_kombinasyonlar(faktorler):
        for pt, M in ((2, noktalar), (0, np.zeros((merkez, k)))):
            for r in M:
                deg = {f["isim"]: _cozkod(c, f) for c, f in zip(r, nicel)}
                deg.update(komb)
                temel.append({"pt": pt, "deg": deg})
    meta = {"tip": "bbd", "ad": "Box-Behnken", "faktorler": _harfle(faktorler),
            "pt_sutun": "PtType", "merkez": merkez}
    meta, satirlar = _sonlandir(meta, temel, **ort)
    return _ozet_ortak(meta, [("Kenar noktaları", len(noktalar)),
                              ("Merkez noktalar", merkez)]), satirlar


def dsd(faktorler, ek_merkez=0, **ort):
    if any(f["tip"] == "nitel" for f in faktorler):
        raise ValueError("Kesin eleme tasarımı bu uygulamada sadece nicel faktörlerle "
                         "oluşturulur. Nitel faktörleri sabit tutun.")
    m = len(faktorler)
    if m < 3:
        raise ValueError("Kesin eleme tasarımı en az 3 nicel faktör gerektirir.")
    D, n = dsd_matris(m)
    temel = []
    for r in D:
        temel.append({"pt": 0 if not np.any(r) else 1,
                      "deg": {f["isim"]: _cozkod(c, f) for c, f in zip(r, faktorler)}})
    for _ in range(ek_merkez):
        temel.append({"pt": 0, "deg": {f["isim"]: _cozkod(0, f) for f in faktorler}})
    meta = {"tip": "dsd", "ad": f"Kesin eleme (DSD), {m} faktör",
            "faktorler": _harfle(faktorler), "pt_sutun": "CenterPt", "konferans": n}
    meta, satirlar = _sonlandir(meta, temel, **ort)
    return _ozet_ortak(meta, [("Konferans matrisi mertebesi", n),
                              ("Seviye", "3 (−1, 0, +1)")]), satirlar


def taguchi(faktorler, dizi=None, tekrar_sutun=1, **ort):
    """Her faktör 2 veya 3 seviyeli. Nicel: f['taguchi_seviye'] (2/3); nitel: seçilen kategoriler."""
    seviye = []
    for f in faktorler:
        if f["tip"] == "nicel":
            s = int(f.get("taguchi_seviye") or 2)
            f["seviyeler_sayi"] = [_yuvarla(v, f) for v in np.linspace(f["dusuk"], f["yuksek"], s)]
        else:
            s = len(f["seviyeler"])
        if s not in (2, 3):
            raise ValueError(f"'{f['isim']}': Taguchi dizileri için 2 veya 3 seviye seçin.")
        seviye.append(s)
    n2, n3 = seviye.count(2), seviye.count(3)
    uygun = taguchi_oner(n2, n3)
    if not uygun:
        raise ValueError("Bu seviye karışımına uygun dizi yok (L18: en fazla 1 adet 2 seviyeli "
                         "+ 7 adet 3 seviyeli).")
    dizi = dizi if dizi in uygun else uygun[0]
    A = taguchi_dizi(dizi)
    # Sütun ataması: 2 seviyeliler 2 seviyeli sütunlara, 3 seviyeliler 3 seviyeli sütunlara
    sut2 = [j for j in range(A.shape[1]) if A[:, j].max() == 2]
    sut3 = [j for j in range(A.shape[1]) if A[:, j].max() == 3]
    atama, i2, i3 = [], 0, 0
    for s in seviye:
        if s == 2:
            atama.append(sut2[i2]); i2 += 1
        else:
            atama.append(sut3[i3]); i3 += 1
    temel = []
    for r in A:
        deg = {}
        for f, j in zip(faktorler, atama):
            sev = (f["seviyeler_sayi"] if f["tip"] == "nicel" else f["seviyeler"])
            deg[f["isim"]] = sev[r[j] - 1]
        temel.append({"pt": 1, "deg": deg})
    meta = {"tip": "taguchi", "ad": f"Taguchi {dizi}", "faktorler": _harfle(faktorler),
            "pt_sutun": None, "dizi": dizi, "tekrar_sutun": int(tekrar_sutun),
            "sutunlar": [j + 1 for j in atama]}
    ort = dict(ort, rastgele=False)
    meta, satirlar = _sonlandir(meta, temel, **ort)
    return _ozet_ortak(meta, [("Dizi", dizi), ("Dizi sütunları", ", ".join(map(str, meta["sutunlar"]))),
                              ("Tekrar sütunu (gürültü koşulu)", tekrar_sutun)]), satirlar


def karisim(faktorler, cesit="centroid", derece=2, eksenel=True, merkez_ekle=True,
            toplam=1.0, **ort):
    if any(f["tip"] == "nitel" for f in faktorler):
        raise ValueError("Karışım tasarımında bileşenler nicel olmalıdır.")
    q = len(faktorler)
    if not 2 <= q <= 10:
        raise ValueError("Karışım tasarımı 2–10 bileşenle kullanılabilir.")
    alt = np.array([max(0.0, f["dusuk"] or 0.0) for f in faktorler])
    kalan = toplam - alt.sum()
    if kalan <= 0:
        raise ValueError("Alt sınırların toplamı karışım toplamından küçük olmalı.")
    merkez = np.full(q, 1 / q)
    noktalar = []  # (sözde bileşen oranları, PtType)
    def ekle(v, pt):
        if not any(np.allclose(v, p) for p, _ in noktalar):
            noktalar.append((v, pt))
    if cesit == "lattice":
        for v in itertools.product(range(derece, -1, -1), repeat=q):
            if sum(v) == derece:
                z = np.array(v) / derece
                ekle(z, 0 if np.allclose(z, merkez) else int((z > 1e-9).sum()))
    else:
        for r in range(1, q + 1):
            for alt_k in itertools.combinations(range(q), r):
                z = np.zeros(q)
                z[list(alt_k)] = 1 / r
                ekle(z, 0 if r == q else r)
    if merkez_ekle:
        ekle(merkez, 0)
    if eksenel:
        for i in range(q):
            z = np.full(q, 1 / (2 * q))
            z[i] = (q + 1) / (2 * q)
            ekle(z, -1)
    temel = []
    for z, pt in noktalar:
        gercek = alt + kalan * z
        temel.append({"pt": pt, "deg": {f["isim"]: round(float(x), 5)
                                        for f, x in zip(faktorler, gercek)}})
    for f, a in zip(faktorler, alt):
        f["kar_alt"] = float(a)
    meta = {"tip": "karisim", "ad": ("Simplex centroid" if cesit == "centroid"
                                     else f"Simplex lattice (derece {derece})"),
            "faktorler": _harfle(faktorler), "pt_sutun": "PtType", "toplam": toplam,
            "kalan": float(kalan), "cesit": cesit, "derece": derece}
    meta, satirlar = _sonlandir(meta, temel, **ort)
    return _ozet_ortak(meta, [("Bileşen", q), ("Karışım toplamı", toplam),
                              ("Alt sınırlar", ", ".join(f"{f['isim']}≥{f['kar_alt']:g}"
                                                         for f in faktorler)),
                              ("Eksenel nokta", "Evet" if eksenel else "Hayır")]), satirlar


def doptimal(faktorler, model="kuadratik", n=None, merkez=0, deneme=6, **ort):
    """Aday noktalardan (nicel: -1, 0, +1; nitel: tüm kategoriler) D-optimal seçim."""
    import doe_analiz as A
    faktorler = _harfle(faktorler)
    seviye = []
    for f in faktorler:
        if f["tip"] == "nicel":
            seviye.append([-1.0, 0.0, 1.0] if model in ("kuadratik", "kare") else [-1.0, 1.0])
        else:
            seviye.append(list(f["seviyeler"]))
    aday = list(itertools.product(*seviye))
    rng = np.random.default_rng(ort.get("tohum"))
    if len(aday) > 3000:
        aday = [aday[i] for i in rng.choice(len(aday), 3000, replace=False)]
    meta_g = {"tip": "doptimal", "faktorler": faktorler}
    terimler = A.varsayilan_terimler(meta_g, model)
    def satira(c):
        return {f["isim"]: (_cozkod(v, f) if f["tip"] == "nicel" else v)
                for f, v in zip(faktorler, c)}
    X_aday, _, _ = A.model_matrisi(meta_g, [satira(c) for c in aday], terimler)
    p = X_aday.shape[1]
    n = int(n or p + 5)
    if n < p:
        raise ValueError(f"Bu model {p} parametre içeriyor; en az {p} deney gerekli.")
    en_iyi, en_iyi_d = None, -np.inf
    ridge = 1e-6 * np.eye(p)
    for _ in range(int(deneme)):
        sec = list(rng.choice(len(aday), n, replace=len(aday) < n))
        # Fedorov değişimi: det(M') = det(M)·(1 + Δ)
        for _tur in range(50):
            X = X_aday[sec]
            Minv = np.linalg.inv(X.T @ X + ridge)
            dc = np.einsum("ij,jk,ik->i", X_aday, Minv, X_aday)
            en = (0.0, None, None)
            for i in range(n):
                xi = X_aday[sec[i]]
                di = xi @ Minv @ xi
                dic = X_aday @ (Minv @ xi)
                delta = dc - (di * dc - dic ** 2) - di
                j = int(np.argmax(delta))
                if delta[j] > en[0]:
                    en = (float(delta[j]), i, j)
            if en[0] <= 1e-9:
                break
            sec[en[1]] = en[2]
        X = X_aday[sec]
        sgn, d = np.linalg.slogdet(X.T @ X)
        d = d if sgn > 0 else -np.inf
        if d > en_iyi_d:
            en_iyi, en_iyi_d = sec, d
    if not np.isfinite(en_iyi_d):
        raise ValueError("Bu model için uygun D-optimal tasarım bulunamadı; deney sayısını artırın.")
    temel = [{"pt": 1, "deg": satira(aday[i])} for i in sorted(en_iyi, key=lambda i: aday[i])]
    if merkez and any(f["tip"] == "nicel" for f in faktorler):
        for komb in _nitel_kombinasyonlar(faktorler):
            for _ in range(merkez):
                deg = {f["isim"]: _cozkod(0, f) for f in faktorler if f["tip"] == "nicel"}
                deg.update(komb)
                temel.append({"pt": 0, "deg": deg})
    meta = {"tip": "doptimal", "ad": f"D-optimal ({model} model)", "faktorler": faktorler,
            "pt_sutun": "PtType", "model": model, "d_kriteri": float(en_iyi_d)}
    meta, satirlar = _sonlandir(meta, temel, **ort)
    return _ozet_ortak(meta, [("Model", model), ("Model parametre sayısı", p),
                              ("log det(X'X)", round(en_iyi_d, 3)),
                              ("Aday nokta", len(aday))]), satirlar


URETICI = {"faktoriyel2": faktoriyel2, "kesirli": kesirli, "pb": pb, "genel": genel,
           "ccd": ccd, "bbd": bbd, "dsd": dsd, "taguchi": taguchi, "karisim": karisim,
           "doptimal": doptimal}


def olustur(tip, faktorler, **secenek):
    faktorler = [dict(f) for f in faktorler]
    return URETICI[tip](faktorler, **secenek)


# ════════════════════════════════════════════════════════════════
# Minitab çalışma sayfası
# ════════════════════════════════════════════════════════════════
def yanit_sutunlari(meta, ciktilar: list[str]) -> list[str]:
    r = meta.get("tekrar_sutun", 1) if meta["tip"] == "taguchi" else 1
    if r > 1:
        return [f"{c}_{i}" for c in ciktilar for i in range(1, r + 1)]
    return list(ciktilar)


def calisma_sayfasi(meta, satirlar, ciktilar, sirala="RunOrder"):
    """Minitab düzeni: StdOrder RunOrder CenterPt/PtType Blocks faktörler yanıtlar."""
    fis = [f["isim"] for f in meta["faktorler"]]
    yan = yanit_sutunlari(meta, ciktilar)
    if meta["tip"] == "taguchi":
        bas = fis + yan
    else:
        bas = ["StdOrder", "RunOrder", meta["pt_sutun"], "Blocks"] + fis + yan
    bas = bas + ["Veri_ID"]
    sira = sorted(satirlar, key=lambda s: s[sirala] if sirala in s else s["StdOrder"])
    rows = [bas]
    for s in sira:
        if meta["tip"] == "taguchi":
            r = [s["deg"][f] for f in fis]
        else:
            r = [s["StdOrder"], s["RunOrder"], s["pt"], s["blok"]] + [s["deg"][f] for f in fis]
        rows.append(r + [""] * len(yan) + [""])
    return rows
