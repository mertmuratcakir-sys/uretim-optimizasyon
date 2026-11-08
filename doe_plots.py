"""
doe_plots.py — Minitab tarzı DOE grafikleri
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import doe_analiz as A

MAVI, KIRMIZI, GRI, YESIL = "#2a6fdb", "#d1342f", "#8a8f98", "#1a9e5a"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})


def _tek_sd(s):
    """Pareto/normal grafiği için tek serbestlik dereceli terimler."""
    out = []
    for k in s["katsayi"]:
        t = k["_terim"]
        if t is None or t == (A.BLK,) or k["T"] is None:
            continue
        out.append((k["Terim"], float(k["T"])))
    return out


def pareto(s):
    v = _tek_sd(s)
    if not v:
        return None
    lenth = s["lenth"]
    ref = lenth["t_kritik"] if lenth else s["tcrit"]
    v = sorted(v, key=lambda x: abs(x[1]))
    fig, ax = plt.subplots(figsize=(7, 0.8 + 0.32 * len(v)))
    renk = [KIRMIZI if ref and abs(t) > ref else MAVI for _, t in v]
    ax.barh([a for a, _ in v], [abs(t) for _, t in v], color=renk)
    if ref:
        ax.axvline(ref, color=KIRMIZI, ls="--", lw=1)
        ax.text(ref, len(v) - 0.4, f" {ref:.3f}", color=KIRMIZI, fontsize=8, va="bottom")
    ax.set_xlabel("Standartlaştırılmış etki" + (" (Lenth PSE)" if lenth else ""))
    ax.set_title(f"Standartlaştırılmış Etkilerin Pareto Grafiği\n(yanıt: {s['yanit']}, α = {s['alfa']})",
                 fontsize=10)
    fig.tight_layout()
    return fig


def normal_etki(s):
    v = _tek_sd(s)
    if len(v) < 3:
        return None
    lenth = s["lenth"]
    ref = lenth["t_kritik"] if lenth else s["tcrit"]
    v = sorted(v, key=lambda x: x[1])
    n = len(v)
    q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    tv = np.array([t for _, t in v])
    anl = np.abs(tv) > (ref or np.inf)
    ax.scatter(tv[~anl], q[~anl], color=MAVI, s=25, label="Anlamlı değil")
    ax.scatter(tv[anl], q[anl], color=KIRMIZI, marker="s", s=30, label="Anlamlı")
    for (a, t), qq, an in zip(v, q, anl):
        if an:
            ax.annotate(a, (t, qq), xytext=(4, 0), textcoords="offset points", fontsize=8)
    sd = np.std(tv[~anl]) if (~anl).sum() > 1 else np.std(tv)
    xs = np.linspace(min(tv.min(), -1), max(tv.max(), 1), 10)
    ax.plot(xs, xs / (sd if sd > 0 else 1), color=GRI, lw=1)
    ax.set_xlabel("Standartlaştırılmış etki")
    ax.set_ylabel("Normal skor")
    ax.set_title(f"Standartlaştırılmış Etkilerin Normal Grafiği (yanıt: {s['yanit']})", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def _seviye_ort(satirlar, yanit, ad):
    d = {}
    for s in satirlar:
        y = A._sayi(s["y"].get(yanit))
        if y is None:
            continue
        d.setdefault(s["deg"][ad], []).append(y)
    try:
        anahtar = sorted(d, key=float)
    except (TypeError, ValueError):
        anahtar = sorted(d, key=str)
    return anahtar, [np.mean(d[k]) for k in anahtar]


def ana_etki(meta, satirlar, yanit, baslik="Ana Etkiler Grafiği", y_et=None):
    fs = meta["faktorler"]
    n = len(fs)
    sut = min(4, n)
    sat = int(np.ceil(n / sut))
    fig, axes = plt.subplots(sat, sut, figsize=(7, 2.3 * sat), squeeze=False, sharey=True)
    genel = [A._sayi(s["y"].get(yanit)) for s in satirlar]
    genel = np.mean([g for g in genel if g is not None])
    for ax, f in zip(axes.flat, fs):
        k, v = _seviye_ort(satirlar, yanit, f["isim"])
        ax.plot(range(len(k)), v, "o-", color=MAVI)
        ax.axhline(genel, color=GRI, ls="--", lw=0.8)
        ax.set_xticks(range(len(k)))
        ax.set_xticklabels([A.seviye_yaz(x) for x in k], fontsize=7, rotation=30)
        ax.set_title(f["isim"], fontsize=9)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel(y_et or f"Ortalama {yanit}")
    fig.suptitle(f"{baslik} — {y_et or yanit}", fontsize=10)
    fig.tight_layout()
    return fig


def etkilesim(meta, satirlar, yanit):
    fs = meta["faktorler"][:5]
    ciftler = list(itertools.combinations(fs, 2))[:6]
    if not ciftler:
        return None
    sut = 2 if len(ciftler) > 1 else 1
    sat = int(np.ceil(len(ciftler) / sut))
    fig, axes = plt.subplots(sat, sut, figsize=(7, 2.6 * sat), squeeze=False)
    renkler = [MAVI, KIRMIZI, YESIL, "#e8710a", "#8e44ad"]
    for ax, (f1, f2) in zip(axes.flat, ciftler):
        k1, _ = _seviye_ort(satirlar, yanit, f1["isim"])
        k2, _ = _seviye_ort(satirlar, yanit, f2["isim"])
        for j, l2 in enumerate(k2[:5]):
            ort = []
            for l1 in k1:
                v = [A._sayi(s["y"].get(yanit)) for s in satirlar
                     if s["deg"][f1["isim"]] == l1 and s["deg"][f2["isim"]] == l2]
                v = [x for x in v if x is not None]
                ort.append(np.mean(v) if v else np.nan)
            ax.plot(range(len(k1)), ort, "o-", color=renkler[j % 5],
                    label=f"{f2['isim']}={A.seviye_yaz(l2)}")
        ax.set_xticks(range(len(k1)))
        ax.set_xticklabels([A.seviye_yaz(x) for x in k1], fontsize=7)
        ax.set_xlabel(f1["isim"], fontsize=8)
        ax.legend(fontsize=6, frameon=False)
    for ax in list(axes.flat)[len(ciftler):]:
        ax.set_visible(False)
    fig.suptitle(f"Etkileşim Grafiği — {yanit} (veri ortalamaları)", fontsize=10)
    fig.tight_layout()
    return fig


def artik4(s):
    if s["mse"] is None:
        return None
    r = s["std_artik"]
    fig, ax = plt.subplots(2, 2, figsize=(7, 5.2))
    n = len(r)
    q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    ax[0, 0].scatter(np.sort(r), q, s=12, color=MAVI)
    ax[0, 0].plot([-3, 3], [-3, 3], color=KIRMIZI, lw=0.8)
    ax[0, 0].set_title("Normal olasılık grafiği", fontsize=9)
    ax[0, 0].set_xlabel("Std. artık"); ax[0, 0].set_ylabel("Normal skor")
    ax[0, 1].scatter(s["yhat"], r, s=12, color=MAVI)
    ax[0, 1].axhline(0, color=GRI, lw=0.8)
    ax[0, 1].set_title("Uyum değerine karşı", fontsize=9)
    ax[0, 1].set_xlabel("Uyum"); ax[0, 1].set_ylabel("Std. artık")
    ax[1, 0].hist(r, bins=max(5, int(np.sqrt(n)) + 1), color=MAVI, edgecolor="white")
    ax[1, 0].set_title("Histogram", fontsize=9); ax[1, 0].set_xlabel("Std. artık")
    sira = np.argsort(s["runorder"])
    ax[1, 1].plot(np.array(s["runorder"])[sira], r[sira], "o-", color=MAVI, ms=3)
    ax[1, 1].axhline(0, color=GRI, lw=0.8)
    ax[1, 1].set_title("Deney sırasına karşı", fontsize=9)
    ax[1, 1].set_xlabel("RunOrder"); ax[1, 1].set_ylabel("Std. artık")
    fig.suptitle(f"Artık Grafikleri — {s['yanit']}", fontsize=10)
    fig.tight_layout()
    return fig


def _izgara(s, fx, fy, sabit, n=45):
    meta = s["meta"]
    a = meta.get("alfa", 1.0) if meta["tip"] == "ccd" else 1.0
    x = np.linspace(fx["merkez"] - a * fx["yari"], fx["merkez"] + a * fx["yari"], n)
    y = np.linspace(fy["merkez"] - a * fy["yari"], fy["merkez"] + a * fy["yari"], n)
    XX, YY = np.meshgrid(x, y)
    ayar = []
    for xv, yv in zip(XX.ravel(), YY.ravel()):
        d = dict(sabit)
        d[fx["isim"]], d[fy["isim"]] = xv, yv
        ayar.append(d)
    Z, _ = A.tahmin(s, ayar)
    return XX, YY, Z.reshape(XX.shape)


def kontur(s, x_ad, y_ad, sabit, hedef=None):
    meta = s["meta"]
    fx, fy = A._faktor(meta, x_ad), A._faktor(meta, y_ad)
    XX, YY, Z = _izgara(s, fx, fy, sabit)
    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(XX, YY, Z, levels=14, cmap="viridis")
    ax.contour(XX, YY, Z, levels=14, colors="white", linewidths=0.4)
    fig.colorbar(cs, ax=ax, label=s["yanit"])
    if hedef and hedef[0] is not None and hedef[1] is not None:
        ax.contour(XX, YY, Z, levels=[hedef[0], hedef[1]], colors=KIRMIZI, linewidths=1.8)
        ax.contourf(XX, YY, Z, levels=[hedef[0], hedef[1]], colors="none", hatches=["//"])
    noktalar = [(r["deg"][x_ad], r["deg"][y_ad]) for r in s["satirlar"]]
    ax.scatter(*zip(*noktalar), s=12, color="white", edgecolor="black", lw=0.5, label="Deney noktaları")
    ax.set_xlabel(x_ad); ax.set_ylabel(y_ad)
    diger = ", ".join(f"{k} = {A.seviye_yaz(round(v, 4) if isinstance(v, float) else v)}"
                      for k, v in sabit.items() if k not in (x_ad, y_ad))
    ax.set_title(f"{s['yanit']} kontur grafiği" + (f"\nsabit: {diger}" if diger else "")
                 + ("  (taralı: hedef aralık)" if hedef else ""), fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def yuzey(s, x_ad, y_ad, sabit):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    meta = s["meta"]
    fx, fy = A._faktor(meta, x_ad), A._faktor(meta, y_ad)
    XX, YY, Z = _izgara(s, fx, fy, sabit, n=30)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XX, YY, Z, cmap="viridis", alpha=0.9, linewidth=0)
    ax.set_xlabel(x_ad); ax.set_ylabel(y_ad); ax.set_zlabel(s["yanit"])
    ax.set_title(f"{s['yanit']} yüzey grafiği", fontsize=10)
    fig.tight_layout()
    return fig


def karisim_ucgen(s, sabit=None):
    meta = s["meta"]
    fs = meta["faktorler"]
    if len(fs) != 3:
        return None
    n = 60
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append((i / n, j / n, (n - i - j) / n))
    pts = np.array(pts)
    ayar = [{f["isim"]: f.get("kar_alt", 0) + meta["kalan"] * z for f, z in zip(fs, p)} for p in pts]
    Z, _ = A.tahmin(s, ayar)
    xx = pts[:, 1] + 0.5 * pts[:, 2]
    yy = pts[:, 2] * np.sqrt(3) / 2
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cs = ax.tricontourf(xx, yy, Z, levels=14, cmap="viridis")
    fig.colorbar(cs, ax=ax, label=s["yanit"])
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color="black", lw=1)
    for (x, y), f in zip([(0, 0), (1, 0), (0.5, np.sqrt(3) / 2)], fs):
        ax.annotate(f["isim"], (x, y), xytext=(0, -12 if y == 0 else 6), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{s['yanit']} karışım kontur grafiği (sözde bileşenler)", fontsize=10)
    fig.tight_layout()
    return fig


def taguchi_grafik(ta):
    meta = ta["meta"]
    out = []
    for olcut, baslik in (("SN", "S/N oranları"), ("Ort", "Ortalamalar")):
        t = ta["tablo"][olcut]
        fs = meta["faktorler"]
        n = len(fs)
        sut = min(4, n)
        sat = int(np.ceil(n / sut))
        fig, axes = plt.subplots(sat, sut, figsize=(7, 2.3 * sat), squeeze=False, sharey=True)
        for ax, f in zip(axes.flat, fs):
            L = A._seviyeler(meta, f)
            v = t["seviye"][f["isim"]]
            ax.plot(range(len(L)), v, "o-", color=MAVI if olcut == "SN" else YESIL)
            ax.axhline(t["genel"], color=GRI, ls="--", lw=0.8)
            ax.set_xticks(range(len(L)))
            ax.set_xticklabels([A.seviye_yaz(x) for x in L], fontsize=7)
            ax.set_title(f["isim"], fontsize=9)
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        ek = f" — {A.SN_ADLARI[ta['hedef']]}" if olcut == "SN" else ""
        fig.suptitle(f"Ana Etkiler Grafiği: {baslik}{ek}", fontsize=10)
        fig.tight_layout()
        out.append(fig)
    return out
