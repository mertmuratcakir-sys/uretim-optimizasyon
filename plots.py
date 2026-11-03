"""
plots.py — Tablo, grafik ve sonuç metinleri (arayüzden bağımsız)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import engine as E

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})

ADLAR = ["GaussianProcess", "RandomForest", "XGBoost", "Lojistik", "SVM"]


def yaris_tablosu(model, guven):
    if not model:
        return pd.DataFrame(columns=["Çıktı"])
    satirlar = []
    for i, y in model["yaris"].items():
        r = {"Çıktı": i, "Tip": y["tip"], "Metrik": y["metrik"]}
        for a in ADLAR:
            r[a] = round(y["skorlar"][a], 3) if a in y["skorlar"] else None
        r["Kazanan"] = f"{E.MODEL_EMOJI.get(y['kazanan'], '')} {y['kazanan']}"
        r["Değerlendirme"] = guven["detay"][i]["yorum"] if guven else ""
        r["Kayıt"] = y["n"]
        satirlar.append(r)
    return pd.DataFrame(satirlar).dropna(axis=1, how="all")


def yaris_grafik(model):
    if not model or not model["yaris"]:
        return None
    ciktilar = [i for i, y in model["yaris"].items() if y["skorlar"]]
    if not ciktilar:
        return None
    var = [a for a in ADLAR if any(a in model["yaris"][i]["skorlar"] for i in ciktilar)]
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.9 * len(ciktilar)))
    h = 0.8 / len(var)
    for k, a in enumerate(var):
        ys = np.arange(len(ciktilar)) + k * h - 0.4 + h / 2
        vals = [model["yaris"][i]["skorlar"].get(a, np.nan) for i in ciktilar]
        ax.barh(ys, np.clip(np.nan_to_num(vals, nan=0), -0.2, 1), height=h * 0.9,
                color=E.MODEL_RENK[a], label=a)
    for j, i in enumerate(ciktilar):
        kaz = model["yaris"][i]["kazanan"]
        ax.text(1.02, j, kaz, va="center", fontsize=8, fontweight="bold",
                color=E.MODEL_RENK.get(kaz, "#333"))
    ax.set_yticks(range(len(ciktilar)))
    ax.set_yticklabels([f"{i}\n({model['yaris'][i]['metrik']})" for i in ciktilar])
    ax.set_xlim(-0.2, 1.4)
    ax.axvline(0.7, ls="--", lw=0.8, color="#999")
    ax.set_xlabel("Çapraz doğrulama skoru (yüksek = iyi) — sağdaki isim kazanan model")
    ax.legend(fontsize=7, ncol=len(var), loc="lower center",
              bbox_to_anchor=(0.45, 1.0), frameon=False)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def opt_grafik(gecmis, onem):
    if not gecmis:
        return None
    iki = bool(onem)
    fig, axes = plt.subplots(2 if iki else 1, 1, figsize=(7, 5.6 if iki else 3))
    ax = axes[0] if iki else axes
    x = np.arange(1, len(gecmis) + 1)
    ax.scatter(x, gecmis, s=8, color="#9ab", label="Deneme")
    ax.plot(x, np.maximum.accumulate(gecmis), color="#e8710a", lw=2, label="En iyi")
    ax.set_xlabel("Deneme")
    ax.set_ylabel("Hedef skoru (0–100)")
    ax.set_title("Optimizasyon ilerlemesi")
    ax.legend(fontsize=8, frameon=False)
    if iki:
        ax2 = axes[1]
        k = list(onem.keys())[::-1]
        ax2.barh(k, [100 * onem[a] for a in k], color="#2a6fdb")
        ax2.set_xlabel("Optimizasyon sırasında etkisi (%)")
        ax2.set_title("Hangi ayar sonucu en çok değiştirdi?")
    fig.tight_layout()
    return fig


def onem_grafik(etki, model):
    if not etki or not etki.get("onem"):
        return None
    onem = etki["onem"]
    n = len(onem)
    en_uzun = max(len(v) for v in onem.values())
    fig, axes = plt.subplots(n, 1, figsize=(7, 0.6 + n * (0.6 + 0.35 * en_uzun)),
                             squeeze=False)
    for ax, (i, d) in zip(axes[:, 0], onem.items()):
        k = list(d.keys())[::-1]
        v = [d[a] for a in k]
        kaz = model["modeller"][i]["kazanan"]
        ax.barh(k, v, color=E.MODEL_RENK.get(kaz, "#e8710a"))
        for y, val in enumerate(v):
            ax.text(val + 1, y, f"%{val:.0f}", va="center", fontsize=8)
        ax.set_xlim(0, 112)
        ax.set_title(f"{i}  —  {kaz}", fontsize=10, loc="left")
    axes[-1, 0].set_xlabel("Çıktıyı belirlemedeki payı (%)")
    fig.tight_layout()
    return fig


def duy_grafik(etki, yapi, cikti):
    if not etki or cikti not in (etki.get("duy") or {}):
        return None
    c = next((x for x in yapi["ciktilar"] if x["isim"] == cikti), None)
    d = etki["duy"][cikti]
    if c is None or not d:
        return None
    n = len(d)
    sut = 2 if n > 1 else 1
    sat = int(np.ceil(n / sut))
    fig, axes = plt.subplots(sat, sut, figsize=(7, 2.7 * sat), squeeze=False)
    for ax, (g, v) in zip(axes.flat, d.items()):
        x, y = np.array(v["x"]), np.array(v["y"])
        if c["tip"] == "nicel":
            if c.get("hedef_min") is not None and c.get("hedef_max") is not None:
                ax.axhspan(c["hedef_min"], c["hedef_max"], color="#1a9e5a", alpha=0.12,
                           label="Hedef aralık")
            sd = v["std"]
            if sd and sd[0] is not None:
                sd = np.array(sd, dtype=float)
                ax.fill_between(x, y - sd, y + sd, color="#e8710a", alpha=0.15,
                                label="±1σ belirsizlik")
            ax.plot(x, y, color="#e8710a", lw=2)
            ax.set_ylabel(cikti)
        else:
            ax.plot(x, 100 * y, color="#8e44ad", lw=2)
            ax.set_ylim(0, 100)
            ax.set_ylabel(f"P({c['hedef_kat']}) %")
        ax.set_xlabel(g)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, frameon=False)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{cikti}: girdi değişince ne oluyor? (diğer girdiler ortalamada)",
                 fontsize=10)
    fig.tight_layout()
    return fig


def kor_grafik(etki):
    if not etki or etki.get("kor") is None:
        return None
    k = etki["kor"]
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.55 * len(k)))
    vals = k.values.astype(float)
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(k.shape[1]))
    ax.set_xticklabels(k.columns, rotation=25, ha="right")
    ax.set_yticks(range(k.shape[0]))
    ax.set_yticklabels(k.index)
    for r in range(k.shape[0]):
        for s in range(k.shape[1]):
            v = vals[r, s]
            ax.text(s, r, "—" if np.isnan(v) else f"{v:+.2f}", ha="center", va="center",
                    fontsize=8, color="white" if not np.isnan(v) and abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Korelasyon (Spearman)")
    fig.tight_layout()
    return fig


def sonuc_md(yapi, baslik, girdi, t, puan):
    s = [f"#### {baslik}", f"**Hedef skoru: {puan:.0f} / 100**", ""]
    if girdi is not None:
        s += ["| Ayar | Değer |", "|---|---|"]
        for g in yapi["girdiler"]:
            s.append(f"| {g['isim']} | **{girdi.get(g['isim'])}** |")
        s.append("")
    s += ["| Çıktı | Tahmin | Hedef | |", "|---|---|---|---|"]
    for c in yapi["ciktilar"]:
        r = t.get(c["isim"])
        if r is None:
            s.append(f"| {c['isim']} | *model yok* | | |")
            continue
        ok = E.hedef_durum(c, r)
        isaret = "✅" if ok else ("⚠️" if ok is False else "")
        if c["tip"] == "nicel":
            deger = f"{r['deger']:.3g}" + (f" ±{r['std']:.2g}" if r.get("std") else "")
            hedef = f"{c['hedef_min']}–{c['hedef_max']}"
        else:
            p = r["olas"].get(c.get("hedef_kat"), 0)
            deger = f"{r['deger']} (P[{c.get('hedef_kat')}] = %{100 * p:.0f})"
            hedef = str(c.get("hedef_kat"))
        s.append(f"| {c['isim']} | {deger} | {hedef} | {isaret} |")
    return "\n".join(s)
