"""Pipeline completo: `PYTHONPATH=src python -m segmentation`.

Genera clientes sintéticos, elige k, segmenta, compara con una mezcla gaussiana, mide estabilidad y
deja perfiles y métricas en output/ y la figura de selección de k en docs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import generar_clientes
from .pipeline import segmentar

RAIZ = Path(__file__).resolve().parents[2]


def figura(tabla_k, k_elegido: int, ruta: Path) -> None:
    """Figura pequeña: codo (inercia), silhouette y BIC de la mezcla gaussiana por k."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), dpi=110)
    series = [
        ("inercia", "Codo: inercia de K-Means", "#9aa0a6"),
        ("silhouette_kmeans", "Silhouette de K-Means", "#1a73e8"),
        ("bic_gmm", "BIC de la mezcla gaussiana", "#c58f3d"),
    ]
    for a, (col, titulo, color) in zip(ax, series):
        a.plot(tabla_k["k"], tabla_k[col], "o-", color=color)
        a.axvline(k_elegido, ls=":", color="#5f6368")
        a.set(title=titulo, xlabel="k")
        a.set_xticks(tabla_k["k"])
        a.grid(alpha=0.3)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="Segmentación de clientes con clustering (datos sintéticos)")
    p.add_argument("--n", type=int, default=3000, help="clientes sintéticos")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=None, help="fijar k; por defecto, el de mayor silhouette")
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--semillas", type=int, default=20, help="corridas para medir estabilidad")
    p.add_argument("--out", type=Path, default=RAIZ / "output")
    p.add_argument("--figura", type=Path, default=RAIZ / "docs" / "seleccion_k.png")
    p.add_argument("--sin-figura", action="store_true")
    a = p.parse_args(argv)

    df = generar_clientes(a.n, a.seed)
    r = segmentar(df, k=a.k, ks=range(2, a.k_max + 1), seed=a.seed, semillas=a.semillas)

    a.out.mkdir(parents=True, exist_ok=True)
    df.assign(segmento=r["etiquetas"]).to_csv(a.out / "clientes_segmentados.csv", index=False)
    r["tabla_k"].to_csv(a.out / "seleccion_k.csv", index=False)
    r["perfil"].to_csv(a.out / "perfiles.csv")
    resumen = {
        clave: r[clave]
        for clave in ["k", "k_silhouette", "k_codo", "k_bic_gmm", "silhouette_kmeans", "silhouette_gmm",
                      "ari_kmeans_vs_gmm", "estabilidad", "ari_kmeans_vs_real", "ari_gmm_vs_real"]
        if clave in r
    }
    resumen["n_clientes"] = len(df)
    resumen["tabla_k"] = r["tabla_k"].to_dict(orient="records")
    resumen["segmentos"] = r["perfil"].reset_index().to_dict(orient="records")
    (a.out / "metricas.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=float))
    if not a.sin_figura:
        figura(r["tabla_k"], r["k"], a.figura)

    print(f"Clientes: {len(df)}")
    print(r["tabla_k"].round(3).to_string(index=False))
    print(f"\nk por silhouette: {r['k_silhouette']} | k por codo: {r['k_codo']} | k por BIC (GMM): {r['k_bic_gmm']}"
          f" | k usado: {r['k']}")
    print(f"Silhouette K-Means {r['silhouette_kmeans']:.3f} | GMM {r['silhouette_gmm']:.3f} | "
          f"ARI K-Means vs GMM {r['ari_kmeans_vs_gmm']:.3f}")
    e = r["estabilidad"]
    print(f"Estabilidad: ARI entre semillas {e['ari_semillas_media']:.3f} (mín {e['ari_semillas_min']:.3f}) | "
          f"bootstrap 80 % {e['ari_bootstrap_media']:.3f} (mín {e['ari_bootstrap_min']:.3f})")
    if "ari_kmeans_vs_real" in r:
        print(f"Recuperación del arquetipo real (solo datos sintéticos): ARI K-Means {r['ari_kmeans_vs_real']:.3f}"
              f" | GMM {r['ari_gmm_vs_real']:.3f}")
    cols = ["nombre", "clientes", "participacion", "recencia_dias", "frecuencia_24m", "monto_total_clp",
            "ticket_medio_clp", "tasa_apertura_email", "clics_campanas_12m", "webinars_12m",
            "antiguedad_meses", "financiado_empresa", "canal_principal"]
    print("\n" + r["perfil"][cols].round(2).to_string())
    return r


if __name__ == "__main__":
    main()
