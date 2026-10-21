"""Escalado, selección de k, K-Means, comparación con mezcla gaussiana, estabilidad y perfilamiento."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Atributos de comportamiento que entran al clustering. El canal queda fuera a propósito: es un
# atributo de origen, no de comportamiento, y se usa para describir los segmentos.
ATRIBUTOS = [
    "recencia_dias", "frecuencia_24m", "monto_total_clp", "tasa_apertura_email",
    "clics_campanas_12m", "webinars_12m", "antiguedad_meses", "financiado_empresa",
]
# Variables muy asimétricas: se comprimen con log1p antes de escalar
LOG = ["recencia_dias", "frecuencia_24m", "monto_total_clp", "clics_campanas_12m", "webinars_12m"]


def preparar(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """log1p en las variables asimétricas y estandarización (media 0, desviación 1)."""
    x = df[ATRIBUTOS].astype(float).copy()
    for c in LOG:
        x[c] = np.log1p(x[c])
    escalador = StandardScaler()
    return escalador.fit_transform(x), escalador


def codo(ks, inercias) -> int:
    """k del codo: el punto más alejado de la recta entre el primer y el último k (método del triángulo).

    Ambos ejes se normalizan a [0, 1] para que la distancia no dependa de las unidades; la recta va de
    (0, 1) a (1, 0) y la distancia de cada punto a ella es (1 - k_norm - inercia_norm) / √2.
    """
    ks, y = np.asarray(ks, float), np.asarray(inercias, float)
    kn = (ks - ks[0]) / (ks[-1] - ks[0])
    yn = (y - y[-1]) / (y[0] - y[-1])
    return int(ks[np.argmax((1 - kn - yn) / np.sqrt(2))])


def seleccionar_k(x: np.ndarray, ks=range(2, 11), seed: int = 0) -> pd.DataFrame:
    """Inercia (codo), silhouette de K-Means y BIC de la mezcla gaussiana para cada k."""
    filas = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
        gmm = GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=seed).fit(x)
        filas.append(
            {
                "k": k,
                "inercia": float(km.inertia_),
                "silhouette_kmeans": float(silhouette_score(x, km.labels_)),
                "bic_gmm": float(gmm.bic(x)),
            }
        )
    return pd.DataFrame(filas)


def estabilidad(x: np.ndarray, k: int, referencia: np.ndarray, semillas: int = 20, seed: int = 0) -> dict:
    """Qué tanto cambian los segmentos al cambiar la semilla o los datos.

    - semillas: K-Means con una sola inicialización por semilla; ARI entre todos los pares.
    - bootstrap: se reentrena con un 80 % de los clientes y se asigna a todos; ARI contra la referencia.
    """
    etiquetas = [KMeans(n_clusters=k, n_init=1, random_state=s).fit_predict(x) for s in range(semillas)]
    ari_pares = [adjusted_rand_score(a, b) for a, b in combinations(etiquetas, 2)]

    rng = np.random.default_rng(seed)
    ari_boot = []
    for s in range(semillas):
        idx = rng.choice(len(x), size=int(0.8 * len(x)), replace=False)
        km = KMeans(n_clusters=k, n_init=10, random_state=s).fit(x[idx])
        ari_boot.append(adjusted_rand_score(referencia, km.predict(x)))
    return {
        "ari_semillas_media": float(np.mean(ari_pares)),
        "ari_semillas_min": float(np.min(ari_pares)),
        "ari_bootstrap_media": float(np.mean(ari_boot)),
        "ari_bootstrap_min": float(np.min(ari_boot)),
    }


# --- perfilamiento y nombres de negocio ----------------------------------------------------------
# Cada nombre tiene una "firma" en unidades estandarizadas (z) sobre la mediana del segmento.
# Los nombres se asignan a los segmentos con el algoritmo húngaro (máxima afinidad total), así cada
# nombre se usa una vez y el resultado no depende del orden de los clusters.
FIRMAS = {
    "Cuentas corporativas recurrentes": {"frecuencia_24m": 1, "financiado_empresa": 1.5, "monto_total_clp": 0.5},
    "Profesionales comprometidos": {"tasa_apertura_email": 1, "clics_campanas_12m": 1, "webinars_12m": 1,
                                    "frecuencia_24m": 0.5},
    "Interesados en evaluación": {"frecuencia_24m": -1.5, "antiguedad_meses": -1, "recencia_dias": -0.5,
                                  "webinars_12m": 0.5},
    "Ex-alumnos dormidos": {"recencia_dias": 1.5, "tasa_apertura_email": -1, "clics_campanas_12m": -1},
    "Compradores de programa premium": {"ticket_medio_clp": 1.5, "frecuencia_24m": -0.3},
}
AFINIDAD_MINIMA = 1.0  # puntaje z ponderado mínimo para aceptar un nombre de negocio
RECOMENDACIONES = {
    "Cuentas corporativas recurrentes": "Gestión por ejecutivo de cuenta: plan anual de formación para la empresa, "
                                        "cupos prepagados y reporte de avance a RR. HH.",
    "Profesionales comprometidos": "Ruta de especialización: ofrecer el siguiente programa con beneficio de "
                                   "continuidad y convertirlos en referidos con un incentivo.",
    "Interesados en evaluación": "Nutrir para convertir: clase de prueba, asesoría de admisión de 15 minutos "
                                 "y una oferta con fecha límite; medir conversión a primera compra.",
    "Ex-alumnos dormidos": "Reactivar con bajo costo: encuesta corta y contenido de actualización para "
                           "ex-alumnos; si no responde en 90 días, bajar la frecuencia de envíos.",
    "Compradores de programa premium": "Cuidar la experiencia: red de egresados, eventos exclusivos y "
                                       "programas cortos de alto valor; no saturar con campañas masivas.",
}


def perfilar(df: pd.DataFrame, etiquetas: np.ndarray) -> pd.DataFrame:
    """Tamaño, medianas de comportamiento, canal dominante, nombre de negocio y recomendación."""
    d = df.assign(segmento=etiquetas)
    d["ticket_medio_clp"] = np.where(d["frecuencia_24m"] > 0, d["monto_total_clp"] / d["frecuencia_24m"].clip(lower=1), 0)
    cols = ATRIBUTOS + ["ticket_medio_clp"]
    med = d.groupby("segmento")[cols].median()
    # z de las medianas de cada segmento respecto de la distribución total
    z = (med - d[cols].median()) / d[cols].std().replace(0, 1)
    z["financiado_empresa"] = (d.groupby("segmento")["financiado_empresa"].mean() - d["financiado_empresa"].mean()) / d["financiado_empresa"].std()

    nombres = list(FIRMAS)
    afinidad = np.array([[sum(w * z.loc[s, c] for c, w in FIRMAS[n].items()) for n in nombres] for s in z.index])
    filas_s, cols_n = linear_sum_assignment(-afinidad)
    # Un nombre solo se acepta si la afinidad es clara; si no, el segmento se marca para revisión manual
    nombre_de, afinidad_de = {}, {}
    for i, j in zip(filas_s, cols_n):
        afinidad_de[z.index[i]] = float(afinidad[i, j])
        if afinidad[i, j] >= AFINIDAD_MINIMA:
            nombre_de[z.index[i]] = nombres[j]

    perfil = med.copy()
    perfil["financiado_empresa"] = d.groupby("segmento")["financiado_empresa"].mean()
    perfil.insert(0, "clientes", d.groupby("segmento").size())
    perfil.insert(1, "participacion", perfil["clientes"] / len(d))
    perfil["canal_principal"] = d.groupby("segmento")["canal_adquisicion"].agg(lambda s: s.value_counts().idxmax())
    perfil["monto_total_segmento_clp"] = d.groupby("segmento")["monto_total_clp"].sum()
    perfil["nombre"] = [nombre_de.get(s, f"Segmento sin firma clara {s}") for s in perfil.index]
    perfil["afinidad_nombre"] = [afinidad_de.get(s, float("nan")) for s in perfil.index]
    perfil["recomendacion"] = perfil["nombre"].map(RECOMENDACIONES).fillna("Revisar manualmente: perfil sin firma clara.")
    return perfil.sort_values("clientes", ascending=False)


def segmentar(df: pd.DataFrame, k: int | None = None, ks=range(2, 11), seed: int = 0, semillas: int = 20) -> dict:
    """Pipeline completo. Si no se fija k, se elige el de mayor silhouette."""
    x, _ = preparar(df)
    tabla_k = seleccionar_k(x, ks, seed)
    k_silhouette = int(tabla_k.loc[tabla_k["silhouette_kmeans"].idxmax(), "k"])
    k_codo = codo(tabla_k["k"], tabla_k["inercia"])
    k_bic = int(tabla_k.loc[tabla_k["bic_gmm"].idxmin(), "k"])
    k = k or k_silhouette

    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
    gmm = GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=seed).fit(x)
    et_gmm = gmm.predict(x)

    resultado = {
        "k": k,
        "k_silhouette": k_silhouette,
        "k_codo": k_codo,
        "k_bic_gmm": k_bic,
        "tabla_k": tabla_k,
        "etiquetas": km.labels_,
        "silhouette_kmeans": float(silhouette_score(x, km.labels_)),
        "silhouette_gmm": float(silhouette_score(x, et_gmm)),
        "ari_kmeans_vs_gmm": float(adjusted_rand_score(km.labels_, et_gmm)),
        "estabilidad": estabilidad(x, k, km.labels_, semillas, seed),
        "perfil": perfilar(df, km.labels_),
    }
    if "_arquetipo_real" in df:  # solo existe en datos sintéticos
        resultado["ari_kmeans_vs_real"] = float(adjusted_rand_score(df["_arquetipo_real"], km.labels_))
        resultado["ari_gmm_vs_real"] = float(adjusted_rand_score(df["_arquetipo_real"], et_gmm))
    return resultado
