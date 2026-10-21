"""Generación de clientes sintéticos de programas de formación ejecutiva.

Cada cliente nace de un arquetipo latente (que el clustering NO ve) y sus atributos se sortean con
ruido y solapamiento entre arquetipos, para que segmentar no sea trivial. Conocer el arquetipo real solo
es posible porque los datos son sintéticos: permite medir cuánto recupera el clustering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CANALES = ["organico", "referido", "linkedin", "evento", "convenio_empresa"]

# Arquetipo: proporción, parámetros de comportamiento y mezcla de canales de adquisición.
# recencia: media de días desde la última compra o interacción; freq: media de programas en 24 meses;
# ticket: precio medio por programa (CLP); apertura: tasa de apertura de correos (beta a, b);
# clics y webinars: medias anuales; financiado: probabilidad de que la empresa pague.
ARQUETIPOS = {
    "corporativo_recurrente": dict(p=0.16, recencia=60, freq=3.2, ticket=1_400_000, apertura=(5, 5),
                                   clics=6, webinars=1.5, antig=40, financiado=0.85,
                                   canales=[0.05, 0.10, 0.10, 0.15, 0.60]),
    "profesional_comprometido": dict(p=0.22, recencia=90, freq=1.8, ticket=900_000, apertura=(8, 3),
                                     clics=14, webinars=4.0, antig=28, financiado=0.25,
                                     canales=[0.25, 0.25, 0.30, 0.15, 0.05]),
    "interesado_en_evaluacion": dict(p=0.27, recencia=25, freq=0.15, ticket=700_000, apertura=(6, 4),
                                     clics=9, webinars=2.5, antig=5, financiado=0.10,
                                     canales=[0.30, 0.10, 0.40, 0.18, 0.02]),
    "exalumno_dormido": dict(p=0.25, recencia=420, freq=1.1, ticket=800_000, apertura=(1.5, 8),
                             clics=0.7, webinars=0.2, antig=48, financiado=0.30,
                             canales=[0.30, 0.25, 0.20, 0.15, 0.10]),
    "comprador_premium": dict(p=0.10, recencia=150, freq=1.1, ticket=4_500_000, apertura=(3, 5),
                              clics=3, webinars=0.8, antig=20, financiado=0.45,
                              canales=[0.15, 0.45, 0.15, 0.20, 0.05]),
}


def generar_clientes(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Genera `n` clientes con atributos de recencia, frecuencia, monto, canal y engagement."""
    rng = np.random.default_rng(seed)
    nombres = list(ARQUETIPOS)
    probs = np.array([ARQUETIPOS[a]["p"] for a in nombres])
    arq = rng.choice(len(nombres), size=n, p=probs / probs.sum())

    filas = []
    for i, k in enumerate(arq):
        a = ARQUETIPOS[nombres[k]]
        frecuencia = int(rng.poisson(a["freq"]))
        # El monto existe solo si compró; cada programa tiene precio con dispersión lognormal
        monto = float(sum(a["ticket"] * rng.lognormal(0, 0.35) for _ in range(frecuencia)))
        filas.append(
            {
                "id_cliente": f"C{i:05d}",
                "recencia_dias": int(np.clip(rng.gamma(2.0, a["recencia"] / 2.0), 1, 900)),
                "frecuencia_24m": frecuencia,
                "monto_total_clp": round(monto, -3),
                "tasa_apertura_email": round(float(rng.beta(*a["apertura"])), 3),
                "clics_campanas_12m": int(rng.poisson(a["clics"] * rng.lognormal(0, 0.4))),
                "webinars_12m": int(rng.poisson(a["webinars"])),
                "antiguedad_meses": int(np.clip(rng.normal(a["antig"], a["antig"] * 0.35), 1, 120)),
                "financiado_empresa": int(rng.random() < a["financiado"]) if frecuencia > 0 else 0,
                "canal_adquisicion": CANALES[rng.choice(len(CANALES), p=a["canales"])],
                "_arquetipo_real": nombres[k],  # solo para evaluar; el clustering nunca lo ve
            }
        )
    return pd.DataFrame(filas)
