import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentation.__main__ import main  # noqa: E402
from segmentation.data import generar_clientes  # noqa: E402
from segmentation.pipeline import ATRIBUTOS, codo, preparar, segmentar  # noqa: E402


class DatosTests(unittest.TestCase):
    def test_reproducible_con_semilla(self):
        pd.testing.assert_frame_equal(generar_clientes(400, seed=3), generar_clientes(400, seed=3))
        self.assertFalse(generar_clientes(400, seed=3).equals(generar_clientes(400, seed=4)))

    def test_rangos_coherentes(self):
        df = generar_clientes(600, seed=1)
        self.assertTrue((df["recencia_dias"] >= 1).all())
        self.assertTrue(df["tasa_apertura_email"].between(0, 1).all())
        # sin compras no hay monto ni financiamiento
        sin_compra = df["frecuencia_24m"] == 0
        self.assertTrue((df.loc[sin_compra, "monto_total_clp"] == 0).all())
        self.assertTrue((df.loc[sin_compra, "financiado_empresa"] == 0).all())

    def test_el_arquetipo_real_no_entra_al_clustering(self):
        self.assertNotIn("_arquetipo_real", ATRIBUTOS)
        x, _ = preparar(generar_clientes(200, seed=2))
        self.assertEqual(x.shape[1], len(ATRIBUTOS))
        self.assertTrue(np.allclose(x.mean(axis=0), 0, atol=1e-9))


class PipelineTests(unittest.TestCase):
    def test_codo_en_curva_conocida(self):
        # caída fuerte hasta k=3 y luego plana: el codo es 3
        self.assertEqual(codo([1, 2, 3, 4, 5, 6], [100, 50, 12, 10, 9, 8]), 3)

    def test_pipeline_corre_en_pequeno(self):
        df = generar_clientes(400, seed=5)
        r = segmentar(df, ks=range(2, 5), seed=0, semillas=3)
        self.assertIn(r["k"], [2, 3, 4])
        self.assertEqual(len(r["etiquetas"]), len(df))
        self.assertEqual(r["perfil"]["clientes"].sum(), len(df))
        self.assertAlmostEqual(r["perfil"]["participacion"].sum(), 1.0)
        self.assertTrue(-1 <= r["silhouette_kmeans"] <= 1)
        self.assertTrue(r["perfil"]["recomendacion"].str.len().gt(0).all())
        # los nombres de negocio no se repiten
        self.assertEqual(r["perfil"]["nombre"].nunique(), len(r["perfil"]))

    def test_cli_escribe_metricas(self):
        with tempfile.TemporaryDirectory() as tmp:
            main(["--n", "300", "--k-max", "4", "--semillas", "2", "--out", tmp, "--sin-figura"])
            self.assertTrue((Path(tmp) / "metricas.json").exists())
            self.assertTrue((Path(tmp) / "perfiles.csv").exists())


if __name__ == "__main__":
    unittest.main()
