from __future__ import annotations

from typing import Dict, Hashable, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

GeneLengthsLike = Union[
    pd.Series,
    pd.DataFrame,
    Mapping[Hashable, float],
    Dict[Hashable, float],
]


def _coerce_gene_lengths(
    gene_lengths: GeneLengthsLike,
    target_index: pd.Index,
    length_column: Optional[str] = None,
    min_length: float = 0.0,
) -> pd.Series:
    """
    Normaliza distintas representaciones de longitudes génicas a una pd.Series
    alineada con un índice de genes concreto.
    """
    # 1) Convertir a Series
    if isinstance(gene_lengths, pd.Series):
        gl = gene_lengths.copy()
    elif isinstance(gene_lengths, pd.DataFrame):
        if length_column is None:
            if gene_lengths.shape[1] != 1:
                raise ValueError(
                    "gene_lengths es un DataFrame con varias columnas; "
                    "indica explícitamente 'length_column'."
                )
            gl = gene_lengths.iloc[:, 0]
        else:
            if length_column not in gene_lengths.columns:
                raise ValueError(
                    f"Columna '{length_column}' no encontrada en gene_lengths."
                )
            gl = gene_lengths[length_column]
    else:
        # dict, Mapping, etc.
        gl = pd.Series(gene_lengths)

    # 2) Limpiar y asegurar tipo float
    gl = gl.dropna().astype(float)

    # 3) Alinear con el índice de genes (intersección)
    common_genes = target_index.intersection(gl.index)
    if common_genes.empty:
        raise ValueError(
            "No hay intersección entre los genes de 'counts' y los de 'gene_lengths'."
        )

    gl = gl.loc[common_genes]

    # 4) Filtrar genes demasiado cortos
    if min_length > 0:
        gl = gl[gl > min_length]
        if gl.empty:
            raise ValueError(
                "Tras aplicar el filtro 'min_length' no queda ningún gen válido."
            )

    return gl


def compute_tpm(
    counts: pd.DataFrame,
    gene_lengths: GeneLengthsLike,
    *,
    feature_axis: int = 0,
    lengths_unit: str = "bp",
    length_column: Optional[str] = None,
    min_length: float = 0.0,
    scale: float = 1e6,
    return_rpk: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Calcula TPM (Transcripts Per Million) a partir de una matriz de conteos brutos.

    La definición estándar de TPM es:

        1) Para cada gen g, en cada muestra s:
               RPK[g, s] = counts[g, s] / (length_g_en_kb)

        2) Para cada muestra s:
               sum_RPK[s] = sum_g RPK[g, s]

        3) Para cada gen g, muestra s:
               TPM[g, s] = (RPK[g, s] / sum_RPK[s]) * 1e6

    Esta función implementa ese esquema en forma general, permitiendo:

        - genes en filas o en columnas (feature_axis=0 o 1),
        - longitudes como Series, DataFrame o dict,
        - unidades de longitud en pares de bases (bp) o kilobases (kb),
        - filtrado opcional de genes demasiado cortos,
        - opción de devolver también la matriz de RPK.

    Parámetros
    ----------
    counts : pd.DataFrame
        Matriz de conteos brutos. Debe contener sólo valores no negativos
        (enteros o floats). La orientación se controla con `feature_axis`:

        - feature_axis = 0 -> genes en filas, muestras en columnas.
        - feature_axis = 1 -> genes en columnas, muestras en filas.

    gene_lengths : GeneLengthsLike
        Longitud de los genes. Puede ser:

        - pd.Series: índice = IDs de gen, valores = longitud.
        - pd.DataFrame: se tomará una columna con las longitudes.
        - dict / Mapping: claves = IDs de gen, valores = longitud.

        Los IDs de gen deben coincidir con el índice (si feature_axis=0) o
        con las columnas (si feature_axis=1) de `counts`.

    feature_axis : int, por defecto 0
        Indica en qué eje están los genes:
        - 0 → filas (index).
        - 1 → columnas.

    lengths_unit : {"bp", "kb"}, por defecto "bp"
        Unidad en la que están las longitudes de `gene_lengths`:
        - "bp" → pares de bases; internamente se convierten a kb dividiendo entre 1000.
        - "kb" → ya en kilobases; no se reescala.

    length_column : str, opcional
        Si `gene_lengths` es un DataFrame, esta columna se toma como longitud.
        Si es None y el DataFrame tiene una sola columna, se usa esa columna única.

    min_length : float, por defecto 0.0
        Longitud mínima (en la misma unidad especificada en `lengths_unit`)
        para que un gen se considere válido. Genes con longitud <= min_length
        se descartan antes de calcular RPK/TPM.

    scale : float, por defecto 1e6
        Factor de escalado. Por definición TPM usa 1e6, pero se permite cambiarlo
        si se desea otra escala.

    return_rpk : bool, por defecto False
        Si es True, devuelve una tupla (tpm, rpk).
        Si es False, devuelve sólo la matriz TPM.

    Devuelve
    --------
    tpm : pd.DataFrame
        Matriz de TPM con la misma orientación que `counts`:
        - si feature_axis=0 -> genes en filas, muestras en columnas.
        - si feature_axis=1 -> genes en columnas, muestras en filas.

    rpk : pd.DataFrame, opcional
        Sólo si `return_rpk=True`. Misma orientación que `tpm`.

    Notas
    -----
    - La función alinea automáticamente los genes de `counts` y `gene_lengths`
      usando la intersección de IDs. Cualquier gen sin longitud disponible
      se descarta.

    - En cada muestra, si la suma de RPK es 0 (por ejemplo, todas las cuentas son 0
      para los genes considerados), la TPM resultante de esa muestra se rellena con 0.

    - No se hace ningún chequeo de normalización previa (por ejemplo, filtrado
      de genes de muy baja expresión); se asume que eso se hace aguas arriba
      si es necesario.

    Ejemplos
    --------
    >>> # genes en filas (por defecto), longitudes en bp
    >>> tpm = compute_tpm(counts, gene_lengths_bp)

    >>> # genes en columnas
    >>> tpm = compute_tpm(counts, gene_lengths_bp, feature_axis=1)

    >>> # gene_lengths como DataFrame con columna 'gene_length'
    >>> tpm = compute_tpm(
    ...     counts,
    ...     gene_lengths_df,
    ...     length_column="gene_length",
    ...     lengths_unit="bp",
    ... )

    >>> # Obtener también la matriz de RPK
    >>> tpm, rpk = compute_tpm(counts, gene_lengths_bp, return_rpk=True)
    """
    if feature_axis not in (0, 1):
        raise ValueError("feature_axis debe ser 0 (genes en filas) o 1 (genes en columnas).")

    # Trabajar internamente con genes en filas (eje 0)
    if feature_axis == 0:
        counts_gx = counts.copy()
        feature_index = counts_gx.index
    else:
        counts_gx = counts.T.copy()
        feature_index = counts_gx.index  # ahora las filas son genes

    # Alinear longitudes con el índice de genes
    gl = _coerce_gene_lengths(
        gene_lengths=gene_lengths,
        target_index=feature_index,
        length_column=length_column,
        min_length=min_length,
    )

    # Subconjunto común de genes entre counts y gene_lengths
    common_genes = gl.index
    counts_gx = counts_gx.loc[common_genes]

    # Asegurar tipo numérico en counts
    counts_gx = counts_gx.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Convertir longitudes a kb si es necesario
    if lengths_unit.lower() == "bp":
        length_kb = gl / 1000.0
    elif lengths_unit.lower() == "kb":
        length_kb = gl.astype(float)
    else:
        raise ValueError("lengths_unit debe ser 'bp' o 'kb'.")

    # 1) RPK = counts / length_kb (misma longitud para todas las muestras)
    rpk = counts_gx.div(length_kb, axis=0)

    # 2) Suma de RPK por muestra
    rpk_sum = rpk.sum(axis=0)

    # Evitar división por 0
    rpk_sum = rpk_sum.replace(0, np.nan)

    # 3) TPM = (RPK / sum_RPK_muestra) * scale
    tpm = rpk.div(rpk_sum, axis=1) * float(scale)
    tpm = tpm.fillna(0.0)

    # Volver a la orientación original
    if feature_axis == 1:
        tpm = tpm.T
        rpk = rpk.T

    if return_rpk:
        return tpm, rpk
    return tpm
