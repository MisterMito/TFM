from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union, Pattern
import re
import pandas as pd


PathLike = Union[str, Path]


@dataclass
class MetadataSpec:
    """
    Especificación genérica de cómo leer y procesar una tabla de metadatos
    de pacientes / muestras para un estudio concreto.

    """
    path: PathLike
    # Formato de carga
    file_type: str = "excel"              # "excel" | "csv"
    sheet_name: Union[str, int] = 0       # solo aplica a Excel
    header: int = 0

    # Columnas clave en la tabla original
    sample_id_col: str = "Sample ID"
    group_col: str = "Group"              # se convertirá en "Patient_group"
    stage_col: Optional[str] = "Stage"
    sex_col: Optional[str] = "Sex"
    age_col: Optional[str] = "Age"
    score_cols: Sequence[str] = field(default_factory=list)

    # Columnas donde los decimales vienen con coma (ej: "0,4859")
    decimal_comma_cols: Sequence[str] = field(default_factory=list)

    # Configuración de etiquetas (target)
    # Dict opcional que mapea Patient_group -> etiqueta de clase (ej. Malignant/nonMalignant)
    group_to_class_map: Optional[Dict[str, str]] = None
    class_col_name: str = "Class_group"

    # Valores por defecto si group_to_class_map no está definido para un grupo
    default_class_label: Optional[str] = None  # p.ej. "nonMalignant"

    # Opciones varias
    # nombre de la columna "Patient_group" normalizada
    patient_group_col_name: str = "Patient_group"


def load_metadata(spec: MetadataSpec) -> pd.DataFrame:
    """
    Carga la tabla de metadatos según lo indicado en MetadataSpec y aplica
    algunas normalizaciones básicas (decimales, renombrado de columnas, etc.).

    No añade todavía la columna de clase; eso lo hace `add_class_labels`.
    """
    path = Path(spec.path)

    if spec.file_type.lower() == "excel":
        df = pd.read_excel(path, sheet_name=spec.sheet_name, header=spec.header)
    elif spec.file_type.lower() == "csv":
        df = pd.read_csv(path, header=spec.header)
    else:
        raise ValueError(f"file_type '{spec.file_type}' no soportado")

    # Normalizar columnas de decimal-coma
    for col in spec.decimal_comma_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )

    # Renombrar columna de grupo a "Patient_group" para tener un nombre estándar
    if spec.group_col in df.columns:
        df = df.rename(columns={spec.group_col: spec.patient_group_col_name})

    return df


def add_class_labels(df: pd.DataFrame, spec: MetadataSpec) -> pd.DataFrame:
    """
    Añade una columna de clase (binaria o multi–clase) a partir de un mapping configurable.

    - Usa spec.patient_group_col_name (por defecto "Patient_group") como origen.
    - Usa spec.group_to_class_map para mapear grupos -> etiqueta.
    - Si default_class_label está definido y algún grupo no está en el mapping,
      se usará este valor por defecto.
    """
    df = df.copy()

    group_col = spec.patient_group_col_name
    class_col = spec.class_col_name

    if group_col not in df.columns:
        raise KeyError(
            f"La columna '{group_col}' no existe en df. "
            f"Comprueba que 'group_col' y 'patient_group_col_name' en MetadataSpec sean correctos."
        )

    if spec.group_to_class_map is None:
        if spec.default_class_label is None:
            # Si no hay mapping ni valor por defecto, no hacemos nada
            return df
        else:
            # Asignamos todo al mismo label
            df[class_col] = spec.default_class_label
            return df

    def map_group(g: str) -> str:
        if g in spec.group_to_class_map:
            return spec.group_to_class_map[g]
        if spec.default_class_label is not None:
            return spec.default_class_label
        raise KeyError(
            f"Grupo '{g}' no está en group_to_class_map y no se ha definido default_class_label."
        )

    df[class_col] = df[group_col].astype(str).map(map_group)
    return df


def align_and_merge_by_sample_id(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    meta_sample_id_col: str = "Sample ID",
    sample_id_index_regex: Optional[Union[str, Pattern[str]]] = r"^\d+-(.+)$",
    assume_same_order_fallback: bool = True,
    how: str = "inner",
) -> pd.DataFrame:
    """
    Alinea una matriz de expresión (expr_df) con una tabla de metadatos (meta_df)
    garantizando, en la medida de lo posible, que ninguna muestra se pierda.

    Estrategia:
      1) expr_df.index se asume que contiene algún tipo de identificador de muestra,
         potencialmente con prefijos numéricos (ej. '1-Vumc-HD-101-TR922').
      2) meta_df[meta_sample_id_col] contiene los IDs "canónicos" (ej. 'Vumc-HD-101-TR922').
      3) Para cada fila i:
           a) Se intenta extraer una versión "limpia" del índice usando sample_id_index_regex.
           b) Si la versión limpia existe en el conjunto de IDs de meta_df, se usa esa.
           c) Si NO existe y assume_same_order_fallback=True,
              se toma el Sample ID de meta_df en la MISMA posición i (como hace tu compañera).
           d) Si NO existe y assume_same_order_fallback=False, se deja como NaN.
      4) Se añade una columna meta_sample_id_col a expr_df con el ID resultante.
      5) Se hace un merge por esa columna.

    Esto reproduce el truco "asumimos el mismo orden" pero encapsulado y con una
    regla generalizable para la mayoría de los casos (quitar prefijos numéricos).
    """
    expr = expr_df.copy()
    meta = meta_df.copy()

    if meta_sample_id_col not in meta.columns:
        raise KeyError(
            f"La columna '{meta_sample_id_col}' no existe en meta_df."
        )

    expr_ids = list(expr.index)
    meta_ids = list(meta[meta_sample_id_col])

    n_expr = len(expr_ids)
    n_meta = len(meta_ids)
    if n_expr != n_meta:
        # No interrumpo, pero lo señalo
        print(
            f"[align_and_merge_by_sample_id] Advertencia: "
            f"len(expr_df)={n_expr} != len(meta_df)={n_meta}. "
            "El fallback por posición puede no ser correcto para todas las filas."
        )

    if isinstance(sample_id_index_regex, str):
        pattern = re.compile(sample_id_index_regex)
    else:
        pattern = sample_id_index_regex

    meta_ids_set = set(meta_ids)

    aligned_ids = []
    n_clean_match = 0
    n_fallback = 0
    n_unmatched = 0

    for i, expr_id in enumerate(expr_ids):
        expr_id_str = str(expr_id)

        # 1) Intentar limpiar usando la regex (por defecto: quitar '^\d+-')
        m = pattern.search(expr_id_str) if pattern is not None else None
        if m:
            cleaned = m.group(1)
        else:
            cleaned = expr_id_str

        if cleaned in meta_ids_set:
            aligned_ids.append(cleaned)
            n_clean_match += 1
        else:
            # 2) Fallback: usar el meta_id en la misma posición
            if assume_same_order_fallback and i < n_meta:
                fallback_id = meta_ids[i]
                aligned_ids.append(fallback_id)
                n_fallback += 1
            else:
                aligned_ids.append(None)
                n_unmatched += 1

    print(
        "[align_and_merge_by_sample_id] Resumen alineación:\n"
        f"  - matches por regex/limpieza: {n_clean_match}\n"
        f"  - matches por fallback de orden: {n_fallback}\n"
        f"  - sin emparejar: {n_unmatched}"
    )

    expr[meta_sample_id_col] = aligned_ids

    merged = pd.merge(
        expr,
        meta,
        on=meta_sample_id_col,
        how=how,
    )

    return merged
