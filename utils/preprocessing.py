import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Optional, Tuple

from .config import (
    SPLIT, COL_AÑO, COL_PESO, COL_TARGET, PARAMETERS, PATHS,
    VARS_CATEGORICAS, AÑO_CORTE_VEN, PAISES_EXCLUIR_EVAL,
    COL_PAIS,
)

# Nombre legible de cada conjunto del split único
NOMBRES_CONJUNTOS = {
    "train": "Entrenamiento",
    "val":   "Validación",
    "test":  "Prueba",
}


def limpiar_nsnr(df: pd.DataFrame, cols: list, codigos: list) -> pd.DataFrame:
    """Convierte a NaN los valores de no respuesta (NS/NR) y negativos."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        mask = df[col].isin(codigos) | (pd.to_numeric(df[col], errors="coerce") < 0)
        df.loc[mask, col] = np.nan
    return df


def construir_split(
    df: pd.DataFrame,
    features: list,
    pesos_clase: dict,
) -> Tuple:
    """
    Construye los conjuntos de entrenamiento, validación y prueba
    a partir del split único definido en SPLIT (config.py).

    Reglas de exclusión aplicadas:
    - Venezuela: excluida de val y test (sesgo de respuesta post-2017);
      en train solo hasta AÑO_CORTE_VEN=2017.
    - Nicaragua: excluida de val y test (sin datos en 2023-2024;
      consistencia de dominio).
    - Ambas siguen presentes en train.

    Parámetros
    ----------
    df           : DataFrame integrado con todas las olas
    features     : lista de features a usar (28 variables)
    pesos_clase  : dict {clase: peso} para balanceo

    Retorna
    -------
    X_tr, y_tr, X_val, y_val, X_te, y_te, w_tr, w_val, w_te
    """
    feats = [f for f in features if f in df.columns and f != COL_PESO]

    # Conjuntos base por año
    df_tr  = df[df[COL_AÑO].isin(SPLIT["train"])].copy()
    df_val = df[df[COL_AÑO].isin(SPLIT["val"])].copy()
    df_te  = df[df[COL_AÑO].isin(SPLIT["test"])].copy()

    # Exclusión de Venezuela y Nicaragua de val y test
    # (Venezuela ya fue recortada en NB02 celda de exclusiones;
    #  Nicaragua no tiene datos en 2023-2024 de todas formas,
    #  pero la exclusión explícita garantiza consistencia)
    for df_sub in [df_val, df_te]:
        if COL_PAIS in df_sub.columns:
            mask_excl = df_sub[COL_PAIS].isin(PAISES_EXCLUIR_EVAL)
            df_sub.drop(index=df_sub[mask_excl].index, inplace=True)

    X_tr,  y_tr  = df_tr[feats],  df_tr[COL_TARGET].astype(int)
    X_val, y_val = df_val[feats], df_val[COL_TARGET].astype(int)
    X_te,  y_te  = df_te[feats],  df_te[COL_TARGET].astype(int)

    def _pesos(df_sub, y):
        w_m = (df_sub[COL_PESO].fillna(1.0)
               if COL_PESO in df_sub.columns
               else pd.Series(np.ones(len(df_sub)), index=df_sub.index))
        w_m = w_m / w_m.mean()
        w_c = y.map(pesos_clase)
        return (w_m.values * w_c.values).astype(float)

    w_tr  = _pesos(df_tr,  y_tr)
    w_val = _pesos(df_val, y_val)
    w_te  = _pesos(df_te,  y_te)

    return X_tr, y_tr, X_val, y_val, X_te, y_te, w_tr, w_val, w_te


def resumen_split(X_tr, y_tr, X_val, y_val, X_te, y_te):
    print(f"{'─'*52}")
    print(f"  Split único")
    print(f"{'─'*52}")
    print(f"  Train : {len(X_tr):>8,} registros | {X_tr.shape[1]} features")
    print(f"  Val   : {len(X_val):>8,} registros")
    print(f"  Test  : {len(X_te):>8,} registros")
    print(f"  Ratio train/test: {len(X_tr)/len(X_te):.1f}x")
    print(f"  Clases train : {dict(y_tr.value_counts().sort_index())}")
    print(f"  Clases val   : {dict(y_val.value_counts().sort_index())}")
    print(f"  Clases test  : {dict(y_te.value_counts().sort_index())}")
    miss_tr  = X_tr.isnull().mean().mean() * 100
    miss_val = X_val.isnull().mean().mean() * 100
    miss_te  = X_te.isnull().mean().mean() * 100
    print(f"  NaN train: {miss_tr:.1f}%  |  NaN val: {miss_val:.1f}%  |  NaN test: {miss_te:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# TABLAS DESCRIPTIVAS DE LOS CONJUNTOS DE DATOS
#
# Se usan dos veces en el NB02: antes de las exclusiones (justo después de la
# fusión LB × V-Dem) y después de todas las exclusiones (justo antes del
# entrenamiento), de modo que el efecto de cada exclusión quede documentado.
# ─────────────────────────────────────────────────────────────────────────────

def subconjuntos_split(
    df: pd.DataFrame,
    aplicar_exclusiones_eval: bool = False,
    incluir_fuera_split: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Parte el DataFrame en los conjuntos del split único definido en SPLIT.

    Parámetros
    ----------
    df                       : DataFrame con las columnas de año y país.
    aplicar_exclusiones_eval : si True replica las reglas de `construir_split`:
                               Venezuela solo hasta AÑO_CORTE_VEN en train y
                               PAISES_EXCLUIR_EVAL fuera de val y test.
                               Si False no se excluye ningún país (situación
                               previa a las exclusiones).
    incluir_fuera_split      : añade la clave 'Fuera del split' con los
                               registros cuyos años no pertenecen a ningún
                               conjunto (solo si existen).

    Retorna
    -------
    dict {nombre_conjunto: sub_DataFrame} en orden train → val → test.
    """
    tiene_pais = COL_PAIS in df.columns
    subconjuntos: Dict[str, pd.DataFrame] = {}

    for clave in ["train", "val", "test"]:
        sub = df[df[COL_AÑO].isin(SPLIT[clave])].copy()
        if aplicar_exclusiones_eval and tiene_pais:
            if clave == "train":
                sub = sub[~((sub[COL_PAIS] == "Venezuela") &
                            (sub[COL_AÑO] > AÑO_CORTE_VEN))]
            else:
                sub = sub[~sub[COL_PAIS].isin(PAISES_EXCLUIR_EVAL)]
        subconjuntos[NOMBRES_CONJUNTOS[clave]] = sub

    if incluir_fuera_split:
        años_split = set(SPLIT["train"]) | set(SPLIT["val"]) | set(SPLIT["test"])
        fuera = df[~df[COL_AÑO].isin(años_split)].copy()
        if len(fuera) > 0:
            subconjuntos["Fuera del split"] = fuera

    return subconjuntos


def _rangos_olas(años: list) -> str:
    """Representación compacta de una lista de años: '1995–1998, 2000, 2002–2004'."""
    if not años:
        return ""
    tramos, inicio, previo = [], años[0], años[0]
    for a in años[1:]:
        if a == previo + 1:
            previo = a
            continue
        tramos.append((inicio, previo))
        inicio = previo = a
    tramos.append((inicio, previo))
    return ", ".join(str(i) if i == f else f"{i}–{f}" for i, f in tramos)


def resumen_conjuntos(
    df: pd.DataFrame,
    aplicar_exclusiones_eval: bool = False,
) -> pd.DataFrame:
    """
    Tabla resumen de los conjuntos de datos.

    Columnas: conjunto, olas (lista completa), olas_rango (forma compacta),
    n_olas, n_registros, n_paises. Incluye una fila TOTAL con la unión de
    los tres conjuntos del split.

    Parámetros
    ----------
    df                       : DataFrame a describir.
    aplicar_exclusiones_eval : ver `subconjuntos_split`.
    """
    subconjuntos = subconjuntos_split(df, aplicar_exclusiones_eval)
    tiene_pais   = COL_PAIS in df.columns

    def _fila(nombre, sub):
        olas = sorted(int(a) for a in sub[COL_AÑO].dropna().unique())
        return {
            "conjunto"   : nombre,
            "olas"       : ", ".join(str(a) for a in olas),
            "olas_rango" : _rangos_olas(olas),
            "n_olas"     : len(olas),
            "n_registros": len(sub),
            "n_paises"   : int(sub[COL_PAIS].nunique()) if tiene_pais else np.nan,
        }

    filas = [_fila(nombre, sub) for nombre, sub in subconjuntos.items()]

    # Fila TOTAL: unión de train + val + test (excluye 'Fuera del split')
    claves_split = [NOMBRES_CONJUNTOS[k] for k in ["train", "val", "test"]]
    df_total     = pd.concat([subconjuntos[k] for k in claves_split
                              if k in subconjuntos])
    filas.append(_fila("TOTAL", df_total))

    return pd.DataFrame(filas)


def conjuntos_por_pais(
    df: pd.DataFrame,
    aplicar_exclusiones_eval: bool = False,
) -> pd.DataFrame:
    """
    Tabla de registros por país y conjunto.

    Filas: país. Columnas: Entrenamiento, Validación, Prueba, Total.

    Parámetros
    ----------
    df                       : DataFrame a describir.
    aplicar_exclusiones_eval : ver `subconjuntos_split`.
    """
    if COL_PAIS not in df.columns:
        return pd.DataFrame(columns=["pais"] + list(NOMBRES_CONJUNTOS.values()) + ["Total"])

    subconjuntos = subconjuntos_split(df, aplicar_exclusiones_eval,
                                      incluir_fuera_split=False)
    conteos = {nombre: sub[COL_PAIS].value_counts()
               for nombre, sub in subconjuntos.items()}

    tabla = (pd.DataFrame(conteos)
             .reindex(columns=list(NOMBRES_CONJUNTOS.values()))
             .fillna(0).astype(int))
    tabla["Total"] = tabla.sum(axis=1)
    tabla = tabla.sort_values("Total", ascending=False)
    tabla.index.name = "pais"

    # Fila TOTAL al final
    tabla.loc["TOTAL"] = tabla.sum(axis=0)
    return tabla.reset_index()


def tablas_conjuntos(
    df: pd.DataFrame,
    etapa: str = "antes_exclusiones",
    titulo: Optional[str] = None,
    aplicar_exclusiones_eval: bool = False,
    guardar: bool = True,
    verboso: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera, imprime y guarda las dos tablas descriptivas de los conjuntos.

    Parámetros
    ----------
    etapa                    : sufijo de los archivos CSV
                               ('antes_exclusiones' | 'despues_exclusiones').
    titulo                   : encabezado para la salida por consola.
    aplicar_exclusiones_eval : ver `subconjuntos_split`.
    guardar                  : escribe los CSV en results/tables/.

    Retorna
    -------
    (df_resumen, df_paises)
    """
    df_resumen = resumen_conjuntos(df, aplicar_exclusiones_eval)
    df_paises  = conjuntos_por_pais(df, aplicar_exclusiones_eval)

    if verboso:
        encabezado = titulo or f"Conjuntos de datos — {etapa}"
        print("=" * 72)
        print(encabezado)
        print("=" * 72)
        print()
        print("Tabla resumen de los conjuntos de datos:")
        print(df_resumen[["conjunto", "olas_rango", "n_olas",
                          "n_registros", "n_paises"]].to_string(index=False))
        print()
        print("Olas de cada conjunto (detalle):")
        for _, fila in df_resumen.iterrows():
            if fila["conjunto"] == "TOTAL":
                continue
            print(f"  {fila['conjunto']:<16}: {fila['olas']}")
        print()
        print("Registros por país y conjunto:")
        print(df_paises.to_string(index=False))

    if guardar:
        carpeta = PATHS["FOLDER_RESULTS_TABLES"]
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta_resumen = carpeta / f"conjuntos_resumen_{etapa}.csv"
        ruta_paises  = carpeta / f"conjuntos_por_pais_{etapa}.csv"
        df_resumen.to_csv(ruta_resumen, index=False)
        df_paises.to_csv(ruta_paises, index=False)
        if verboso:
            print()
            print(f"✓ Tabla guardada: results/tables/{ruta_resumen.name}")
            print(f"✓ Tabla guardada: results/tables/{ruta_paises.name}")

    return df_resumen, df_paises


def imputar(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    X_te: pd.DataFrame,
    semilla: int = None,
) -> Tuple:
    """
    Imputación diferenciada por tipo de variable:
    - Numéricas: MICE (IterativeImputer con BayesianRidge)
    - Categóricas nominales (S_200): moda (SimpleImputer)

    El imputer se ajusta ÚNICAMENTE sobre X_tr y se aplica
    sin re-ajuste sobre X_val y X_te para evitar data leakage.
    """
    if semilla is None:
        semilla = PARAMETERS["SEED"]
    cols_cat = [c for c in VARS_CATEGORICAS if c in X_tr.columns]
    cols_num = [c for c in X_tr.columns if c not in cols_cat]

    imp_num = IterativeImputer(
        estimator=BayesianRidge(), max_iter=10,
        random_state=semilla, verbose=0,
    )
    X_tr_num  = pd.DataFrame(imp_num.fit_transform(X_tr[cols_num]),
                              columns=cols_num, index=X_tr.index)
    X_val_num = pd.DataFrame(imp_num.transform(X_val[cols_num]),
                              columns=cols_num, index=X_val.index)
    X_te_num  = pd.DataFrame(imp_num.transform(X_te[cols_num]),
                              columns=cols_num, index=X_te.index)

    imp_cat = SimpleImputer(strategy="most_frequent")
    if cols_cat:
        X_tr_cat  = pd.DataFrame(imp_cat.fit_transform(X_tr[cols_cat]),
                                  columns=cols_cat, index=X_tr.index)
        X_val_cat = pd.DataFrame(imp_cat.transform(X_val[cols_cat]),
                                  columns=cols_cat, index=X_val.index)
        X_te_cat  = pd.DataFrame(imp_cat.transform(X_te[cols_cat]),
                                  columns=cols_cat, index=X_te.index)
        X_tr_imp  = pd.concat([X_tr_num,  X_tr_cat],  axis=1)[X_tr.columns]
        X_val_imp = pd.concat([X_val_num, X_val_cat], axis=1)[X_val.columns]
        X_te_imp  = pd.concat([X_te_num,  X_te_cat],  axis=1)[X_te.columns]
    else:
        X_tr_imp, X_val_imp, X_te_imp = X_tr_num, X_val_num, X_te_num

    assert X_tr_imp.isnull().sum().sum()  == 0, "NaN residuales tras imputación (train)"
    assert X_val_imp.isnull().sum().sum() == 0, "NaN residuales tras imputación (val)"
    assert X_te_imp.isnull().sum().sum()  == 0, "NaN residuales tras imputación (test)"

    # Documentar variables con 100% NaN en val o test (valores completamente sintéticos)
    vars_100nan_val  = [c for c in cols_num if X_val[c].isna().all()]
    vars_100nan_test = [c for c in cols_num if X_te[c].isna().all()]
    if vars_100nan_val:
        import warnings as _w
        _w.warn(
            f"imputar(): variables con 100% NaN en validación (valores sintéticos): "
            f"{vars_100nan_val}", UserWarning, stacklevel=2,
        )
    if vars_100nan_test:
        import warnings as _w
        _w.warn(
            f"imputar(): variables con 100% NaN en prueba (valores sintéticos): "
            f"{vars_100nan_test}", UserWarning, stacklevel=2,
        )

    return X_tr_imp, X_val_imp, X_te_imp, imp_num, imp_cat


def normalizar(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    X_te: pd.DataFrame,
    metodo: str = "minmax",
) -> Tuple:
    """Normaliza las columnas numéricas. El scaler se ajusta solo sobre X_tr."""
    cols_num = [c for c in X_tr.columns if c not in VARS_CATEGORICAS]
    scaler   = MinMaxScaler() if metodo == "minmax" else StandardScaler()
    X_tr_sc  = X_tr.copy()
    X_val_sc = X_val.copy()
    X_te_sc  = X_te.copy()
    X_tr_sc[cols_num]  = scaler.fit_transform(X_tr[cols_num])
    X_val_sc[cols_num] = scaler.transform(X_val[cols_num])
    X_te_sc[cols_num]  = scaler.transform(X_te[cols_num])
    return X_tr_sc, X_val_sc, X_te_sc, scaler


def preparar_features_modelo(
    X,
    art: dict,
    columnas: Optional[list] = None,
) -> pd.DataFrame:
    """
    Ajusta una matriz numérica al formato exacto que espera un modelo ya
    entrenado: orden de columnas y dtype de las variables categóricas.

    Necesario cuando se llama a `predict`/`predict_proba` con datos que vienen
    de una fuente que solo maneja números —LIME, ALE, o un array de NumPy—,
    porque cada librería exige una codificación distinta de las categóricas:

    - **CatBoost**: la categórica debe ser texto, con los NaN como la categoría
      explícita "-999", igual que en el entrenamiento.
    - **LightGBM**: la categórica debe ser `pandas.Categorical` con exactamente
      las mismas categorías del entrenamiento. Si no, LightGBM aborta con
      "train and valid dataset categorical_feature do not match". Las
      categorías se recuperan del propio booster (`pandas_categorical`), que
      es la fuente de verdad; si no están disponibles se derivan de los datos.
    - **XGBoost, OLO y TabNet**: reciben la matriz numérica sin cambios.

    Parámetros
    ----------
    X        : DataFrame o array con las features en el orden de `columnas`.
    art      : artefacto del pipeline (`cargar_pipeline`).
    columnas : nombres de las columnas de X. Si X ya es un DataFrame se usan
               sus columnas; si es un array es obligatorio.

    Retorna
    -------
    DataFrame listo para `predict`/`predict_proba`.
    """
    feats = art["features"]

    if isinstance(X, pd.DataFrame):
        X_out = X.copy()
    else:
        if columnas is None:
            columnas = feats
        X_out = pd.DataFrame(np.asarray(X), columns=list(columnas))

    faltantes = [c for c in feats if c not in X_out.columns]
    if faltantes:
        raise ValueError(
            f"preparar_features_modelo(): faltan {len(faltantes)} features "
            f"que el modelo espera: {faltantes[:8]}"
        )
    X_out = X_out[feats]

    modelo   = art["modelo"]
    tipo     = art.get("tipo_modelo", "trees")
    nombre   = art.get("nombre_modelo", "")
    cat_cols = [c for c in art.get("vars_categoricas", []) if c in feats]

    if tipo != "trees" or not cat_cols:
        return X_out

    es_catboost = nombre == "CatBoost" or hasattr(modelo, "get_cat_feature_indices")
    es_lightgbm = nombre == "LightGBM" or hasattr(modelo, "booster_")

    if es_catboost:
        for col in cat_cols:
            valores = pd.to_numeric(X_out[col], errors="coerce")
            X_out[col] = valores.fillna(-999).round().astype(int).astype(str)

    elif es_lightgbm:
        cats_entrenamiento = getattr(
            getattr(modelo, "booster_", None), "pandas_categorical", None)
        for j, col in enumerate(cat_cols):
            if cats_entrenamiento and j < len(cats_entrenamiento):
                cats = list(cats_entrenamiento[j])
            else:
                cats = sorted(pd.to_numeric(X_out[col], errors="coerce")
                              .dropna().unique().tolist())
            valores = pd.to_numeric(X_out[col], errors="coerce")
            # Las categorías de entrenamiento son numéricas: redondear para
            # que los valores reconstruidos (p. ej. 3.0000001) coincidan.
            if all(isinstance(c, (int, float, np.integer, np.floating))
                   for c in cats):
                valores = valores.round()
            X_out[col] = pd.Categorical(valores, categories=cats, ordered=False)

    return X_out


def aplicar_transformaciones_deterministas(
    df_in: pd.DataFrame,
    transformaciones: dict,
    año_encuesta: int,
) -> pd.DataFrame:
    """
    Aplica transformaciones de escala deterministas por ola.

    Transformaciones aplicadas:
    1. NS/NR → NaN (limpiar_nsnr)
    2. Armonización de escala para evaluaciones económicas comparativas
       (D_001_021, D_001_041, D_001_091): olas ≤ 2000 usan escala de 3
       puntos (fórmula 4-x); olas ≥ 2001 usan escala de 5 puntos (6-x).
    3. Recodificaciones binarias con np.select (evita NaN silenciosos de .map())
    4. Victimización delictiva I_001_001: armonización de 3 escalas distintas
       (≤2008: binaria; 2009: 3 categorías; ≥2010: 4 categorías) → binaria {0,1}
    5. Corrupción experiencial G_002_011: caso especial para ola 2013

    NO se invierten variables Likert ni índices V-Dem.
    Ver documento metodológico sección 5 para justificación.
    """
    df = df_in.copy()
    tr = transformaciones

    cols = [c for c in df.columns
            if c not in ("año", "pais_iso3", "pais_nombre", "ola")]
    df = limpiar_nsnr(df, cols, tr["nsnr"])

    # Paso 2: armonización evaluaciones económicas comparativas
    for col in ["D_001_021", "D_001_041", "D_001_091"]:
        if col not in df.columns:
            continue
        if año_encuesta <= 2000:
            mask = df[col].between(1, 3)
            df.loc[mask, col] = 4 - df.loc[mask, col]
        else:
            mask = df[col].between(1, 5)
            df.loc[mask, col] = 6 - df.loc[mask, col]

    # Paso 3: recodificaciones binarias con np.select
    for col, mapeo in tr.get("binarias", {}).items():
        if col not in df.columns:
            continue
        mapeo_float = {float(k): v for k, v in mapeo.items()}
        condiciones = [df[col] == k for k in mapeo_float]
        valores     = list(mapeo_float.values())
        df[col]     = np.select(condiciones, valores, default=np.nan)

    # Paso 4: victimización delictiva — armonización longitudinal
    # Tres esquemas de pregunta distintos según el año de la ola:
    #   ≤ 2008: binaria (1=Sí, 2=No) → recodifica a {1, 0}
    #     2009: 3 categorías (1=Sí últimos 12 meses, 2=Sí antes, 3=No)
    #           → colapsa {1,2}→1, {3}→0
    #   ≥ 2010: 4 categorías (1=Sí últimos 12m, 2=Sí últimos 3a,
    #           3=Sí hace más de 3a, 4=No) → colapsa {1,2,3}→1, {4}→0
    if "I_001_001" in df.columns:
        col   = "I_001_001"
        nueva = np.full(len(df), np.nan)
        if año_encuesta <= 2008:
            nueva[df[col].values == 1] = 1
            nueva[df[col].values == 2] = 0
        elif año_encuesta == 2009:
            nueva[np.isin(df[col].values, [1, 2])] = 1
            nueva[df[col].values == 3]              = 0
        else:
            nueva[np.isin(df[col].values, [1, 2, 3])] = 1
            nueva[df[col].values == 4]                 = 0
        df[col] = nueva

    # Paso 5: corrupción experiencial — caso especial ola 2013
    if "G_002_011" in df.columns:
        col   = "G_002_011"
        nueva = np.full(len(df), np.nan)
        if año_encuesta == 2013:
            nueva[df[col].values == 1] = 1
            nueva[(df[col].values > 1) & (~np.isnan(df[col].values.astype(float)))] = 0
        else:
            nueva[df[col].values == 1] = 1
            nueva[df[col].values == 2] = 0
        df[col] = nueva

    return df
