import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple

from .config import (
    SUBPERIODOS, COL_AÑO, COL_PESO, COL_TARGET, PARAMETERS, VARS_CATEGORICAS,
)


def limpiar_nsnr(df: pd.DataFrame, cols: list, codigos: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        mask = df[col].isin(codigos) | (pd.to_numeric(df[col], errors="coerce") < 0)
        df.loc[mask, col] = np.nan
    return df


def construir_split(
    df: pd.DataFrame,
    subperiodo: str,
    features: list,
    pesos_clase: dict,
) -> Tuple:
    cfg   = SUBPERIODOS[subperiodo]
    feats = [f for f in features if f in df.columns and f != COL_PESO]

    df_tr  = df[df[COL_AÑO].isin(cfg["train_olas"])].copy()
    df_val = df[df[COL_AÑO].isin(cfg["validate_ola"])].copy()
    df_te  = df[df[COL_AÑO].isin(cfg["test_ola"])].copy()

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


def resumen_split(nombre, X_tr, y_tr, X_val, y_val, X_te, y_te):
    print(f"{'─'*52}")
    print(f"  {nombre}")
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


def imputar(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    X_te: pd.DataFrame,
    semilla: int = None,
) -> Tuple:
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

    # Documentar variables que fueron imputadas totalmente en val o test
    # (100% NaN antes de imputar → todos los valores son sintéticos)
    vars_100nan_val  = [c for c in cols_num if X_val[c].isna().all()]
    vars_100nan_test = [c for c in cols_num if X_te[c].isna().all()]
    if vars_100nan_val:
        import warnings as _w
        _w.warn(
            f"imputar(): las siguientes variables tienen 100% NaN en el conjunto "
            f"de validación y fueron imputadas totalmente: {vars_100nan_val}. "
            f"Los valores resultantes son completamente sintéticos.",
            UserWarning, stacklevel=2,
        )
    if vars_100nan_test:
        import warnings as _w
        _w.warn(
            f"imputar(): las siguientes variables tienen 100% NaN en el conjunto "
            f"de prueba y fueron imputadas totalmente: {vars_100nan_test}. "
            f"Los valores resultantes son completamente sintéticos.",
            UserWarning, stacklevel=2,
        )

    return X_tr_imp, X_val_imp, X_te_imp, imp_num, imp_cat


def normalizar(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    X_te: pd.DataFrame,
    metodo: str = "minmax",
) -> Tuple:
    cols_num = [c for c in X_tr.columns if c not in VARS_CATEGORICAS]
    scaler   = MinMaxScaler() if metodo == "minmax" else StandardScaler()
    X_tr_sc  = X_tr.copy()
    X_val_sc = X_val.copy()
    X_te_sc  = X_te.copy()
    X_tr_sc[cols_num]  = scaler.fit_transform(X_tr[cols_num])
    X_val_sc[cols_num] = scaler.transform(X_val[cols_num])
    X_te_sc[cols_num]  = scaler.transform(X_te[cols_num])
    return X_tr_sc, X_val_sc, X_te_sc, scaler


def aplicar_transformaciones_deterministas(
    df_in: pd.DataFrame,
    transformaciones: dict,
    año_encuesta: int,
) -> pd.DataFrame:
    df = df_in.copy()
    tr = transformaciones

    cols = [c for c in df.columns
            if c not in ("año", "pais_iso3", "pais_nombre", "ola")]
    df = limpiar_nsnr(df, cols, tr["nsnr"])

    if tr.get("a007071_nsnr_97") and "A_007_071" in df.columns:
        df.loc[df["A_007_071"] == 97, "A_007_071"] = np.nan
    if tr.get("c003003_nan5") and "C_003_003_011" in df.columns:
        df.loc[df["C_003_003_011"] == 5, "C_003_003_011"] = np.nan

    for col in tr.get("likert4", []):
        if col not in df.columns:
            continue
        mask = df[col].notna() & df[col].between(1, 4)
        df.loc[mask, col] = 5 - df.loc[mask, col]

    for col in tr.get("likert4_interes", []):
        if col not in df.columns:
            continue
        mask = df[col].notna() & df[col].between(1, 4)
        df.loc[mask, col] = 5 - df.loc[mask, col]

    for col in ["D_001_021", "D_001_041", "D_001_091"]:
        if col not in df.columns:
            continue
        if año_encuesta <= 2000:
            mask = df[col].between(1, 3)
            df.loc[mask, col] = 4 - df.loc[mask, col]
        else:
            mask = df[col].between(1, 5)
            df.loc[mask, col] = 6 - df.loc[mask, col]

    # Recodificaciones binarias: np.select en lugar de .map() para evitar
    # que valores no contemplados se conviertan silenciosamente a NaN.
    for col, mapeo in tr.get("binarias", {}).items():
        if col not in df.columns:
            continue
        mapeo_float = {float(k): v for k, v in mapeo.items()}
        condiciones = [df[col] == k for k in mapeo_float]
        valores     = list(mapeo_float.values())
        df[col]     = np.select(condiciones, valores, default=np.nan)

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

    if "G_002_011" in df.columns:
        col   = "G_002_011"
        nueva = np.full(len(df), np.nan)
        if año_encuesta == 2013:
            nueva[df[col].values == 1] = 1
            nueva[(df[col].values >  1) & (~np.isnan(df[col].values.astype(float)))] = 0
        else:
            nueva[df[col].values == 1] = 1
            nueva[df[col].values == 2] = 0
            # Valores fuera de {1, 2} quedan como NaN (explícito)
        df[col] = nueva

    for col in tr.get("vdem_invertir", []):
        if col in df.columns:
            df[col] = 1.0 - df[col]

    return df
