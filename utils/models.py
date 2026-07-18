import json
import joblib
import numpy as np
import pandas as pd
import torch
import optuna
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from datetime import datetime
from typing import Dict, Tuple

from .config import PATHS, PARAMETERS, N_CLASES, VARS_CATEGORICAS
from .metrics import evaluar
from .preprocessing import aplicar_transformaciones_deterministas

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import mord as _mord
    MORD_OK = True
except ImportError:
    MORD_OK = False

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    CatBoostClassifier = Pool = None

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    TabNetClassifier = None


def entrenar_olo(
    X_tr, y_tr, X_val, y_val, X_te, y_te, w_tr, w_val, sp: str, cfg: dict,
) -> Tuple:
    nombre   = "OLO"
    ruta_hp  = PATHS["FOLDER_MODELS"] / f"hp_{nombre}_{sp}.json"
    seed     = PARAMETERS["SEED"]

    print(f"{'='*52}  Entrenando {nombre} — {sp}  {'='*52}")

    if not cfg["ejecutar_hp"] and ruta_hp.exists():
        best_hp = json.loads(ruta_hp.read_text())
        print(f"  HPs cargados: {best_hp}")
    else:
        X_tr_np, y_tr_np   = np.array(X_tr), np.array(y_tr)
        X_val_np, y_val_np = np.array(X_val), np.array(y_val)

        def obj(trial):
            alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
            if MORD_OK:
                m = _mord.LogisticIT(alpha=alpha, max_iter=500)
                m.fit(X_tr_np, y_tr_np, sample_weight=w_tr)
            else:
                m = LogisticRegression(C=1/alpha, solver="lbfgs",
                                       max_iter=500, random_state=seed)
                m.fit(X_tr_np, y_tr_np, sample_weight=w_tr)
            return cohen_kappa_score(y_val_np, m.predict(X_val_np), weights="quadratic")

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(obj, n_trials=cfg["n_trials"], show_progress_bar=False)
        best_hp = study.best_params
        print(f"  Mejor Kappa Val: {study.best_value:.4f} | {best_hp}")
        ruta_hp.write_text(json.dumps(best_hp))

    alpha = best_hp.get("alpha", 1.0)
    if MORD_OK:
        clf = _mord.LogisticIT(alpha=alpha, max_iter=500)
        clf.fit(np.array(X_tr), np.array(y_tr), sample_weight=w_tr)
    else:
        clf = LogisticRegression(C=1/alpha, solver="lbfgs",
                                 max_iter=500, random_state=seed)
        clf.fit(np.array(X_tr), np.array(y_tr), sample_weight=w_tr)

    y_pred_val = clf.predict(np.array(X_val))
    y_prob_val = clf.predict_proba(np.array(X_val))
    y_pred_te  = clf.predict(np.array(X_te))
    y_prob_te  = clf.predict_proba(np.array(X_te))

    joblib.dump(clf, PATHS["FOLDER_MODELS"] / f"{nombre}_{sp}.pkl")
    print(f"  ✓ Guardado: {nombre}_{sp}.pkl")
    m_val = evaluar(y_val, y_pred_val, y_prob_val, nombre, sp, split="val")
    m_te  = evaluar(y_te,  y_pred_te,  y_prob_te,  nombre, sp, split="test")
    return clf, m_val, m_te


def entrenar_xgboost(
    X_tr, y_tr, X_val, y_val, X_te, y_te, w_tr, w_val, sp: str, cfg: dict,
) -> Tuple:
    nombre  = "XGBoost"
    ruta_hp = PATHS["FOLDER_MODELS"] / f"hp_{nombre}_{sp}.json"
    seed    = PARAMETERS["SEED"]

    print(f"\n{'='*52}\n  Entrenando {nombre} — {sp}\n{'='*52}")

    if not cfg["ejecutar_hp"] and ruta_hp.exists():
        best_hp = json.loads(ruta_hp.read_text())
        print(f"  HPs cargados: {best_hp}")
    else:
        def obj(trial):
            p = {
                "n_estimators"    : trial.suggest_int("n_estimators", 200, 1000, step=100),
                "max_depth"       : trial.suggest_int("max_depth", 3, 8),
                "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "objective": "multi:softprob", "num_class": N_CLASES,
                "tree_method": "hist",
                "device"     : cfg["device_cuda"] if cfg["usar_gpu"] else "cpu",
                "random_state": seed, "n_jobs": cfg["n_jobs"], "verbosity": 0,
            }
            m = xgb.XGBClassifier(**p)
            m.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_val, y_val)], verbose=False)
            return cohen_kappa_score(y_val, m.predict(X_val), weights="quadratic")

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(obj, n_trials=cfg["n_trials"], show_progress_bar=False)
        best_hp = study.best_params
        print(f"  Mejor Kappa Val: {study.best_value:.4f} | {best_hp}")
        ruta_hp.write_text(json.dumps(best_hp))

    clf = xgb.XGBClassifier(
        **best_hp,
        objective="multi:softprob", num_class=N_CLASES,
        tree_method="hist",
        device=cfg["device_cuda"] if cfg["usar_gpu"] else "cpu",
        random_state=seed, n_jobs=cfg["n_jobs"], verbosity=0,
    )
    clf.fit(X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_val, y_val)], verbose=False)

    y_pred_val = clf.predict(X_val)
    y_prob_val = clf.predict_proba(X_val)
    y_pred_te  = clf.predict(X_te)
    y_prob_te  = clf.predict_proba(X_te)

    joblib.dump(clf, PATHS["FOLDER_MODELS"] / f"{nombre}_{sp}.pkl")
    print(f"  ✓ Guardado: {nombre}_{sp}.pkl")
    m_val = evaluar(y_val, y_pred_val, y_prob_val, nombre, sp, split="val")
    m_te  = evaluar(y_te,  y_pred_te,  y_prob_te,  nombre, sp, split="test")
    return clf, m_val, m_te


def entrenar_catboost(
    X_tr, y_tr, X_val, y_val, X_te, y_te, w_tr, w_val, sp: str, cfg: dict,
) -> Tuple:
    nombre  = "CatBoost"
    ruta_hp = PATHS["FOLDER_MODELS"] / f"hp_{nombre}_{sp}.json"
    seed    = PARAMETERS["SEED"]

    print(f"\n{'='*52}\n  Entrenando {nombre} — {sp}\n{'='*52}")

    def prep_cat(X):
        X = X.copy()
        for col in VARS_CATEGORICAS:
            if col in X.columns:
                X[col] = X[col].fillna(-999).astype(int).astype(str)
        return X

    X_tr_c  = prep_cat(X_tr)
    X_val_c = prep_cat(X_val)
    X_te_c  = prep_cat(X_te)
    cat_idx = [i for i, c in enumerate(X_tr.columns) if c in VARS_CATEGORICAS]

    if not cfg["ejecutar_hp"] and ruta_hp.exists():
        best_hp = json.loads(ruta_hp.read_text())
        print(f"  HPs cargados: {best_hp}")
    else:
        def obj(trial):
            p = {
                "iterations"         : trial.suggest_int("iterations", 300, 1000, step=100),
                "depth"              : trial.suggest_int("depth", 4, 8),
                "learning_rate"      : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg"        : trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "border_count"       : trial.suggest_int("border_count", 32, 128),
                "random_strength"    : trial.suggest_float("random_strength", 0.0, 10.0),
            }
            pool_tr  = Pool(X_tr_c, label=y_tr.values, weight=w_tr, cat_features=cat_idx)
            pool_val = Pool(X_val_c, label=y_val.values, cat_features=cat_idx)
            m = CatBoostClassifier(**p, loss_function="MultiClass",
                                   random_seed=seed, verbose=False,
                                   task_type="GPU" if cfg["usar_gpu"] else "CPU")
            m.fit(pool_tr, eval_set=pool_val)
            return cohen_kappa_score(y_val, m.predict(X_val_c).flatten(), weights="quadratic")

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(obj, n_trials=cfg["n_trials"], show_progress_bar=False)
        best_hp = study.best_params
        print(f"  Mejor Kappa Val: {study.best_value:.4f} | {best_hp}")
        ruta_hp.write_text(json.dumps(best_hp))

    pool_tr  = Pool(X_tr_c, label=y_tr.values, weight=w_tr, cat_features=cat_idx)
    pool_val = Pool(X_val_c, label=y_val.values, cat_features=cat_idx)
    clf = CatBoostClassifier(**best_hp, loss_function="MultiClass",
                              random_seed=seed, verbose=False,
                              task_type="GPU" if cfg["usar_gpu"] else "CPU")
    clf.fit(pool_tr, eval_set=pool_val)

    y_pred_val = clf.predict(X_val_c).flatten()
    y_prob_val = clf.predict_proba(X_val_c)
    y_pred_te  = clf.predict(X_te_c).flatten()
    y_prob_te  = clf.predict_proba(X_te_c)

    clf.save_model(str(PATHS["FOLDER_MODELS"] / f"{nombre}_{sp}.cbm"))
    print(f"  ✓ Guardado: {nombre}_{sp}.cbm")
    m_val = evaluar(y_val, y_pred_val, y_prob_val, nombre, sp, split="val")
    m_te  = evaluar(y_te,  y_pred_te,  y_prob_te,  nombre, sp, split="test")
    return clf, m_val, m_te


def entrenar_lightgbm(
    X_tr, y_tr, X_val, y_val, X_te, y_te, w_tr, w_val,
    pesos_clase: dict, sp: str, cfg: dict,
    
) -> Tuple:
    nombre  = "LightGBM"
    ruta_hp = PATHS["FOLDER_MODELS"] / f"hp_{nombre}_{sp}.json"
    seed    = PARAMETERS["SEED"]

    print(f"\n{'='*52}\n  Entrenando {nombre} — {sp}\n{'='*52}")

    def prep_lgb(Xa, Xb, Xc):
        Xa, Xb, Xc = Xa.copy(), Xb.copy(), Xc.copy()
        for col in VARS_CATEGORICAS:
            if col not in Xa.columns:
                continue
            cats = sorted(set(Xa[col].dropna().tolist() +
                              Xb[col].dropna().tolist() +
                              Xc[col].dropna().tolist()))
            ct = pd.CategoricalDtype(categories=cats, ordered=False)
            Xa[col] = Xa[col].astype(ct)
            Xb[col] = Xb[col].astype(ct)
            Xc[col] = Xc[col].astype(ct)
        return Xa, Xb, Xc

    X_tr_l, X_val_l, X_te_l = prep_lgb(X_tr, X_val, X_te)

    if not cfg["ejecutar_hp"] and ruta_hp.exists():
        best_hp = json.loads(ruta_hp.read_text())
        print(f"  HPs cargados: {best_hp}")
    else:
        def obj(trial):
            p = {
                "n_estimators"     : trial.suggest_int("n_estimators", 200, 1000, step=100),
                "num_leaves"       : trial.suggest_int("num_leaves", 20, 150),
                "max_depth"        : trial.suggest_int("max_depth", 3, 8),
                "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            }
            m = lgb.LGBMClassifier(
                **p, objective="multiclass", num_class=N_CLASES,
                class_weight=pesos_clase, random_state=seed,
                n_jobs=cfg["n_jobs"], verbose=-1,
                device=cfg["device_cuda"] if cfg["usar_gpu"] else "cpu",
            )
            m.fit(X_tr_l, y_tr, sample_weight=w_tr,
                  eval_set=[(X_val_l, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)])
            return cohen_kappa_score(y_val, m.predict(X_val_l), weights="quadratic")

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(obj, n_trials=cfg["n_trials"], show_progress_bar=False)
        best_hp = study.best_params
        print(f"  Mejor Kappa Val: {study.best_value:.4f} | {best_hp}")
        ruta_hp.write_text(json.dumps(best_hp))

    clf = lgb.LGBMClassifier(
        **best_hp, objective="multiclass", num_class=N_CLASES,
        class_weight=pesos_clase, random_state=seed,
        n_jobs=cfg["n_jobs"], verbose=-1,
        device=cfg["device_cuda"] if cfg["usar_gpu"] else "cpu",
    )
    clf.fit(X_tr_l, y_tr, sample_weight=w_tr,
            eval_set=[(X_val_l, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)])

    y_pred_val = clf.predict(X_val_l)
    y_prob_val = clf.predict_proba(X_val_l)
    y_pred_te  = clf.predict(X_te_l)
    y_prob_te  = clf.predict_proba(X_te_l)

    joblib.dump(clf, PATHS["FOLDER_MODELS"] / f"{nombre}_{sp}.pkl")
    print(f"  ✓ Guardado: {nombre}_{sp}.pkl")
    m_val = evaluar(y_val, y_pred_val, y_prob_val, nombre, sp, split="val")
    m_te  = evaluar(y_te,  y_pred_te,  y_prob_te,  nombre, sp, split="test")
    return clf, m_val, m_te


def entrenar_tabnet(
    X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_te,
    sp: str, cat_idxs: list, cat_dims: list, cfg: dict,
) -> Tuple:
    nombre  = "TabNet"
    ruta_hp = PATHS["FOLDER_MODELS"] / f"hp_{nombre}_{sp}.json"
    seed    = PARAMETERS["SEED"]

    print(f"\n{'='*52}\n  Entrenando {nombre} — {sp}\n{'='*52}")
    print(f"  Dispositivo: {cfg['dispositivo_tn']}")

    if not cfg["ejecutar_hp"] and ruta_hp.exists():
        best_hp = json.loads(ruta_hp.read_text())
        print(f"  HPs cargados: {best_hp}")
    else:
        def obj(trial):
            p = {
                "n_d"          : trial.suggest_int("n_d", 8, 64, step=8),
                "n_a"          : trial.suggest_int("n_a", 8, 64, step=8),
                "n_steps"      : trial.suggest_int("n_steps", 3, 7),
                "gamma"        : trial.suggest_float("gamma", 1.0, 2.0),
                "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
                "momentum"     : trial.suggest_float("momentum", 0.01, 0.4),
                "mask_type"    : trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
            }
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            m = TabNetClassifier(
                **p,
                optimizer_fn=torch.optim.Adam,
                optimizer_params={"lr": lr},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                scheduler_params={"step_size": 10, "gamma": 0.9},
                cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=3,
                verbose=0, device_name=cfg["dispositivo_tn"], seed=seed,
            )
            m.fit(
                X_tr_sc.astype(np.float32), y_tr,
                eval_set=[(X_val_sc.astype(np.float32), y_val)],
                eval_metric=["balanced_accuracy"],
                max_epochs=100, patience=15,
                batch_size=1024, virtual_batch_size=128, weights=1,
            )
            return cohen_kappa_score(y_val,
                                     m.predict(X_val_sc.astype(np.float32)),
                                     weights="quadratic")

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
        study.optimize(obj, n_trials=min(cfg["n_trials"], 20), show_progress_bar=False)
        best_hp = study.best_params
        print(f"  Mejor Kappa Val: {study.best_value:.4f} | {best_hp}")
        ruta_hp.write_text(json.dumps(best_hp))

    lr_opt = best_hp.pop("lr", 1e-3)
    clf = TabNetClassifier(
        **best_hp,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": lr_opt},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=3,
        verbose=0, device_name=cfg["dispositivo_tn"], seed=seed,
    )
    clf.fit(
        X_tr_sc.astype(np.float32), y_tr,
        eval_set=[(X_val_sc.astype(np.float32), y_val)],
        eval_metric=["balanced_accuracy"],
        max_epochs=200, patience=20,
        batch_size=1024, virtual_batch_size=128, weights=1,
    )
    best_hp["lr"] = lr_opt

    y_pred_val = clf.predict(X_val_sc.astype(np.float32))
    y_prob_val = clf.predict_proba(X_val_sc.astype(np.float32))
    y_pred_te  = clf.predict(X_te_sc.astype(np.float32))
    y_prob_te  = clf.predict_proba(X_te_sc.astype(np.float32))

    clf.save_model(str(PATHS["FOLDER_MODELS"] / f"{nombre}_{sp}"))
    print(f"  ✓ Guardado: {nombre}_{sp}.zip")
    print("  ⚠ Limitación: TabNet usa pesos por clase, no sample_weight individual.")
    m_val = evaluar(y_val, y_pred_val, y_prob_val, nombre, sp, split="val")
    m_te  = evaluar(y_te,  y_pred_te,  y_prob_te,  nombre, sp, split="test")
    return clf, m_val, m_te


def predecir(
    datos_crudos: dict,
    nombre_modelo: str = "XGBoost",
    # Naming simplificado: pipeline_{modelo}.pkl
    año_encuesta: int  = 2024,
) -> dict:
    ruta = PATHS["FOLDER_MODELS"] / f"pipeline_{nombre_modelo}.pkl"
    assert ruta.exists(), f"Pipeline no encontrado: {ruta}"
    art  = joblib.load(ruta)
    tipo = art["tipo_modelo"]

    df_in = pd.DataFrame([datos_crudos])
    df_t  = aplicar_transformaciones_deterministas(df_in, art["transformaciones"], año_encuesta)
    df_t  = df_t.reindex(columns=art["features"])

    feats = art["features"]

    if tipo == "olo":
        X_imp = pd.DataFrame(art["imp_num"].transform(df_t[feats]), columns=feats)
        X_sc  = pd.DataFrame(art["scaler"].transform(X_imp), columns=feats)
        y_pred = art["modelo"].predict(X_sc.values)
        y_prob = art["modelo"].predict_proba(X_sc.values)[0]

    elif tipo == "tabnet":
        X_imp = pd.DataFrame(art["imp_num"].transform(df_t[feats]), columns=feats)
        X_sc  = pd.DataFrame(art["scaler"].transform(X_imp), columns=feats)
        y_pred = art["modelo"].predict(X_sc.values.astype(np.float32))
        y_prob = art["modelo"].predict_proba(X_sc.values.astype(np.float32))[0]

    else:
        X_in = df_t[feats].copy()
        if nombre_modelo == "CatBoost":
            for col in art.get("vars_categoricas", []):
                if col in X_in.columns:
                    X_in[col] = X_in[col].fillna(-999).astype(int).astype(str)
        y_raw  = art["modelo"].predict(X_in)
        y_pred = y_raw.flatten() if hasattr(y_raw, "flatten") else y_raw
        y_prob = art["modelo"].predict_proba(X_in)
        if y_prob.ndim == 2:
            y_prob = y_prob[0]

    clase = int(y_pred[0]) if hasattr(y_pred, "__len__") else int(y_pred)
    ets   = art["etiquetas_target"]
    return {
        "clase_predicha" : clase,
        "etiqueta"       : ets[clase],
        "probabilidades" : {ets[i]: float(p) for i, p in enumerate(y_prob)},
        "modelo"         : nombre_modelo,
        # Campo de split ya no aplica en diseño de validación único
    }
