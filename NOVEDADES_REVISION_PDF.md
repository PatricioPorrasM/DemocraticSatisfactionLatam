# Novedades tras revisar el PDF con los cambios aplicados

Revisión del PDF de la tesis (octubre de 2026) que ya incorpora el
[plan de cambios](PLAN_CAMBIOS_TESIS.md), cotejada contra los artefactos de `results/`,
`models/` y las salidas de `notebooks/output/`.

**Veredicto: el plan se aplicó de forma completa y correcta.** Recalculé todos los valores
numéricos del documento contra los artefactos y las 26 tablas del capítulo 5 coinciden celda por
celda. Quedan **tres novedades** que este documento detalla: una contradicción lógica que el plan
anterior no detectó y dos cifras que no reconcilian, de las cuales **dos son errores míos** que
arrastré como "verificado" en la versión anterior del plan.

Cierro además el último punto abierto: **la Tabla 4.1 queda verificada** contra el entorno real.

---

## 1. Novedad A — §5.5.4 contradice a la Tabla 5.9

**Gravedad: alta.** Es la única inconsistencia de razonamiento, no de redondeo, y afecta al
título de una subsección, a su párrafo de apertura y al rótulo de una figura.

### El problema

§5.5.4 se titula *"Concordancia entre modelos indistinguibles"* y abre así:

> Como las mejores configuraciones de los tres modelos de *gradient boosting* presentan
> diferencias de kappa que el bootstrap pareado no distingue del ruido (Tabla 5.9), cabe
> preguntarse si el ranking de importancia describe los datos o la elección de algoritmo.

Esa premisa es verdadera para las configuraciones **con pesos de clase**, que son las "mejores"
de XGBoost y LightGBM:

| Configuración | Δκw vs. principal | IC 95 % | Tabla 5.9 |
|---|---|---|---|
| XGBoost [pesos de clase] | +0,0046 | [−0,0064; +0,0146] | no distinguible |
| LightGBM [pesos de clase] | +0,0069 | [−0,0058; +0,0189] | no distinguible |

Pero **los valores SHAP que se comparan no son esos**. La concordancia de la Figura 5.10 y la
Tabla 5.22 se calcula sobre las tres configuraciones con **SMOTE-NC**, y esas dos sí difieren de
la principal con intervalos que excluyen el cero:

| Configuración | Δκw vs. principal | IC 95 % | Tabla 5.9 |
|---|---|---|---|
| XGBoost [SMOTE-NC] | **+0,0236** | [+0,0181; +0,0294] | **distinguible** |
| LightGBM [SMOTE-NC] | **+0,0334** | [+0,0192; +0,0469] | **distinguible** |

### Evidencia

- `results/shap/` contiene exactamente tres archivos y los tres son de SMOTE-NC:
  `shap_CatBoost_smotenc.parquet`, `shap_XGBoost_smotenc.parquet`,
  `shap_LightGBM_smotenc.parquet`.
- La salida del NB04 ejecutado lo confirma: *"Cargando SHAP guardados: XGBoost [smotenc]"*,
  *"Cargando SHAP guardados: LightGBM [smotenc]"*, y *"Estado de valores SHAP disponibles
  [estrategia: smotenc]"*.
- El flujo usa `ESTRATEGIA_XAI`, que es la estrategia de la configuración principal (SMOTE-NC),
  para los tres modelos de `MODELOS_CONCORDANCIA`.

El propio §6.3, en su limitación 11, ya describe el diseño correctamente: *"se extiende a los
otros dos modelos de gradient boosting **con la misma estrategia de balanceo**"*. La
contradicción está solo en §5.5.4.

### Por qué el diseño está bien y la justificación mal

Mantener constante la estrategia de balanceo en la ganadora es la decisión metodológica
**correcta**: si cada modelo entrara con su propia estrategia, la comparación de rankings
mezclaría dos factores —algoritmo y tratamiento del desbalance— y no podría atribuirse a
ninguno. Lo que no se sostiene es justificar esa comparación por una indistinguibilidad de
rendimiento que, para esas configuraciones concretas, no existe.

Además, el hallazgo se vuelve **más interesante** al corregirlo: el perfil de determinantes se
mantiene concordante *aunque* el desempeño difiera, lo que es un resultado más fuerte que la
concordancia entre modelos equivalentes.

### Cambios a aplicar

**Título de §5.5.4** (y, por tanto, el índice general, que LaTeX regenera):

> ~~Concordancia entre modelos indistinguibles~~ →
> **Concordancia de la importancia entre los modelos de gradient boosting**

**Párrafo de apertura**, texto de reemplazo:

> Los valores SHAP de esta subsección se calculan para los tres modelos de *gradient boosting*
> manteniendo constante la estrategia de balanceo en la ganadora del Experimento 1 (SMOTE-NC) y
> la formulación ordinal de cuatro clases, de modo que lo único que varía entre los tres sea el
> algoritmo. Cabe preguntarse, entonces, si el ranking de importancia describe los datos o la
> elección de algoritmo. Conviene precisar que estas tres configuraciones no son equivalentes en
> rendimiento: con SMOTE-NC, las diferencias de kappa de XGBoost y de LightGBM respecto de la
> configuración principal son de +0,0236 y +0,0334, y sus intervalos del 95 % excluyen el cero
> (Tabla 5.9). La concordancia que se reporta a continuación no se apoya, por tanto, en una
> indistinguibilidad de desempeño entre las configuraciones comparadas; su interés es justamente
> el inverso: el perfil de determinantes se mantiene estable aun cuando el desempeño difiere.

**Última frase de la subsección** — se mantiene, pero conviene ajustarla para que no arrastre el
encuadre anterior:

> La lectura sustantiva —el predominio de la percepción política y de la evaluación económica
> sobre la confianza institucional— no depende por tanto del algoritmo elegido; la identidad y el
> orden precisos del top-5, sí.

Esta frase ya es correcta y **no requiere cambios**.

### Trabajo manual: una figura

**`results/figures/04_shap_concordancia_modelos.png` (Figura 5.10) lleva el rótulo incorrecto
incrustado en la imagen.** Su subtítulo es *"modelos con rendimiento indistinguible · smotenc"*.
Los valores de la figura son correctos (0,963 / 0,861 / 0,924); solo el rótulo debe cambiar.

Para corregirlo hay que editar la celda 14 del NB04 y reejecutarlo (13 s):

```python
    plot_spearman_estabilidad(
        mat_conc.astype(float),
        titulo=("Concordancia entre rankings de importancia SHAP\n"
                f"modelos con rendimiento indistinguible · {ESTRATEGIA_XAI}"),   # ← cambiar
        nombre_archivo="04_shap_concordancia_modelos",
    )
```

Rótulo sugerido:

```python
                f"modelos de gradient boosting · {ESTRATEGIA_XAI}"),
```

**Nota:** esta es la única figura del documento que requiere acción; la sección 7 del plan
anterior sigue siendo válida para las once restantes.

### Comentario del código fuente

`utils/config.py`, líneas 127–132, arrastra el mismo encuadre. El comentario describe bien el
diseño ("Los tres se explican con la MISMA estrategia de balanceo y la MISMA formulación del
target … lo único que varía es el modelo") pero los llama "configuraciones indistinguibles".
Conviene alinearlo con la redacción nueva. No afecta al cálculo.

---

## 2. Novedad B — §5.3.4: el rango debe ser 0,0137, no 0,0138

**Gravedad: media.** Es una inconsistencia interna, porque el documento reporta la misma
cantidad con dos valores distintos.

§5.3.4 dice:

> En el conjunto de prueba definitivo, las configuraciones seleccionadas en validación de los
> cinco modelos quedan dentro de un rango de **0,0138** puntos de kappa.

Ese rango es la diferencia entre la mejor y la peor de las cinco configuraciones seleccionadas en
validación, es decir CatBoost [SMOTE-NC] menos TabNet [SMOTE-NC]:

```
0,538562 − 0,524818 = 0,013744  →  0,0137
```

**Es exactamente la misma cantidad que la Tabla 5.9 reporta como TabNet [SMOTE-NC] = +0,0137.**
El 0,0138 surge de restar los valores ya redondeados de la Tabla 5.5 (0,5386 − 0,5248), una
convención que el propio §5.3.4 descarta dos frases más abajo al declarar que la brecha de
0,1434 está "calculada antes del redondeo", y que las notas de las Tablas 5.10 y 5.11 también
descartan ("Las diferencias se calculan antes del redondeo").

**Cambio:** en §5.3.4, "un rango de **0,0138** puntos de kappa" → "un rango de **0,0137** puntos
de kappa".

No aparece en ningún otro lugar del documento: el capítulo 6 no repite esta cifra.

---

## 3. Novedad C — §5.3.1: la ganancia de la línea base es +0,0730, no +0,0729

**Gravedad: baja.** Un dígito en la cuarta decimal.

§5.3.1 dice:

> La mejora más pronunciada respecto de la ausencia de balanceo se observa en la línea base
> ordinal, cuyo kappa aumenta de 0,4578 a 0,5308 (**+0,0729**).

```
0,530850 − 0,457847 = 0,073003  →  +0,0730
```

El valor no se obtiene ni con los valores exactos ni con los redondeados de la Tabla 5.5
(0,5308 − 0,4578 = 0,0730), de modo que +0,0729 no reconcilia por ninguna vía.

**Cambio:** "(+0,0729)" → "**(+0,0730)**".

Las otras cuatro ganancias del mismo párrafo son correctas y se confirman con los valores
exactos: OLO +0,0730, XGBoost +0,0330, CatBoost +0,0336, LightGBM +0,0251, TabNet +0,0197. El
rango declarado —"entre 0,0197 (TabNet) y 0,0336 puntos (CatBoost)"— es correcto y **no cambia**.

---

## 4. Punto cerrado — Tabla 4.1 verificada

Era el último ítem que no había podido cotejar. La salida del NB02 ejecutado
(`notebooks/output/02_preprocesamiento_entrenamiento.ipynb`) imprime el entorno real de la
corrida, y **las trece filas de la Tabla 4.1 coinciden**:

| Tabla 4.1 | Corrida real | |
|---|---|---|
| Python 3.12.3 | `Python : 3.12.3` | ✓ |
| Kernel 7.0.0-28-generic | `Sistema : Linux 7.0.0-28-generic` | ✓ |
| pandas 3.0.3 | 3.0.3 | ✓ |
| numpy 2.4.6 | 2.4.6 | ✓ |
| scikit-learn 1.9.0 | 1.9.0 | ✓ |
| xgboost 3.3.0 | 3.3.0 | ✓ |
| lightgbm 4.6.0 | 4.6.0 | ✓ |
| catboost 1.2.10 | 1.2.10 | ✓ |
| torch 2.13.0+cu130 | 2.13.0+cu130 | ✓ |
| optuna 4.9.0 | 4.9.0 | ✓ |
| mord 0.7 | 0.7 | ✓ |
| shap 0.52.0 | 0.52.0 | ✓ |
| imbalanced-learn 0.14.2 | 0.14.2 | ✓ |

La misma salida confirma `GPU usada : True`, `Semilla global : 42` y
`Backend OLO : mord.LogisticIT`, coherentes con §4.2.3 y §4.9.1.

---

## 5. Correcciones a mi propio plan

Dos de las tres novedades son errores míos, no del agente que aplicó los cambios:

| # | Qué hice mal | Dónde |
|---|---|---|
| 1 | Propuse "un rango de **0,0138** puntos" como valor de reemplazo en §5.3.4. Lo calculé restando los valores redondeados de la Tabla 5.5 en lugar de los exactos, y no advertí que es la misma cantidad que la Tabla 5.9 ya reportaba como +0,0137. | Plan v2, §3, tabla de §5.3.4 |
| 2 | Listé "+0,0729" entre los valores "verificados y sin cambios" de §5.3.1. No lo recalculé; el valor exacto es +0,0730. | Plan v2, §3, "Verificado y sin cambios en §5.3.1" |

La novedad A (§5.5.4) no era un error del plan sino una omisión: el plan nunca revisó la
coherencia entre el encuadre de §5.5.4 y la estrategia de balanceo efectivamente usada en el
cálculo de la concordancia.

---

## 6. Inventario de lo verificado y correcto

Para que conste qué se comprobó y contra qué. Todo lo que sigue **no requiere ninguna acción**.

### Capítulo 4

- **§4.6**: las cinco correlaciones del grupo de señal baja (−0,0412 / −0,0096 / +0,0137 /
  +0,0081 / +0,0480), el conteo "cinco … siete", la viñeta nueva de `C_003_003_011`, la
  cardinalidad 696 de `X_004` con el 50,4 % (139 de 276) y su ρ = −0,002 (p = 0,32), `X_008`
  ρ = +0,045 y `H_001_011` ρ = +0,117.
- **§4.6, resto**: Figuras 4.3, 4.4 y 4.5 y sus valores; los 46 pares con |ρ| > 0,85 y el máximo
  de 0,9901; la reducción a un par con máximo 0,8641; el máximo 0,8552 sobre 360 observaciones.
- **§4.9.3**: pesos de clase 2,6524 / 0,9245 / 0,5756 / 1,2439. El párrafo sobre la ponderación
  de TabNet y `X_020` se conservó correctamente al final de la subsección.
- **§4.9.4 y Tabla 4.14**: la reestructuración a una configuración × dos formulaciones, el
  conteo de dieciséis pipelines (confirmado: `models/` contiene exactamente 16 archivos
  `pipeline_*.pkl`) y los 35.084 registros de prueba.
- **§4.11.1**: la reformulación del `KernelExplainer` como rama no ejecutada quedó exacta.
- **§4.11.4**: subsección nueva; su definición de la media por variable dentro del bloque —frente
  a la suma de la importancia global— es correcta y necesaria para leer las Tablas 5.18 y 5.24
  sin contradicción.

### Capítulo 5

- **§5.1**: Venezuela 46,5 % en 2024; KS "entre 0,056 y 0,063"; los 1.184 registros de 2018 con
  726 (61,3 %) y 148 (12,5 %) repartidos en 57 y 91; la serie 21,9 → 39,7 → 49,2 → 49,9 %; toda
  la Tabla 5.1 con sus cinco pérdidas y el total de 58.876; el 4,6 %; los faltantes 12,84 % /
  13,8 % / 6,3 % / 5,3 % y el 50,69 % / 50,66 % / 0,33 % en prueba.
- **§5.2**: Tabla 5.2 con las tres razones (4,6x / 4,7x / 3,7x), KS 0,0549, ρ(año, target)
  0,0217, Figura 5.1 y las 28 correlaciones de la Tabla 5.3, más los valores agregados por
  país-año de V-Dem.
- **§5.3.1**: duración total 12 h 15 min 54 s, el 99,4 % del NB02 y el desglose por cuaderno (los
  componentes redondeados suman exactamente el total declarado); los tres conteos de
  configuraciones; los dos top-3 de validación; las medias por estrategia; la excepción de
  TabNet; las quince combinaciones con val < test; las Tablas 5.4, 5.5, 5.6 y 5.7 completas con
  sus negritas; y el párrafo de métricas complementarias.
- **Tablas 5.8 y 5.9**: las quince filas de intervalos bootstrap y las catorce diferencias
  pareadas, con su reordenamiento, coinciden una a una con
  `bootstrap_ic_modelos.csv` y `bootstrap_pareado_vs_principal.csv`.
- **Tablas 5.10 y 5.11** (nuevas): coinciden exactamente con `h2_tabnet_vs_referencias.csv` y
  `efecto_balanceo_clase_minoritaria.csv`, incluidas las dos filas de sensibilidad.
- **Tablas 5.12 y 5.13**: los siete hiperparámetros de CatBoost y los de los otros cuatro
  modelos, con TabNet entrando por SMOTE-NC, más los cuatro kappas de validación del pie.
- **§5.3.2 y Tabla 5.14**: las siete métricas de ambas formulaciones; el kappa de validación del
  brazo binario (0,4211, confirmado en `models/hp_CatBoost_smotenc_binario.json`); la expansión
  378.592 → 481.068 con 240.534 por clase; la razón 1,7 → 2,1 con la precisión "entrenamiento
  original", que resuelve la ambigüedad que el plan había señalado; y la coincidencia de kappa
  lineal y cuadrático en 0,4759.
- **§5.3.3**: los F1 por clase, la brecha 0,1863, las asimetrías precisión/recall, el 51,58 % /
  40,89 % / 92,47 %, los 2.641 (7,53 %) y 328 (0,93 %), el 29,1 % / 43,5 % y el 0,7377.
- **§5.3.4 y Tablas 5.15–5.16**: las cinco filas de los cuatro cortes; el pie nuevo que declara
  los pesos de clase constantes entre pliegues; el rango 0,0320–0,0380; el par (0,0763 y 0,0821)
  correctamente intacto; las cinco desviaciones del presupuesto homogéneo; la brecha 0,1434
  "antes del redondeo" y el 0,9957 frente a 0,6162.
- **§5.4 y Tabla 5.17**: las cinco filas; la diferencia 0,1384; ρ(MAE, κ) = +0,70; los HHI
  48,6 % y 39,0 % con ρ = −1,00 y −0,70; y los cuatro países extremos bajo la configuración
  principal (Perú 0,4469; Argentina 0,6678; Rep. Dominicana 0,6677; Costa Rica 0,6627).
- **§5.5.1 y Tablas 5.18–5.20**: los seis bloques con sus intervalos; las veinte variables con
  |SHAP|, intervalos, rangos y porcentajes de top-5; la amplitud media de 0,4 posiciones y los
  cuatro pares consecutivos solapados (verifiqué que son exactamente esos cuatro); las ocho
  variables de contribución nula y su descripción; los porcentajes de ausencias; y la razón 16,9.
- **§5.5.2**: las cinco curvas ALE corresponden a las cinco variables generadas por el flujo.
- **§5.5.3 y Tabla 5.21**: las seis filas de errores graves, los 1.774 frente a 867 y los nueve
  pesos LIME de los tres grupos.
- **§5.5.4**: los valores son correctos (0,963 / 0,861 / 0,924; W = 0,9367) y la Tabla 5.22
  reproduce los tres top-5. Lo que falla es solo el encuadre (novedad A).
- **§5.5.5 y Tablas 5.23–5.24** (nuevas): la matriz bloque × subregión ×1000 y los seis
  coeficientes de variación, con la razón media de 2,03 y el rango 0,0356–0,0381.
- **§5.5.6 y Tabla 5.25** (nueva): las diez correlaciones, el rango 0,9669–0,9989, la media
  0,9899 y los dos pares extremos.
- **§5.6 y Tabla 5.26**: recalculé los cuatro tripletes de convergencias sobre las diez primeras
  variables de `tabla_convergencias_CatBoost.csv` y dan 7/1/2, 5/3/2, 4/1/5 y 4/5/1, exactamente
  como la tabla. También el detalle de Easton (posiciones 11, 13 y 14; tres confianzas nulas) y
  el de Norris (tres indicadores por debajo de 0,008).

### Capítulo 6

- **OE2**: el MAE 0,7364, el trío XGBoost/XGBoost/LightGBM de máximos, el "seis de las catorce" y
  la precisión de que la ventaja solo es distinguible frente a TabNet con SMOTE-NC.
- **OE3**: el "cuatro de los cinco modelos", el rango +0,0823 a +0,1826 con su excepción, y el
  párrafo binario reescrito con 0,7377 / 0,8261 / 0,4759.
- **OE4 y OE5**: verificados contra las tablas del capítulo 5.
- **§6.1**: la apertura con las cinco hipótesis; H1 con 0,7364 y [−0,0074; +0,0212]; **H2
  reescrita**, que además mejora mi borrador al precisar que un intervalo que incluye el cero
  "indica ausencia de una diferencia detectable, no equivalencia estadística" —mi redacción decía
  "equivalencia estadística", que es inferencialmente incorrecto—; H3 sin cambios; y los párrafos
  nuevos de H4 y H5 con sus cifras.
- **§6.2**: el "seis de las catorce configuraciones comparadas".
- **§6.3**: la limitación 11 sobre el alcance de la explicabilidad, correctamente redactada.
- **§6.4**: la línea 6 sobre separar imputación y remuestreo es un añadido del documento, no del
  plan, y es pertinente: se corresponde con la limitación 7.

### Referencias cruzadas

Verifiqué las treinta y dos referencias a tablas del cuerpo del texto tras la renumeración de
5.4–5.26. **Todas apuntan al destino correcto**, incluidas las que cruzan capítulos: §4.3.4 →
Tablas 5.8 y 5.9; §5.4 → Tabla 5.21; §5.5.1 → "posiciones quinta y novena de la Tabla 5.19";
§5.5.4 → Tabla 5.9; §6.1 → Tablas 5.7, 5.9, 5.10, 5.18, 5.24 y 5.25. El índice de figuras sigue
conteniendo 2.1–2.8, 4.1–4.5 y 5.1–5.10, sin cambios.

---

## 7. Resumen de acciones

| Acción | Destinatario | Esfuerzo |
|---|---|---|
| Retitular §5.5.4 y reemplazar su párrafo de apertura | Agente LaTeX | 1 título + 1 párrafo |
| §5.3.4: "0,0138" → "0,0137" | Agente LaTeX | 1 valor |
| §5.3.1: "(+0,0729)" → "(+0,0730)" | Agente LaTeX | 1 valor |
| Cambiar el rótulo de `04_shap_concordancia_modelos.png` (celda 14 del NB04) y reejecutar NB04 | Investigador | ~13 s de corrida |
| Alinear el comentario de `utils/config.py` (líneas 127–132) | Investigador | cosmético |
| Borrar el huérfano `results/tables/h2_balanceo_clase_minoritaria.csv` | Investigador | pendiente del plan anterior |
