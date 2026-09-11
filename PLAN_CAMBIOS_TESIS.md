# Plan de cambios sobre la tesis

Cotejo número por número del PDF de la tesis (versión anterior a los últimos cambios
del código fuente) contra los artefactos de la corrida real. Todos los valores nuevos
son trazables a `results/`. La corrida real cerró con 0 errores en los seis notebooks.

Destinatarios:

- **Agente LaTeX**: secciones 1, 2, 3, 4, 4b y 6.
- **Trabajo manual del investigador**: sección 5 (regeneración de figuras).
- **Cambios en el repositorio**: sección 7 (no son trabajo del agente LaTeX).

---

## 0. Decisiones resueltas

Las cuatro decisiones abiertas quedaron cerradas. Ya no hay bloques condicionados: todo el
plan es ejecutable.

| Decisión | Resolución |
|---|---|
| **D1** — Identidad de H2 | H2 es la del PDF: *"TabNet obtendrá un kappa cuadrático superior al de la regresión logística ordinal, pero inferior al del modelo de gradient boosting con mejor desempeño en el conjunto de prueba."* |
| **D2** — Criterio para TabNet | Opción (a): se aplica el criterio de validación de forma consistente en todas las tablas. TabNet queda con SMOTE-NC. |
| **D3** — Evidencia de H4 y H5 | Se agregan las subsecciones correspondientes en el capítulo 5 y los párrafos en §6.1. |
| **D4** — Cardinalidad de X_004 | Se reemplaza 627 por **696** y el "94 %" por **50,4 %**. |

### Impacto de D1 (H2 = la del PDF)

**En la tesis:** H2 pasa a tener un veredicto definido y respaldado, que antes no estaba en el
documento porque no existía el contraste pareado. Se resuelve en la nueva Tabla H2 de la
sección 4 y en el párrafo reescrito de §6.1. El veredicto es **cumplimiento parcial**: la
segunda cláusula se confirma con respaldo estadístico, la primera no se cumple.

**En el código fuente:** aparece una inconsistencia que hay que corregir. El código y el
README implementan una H2 distinta —*"las estrategias de balanceo mejoran el F1 de la clase
minoritaria"*— que ahora **no es una hipótesis de la tesis**. Ver la sección 7 de este
documento para el detalle de los cambios de código.

El análisis de F1 de la clase 0 **no se descarta**: es evidencia válida y ya calculada del
efecto de las estrategias de balanceo, así que se reetiqueta como respaldo del OE3 en lugar de
como contraste de H2. Ver sección 4.

### Impacto de D2 (criterio de validación consistente)

**En la tesis:** afecta Tablas 5.4, 5.5, 5.6, 5.10, 5.13, §5.3.1, §5.3.4, OE3 y el párrafo de
H2. Todo está detallado en las secciones 2 y 3. Dos consecuencias que conviene destacar:

1. Es lo que hace visible y auditable la discrepancia de TabNet, en vez de esconderla eligiendo
   el criterio que convenga en cada tabla.
2. El rango entre las mejores configuraciones de los cinco modelos pasa de 0,0077 a **0,0138**
   puntos de kappa, porque TabNet entra con 0,5248 en lugar de 0,5374.

**En el código fuente:** ningún cambio. Ya verifiqué que `mejor_estrategia_por_modelo.csv`
aplica el criterio de validación y marca la discrepancia de forma explícita:

| modelo | mejor_estrategia_val | kappa_val | kappa_test_de_esa_config | mejor_estrategia_test | coincide_val_test |
|---|---|---|---|---|---|
| CatBoost | smotenc | 0,4840 | 0,5386 | smotenc | sí |
| LightGBM | pesos_clase | 0,4810 | 0,5316 | pesos_clase | sí |
| OLO | smotenc | 0,4801 | 0,5308 | smotenc | sí |
| TabNet | smotenc | 0,4805 | **0,5248** | **pesos_clase** | **NO** |
| XGBoost | pesos_clase | 0,4799 | 0,5340 | pesos_clase | sí |

**Recomendación adicional:** incorporar esa tabla al capítulo 5 (por ejemplo como Tabla 5.5b,
inmediatamente después de la 5.5). Es el artefacto que documenta el criterio y deja trazable
por qué TabNet aparece con SMOTE-NC en las Tablas 5.6 y 5.10 y con pesos de clase en la columna
"Mejor (prueba)" de la 5.5.

---

## 1. Capítulo 4 — Datos y diseño

### §4.6 — exclusiones de Latinobarómetro: nueve valores de ρ

Base: entrenamiento del flujo, 378.592 registros, Spearman con NS/NR como ausentes.

| Variable | ρ en el PDF | ρ correcto |
|---|---|---|
| `A_007_071` Escala izquierda-derecha | −0,027 | **−0,0412** |
| `S_700` | −0,001 | **−0,0096** |
| `S_701` Práctica religiosa | −0,001 | **+0,0137** |
| `C_001_031` | — | **+0,0081** |
| `H_002_101` Confianza Iglesia Católica | +0,030 | **+0,0480** |
| `C_003_003_011` Preocupación desempleo | −0,057 | **−0,0520** |
| `H_001_011` Confianza interpersonal | +0,129 | **+0,1169** |
| `X_008` Tamaño del municipio | +0,059 | **+0,0448** |
| `X_004` | −0,05 | **−0,0017** (p = 0,318, no significativa) |

Consecuencias de texto en la misma sección:

- **`X_004`**: quitar la atribución "identifica el país (ρ = −0,05)" y la cifra de **627
  categorías**, que no se reproduce desde ninguna base (entrenamiento del flujo = 696; todas las
  olas con país y target válidos = 839; archivo base completo = 883). Quitar también el **"94 %
  de categorías nuevas en test"**: el valor real es **50,4 %** (139 de las 276 categorías
  presentes en prueba están ausentes del entrenamiento). La razón real de exclusión es
  cardinalidad y no-solapamiento entre olas; su ρ es −0,0017 y **no es significativa**.
  Sustituir el pasaje por: *"con 696 categorías en el conjunto de entrenamiento, de las cuales el
  50,4 % de las presentes en prueba no aparecen en entrenamiento, y una correlación con el target
  de ρ = −0,002 que no alcanza significancia (p = 0,32)"*.
- **`X_008`**: con ρ = +0,0448 queda **por debajo** del umbral de 0,05, no por encima. La razón
  de exclusión sigue siendo la falta de cobertura en las cinco primeras olas de entrenamiento
  (1995–1998 y 2000); si el texto lo presentaba como "señal suficiente pero sin cobertura",
  reformular a *"ausencia de cobertura en las cinco primeras olas de entrenamiento; su
  ρ = +0,045 tampoco supera el umbral de señal baja"*.
- **`C_003_003_011`**: sigue por encima del umbral (−0,0520), así que el grupo al que pertenece
  no cambia; solo el valor.
- Si el texto agrupa `H_002_101` y `A_007_071` bajo |ρ| < 0,05, la agrupación se mantiene con
  los valores nuevos.

### §4.9.3 — sin cambios

Los pesos de clase (2,6524 / 0,9245 / 0,5756 / 1,2439) están correctos.

### §4.9.4 y Tabla 4.14 — diseño del Experimento 2: reestructuración completa

El diseño cambió: **E2 ya no entrena cinco modelos en binario**. Entrena exactamente una
configuración adicional —la ganadora de E1— reajustada sobre el target binario, con la
estrategia ganadora re-aplicada a ese target.

**Texto de reemplazo para §4.9.4:**

> El segundo experimento contrasta dos formulaciones del target sobre una única configuración:
> la que resulta ganadora en el Experimento 1 según el criterio declarado (mayor kappa
> cuadrático en la validación de 2020 sobre la formulación ordinal de cuatro clases). Esa
> configuración se reajusta sobre el target binario, obtenido al colapsar las categorías 0 y 1
> en "insatisfecho" y las categorías 2 y 3 en "satisfecho", y la estrategia de balanceo ganadora
> se vuelve a aplicar sobre la distribución del target binario, no se reutiliza el conjunto
> remuestreado del caso ordinal. Los hiperparámetros del brazo binario se optimizan de nuevo con
> el mismo presupuesto de Optuna, porque la función de pérdida y el número de clases cambian. El
> diseño evita, por construcción, comparar formulaciones sobre modelos distintos: la única
> variable que cambia entre los dos brazos es la definición del target. En total el Experimento 2
> añade un pipeline a los quince del Experimento 1, de modo que la corrida completa entrena
> dieciséis.

**Tabla 4.14** debe pasar a describir **1 configuración × 2 formulaciones** (no 5 × 2).
Contenido sugerido:

| Elemento | Valor |
|---|---|
| Configuración | La ganadora de E1 (criterio: kappa cuadrático en validación 2020, formulación ordinal) |
| Formulaciones | Ordinal de 4 clases; binaria de 2 clases (0–1 → insatisfecho, 2–3 → satisfecho) |
| Balanceo | La estrategia ganadora de E1, re-aplicada sobre la distribución del target de cada formulación |
| Optimización | Optuna TPE, mismo presupuesto de trials que en E1, objetivo en validación 2020 |
| Pipelines entrenados | 1 (adicional a los 15 de E1) |
| Conjunto de evaluación | Prueba 2023–2024 (35.084 registros) |

Si en el capítulo 4 hay un conteo global de "20 pipelines" o "25 entrenamientos", cambiarlo
a **16**.

---

## 2. Capítulo 5 — Resultados

### §5.1 — un valor

- Venezuela 2024: **"54,5 %" → "46,5 %"**. (El 61,3 % de 2023 está correcto.)
- El resto de §5.1 está verificado y no cambia: 12,84 / 13,8 / 6,3 / 5,3; TV 50,69 %;
  corrupción 50,66 %; `D_001_001` 0,33 %.

### §5.2 — sin cambios

KS validación–prueba 0,0549; ρ(año, target) = 0,0217; n = 431.756; razón 4,7.

### §5.3.1 — tiempos, conteos y afirmaciones

| En el PDF | Reemplazo |
|---|---|
| duración total "13 h 22 min" | **12 h 16 min** (NB02 = 12 h 11 min 33 s) |
| "99,6 %" del tiempo en NB02 | **99,4 %** |
| "las cinco [configuraciones] de E2" | **"la de E2"** |
| top-3 en validación: CatBoost SMOTE-NC (0,4840), CatBoost pesos (0,4829), OLO SMOTE-NC (0,4804) | CatBoost SMOTE-NC (**0,4840**), CatBoost pesos (**0,4829**), **LightGBM pesos de clase (0,4810)** |
| top-3 por mejor configuración de cada modelo: CatBoost SMOTE-NC (0,4840), OLO SMOTE-NC (0,4804), XGBoost pesos (0,4799) | CatBoost SMOTE-NC (**0,4840**), **LightGBM pesos de clase (0,4810)**, **TabNet SMOTE-NC (0,4805)** |
| medias por estrategia: pesos 0,4796 > SMOTE-NC 0,4730 > sin balanceo 0,4472 | pesos **0,4799** > SMOTE-NC **0,4737** > sin balanceo **0,4478** |
| "los cinco modelos conservan la misma mejor estrategia en validación y en prueba" | **"cuatro de los cinco modelos conservan la misma mejor estrategia en validación y en prueba; TabNet es la excepción: SMOTE-NC en validación (0,4805 frente a 0,4783) y pesos de clase en prueba (0,5374 frente a 0,5248)"** |
| "en los cinco casos el intervalo incluye el cero" (ventaja de la principal, rango 0,0011–0,0108) | **"seis de las catorce comparaciones no son distinguibles del cero, con diferencias entre 0,0011 y 0,0108 puntos"** |

**Sin cambios en §5.3.1:** rango de validación 0,3933–0,4840 y de prueba 0,4578–0,5386 (los
quince pares mantienen val < test); ganancia de la línea base ordinal frente a no balancear
+0,0729 (0,4578 → 0,5308); ganancias de los otros cuatro modelos entre 0,0197 (TabNet) y
0,0336 (CatBoost); ventaja distinguible frente a no balancear entre +0,0208 y +0,0807 en las
cinco familias.

### Tabla 5.4 — kappa cuadrático en validación

Cambian 5 celdas y una columna de "mejor".

| Modelo | Sin balanceo | Pesos de clase | SMOTE-NC | Mejor (val) |
|---|---|---|---|---|
| CatBoost | 0,4582 | 0,4829 | **0,4840** | SMOTE-NC |
| LightGBM | **0,4607** ← 0,4578 | **0,4810** ← 0,4796 | 0,4599 | Pesos de clase |
| OLO | 0,3933 | 0,4772 | **0,4801** ← 0,4804 | SMOTE-NC |
| TabNet | 0,4692 | 0,4783 | **0,4805** ← 0,4766 | **SMOTE-NC** ← Pesos de clase |
| XGBoost | 0,4578 | **0,4799** | 0,4641 | Pesos de clase |

Mover la negrita de la fila TabNet de la columna "Pesos de clase" a "SMOTE-NC".

### Tabla 5.5 — kappa cuadrático en prueba

Cambian 3 celdas.

| Modelo | Sin balanceo | Pesos de clase | SMOTE-NC | Mejor |
|---|---|---|---|---|
| OLO | 0,4578 | 0,5278 | **0,5308** | SMOTE-NC |
| XGBoost | 0,5010 | **0,5340** | 0,5150 | Pesos de clase |
| CatBoost | 0,5049 | 0,5373 | **0,5386** | SMOTE-NC |
| LightGBM | **0,5065** ← 0,5050 | **0,5316** ← 0,5320 | 0,5051 | Pesos de clase |
| TabNet | 0,5178 | **0,5374** | **0,5248** ← 0,5208 | Pesos de clase |

### Tabla 5.5b — criterio de selección por modelo (tabla nueva, recomendada)

Insertar inmediatamente después de la Tabla 5.5. Fuente:
`results/tables/mejor_estrategia_por_modelo.csv`. Título sugerido: *"Estrategia seleccionada
en validación por modelo y su desempeño en prueba, frente a la estrategia que maximiza el
kappa en prueba."*

| Modelo | Mejor estrategia (val) | κw val | κw prueba de esa config. | Mejor estrategia (prueba) | Coincide |
|---|---|---|---|---|---|
| CatBoost | SMOTE-NC | 0,4840 | 0,5386 | SMOTE-NC | sí |
| LightGBM | Pesos de clase | 0,4810 | 0,5316 | Pesos de clase | sí |
| OLO | SMOTE-NC | 0,4801 | 0,5308 | SMOTE-NC | sí |
| TabNet | SMOTE-NC | 0,4805 | **0,5248** | **Pesos de clase** | **no** |
| XGBoost | Pesos de clase | 0,4799 | 0,5340 | Pesos de clase | sí |

Esta tabla es lo que hace auditable el criterio: deja explícito por qué TabNet aparece con
SMOTE-NC en las Tablas 5.6 y 5.10 (selección en validación, que es el protocolo declarado) y con
pesos de clase en la columna "Mejor" de la Tabla 5.5 (máximo en prueba, que es solo descriptivo).

**Cambio de encabezado en la Tabla 5.5:** renombrar la última columna de "Mejor" a
**"Mejor (prueba)"**, para distinguirla del "Mejor (val)" de la Tabla 5.4.

### Tabla 5.6 — métricas complementarias

Se aplica el criterio de validación (**D2**), por lo que TabNet entra con SMOTE-NC. Ajustar el
epígrafe de la tabla a *"...con la estrategia seleccionada en validación"*.

| Modelo | Estrategia | Exactitud | F1 macro | MAE ordinal | AUROC | Exact. balanceada |
|---|---|---|---|---|---|---|
| OLO | SMOTE-NC | **0,4196** ← 0,4198 | **0,3975** ← 0,3977 | **0,7364** ← 0,7363 | **0,7142** ← 0,7143 | **0,4588** ← 0,4590 |
| XGBoost | Pesos de clase | 0,4905 | **0,4793** (ahora es el máximo) | 0,6415 | **0,7688** | 0,5084 |
| CatBoost | SMOTE-NC | **0,5158** | 0,4782 | **0,5688** | 0,7582 | 0,4676 |
| LightGBM | Pesos de clase | **0,4893** ← 0,4909 | **0,4785** ← 0,4798 | **0,6472** ← 0,6463 | **0,7675** ← 0,7672 | **0,5104** ← 0,5115 (sigue siendo el máximo) |
| TabNet | **SMOTE-NC** ← Pesos | **0,4671** ← 0,4747 | **0,4507** ← 0,4607 | **0,6639** ← 0,6576 | **0,7401** ← 0,7577 | **0,4755** ← 0,4949 |

**Cambio de negrita:** el mejor F1 macro pasa de LightGBM (0,4798) a **XGBoost (0,4793)**. Los
demás máximos no se mueven (exactitud CatBoost, MAE CatBoost, AUROC XGBoost, exactitud
balanceada LightGBM).

### Tabla 5.7 — intervalos bootstrap

Cambian 3 filas y el orden del ranking.

| Configuración | En el PDF | Reemplazo |
|---|---|---|
| LightGBM pesos de clase | 0,5320 [0,4910; 0,5658] ee 0,0193 | **0,5316 [0,4911; 0,5647] ee 0,0191** |
| TabNet SMOTE-NC | 0,5212 [0,4832; 0,5516] ee 0,0179 | **0,5248 [0,4870; 0,5559] ee 0,0179** |
| LightGBM sin balanceo | 0,5050 [0,4584; 0,5390] ee 0,0206 | **0,5065 [0,4625; 0,5406] ee 0,0206** |

**Reordenamiento:** las filas 11 y 12 se invierten — LightGBM sin balanceo (0,5065) queda por
encima de LightGBM SMOTE-NC (0,5051).

### Tabla 5.8 — diferencias pareadas contra la configuración principal

Cambian 3 filas, el orden y una afirmación.

| Comparación | En el PDF | Reemplazo |
|---|---|---|
| LightGBM [sin balanceo] | +0,0335 [+0,0253; +0,0422] | **+0,0320 [+0,0254; +0,0395]** |
| TabNet [SMOTE-NC] | +0,0174 [+0,0040; +0,0303], p 0,997 | **+0,0137 [+0,0017; +0,0250], p 0,988** |
| LightGBM [pesos de clase] | +0,0066 [−0,0056; +0,0173], p 0,865 | **+0,0069 [−0,0058; +0,0189], p 0,855** |

**Reordenamiento:** LightGBM [SMOTE-NC] (+0,0334) pasa por encima de LightGBM [sin balanceo]
(+0,0320).

**Afirmación asociada:** TabNet [SMOTE-NC] **sí es distinguible** de la configuración principal
(IC [+0,0017; +0,0250], excluye el cero). Si el texto decía que las diferencias frente a las
mejores configuraciones "incluyen el cero en los cinco casos", reformular como en §5.3.1: seis
de las catorce comparaciones no distinguibles, diferencias entre 0,0011 y 0,0108.

### Tabla 5.9 — sin cambios

Verificada, incluido el pie con κw = 0,4840 en validación.

### Tabla 5.10 — hiperparámetros

Cambian la fila de LightGBM completa, la fila de TabNet completa y el pie.

**LightGBM, pesos de clase:**

| Hiperparámetro | En el PDF | Reemplazo |
|---|---|---|
| n_estimators | 600 | **300** |
| num_leaves | 38 | **50** |
| max_depth | 6 | 6 (sin cambio) |
| learning_rate | 0,026616 | **0,052016** |
| subsample | 0,767667 | **0,692828** |
| colsample_bytree | 0,759975 | **0,830985** |
| reg_alpha | 0,020212 | **7,5853 × 10⁻⁵** |
| reg_lambda | 0,016194 | **0,024375** |
| min_child_samples | 43 | **57** |

**TabNet** — cambia además la estrategia por el criterio de validación: **SMOTE-NC** ← Pesos de clase

| Hiperparámetro | En el PDF | Reemplazo |
|---|---|---|
| n_d | 16 | **32** |
| n_a | 32 | **24** |
| n_steps | 6 | 6 (sin cambio) |
| gamma | 1,024487 | **1,139494** |
| lambda_sparse | 1,0223 × 10⁻⁶ | **7,5237 × 10⁻⁶** |
| momentum | 0,161620 | **0,152881** |
| mask_type | entmax | entmax (sin cambio) |
| learning_rate | 0,000649 | **0,00025081** |

OLO (α = 0,007459) y XGBoost: sin cambios.

**Pie de tabla** con los kappa de validación: OLO **0,4801** ← 0,4804; XGBoost 0,4799 (sin
cambio); LightGBM **0,4810** ← 0,4796; TabNet **0,4805** ← 0,4783; CatBoost 0,4840 (sin cambio).

### Tabla 5.11 y §5.3.2 — reemplazo íntegro

**Tabla nueva.** Título sugerido: *"Experimento 2: formulaciones del target sobre la
configuración principal (CatBoost con SMOTE-NC, conjunto de prueba 2023–2024, n = 35.084)"*.

| Métrica | Ordinal (4 clases) | Binaria (2 clases) |
|---|---|---|
| Exactitud | 0,5158 | 0,7649 |
| Exactitud balanceada | 0,4676 | 0,7435 |
| F1 macro | 0,4782 | 0,7377 |
| F1 ponderado | 0,5104 | 0,7674 |
| Kappa cuadrático | 0,5386 | 0,4759 |
| MAE ordinal | 0,5688 | 0,2351 |
| AUROC (OvR macro) | 0,7582 | 0,8261 |

Notas de pie que hay que incluir:

- En la formulación binaria el kappa lineal y el cuadrático coinciden (0,4759): con dos clases
  existe un único tipo de desacuerdo, así que la ponderación cuadrática no distingue nada.
- El MAE y el kappa no son directamente comparables entre formulaciones, porque cambian el
  número de clases y la escala del error. La lectura es descriptiva, no un test de superioridad.

**Texto de reemplazo para §5.3.2:**

> El Experimento 2 compara las dos formulaciones del target sobre una sola configuración
> —CatBoost con SMOTE-NC, ganadora del Experimento 1— y evalúa ambas sobre el mismo conjunto de
> prueba de 2023–2024. El brazo binario reoptimiza sus hiperparámetros con el mismo presupuesto
> de Optuna (50 trials, mejor kappa en validación 0,4211) y vuelve a aplicar SMOTE-NC sobre la
> distribución del target binario, lo que expande el entrenamiento de 378.592 a 481.068 registros
> con 240.534 casos por clase. La formulación binaria mejora todas las métricas de clasificación:
> la exactitud pasa de 0,5158 a 0,7649, el F1 macro de 0,4782 a 0,7377 y el AUROC de 0,7582 a
> 0,8261. Ese salto no indica un modelo mejor, sino una tarea distinta: al colapsar cuatro
> categorías en dos desaparecen los desacuerdos entre niveles adyacentes de la escala, que son
> precisamente los que concentran el error en la formulación ordinal. El kappa cuadrático se
> mueve en dirección contraria (0,5386 frente a 0,4759), y la razón es la misma: con dos clases
> la ponderación cuadrática ya no puede premiar los aciertos parciales. La formulación de cuatro
> clases se conserva como principal porque el objetivo de la tesis es explicar la gradación de la
> satisfacción con la democracia, información que la dicotomización elimina.

Hiperparámetros del brazo binario, por si se citan: iterations 800, depth 7,
learning_rate 0,076532, l2_leaf_reg 7,800333, bagging_temperature 0,338405, border_count 109,
random_strength 3,376493; función de pérdida Logloss.

### §5.3.3 — sin cambios

Verificado completo: 51,58 % / 40,89 % / 92,47 %; 2.641 = 7,53 %; 328 = 0,93 %; 1.774
sobreestimaciones frente a 867 subestimaciones; F1 por clase 0,5796 / 0,3933 / 0,4003 con
brecha 0,1863; 29,1 % y 43,5 %.

### Tabla 5.13 y §5.3.4 — validación temporal en pliegues

**Tabla 5.13**, dos filas:

| Modelo | En el PDF | Reemplazo |
|---|---|---|
| CatBoost | 0,4842 ± 0,0381 · rango 0,4467–0,5373 · MAE 0,6620 ± 0,0402 | 0,4842 ± **0,0380** · rango **0,4471**–0,5373 · MAE **0,6646 ± 0,0429** |
| LightGBM | 0,4824 ± 0,0331 · rango 0,4622–0,5320 · MAE 0,6635 ± 0,0347 | **0,4818** ± **0,0333** · rango **0,4624**–**0,5316** · MAE **0,6647 ± 0,0328** |

XGBoost, TabNet y OLO no cambian; el orden de filas tampoco.

**§5.3.4**, desviaciones estándar del presupuesto homogéneo:

| En el PDF | Reemplazo |
|---|---|
| "0,0022 para XGBoost, 0,0032 para LightGBM y 0,0172 para CatBoost" | **"0,0022 para XGBoost, 0,0025 para LightGBM y 0,0169 para CatBoost"** |
| "los tres modelos de gradient boosting presentan desviaciones de 0,0320 a 0,0381" | **"de 0,0320 a 0,0380"** |
| "TabNet y OLO duplican esa dispersión (0,0763 y 0,0821)" | **(0,0668 y 0,0783)** |
| "casi cuatro veces la mayor de los modelos de árboles" | se mantiene (0,0668 / 0,0169 = 3,96) |
| "su kappa cae a 0,3343 frente a 0,4778 de CatBoost: una brecha de 0,1435 puntos" | **"0,1434 puntos"** (0,4778 y 0,3343 no cambian) |
| "las mejores configuraciones de los cinco modelos quedan dentro de un rango de 0,0078 puntos" | **"de 0,0138 puntos"** (de 0,5386 en CatBoost a 0,5248 en TabNet, con la estrategia seleccionada en validación) |

### §5.4 — sin cambios

Perú 0,4469; Argentina 0,6678; Rep. Dominicana 0,6677; Costa Rica 0,6627; ρ(MAE, κ) = +0,70;
HHI 48,6 % y 39,0 %.

### §5.5 y §5.6 — sin cambios

Toda la sección SHAP/ALE/LIME verificada contra los artefactos de la corrida real.

### Figura 5.10 y Tablas 5.1, 5.2, 5.3, 5.14–5.20 — sin cambios

---

## 3. Capítulo 6 — Conclusiones

### OE2

| En el PDF | Reemplazo |
|---|---|
| "Seis de las quince configuraciones evaluadas resultan estadísticamente indistinguibles de la principal" | **"Seis de las catorce configuraciones comparadas resultan estadísticamente indistinguibles de la principal"** |
| "LightGBM alcanza el mejor F1 macro (0,4798) y la mayor exactitud balanceada (0,5115), y XGBoost el AUROC OvR más alto (0,7688)" | **"XGBoost alcanza el mejor F1 macro (0,4793) y el AUROC OvR más alto (0,7688), y LightGBM la mayor exactitud balanceada (0,5104)"** |

### OE3

| En el PDF | Reemplazo |
|---|---|
| "los cinco modelos conservan la misma mejor estrategia en validación y en prueba" | **"cuatro de los cinco modelos conservan la misma mejor estrategia en validación y en prueba; TabNet cambia de SMOTE-NC en validación a pesos de clase en prueba"** |
| "En la formulación binaria, TabNet obtuvo el mejor F1 macro (0,7385), seguido por CatBoost (0,7381), con las cinco configuraciones en un rango de 0,0107 puntos" | **eliminar por completo** (ya no hay cinco configuraciones binarias) y sustituir por el texto de abajo |

Reemplazo del segundo punto:

> En la formulación binaria, la única configuración entrenada —la ganadora del Experimento 1—
> alcanza un F1 macro de 0,7377 y un AUROC de 0,8261, frente a 0,4782 y 0,7582 en la formulación
> ordinal; el kappa cuadrático se mueve en dirección contraria (0,4759 frente a 0,5386), porque
> con dos clases la ponderación cuadrática deja de premiar los aciertos parciales.

Además, si la redacción del OE3 de tres cláusulas todavía no está en el PDF, hay que insertarla,
junto con la frase de limitación sobre el alcance de SHAP (modelo ganador, estrategia ganadora,
formulación ordinal de cuatro clases únicamente).

### OE1, OE4 y OE5 — sin cambios

### §6.1 — contraste de hipótesis

**H1** — un valor: el MAE de la línea base pasa de **0,7363 → 0,7364**. El IC de la diferencia
frente a OLO con SMOTE-NC ([−0,0074; +0,0212]) y el MAE de 0,5688 no cambian. La frase sobre la
menor dispersión de los árboles en los tres pliegues con presupuesto homogéneo se mantiene, pero
con las desviaciones corregidas de §5.3.4.

**H2** — reescritura completa del párrafo. El veredicto es **cumplimiento parcial**. Texto de
reemplazo (apoyado en la nueva Tabla H2 de la sección 4):

> H2 se cumple solo en parte. La segunda cláusula se confirma: con la estrategia seleccionada en
> validación, TabNet alcanza un kappa cuadrático de 0,5248 en prueba frente a 0,5386 de CatBoost,
> el modelo de gradient boosting con mejor desempeño, y el bootstrap pareado por conglomerados
> país-año sitúa esa diferencia en −0,0137 con un intervalo del 95 % de [−0,0250; −0,0017], que
> excluye el cero. TabNet es, por tanto, distinguiblemente inferior al mejor modelo de gradient
> boosting. La primera cláusula no se cumple: TabNet no supera a la regresión logística ordinal,
> sino que queda 0,0060 puntos por debajo de ella (0,5248 frente a 0,5308), con un intervalo de
> [−0,0158; +0,0050] que incluye el cero. Lo que sostienen los datos es equivalencia estadística
> entre la red neuronal tabular y la línea base ordinal, no la superioridad que anticipaba la
> hipótesis. La conclusión sustantiva es que el costo computacional adicional de TabNet —el
> entrenamiento más largo de los cinco modelos y la mayor dispersión entre pliegues— no compra
> desempeño ni frente a la línea base lineal ni frente al gradient boosting.

Si se quiere dejar constancia de la sensibilidad al criterio de selección, añadir:

> El resultado no depende del criterio de selección en su parte sustantiva. Si en lugar de la
> estrategia seleccionada en validación se toma la que maximiza el kappa de TabNet en prueba
> (pesos de clase, 0,5374), el orden nominal que predice H2 sí se observa —por encima de OLO y por
> debajo de CatBoost—, pero ninguna de las dos diferencias resulta distinguible del cero
> (+0,0066 con [−0,0065; +0,0180] frente a OLO; −0,0011 con [−0,0106; +0,0095] frente a CatBoost).
> Bajo ese criterio alternativo H2 tampoco encuentra respaldo estadístico.

**H3** — sin cambios.

**H4 y H5** — no existen en §6.1; se añaden:

> **H4.** El contraste se sostiene. Sobre la configuración principal, la importancia media de los
> bloques de confianza institucional y de corrupción y seguridad varía entre subregiones con
> coeficientes de variación de 2,61 % y 2,50 %, frente al 1,26 % del bloque de características
> sociodemográficas tomado como referencia: una razón de 2,03 a 1. El bloque institucional es, por
> tanto, más sensible al contexto subregional que el sociodemográfico, aunque la magnitud absoluta
> de esa variación es pequeña.
>
> **H5.** Se confirma. Las correlaciones de Spearman entre los rankings de importancia SHAP de las
> cinco subregiones se sitúan entre 0,9669 y 0,9989, con una media de 0,9899. El par más
> discrepante es Brasil frente al Cono Sur (0,9669) y el más concordante Región Andina frente a
> México y el Caribe (0,9989). La jerarquía de determinantes que aprende el modelo es, en
> consecuencia, estable a través de las subregiones.

---

## 4. Material nuevo disponible para insertar

### Tabla H2 — contraste de la hipótesis H2 (tabla nueva)

Fuente: `results/tables/h2_tabnet_vs_referencias.csv`, que produce el **NB03 §7.2** (apartado
añadido; ver sección 7). Bootstrap pareado de clústeres país-año: kappa cuadrático, 32
conglomerados, B = 1000, semilla 42. Los kappa coinciden exactamente con los de
`resultados_modelos.csv`. El archivo aparece tras reejecutar el NB03.

Título sugerido: *"Contraste de H2: kappa cuadrático de TabNet frente a la línea base ordinal y
al mejor modelo de gradient boosting, con la estrategia seleccionada en validación. Bootstrap
pareado por conglomerados país-año, B = 1000."*

| Comparación (A − B) | κw A | κw B | Δ | IC 95 % | p(Δ > 0) | Lectura |
|---|---|---|---|---|---|---|
| TabNet [SMOTE-NC] − OLO [SMOTE-NC] | 0,5248 | 0,5308 | −0,0060 | [−0,0158; +0,0050] | 0,149 | no distinguible del cero |
| TabNet [SMOTE-NC] − CatBoost [SMOTE-NC] | 0,5248 | 0,5386 | −0,0137 | [−0,0250; −0,0017] | 0,012 | TabNet inferior, distinguible |

Filas de sensibilidad, para el párrafo opcional de §6.1 (estrategia que maximiza el kappa de
TabNet en prueba):

| Comparación (A − B) | κw A | κw B | Δ | IC 95 % | p(Δ > 0) | Lectura |
|---|---|---|---|---|---|---|
| TabNet [Pesos de clase] − OLO [SMOTE-NC] | 0,5374 | 0,5308 | +0,0066 | [−0,0065; +0,0180] | 0,847 | no distinguible del cero |
| TabNet [Pesos de clase] − CatBoost [SMOTE-NC] | 0,5374 | 0,5386 | −0,0011 | [−0,0106; +0,0095] | 0,409 | no distinguible del cero |

**Veredicto:** H2 se cumple parcialmente. Segunda cláusula confirmada con respaldo estadístico;
primera cláusula no cumplida, con equivalencia estadística entre TabNet y OLO.

### Tabla de efecto del balanceo sobre la clase minoritaria — evidencia del OE3

Fuente: `results/tables/efecto_balanceo_clase_minoritaria.csv` (renombrada desde
`h2_balanceo_clase_minoritaria.csv`; ver sección 7). Los valores no cambian.

**Reetiquetación:** esta tabla ya no contrasta H2 (ver **D1**). Entra en el documento como
respaldo cuantitativo del OE3 —el efecto de las estrategias de balanceo— y no debe presentarse
como contraste de hipótesis. Ubicación sugerida: capítulo 5, junto a la Tabla 5.5b, o dentro de
§5.3.1.

Título sugerido: *"Efecto de las estrategias de balanceo sobre el F1 de la clase minoritaria
(clase 0), conjunto de prueba. Bootstrap pareado por conglomerados país-año, B = 1000."*

| Modelo | Estrategia | F1 clase 0 | F1 base | Δ | IC 95 % | Conclusión |
|---|---|---|---|---|---|---|
| OLO | Pesos de clase | 0,3944 | 0,2118 | +0,1826 | [+0,1194; +0,2389] | mejora distinguible |
| OLO | SMOTE-NC | 0,3937 | 0,2118 | +0,1819 | [+0,1183; +0,2379] | mejora distinguible |
| CatBoost | Pesos de clase | 0,4363 | 0,2636 | +0,1727 | [+0,1364; +0,2049] | mejora distinguible |
| XGBoost | Pesos de clase | 0,4395 | 0,3024 | +0,1370 | [+0,1139; +0,1559] | mejora distinguible |
| CatBoost | SMOTE-NC | 0,3933 | 0,2636 | +0,1297 | [+0,0941; +0,1585] | mejora distinguible |
| LightGBM | Pesos de clase | 0,4394 | 0,3118 | +0,1276 | [+0,1030; +0,1554] | mejora distinguible |
| TabNet | Pesos de clase | 0,4269 | 0,3035 | +0,1234 | [+0,0895; +0,1650] | mejora distinguible |
| TabNet | SMOTE-NC | 0,3897 | 0,3035 | +0,0862 | [+0,0543; +0,1285] | mejora distinguible |
| XGBoost | SMOTE-NC | 0,3848 | 0,3024 | +0,0823 | [+0,0575; +0,1058] | mejora distinguible |
| LightGBM | SMOTE-NC | 0,2984 | 0,3118 | −0,0134 | [−0,0422; +0,0220] | no distinguible |

Texto de acompañamiento sugerido (para el OE3, no para H2):

> El efecto de las estrategias de balanceo sobre la clase minoritaria es inequívoco: nueve de las
> diez comparaciones pareadas muestran una mejora distinguible del F1 de la clase 0, con
> incrementos entre +0,0823 y +0,1826 puntos. La única excepción es LightGBM con SMOTE-NC, cuya
> diferencia de −0,0134 tiene un intervalo de [−0,0422; +0,0220] que incluye el cero. Ninguna
> combinación de modelo y estrategia produce un deterioro distinguible, de modo que el balanceo
> mejora la detección de la clase de menor prevalencia sin costo medible en ningún caso.

---

## 4b. Nuevas subsecciones del capítulo 5 para H4 y H5

Ubicación: al final de §5.5, después de la última subsección de explicabilidad existente.
Ambas se apoyan en los valores SHAP de la configuración principal (CatBoost con SMOTE-NC,
formulación ordinal de cuatro clases), que es el único alcance con el que se calcula SHAP.

### §5.5.x — "Variación de la importancia por bloque temático entre subregiones" (H4)

Texto sugerido:

> El contraste de H4 requiere comparar cuánto varía la importancia de cada bloque temático al
> pasar de una subregión a otra. El estadístico empleado es el coeficiente de variación de la
> importancia SHAP media del bloque entre las cinco subregiones, no su rango absoluto: los bloques
> difieren en un orden de magnitud entre sí, de modo que el rango crece con la magnitud del bloque
> y no mide variación comparable. El bloque de características sociodemográficas se toma como
> referencia por ser el de contribución más baja y el menos ligado al contexto institucional.
>
> Los bloques de confianza institucional y de corrupción y seguridad presentan coeficientes de
> variación de 2,61 % y 2,50 %, frente al 1,26 % del bloque de referencia, lo que arroja una razón
> media de 2,03 a 1. La hipótesis se sostiene en la dirección predicha: la importancia atribuida a
> las variables institucionales es más sensible al contexto subregional que la de las variables
> sociodemográficas. La magnitud absoluta de esa variación es, sin embargo, pequeña —la importancia
> media del bloque de confianza institucional se mueve entre 0,0356 en Brasil y 0,0381 en la Región
> Andina—, de modo que la evidencia respalda la existencia del efecto pero no su relevancia
> práctica.
>
> El bloque de contexto democrático presenta un coeficiente de variación de 52,01 %, el más alto
> del conjunto, pero no forma parte del contraste: sus variables provienen de V-Dem y son
> constantes dentro de cada país-año, por lo que su dispersión entre subregiones refleja las
> diferencias de régimen político registradas en la fuente y no heterogeneidad en el comportamiento
> del modelo.

**Tabla 1 de la subsección** — matriz completa, que permite verificar los coeficientes de
variación. Fuente: `results/tables/shap_bloques_por_subregion.csv`, que ahora persiste el
**NB05 §7** (ver sección 7). Valores multiplicados por 1000 para legibilidad.

| Bloque | Cono Sur | R. Andina | Brasil | Centroamérica | México y Caribe |
|---|---|---|---|---|---|
| Percepción política | 119,47 | 123,48 | 118,91 | 119,58 | 118,74 |
| Evaluación económica | 71,96 | 77,41 | 77,63 | 80,25 | 79,91 |
| Contexto democrático | 47,29 | 22,65 | 9,97 | 34,81 | 21,89 |
| Confianza institucional | 36,67 | 38,09 | 35,55 | 37,41 | 36,52 |
| Corrupción y seguridad | 31,44 | 32,29 | 30,81 | 30,54 | 30,41 |
| Características sociodemográficas | 0,69 | 0,68 | 0,68 | 0,68 | 0,67 |

**Tabla 2 de la subsección** — resumen con los coeficientes de variación (la Tabla H4 de abajo).

### §5.5.y — "Concordancia de los rankings de importancia entre subregiones" (H5)

Texto sugerido:

> H5 se contrasta correlacionando, por pares de subregiones, los rankings completos de importancia
> SHAP de las variables. Las diez correlaciones de Spearman fuera de la diagonal se sitúan entre
> 0,9669 y 0,9989, con una media de 0,9899. El par más discrepante es Brasil frente al Cono Sur
> (0,9669) y el más concordante la Región Andina frente a México y el Caribe (0,9989). La
> hipótesis se confirma: la jerarquía de determinantes que aprende el modelo es prácticamente
> invariante entre subregiones, aunque el nivel de error predictivo sí difiera entre ellas, como
> muestra §5.4. En otras palabras, el modelo no cambia qué variables considera importantes al
> cambiar de subregión; cambia solo cuán bien acierta.

**Tabla de la subsección** — la matriz de Spearman (Tabla H5 de abajo).

### Tabla H4

Fuente: `results/tables/h4_variacion_bloques_subregion.csv`

Título sugerido: *"Variación de la importancia SHAP por bloque temático entre subregiones.
CatBoost con SMOTE-NC, formulación ordinal."*

| Bloque | Variables | Importancia media | CV (%) | Mín. | Máx. |
|---|---|---|---|---|---|
| Contexto democrático | 4 | 0,0273 | 52,01 | 0,0100 (Brasil) | 0,0473 (Cono Sur) |
| Evaluación económica | 5 | 0,0774 | 4,29 | 0,0720 (Cono Sur) | 0,0803 (Centroamérica) |
| Confianza institucional | 7 | 0,0368 | 2,61 | 0,0356 (Brasil) | 0,0381 (R. Andina) |
| Corrupción y seguridad | 3 | 0,0311 | 2,50 | 0,0304 (México y Caribe) | 0,0323 (R. Andina) |
| Percepción política | 4 | 0,1200 | 1,63 | 0,1187 (México y Caribe) | 0,1235 (R. Andina) |
| Características sociodemográficas | 5 | 0,0007 | 1,26 | 0,0007 (México y Caribe) | 0,0007 (Cono Sur) |

**Nota necesaria:** el bloque de contexto democrático (CV 52 %) no forma parte del contraste de
H4; su variación refleja que las variables de V-Dem son constantes por país-año, no
heterogeneidad de comportamiento del modelo.

### Tabla H5

Fuente: `results/tables/spearman_subregiones.csv`

Título sugerido: *"Correlación de Spearman entre los rankings de importancia SHAP de las
subregiones."*

| | Cono Sur | R. Andina | Brasil | Centroamérica | México y Caribe |
|---|---|---|---|---|---|
| Cono Sur | 1,0000 | 0,9905 | 0,9669 | 0,9955 | 0,9894 |
| Región Andina | 0,9905 | 1,0000 | 0,9905 | 0,9972 | 0,9989 |
| Brasil | 0,9669 | 0,9905 | 1,0000 | 0,9821 | 0,9905 |
| Centroamérica | 0,9955 | 0,9972 | 0,9821 | 1,0000 | 0,9972 |
| México y Caribe | 0,9894 | 0,9989 | 0,9905 | 0,9972 | 1,0000 |

Mínimo 0,9669; media de los 10 pares fuera de la diagonal 0,9899.

---

## 5. Figuras a regenerar (trabajo manual)

Dependen de valores que cambiaron:

1. Gráficos de barras o heatmaps de **kappa por modelo × estrategia** en validación o prueba
   (Tablas 5.4 / 5.5): cambian 5 celdas de validación y 3 de prueba, y la mejor estrategia de
   TabNet.
2. **Forest plot / gráfico de intervalos bootstrap** (Tablas 5.7 y 5.8): tres configuraciones
   cambian de valor y el orden del ranking se altera en dos posiciones; TabNet [SMOTE-NC] pasa de
   tocar el cero a excluirlo.
3. Cualquier figura de **comparación E1 vs. E2** o de las cinco configuraciones binarias:
   **ya no existe ese objeto**. Reemplazar por la comparación de dos formulaciones sobre una sola
   configuración, o eliminar.
4. Gráfico de **validación temporal en pliegues** (Tabla 5.13): cambian las series de CatBoost y
   LightGBM.
5. Gráfico de **métricas complementarias por modelo** (Tabla 5.6), si existe: la serie de TabNet
   cambia entera y la de LightGBM en cinco puntos.

**No requieren regeneración:** matrices de confusión, figuras SHAP/ALE/LIME, MAE por país y
subregión, mapas de calor de correlaciones, gráficos de faltantes y de distribución del target.

---

## 6. Lo que el agente LaTeX no debe tocar

Verificado contra los artefactos reales y correcto:

- Tablas 5.1, 5.2, 5.3, 5.9, 5.14, 5.15, 5.16, 5.17, 5.18, 5.19, 5.20.
- Figura 5.10.
- §5.3.3 completa, §5.4 completa, §5.5 completa, §5.6, §5.2.
- Los pesos de clase de §4.9.3 (2,6524 / 0,9245 / 0,5756 / 1,2439).
- Toda la sección de faltantes de §5.1 salvo el dato de Venezuela 2024.
- Los valores de poliarquía de V-Dem: Venezuela 0,327 → 0,233 → 0,196; media de 17 países 0,634;
  Nicaragua 2020 0,215 frente a 0,633.
- OE1, OE4, OE5 y el párrafo de H3 en §6.1.

---

## 7. Cambios en el código fuente

No son trabajo del agente LaTeX. Quedan aquí registrados porque **D1** dejaba al código
implementando una hipótesis que no es la de la tesis. **Todos están aplicados.**

### 7.1 Aplicado

| Archivo | Cambio |
|---|---|
| `utils/config.py` | Las nueve correlaciones de `VARS_EXCLUIR_LB` estaban medidas sobre una definición más laxa del entrenamiento (429.948 filas). Ahora citan el entrenamiento exacto del flujo (378.592), y la cardinalidad de `X_004` es 696 con su ρ no significativo, en coherencia con **D4**. |
| `utils/config.py` | Añadidos `MODELO_H2`, `MODELO_BASE_H2` y `MODELOS_GB_H2` a `PARAMETERS`. Las tres piezas del contraste de H2 quedan en el único lugar donde se editan parámetros, en vez de escritas a mano en el notebook. |
| `README.md` | La fila de H2 en la tabla de hipótesis pasa a la formulación del PDF y apunta a NB03 §7.2. La descripción del NB03 se divide en dos párrafos: efecto del balanceo (§7.1, OE3) y contraste de H2 (§7.2). |
| `notebooks/03_evaluacion_comparativa.ipynb` §7 | Retirada la mención a H2 del enunciado de la sección. |
| `notebooks/03_evaluacion_comparativa.ipynb` §7.1 | Reetiquetada: el análisis de F1 de la clase 0 pasa de "contraste formal de H2" a "efecto de las estrategias de balanceo sobre la clase minoritaria", con la advertencia explícita de que respalda el OE3 y no contrasta ninguna hipótesis. El cálculo no cambia; la regla declarada pasa de "regla de decisión" a "regla de lectura". |
| `notebooks/03_evaluacion_comparativa.ipynb` §7.1 | Salida renombrada: `h2_balanceo_clase_minoritaria.csv` → **`efecto_balanceo_clase_minoritaria.csv`**. Variables internas renombradas en consecuencia (`df_h2` → `df_balanceo`, `BASE_H2` → `BASE_BALANCEO`). |
| `notebooks/03_evaluacion_comparativa.ipynb` §7.2 | **Apartado nuevo** con el contraste de la H2 real: bootstrap pareado del kappa cuadrático de TabNet frente a la línea base ordinal y frente al mejor gradient boosting, los tres con la estrategia que ganó en validación. Incluye la regla de decisión por cláusula declarada antes de los resultados, el veredicto automático (confirmada / parcial / no sostenida) y el análisis de sensibilidad con la estrategia óptima en prueba cuando difiere. Salida: **`h2_tabnet_vs_referencias.csv`**. |
| `notebooks/05_estabilidad_temporal_regional.ipynb` §7 | La matriz de importancia SHAP por bloque × subregión se construía en memoria y se descartaba. Ahora se imprime y se persiste en **`shap_bloques_por_subregion.csv`**, que es la base aritmética del coeficiente de variación de H4: sin ella el CV no es verificable desde los artefactos. |

La referencia de gradient boosting de §7.2 no está escrita a mano: se elige, entre los modelos
de `MODELOS_GB_H2` que tienen predicciones disponibles y con la estrategia que cada uno ganó en
validación, el de mayor kappa en prueba —literalmente lo que pide el enunciado de la hipótesis—.
Si falta el pipeline de alguno, el apartado avisa de que la referencia se eligió sobre un
subconjunto en lugar de abortar el contraste.

### 7.2 Verificación

Ejecuté las dos celdas nuevas contra los artefactos de la corrida real. La de H2 reproduce los
cuatro pares de la Tabla H2 con veredicto "SE CUMPLE PARCIALMENTE (1/2 cláusulas)", y la de
NB05 reproduce la matriz bloque × subregión y deja
`h4_variacion_bloques_subregion.csv` byte a byte idéntico al de la corrida real, lo que confirma
que el cambio no altera lo ya calculado.

`results/tables/shap_bloques_por_subregion.csv` quedó escrito por esa verificación local. Su
contenido es el que producirá el NB05 en el servidor; está disponible desde ya para el agente
LaTeX.

### 7.3 Pendiente

Reejecutar **NB03 y NB05 únicamente** —no el NB02, que es el que consume las doce horas—. Ninguna
de las dos celdas nuevas reentrena nada: leen los pipelines y los valores SHAP ya serializados.

Después de esa corrida queda un archivo huérfano, `results/tables/h2_balanceo_clase_minoritaria.csv`,
que ya no se regenera bajo ese nombre. Conviene borrarlo para que no quede en `results/` una tabla
cuyo nombre apunta a una hipótesis que no es la de la tesis.

---

## Resumen de escala

- **19 valores** cambian en el capítulo 5, más **dos filas completas** de hiperparámetros.
- **Una tabla y dos secciones** se reestructuran por el rediseño de E2 (Tabla 5.11 / §5.3.2, más
  Tabla 4.14 / §4.9.4).
- **Nueve correlaciones** y una cardinalidad en §4.6.
- **Seis afirmaciones de texto** dejan de ser ciertas y hay que reescribirlas.
- **El párrafo de H2 en §6.1 se reescribe por completo**, con veredicto de cumplimiento parcial.
- **Dos subsecciones nuevas** en el capítulo 5 para H4 y H5, con texto redactado.
- **Seis tablas nuevas** para insertar: H2 (contraste), efecto del balanceo (OE3), 5.5b
  (criterio de selección), bloque × subregión, H4 (CV por bloque) y H5 (Spearman).
- **Ocho cambios de código aplicados** (sección 7), ninguno a cargo del agente LaTeX. Queda
  pendiente reejecutar NB03 y NB05 para materializar las dos tablas nuevas.
