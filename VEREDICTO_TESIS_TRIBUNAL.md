# ¿Está el PDF listo para el tribunal?

**Revisión del 2026-09-09 sobre el PDF con todas las modificaciones aplicadas.**

## Veredicto

**Todavía no.** Hay **siete defectos que se deben corregir** antes de presentarlo, y ninguno
requiere volver a correr nada: seis son de texto y uno es una cita de LaTeX sin resolver.

El cuerpo cuantitativo, en cambio, está **sólido**. Se comprobaron **más de 540 cifras** contra
los artefactos de la corrida y prácticamente todas coinciden dígito a dígito. Los ajustes de la
revisión anterior se aplicaron correctamente: el p-valor del KS, los 20 ensayos de la OLO, la
Tabla 5.10 con los hiperparámetros de los cuatro modelos que faltaban, los promedios de
poliarquía, el missingness sobre las 28 predictoras, la distribución de la variante binaria, los
pies de las Figuras 5.2–5.4, la base de cálculo de la multicolinealidad V-Dem y el conteo de
métricas. Todos verificados.

Los defectos que quedan se concentran en **dos pasajes concretos**: el párrafo de Venezuela de
§5.1 y la reescritura de §4.6. Más una cita rota y una omisión estructural.

---

# BLOQUE 1 — Los siete que hay que corregir

## 🔴 1. Cita sin resolver en la página 2 del cuerpo

§1.1 imprime literalmente:

> «Los estudios de Ergun et al. **[?]** y Rosa et al. [7] han demostrado…»

El `[?]` es una clave de `\cite` que LaTeX no encontró. Ergun et al. **sí está** en la
bibliografía, como referencia **[51]**. Es un error de compilación visible en la segunda página
de texto: es lo primero que verá el tribunal.

**Corrección:** cambiar la clave por la de la referencia [51] y recompilar dos veces para que
BibTeX resuelva.

## 🔴 2. §1.1 atribuye a Ergun et al. algo que Ergun et al. no hizo

Es la misma frase, y el problema de fondo es peor que la cita:

> «Los estudios de Ergun et al. [?] y Rosa et al. [7] han demostrado que los modelos de
> **gradient boosting y las redes neuronales** superan sistemáticamente a la regresión
> logística…»

**Esto contradice al propio documento.** §3.2 dice de ese mismo trabajo:

> «los autores estiman distintas especificaciones de **logit ordenado multinivel**… En lugar de
> métricas como accuracy, F1 o AUC, el artículo reporta coeficientes, errores estándar robustos
> y significancia estadística.»

Y la Tabla 3.1 lo registra como «Logit ordenado multinivel», objetivo «Regresión ordinal
explicativa», XAI «No». Ergun et al. no aplicaron ningún modelo de aprendizaje automático.

Es una contradicción interna sobre un antecedente que la tesis usa como justificación, y un
lector de ciencia política que conozca ese artículo la detecta de inmediato.

**Texto de reemplazo:**

> Frente a este panorama, la ciencia política computacional ha comenzado a explorar si los
> modelos de aprendizaje automático pueden capturar mejor que los modelos estadísticos clásicos
> la complejidad no lineal de los determinantes de la satisfacción democrática. Ergun et al.
> [51] muestran, mediante un logit ordenado multinivel, que las características del sistema
> político se asocian con la satisfacción declarada; Rosa et al. [7], Tauil et al. [53] y
> Ferreyra et al. [52] han demostrado que los modelos de gradient boosting, los ensambles y las
> redes neuronales alcanzan un rendimiento predictivo competitivo sobre variables actitudinales
> de encuestas con múltiples interacciones entre predictores. Sin embargo, estos trabajos
> presentan limitaciones importantes: se circunscriben a pocos países y períodos cortos,
> utilizan formulaciones binarias de la variable objetivo que reducen la información disponible
> y rara vez incorporan análisis sistemáticos de explicabilidad.

## 🔴 3. §5.1 — la cifra de Venezuela 2018 no es reproducible, y la afirmación que la acompaña es falsa

Este es el hallazgo más serio, porque el pasaje sostiene la decisión metodológica más
consecuente de la tesis: excluir Venezuela a partir de 2018.

**Texto actual:**

> «Previo al proceso de exclusión se tenía adicionalmente en 2018: el **73,7 %** de las personas
> encuestadas se ubicó ese año en la categoría "Muy satisfecho", **sin observaciones en "Para
> nada satisfecho" ni "No muy satisfecho"**.»

**Distribución real de Venezuela en 2018** (`data/base/latinobarometro.csv`, n = 1.200):

| Categoría | Casos | % de válidos |
|---|---|---|
| Para nada satisfecho | **57** | 4,8 % |
| No muy satisfecho | **91** | 7,7 % |
| Más bien satisfecho | 310 | 26,2 % |
| **Muy satisfecho** | **726** | **61,3 %** |
| NS/NR | 16 | — |

Las dos afirmaciones fallan:

1. **El porcentaje es 61,3 %, no 73,7 %.** Se probaron seis bases de cálculo distintas —sobre
   respuestas válidas, sobre el total de filas, ponderada por `X_020`, sobre solo las dos
   categorías de satisfacción, excluyendo la categoría 1— y **ninguna da 73,7 %**.
2. **Sí hay observaciones en las dos categorías de insatisfacción**: 57 y 91. La frase «sin
   observaciones» es exactamente el tipo de dato llamativo que un revisor comprueba.

**Texto de reemplazo (verificado):**

> Previo al proceso de exclusión se tenía adicionalmente en 2018: el 61,3 % de las personas
> encuestadas se ubicó ese año en la categoría "Muy satisfecho" y solo el 12,5 % en las dos
> categorías de insatisfacción, frente al 42,4 % que declaraba insatisfacción en 2013.

> Los porcentajes de 2013, 2015, 2016 y 2017 que cita el mismo párrafo —**21,9 %, 39,7 %,
> 49,2 % y 49,9 %**— son **exactos**. Solo 2018 y 2024 fallan.

## 🔴 4. §5.1 — la cifra de Venezuela 2024 tampoco es reproducible

> «En 2024, la categoría de máxima satisfacción todavía concentra el **54,5 %** de las
> respuestas.»

**Real: 46,5 %** (544 de 1.170 respuestas válidas; 1.200 filas sin ningún NS/NR ese año).
Igual que en el caso anterior, ninguna de las bases alternativas da 54,5 %.

**Texto de reemplazo:** «En 2024, la categoría de máxima satisfacción todavía concentra el
46,5 % de las respuestas.»

El contraste que el párrafo quiere establecer se sostiene igual de bien: 46,5 % de máxima
satisfacción declarada frente a un índice de poliarquía de 0,196.

## 🔴 5. §4.6 — una de las seis variables de «señal baja» no cumple el criterio

§4.6 afirma:

> «Seis se excluyeron por señal baja (|ρ| < 0,05 con la variable objetivo en el conjunto de
> entrenamiento): la escala izquierda-derecha (A_007_071, ρ = −0,0082), la religión (S_700,
> ρ = −0,0077), la práctica religiosa (S_701, ρ = +0,0233), el problema más importante del país
> (C_001_031, ρ = +0,0395), la confianza en la Iglesia (H_002_101, ρ = +0,0411) y la
> preocupación por quedar sin trabajo (**C_003_003_011, ρ = −0,0442**).»

El problema es de reproducibilidad. Estas seis cifras solo se obtienen si los **códigos
negativos de NS/NR del archivo base se tratan como valores válidos**, algo que el flujo de la
tesis no hace en ninguna otra parte. Con el tratamiento correcto (NS/NR como ausente) los
valores cambian, y **uno cruza el umbral**:

| Variable | ρ que cita el PDF | ρ con NS/NR como ausente | ¿|ρ| < 0,05? |
|---|---|---|---|
| `A_007_071` escala izquierda-derecha | −0,0082 | −0,0261 | sí |
| `S_700` religión | −0,0077 | −0,0006 | sí |
| `S_701` práctica religiosa | +0,0233 | −0,0005 | sí |
| `C_001_031` problema más importante | +0,0395 | +0,0156 | sí |
| `H_002_101` confianza en la Iglesia | +0,0411 | +0,0301 | sí |
| **`C_003_003_011`** preocupación por desempleo | **−0,0442** | **−0,0572** | **no** |

Cinco de las seis aguantan con cualquiera de los dos tratamientos. La sexta no: la
preocupación por quedar sin trabajo tiene |ρ| = 0,057 y **el propio criterio que la frase
declara la excluiría de la lista**.

**Corrección recomendada** — publicar los valores con NS/NR tratado como ausente y mover
`C_003_003_011` fuera del grupo de señal baja:

> Seis se excluyeron por señal baja (|ρ| < 0,05 con la variable objetivo en el conjunto de
> entrenamiento, tratando los códigos de no respuesta como valores ausentes): la práctica
> religiosa (S_701, ρ = −0,0005), la religión (S_700, ρ = −0,0006), el problema más importante
> del país (C_001_031, ρ = +0,0156), la escala izquierda-derecha (A_007_071, ρ = −0,0261), la
> confianza en la Iglesia (H_002_101, ρ = +0,0301) y la región geográfica (X_004, ρ = −0,0473).

Y trasladar `C_003_003_011` (ρ = −0,0572) al segundo grupo, con el motivo conceptual que
corresponda. Nótese que con este tratamiento **X_004 sí cumple el umbral**, lo que además
resuelve el apartado 6.

> ⚠️ Esta discrepancia se originó en un cálculo que le entregué sin enmascarar los códigos de
> NS/NR. La responsabilidad del error es mía; la corrección es la que figura arriba.

## 🔴 6. §4.6 — el p-valor de X_004 es estadísticamente imposible, y su cardinalidad no coincide

> «X_004 (región o área geográfica): es una variable nominal con **627 categorías únicas**…
> Adicionalmente, el diagnóstico aportado registra una asociación débil y no significativa con
> la variable objetivo (**r = 0,019; p = 0,132**).»

Dos problemas, y el segundo es el que delata:

- **`p = 0,132` no puede ser correcto.** Con n ≈ 386.000 observaciones, una correlación de
  0,019 tendría p ≈ 10⁻³¹, no 0,132. El valor reproducible es **ρ = −0,0473 con p ≈ 4×10⁻¹⁸⁰**.
  Un revisor con formación estadística ve «p = 0,132» junto a un n de seis cifras y sabe
  inmediatamente que algo no cuadra. Y la palabra «no significativa» es indefendible con
  cualquier n de este orden.
- **Las categorías son 730, no 627.** Contadas sobre las olas de entrenamiento con NS/NR como
  ausente: 730. Sin enmascarar: 735. Sobre las 24 olas completas: 886. El 627 no corresponde a
  ninguna base del conjunto entregado.

**Texto de reemplazo (todo verificado):**

> `X_004` (región o área geográfica): es una variable nominal con 730 categorías únicas en el
> conjunto de entrenamiento, el 95,3 % de las cuales aparece en un solo país, de modo que su
> inclusión permitiría al modelo recuperar la identidad nacional y contravendría la decisión de
> canalizar el contexto de país exclusivamente a través de los indicadores de V-Dem. Además, de
> las 282 categorías presentes en el conjunto de prueba, 137 (el 48,6 %) no aparecen en
> entrenamiento, lo que impide establecer la correspondencia entre ambos conjuntos. Su
> correlación con la variable objetivo es, asimismo, inferior al umbral (ρ = −0,0473).

El 95,3 % y el 48,6 % están comprobados sobre el archivo base.

## 🔴 7. Las tres hipótesis se plantean y nunca se resuelven

§1.2 enuncia H1, H2 y H3 de forma explícita y numerada. Después:

- §5.3.1 dice «Este resultado también acota el alcance de **H1 y H2** (véase la discusión en
  los apartados correspondientes)» — pero **no dice qué apartados**, y no hay ninguno que
  entregue el veredicto.
- §5.5.1 dice «El análisis SHAP global permite contrastar la hipótesis **H3**», presenta los
  datos… y no concluye si H3 se sostiene.
- **El capítulo 6 no menciona H1, H2 ni H3 en ninguna línea.** Recorre OE1 a OE5 y termina.

Una tesis que declara tres hipótesis de trabajo tiene que decir si se confirmaron. Es la
primera pregunta que hará el tribunal, y ahora mismo el documento no la responde.

**Los tres veredictos, con los datos ya presentes en el PDF:**

| Hipótesis | Qué predijo | Qué salió | Veredicto |
|---|---|---|---|
| **H1** | El *gradient boosting* superará a la OLO en las métricas predictivas | En kappa la ventaja es de 0,0077 con IC [−0,0074; +0,0212], que incluye el cero. En **MAE ordinal** la ventaja es de 0,1676 y **sí** es distinguible. Y en estabilidad entre pliegues la separación es nítida (sd 0,0022–0,0172 frente a 0,0783) | **Parcialmente confirmada**: no en acuerdo ordinal, sí en distancia del error y en estabilidad |
| **H2** | TabNet > OLO pero < mejor *gradient boosting* | TabNet 0,5374 · OLO 0,5308 · CatBoost 0,5386. El orden predicho se cumple, pero las tres diferencias caen dentro del ruido de muestreo | **Compatible en el orden, no verificable en la magnitud** |
| **H3** | Confianza institucional, corrupción y evaluación económica concentrarán la mayor contribución SHAP | El bloque dominante es **percepción política** (0,4818), que H3 no menciona. Evaluación económica 2.ª (0,3853) y confianza institucional 3.ª (0,2601) sí figuran, pero **corrupción y seguridad queda 5.ª de 6** (0,0938) | **Parcialmente refutada** |

Añadir un apartado breve —«6.x Contraste de las hipótesis»— con esta tabla cierra el hueco sin
tocar ningún resultado.

---

# BLOQUE 2 — Cuatro observaciones que conviene atender

## 🟡 8. La Tabla 5.11 compara dos columnas obtenidas con estrategias distintas

La tabla invita a comparar, fila por fila, la formulación ordinal con la binaria. Pero:

- La columna **binaria** se obtuvo con **`pesos_clase` para los cinco modelos** (verificado en
  `results/resultados_modelos.csv`: las diez filas de `variante_target = binario` tienen
  `estrategia_balanceo = pesos_clase`).
- La columna **ordinal** usa **la mejor estrategia de cada modelo**: `smotenc` para OLO y
  CatBoost, `pesos_clase` para los otros tres.

Para OLO y CatBoost, por tanto, las dos celdas de la misma fila vienen de configuraciones
distintas, y §4.9.4 declara que E2 usa «la mejor estrategia de balanceo identificada en E1», en
singular.

**Corrección más limpia** — poner en la columna ordinal los valores con `pesos_clase`, de modo
que ambas columnas queden controladas. Solo cambian dos celdas:

| Modelo | κw ordinal actual | κw ordinal con `pesos_clase` |
|---|---|---|
| OLO | 0,5308 | **0,5278** |
| XGBoost | 0,5340 | 0,5340 (no cambia) |
| CatBoost | 0,5386 | **0,5373** |
| LightGBM | 0,5320 | 0,5320 (no cambia) |
| TabNet | 0,5374 | 0,5374 (no cambia) |

Alternativa sin cambiar cifras: añadir al pie «La variante binaria se entrenó con pesos de
clase en los cinco modelos; la columna ordinal reporta la mejor estrategia de cada modelo, de
modo que ambas columnas no constituyen una comparación pareada de formulaciones».

## 🟡 9. §4.6 mezcla dos notaciones con valores que no se reproducen

La sección usa **ρ** para las seis de señal baja y **r** para las tres restantes, con cifras que
provienen de una fuente distinta:

| Variable | Valor que cita el PDF | Valor reproducible |
|---|---|---|
| `X_008` tamaño de la ciudad | r = 0,048 | **ρ = +0,0580** |
| `H_001_011` confianza interpersonal | r = 0,112 | **ρ = +0,1292** |
| `X_004` región | r = 0,019 | **ρ = −0,0473** |

Conviene unificar el símbolo (todo ρ de Spearman, como declara el resto de la sección) y usar
los valores reproducibles. Si las cifras con «r» vienen de un diagnóstico previo con otra base,
hay que decirlo o retirarlas.

## 🟡 10. §4.6 — X_008 carece de cobertura en cinco olas, no en seis

> «Además, carece de cobertura en las **seis** primeras olas de entrenamiento, correspondientes
> al período 1995–2000.»

El período 1995–2000 es correcto, pero contiene **cinco** olas de encuesta, no seis: 1995, 1996,
1997, 1998 y 2000 —el Latinobarómetro no realizó encuesta en 1999, como el propio documento
señala en §4.3.1—. La primera ola con dato válido en `X_008` es 2001.

**Corrección:** «carece de cobertura en las cinco primeras olas de entrenamiento (1995–1998 y
2000)».

## 🟡 11. §5.1 — dos de los cuatro KS de los noventa superan el umbral que cita el texto

> «El test de Kolmogorov–Smirnov entre la distribución venezolana y la del resto de los países
> alcanza un estadístico de 0,2552 en 2017 (p < 0,001), frente a valores **inferiores a 0,08**
> en las olas de los años noventa.»

El 0,2552 de 2017 es **exacto** (confirmado en la salida del NB02). Los valores de los noventa:

| Ola | KS real |
|---|---|
| 1995 | 0,0592 ✓ |
| 1996 | 0,0749 ✓ |
| 1997 | **0,0848** ✗ |
| 1998 | **0,0827** ✗ |

**Corrección:** «frente a valores comprendidos entre 0,059 y 0,085 en las olas de los años
noventa». El contraste con 0,2552 sigue siendo de un factor de tres.

---

# BLOQUE 3 — Cosmética (no bloquean, pero se ven)

- **Abstract:** «(κw = 0,5386) and the lowest ordinal MAE (0.5688)» mezcla coma y punto decimal
  en la misma frase. En texto inglés deben ser ambos punto: `0.5386` y `0.5688`.
- **§5.2:** «con n = **431,756** el contraste resulta significativo» usa coma decimal donde el
  resto del documento escribe 431.756. Se lee como «431 con 756 milésimas».
- **Tabla 5.16:** tres límites superiores de IC están redondeados con un error de una unidad en
  la cuarta decimal: democracia electoral (0,0081 → **0,0080**), integridad institucional
  (0,0011 → **0,0010**) y expectativa económica personal (0,0081, valor real 0,008050, en el
  límite del redondeo). Afecta a las variables de menor importancia y a ninguna conclusión.

---

# Lo que se comprobó y está correcto

Más de 540 cifras verificadas una a una contra `results/`, `models/`, `data/processed/` y las
salidas de los notebooks. Todo lo que sigue **coincide y no hay que tocarlo**:

| Elemento | Resultado de la comprobación |
|---|---|
| **Tablas 5.4, 5.5, 5.6** | Los 55 valores de kappa y métricas complementarias coinciden con `resultados_modelos.csv` |
| **Tabla 5.7** | Los 60 valores (κ, IC inferior, IC superior, EE) coinciden con `bootstrap_ic_modelos.csv` · B = 1.000, 32 clústeres |
| **Tabla 5.8** | Las 14 filas × 5 campos, incluidas las conclusiones «distinguible / no distinguible», coinciden con `bootstrap_pareado_vs_principal.csv` |
| **Tablas 5.9 y 5.10** | Los 33 valores óptimos coinciden con `hp_optimizados` de los cinco `models/hp_*.json`. Los kappas y los ensayos de los pies también (20 para OLO y TabNet, 50 para los tres de *gradient boosting*) |
| **Tablas 4.9–4.12** | Los 33 rangos y escalas coinciden con `espacio_busqueda`. Los 25 valores seleccionados caen dentro de sus intervalos y respetan los pasos discretos |
| **Tabla 5.11 (cifras)** | Los 15 valores binarios coinciden exactamente; el problema es de estrategia, no de aritmética (ver apartado 8) |
| **Tablas 5.12 y 5.13** | Las cuatro ventanas de pliegue y los 30 valores de media, sd, mín, máx y MAE coinciden con `validacion_temporal_*.csv` |
| **§5.3.4 presupuesto homogéneo** | Las cinco sd sobre los tres pliegues históricos (0,0022 · 0,0032 · 0,0172 · 0,0668 · 0,0783) coinciden. Los 15 ensayos de los pliegues históricos, confirmados |
| **§5.3.4 pliegue 1** | OLO κ = 0,3343 y MAE = 0,9957 frente a CatBoost 0,4778 y 0,6162 · exactos |
| **Tabla 5.14** | Los 20 valores de las cinco subregiones coinciden; los n suman exactamente 35.084 |
| **§5.4 inversión MAE/kappa** | ρ(MAE, κ) = **+0,7000** exacto · ρ(HHI, MAE) = **−1,0000** exacto · ρ(HHI, κ) = **−0,7000** exacto · concentración modal 48,6 % y 39,0 % exactas |
| **§5.4 países** | Perú 0,4469 · Argentina 0,6678 · Rep. Dominicana 0,6677 · Costa Rica 0,6627 · exactos |
| **Tabla 5.15** | Los 24 valores de los seis bloques (nº de variables, |SHAP| total, IC) coinciden con `shap_bloques_ic_*.csv` |
| **Tabla 5.16** | 137 de 140 valores exactos (las 3 excepciones son redondeos de la 4.ª decimal). Los 20 rangos, los IC de rango y los % top-5 coinciden |
| **§5.5.1 estabilidad** | Amplitud media del intervalo de rango = **0,4000** exacta · los **cuatro** pares con IC solapados son exactamente los cuatro que el texto nombra |
| **§5.5.1 SHAP nulo** | Las **ocho** variables de contribución nula son exactamente las que el texto describe. Los porcentajes de ausencias que las contrastan (0,01 / 0,36 / 0,49 / 1,32 frente a 32,9 / 40,8) verificados |
| **Tabla 5.17 y §5.5.1** | Los 9 valores micro/macro coinciden · la razón «casi diecisiete veces (16,9)» da **16,93** |
| **Tabla 5.18** | Las 6 filas coinciden en distancia, casos, porcentaje y países. Totales: 2.641 (7,53 %), 1.774 sobreestimaciones, 867 subestimaciones, 328 de distancia máxima (0,93 %) · todos exactos |
| **§5.3.3 por categoría** | Los 8 valores de precisión, recall y F1 coinciden con `metricas_por_clase_*.csv` · la brecha de 0,1863 puntos es exacta |
| **§5.5.3 LIME** | Los 9 pesos medios de los tres grupos coinciden dígito a dígito (0,0461 · 0,0430 · 0,0414 / 0,0306 · 0,0285 · 0,0256 / 0,0243 · 0,0243 · 0,0225) |
| **Tabla 5.19 y Figura 5.10** | Los 15 nombres del top-5 y las tres correlaciones de Spearman (0,9625 · 0,8614 · 0,9243) coinciden. **W de Kendall = 0,9367** confirmada en la salida del NB04 |
| **Tabla 5.20** | Los **12 conteos** de convergencias, parciales y divergencias se reproducen exactamente contando las diez primeras filas de `tabla_convergencias_CatBoost.csv` |
| **Tabla 5.1** | La cadena de exclusiones es aritméticamente consistente: 32.272 + 21.036 + 4.707 + 861 = 58.876, y 489.771 − 58.876 = 430.895 |
| **Tabla 5.2** | Los 21 valores (4 porcentajes, n, razón y países por conjunto) coinciden con los tres `data/processed/*.parquet` |
| **Tabla 5.3** | Las 28 correlaciones individuales y las 4 de nivel país-año coinciden con `eda_correlaciones_features.csv` |
| **§5.1 missingness** | 12,84 % global y 13,8 / 6,3 / 5,3 por conjunto · exactos. Confianza en televisión 50,69 % y conocimiento de corrupción 50,66 % en prueba (ambas > 50 %), situación económica 0,33 % (< 0,4 %) |
| **§5.2 KS val–test** | Estadístico **0,0549** con p = 0,0000 · confirmado en la salida del NB02 (era el error crítico de la revisión anterior; **está corregido**) |
| **§5.2 ρ(año, target)** | 0,0217 con n = 431.756 · coincide con `eda_correlacion_año_target.csv` |
| **§4.9.3 pesos de clase** | 2,6524 · 0,9245 · 0,5756 · 1,2439 · los cuatro exactos sobre los 378.592 registros de entrenamiento |
| **§5.3.1 tiempos** | 13 h 22 min (real 13 h 21 min 37 s) · el NB02 concentra el **99,6 %** exacto · los otros cinco suman 3,2 min («menos de cuatro minutos») |
| **§4.6 multicolinealidad V-Dem** | 19 variables, 46 pares > 0,85, máximo 0,9901 sobre 540 país-año; 1 par y 0,8641 al conservar cuatro; 0,8552 sobre los 360 de entrenamiento · todos exactos y con la base ya declarada |
| **Figura 4.4** | n = 360 pares país-año exactos |
| **§2.2.1 cobertura** | 8 países en 1995, 17 desde 1996, 18 desde 2004 · verificado |
| **§5.1 poliarquía** | Venezuela 0,196 en 2024 y promedio 0,634 de los diecisiete restantes; Nicaragua 0,215 en 2020 y promedio 0,633 · **corregidos y ahora exactos** |
| **§5.1 Venezuela 2013–2017** | 21,9 % · 39,7 % · 49,2 % · 49,9 % · los cuatro exactos (fallan solo 2018 y 2024) |
| **§4.6 disponibilidad** | A_003_021, D_001_061 y D_001_131 con **0,00 %** de dato válido en 2023–2024 · verificado. Coberturas del 80 % y 88 % en entrenamiento (reales 78,4 % y 88,6 %) · D_001_061 sin dato desde 2016 · correcto |
| **§4.6 X_004 correspondencia** | 137 de las 282 categorías de prueba (48,6 %) no aparecen en entrenamiento · la afirmación cualitativa se sostiene |
| **§2.4 y §4.10** | El conteo de ocho métricas es coherente entre la Tabla 2.1, el texto de §2.4 y la enumeración de §4.10 · **corregido** |

---

# Checklist

## Antes de imprimir (siete)

- [ ] **1.** §1.1: resolver la cita `[?]` → referencia [51] y recompilar con BibTeX
- [ ] **2.** §1.1: reescribir la atribución a Ergun et al. (usa logit ordenado multinivel, no ML)
- [ ] **3.** §5.1: Venezuela 2018 → 61,3 %, y retirar «sin observaciones en Para nada ni No muy satisfecho»
- [ ] **4.** §5.1: Venezuela 2024 → 46,5 %
- [ ] **5.** §4.6: recalcular las seis correlaciones con NS/NR como ausente y sacar `C_003_003_011` del grupo de señal baja
- [ ] **6.** §4.6: X_004 → 730 categorías y ρ = −0,0473; **eliminar «p = 0,132» y «no significativa»**
- [ ] **7.** Capítulo 6: añadir el apartado de contraste de H1, H2 y H3

## Recomendables (cuatro)

- [ ] **8.** Tabla 5.11: igualar la estrategia de ambas columnas o declarar la diferencia al pie
- [ ] **9.** §4.6: unificar la notación en ρ y usar los valores reproducibles de X_008 y H_001_011
- [ ] **10.** §4.6: X_008 → cinco olas, no seis
- [ ] **11.** §5.1: KS de los noventa → «entre 0,059 y 0,085»

## Cosmética (tres)

- [ ] **12.** Abstract: `0.5386` con punto
- [ ] **13.** §5.2: `n = 431.756` con punto de millar
- [ ] **14.** Tabla 5.16: tres IC superiores en la 4.ª decimal

## Revisión final

- [ ] Recompilar dos veces y comprobar que **no queda ningún `[?]` ni `??`** en el PDF
- [ ] Verificar que las cifras corregidas de §5.1 y §4.6 no se repiten sin corregir en otro capítulo
