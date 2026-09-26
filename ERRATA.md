# Errata — Inversión de las etiquetas de la variable objetivo

**Fecha de detección:** septiembre de 2026
**Alcance:** rotulación de las cuatro categorías de `A_003_031` (satisfacción con la democracia)
**Estado de los resultados numéricos:** sin cambios

---

## 1. Qué estaba invertido

La escala original del Latinobarómetro para `A_003_031` es **descendente**: el código 1
corresponde al nivel máximo de satisfacción y el 4 al mínimo.

| Código original | Etiqueta real (codebook) | `target` tras el desplazamiento |
|---|---|---|
| 1 | Muy satisfecho | 0 |
| 2 | Más bien satisfecho | 1 |
| 3 | No muy satisfecho | 2 |
| 4 | Nada satisfecho | 3 |

El proyecto y el documento de tesis declararon el orden **ascendente**, es decir, el
inverso:

| Clase | Declarado (incorrecto) | Correcto |
|---|---|---|
| 0 | Para nada satisfecho | **Muy satisfecho** |
| 1 | No muy satisfecho | **Más bien satisfecho** |
| 2 | Más bien satisfecho | **No muy satisfecho** |
| 3 | Muy satisfecho | **Nada satisfecho** |

El error está en dos lugares y es el mismo en ambos:

- `utils/config.py`, diccionario `ETIQUETAS` (línea 682).
- Tesis, sección 4.3.1, donde se describe la codificación de la variable dependiente.

**Origen probable.** En la ola 2013 el código 4 se etiqueta literalmente
«Para nada satisfecho». Al construir el diccionario se tomó esa etiqueta —la del
extremo inferior de la escala— y se asignó a la clase 0, invirtiendo el orden
completo.

**La recodificación en sí es correcta.** El NB02 aplica un desplazamiento puro que
preserva el orden, sin inversión:

```python
df[COL_TARGET] = df["A_003_031"].map({1.0:0, 2.0:1, 3.0:2, 4.0:3})
```

Por tanto el problema es exclusivamente de rotulación: el modelo aprendió sobre una
escala ordinal correcta y coherente, y lo único equivocado es el nombre que se le dio
a cada extremo.

---

## 2. Cómo se verificó

Cuatro comprobaciones independientes, todas concordantes.

### 2.1 Etiquetas de valor de los archivos Stata (evidencia directa)

Se leyeron los metadatos de los `.dta` originales con `pyreadstat`
(`metadataonly=True`) en **las 24 olas** del período. Veintitrés conservan etiquetas
de valor legibles y las veintitrés siguen el mismo orden descendente, sin una sola
excepción:

```
1 = Muy satisfecho        2 = Mas bien satisfecho
3 = No muy satisfecho     4 = Nada satisfecho
```

La única ola sin etiquetas de valor en el archivo Stata es 2011, de modo que no
aporta ni contradice.

El texto del código 4 varía entre olas: en 1995–1998, 2000, 2003–2010, 2015–2018,
2020, 2023 y 2024 aparece como «Nada satisfecho», y solo en 2001, 2002 y 2013 como
«Para nada satisfecho». Esa segunda variante es la que se asignó por error a la
clase 0.

### 2.2 Proporción de satisfechos frente a la fuente citada

La sección 1.1 de la tesis cita el informe de Latinobarómetro 2024: la proporción de
ciudadanos satisfechos con la democracia «oscila entre el 20 % y el 50 %».

| Conjunto | Con las etiquetas declaradas | Con el codebook |
|---|---|---|
| Entrenamiento 1995–2018 | 63,5 % satisfechos | **36,5 %** |
| Prueba 2023–2024 | 67,6 % satisfechos | **32,4 %** |

Solo la lectura del codebook cae dentro de la banda que la propia tesis cita como
referencia.

### 2.3 Comparación con Rosa et al. (2023)

La tabla 3.1 registra que Rosa et al. trabajan con 615 personas insatisfechas y 257
satisfechas en Brasil 2020, es decir un 70,5 % de insatisfacción. Con el codebook, el
conjunto de prueba de este trabajo arroja un 67,6 % de insatisfacción: valores
próximos. Con las etiquetas declaradas arrojaría un 67,6 % de *satisfacción*, lo que
contradiría el antecedente.

### 2.4 Distribución de Venezuela (evidencia decisiva)

Se recuperó la distribución venezolana por código original desde
`data/base/latinobarometro.csv`:

| Año | 1 Muy | 2 Más bien | 3 No muy | **4 Nada** |
|---|---|---|---|---|
| 2013 | 20,1 | 22,8 | 35,2 | **21,9** |
| 2015 | 10,0 | 19,9 | 30,4 | **39,7** |
| 2016 | 8,6 | 15,3 | 26,9 | **49,2** |
| 2017 | 9,0 | 13,0 | 28,0 | **49,9** |
| 2018 | 4,8 | 7,7 | 26,2 | **61,3** |

La serie 21,9 → 39,7 → 49,2 → 49,9 que la sección 5.1 describe como «la proporción
de personas que declara la máxima satisfacción» corresponde al código 4, es decir a
**«Nada satisfecho»**. El conteo de la ola 2018 confirma la correspondencia de forma
exacta: `{1: 57, 2: 91, 3: 310, 4: 726}`, y 726/1.184 = 61,3 %, el valor que la
sección 5.1 atribuye a «Muy satisfecho».

---

## 3. Qué NO cambia

### 3.1 Todas las métricas de rendimiento

Las métricas empleadas dependen de la distancia ordinal |i − j| o del acierto exacto,
de modo que son **invariantes** a la inversión del orden de las categorías. Se verificó
numéricamente invirtiendo un par (y_true, y_pred) de 20.000 observaciones:

| Métrica | Original | Reetiquetado | |
|---|---|---|---|
| Kappa cuadrático | 0,797276 | 0,797276 | invariante |
| Kappa lineal | 0,595074 | 0,595074 | invariante |
| MAE ordinal | 0,505500 | 0,505500 | invariante |
| Accuracy | 0,494500 | 0,494500 | invariante |
| F1 macro | 0,495010 | 0,495010 | invariante |

En consecuencia se mantienen sin alteración:

- κw = 0,5386 y MAE ordinal = 0,5688 de la configuración principal.
- Las tablas 5.4 a 5.17 completas (validación, prueba, estrategia seleccionada,
  métricas complementarias, bootstrap, contraste de H2, efecto del balanceo,
  hiperparámetros, E2, pliegues temporales y subregiones).
- Los intervalos de confianza y los errores estándar bootstrap.
- La selección de CatBoost [SMOTE-NC] como configuración principal.
- Los veredictos de H1 y H2.
- Las métricas ponderadas por `X_020`.

### 3.2 Los resultados de explicabilidad

- **Importancias SHAP.** Se calculan sobre el valor absoluto promediado entre las
  cuatro clases, `|SHAP|_j = (1/N)Σ_i (1/4)Σ_k |φ_jk(x_i)|`. Reordenar las clases
  permuta los sumandos sin alterar la suma, de modo que tanto los valores como el
  ranking por variable y la agregación por bloque son idénticos.
- **Tabla 5.18** (importancia por bloque e intervalos) y **Tabla 5.19** (veinte
  variables principales, intervalos y estabilidad de posiciones): sin cambios.
- **Tabla 5.20** (micro/macro), **Tabla 5.22** (top 5 por modelo) y **Figura 5.10**
  (concordancia entre modelos de gradient boosting): sin cambios.
- **Tablas 5.23, 5.24 y 5.25** (variación por subregión y concordancia de rankings):
  sin cambios.
- **Veredictos de H3, H4 y H5:** sin cambios, porque los tres se contrastan sobre
  magnitudes de |SHAP| y no sobre la dirección del target.
- **Curvas ALE (Figuras 5.5 a 5.9 y 2.7).** Las curvas en sí no cambian, porque el
  efecto local acumulado se calcula sobre una salida que la inversión de etiquetas no
  altera. Su lectura en la sección 5.5.2 está redactada en términos de «cambio en la
  salida del modelo», sin afirmaciones direccionales sobre el nivel de satisfacción,
  de modo que el texto tampoco requiere corrección. Lo que sí debe precisarse es
  **sobre qué salida** se calculan: véase la sección 4.4.

### 3.3 Las correlaciones entre variables del Latinobarómetro

Las 24 filas de variables LB de la Tabla 5.3 y la narrativa de la sección 5.5 se
mantienen válidas, pero el argumento requiere precisión: **no es cierto que todas las
escalas conserven la convención descendente del Latinobarómetro**. Cinco variables
quedan ascendentes después del preprocesamiento, por transformaciones aplicadas de
forma deliberada en el NB02.

| Variable | Transformación | Dirección resultante |
|---|---|---|
| `D_001_021` | `4 − x` (olas ≤ 2000) y `6 − x` (olas ≥ 2001) | ascendente: alto = mejor economía |
| `D_001_041` | ídem | ascendente: alto = mejor expectativa del país |
| `D_001_091` | ídem | ascendente: alto = mejor expectativa personal |
| `B_006_061` | recodificación binaria `{1→1, 2→0}` | ascendente: 1 = aprueba al Gobierno |
| `B_001_101` | recodificación binaria `{1→1, 2→0}` | ascendente: 1 = gobernado **para los poderosos** |

Las 19 variables LB restantes sí conservan la convención descendente. Se verificó en
la ola 2023 que la confianza institucional va `1 = Mucha … 4 = Ninguna` y que el
apoyo a la democracia va `1 = La democracia es preferible …`.

La conclusión de que **todas** las correlaciones de la Tabla 5.3 admiten una lectura
intuitiva se sostiene, y se comprobó recalculándolas sobre `train.parquet`: los
valores reproducen la tabla al cuarto decimal. El razonamiento es de dos tipos según
la dirección de cada variable:

- **Variables descendentes** (target y predictor invertidos en el mismo sentido): el
  signo conserva su interpretación. Ejemplo: ρ(Confianza en el Gobierno, target) =
  +0,3730 significa «más desconfianza va con más insatisfacción» bajo ambas lecturas.
- **Variables ascendentes** (solo el target invertido): el signo negativo es el
  intuitivo. Ejemplo: ρ(`B_006_061`, target) = −0,3273 significa «más aprobación va
  con más satisfacción».

El caso que exige más cuidado es `B_001_101`, con ρ = +0,3014. El valor alto de esa
variable no es «país para todos» sino **«gobernado por unos cuantos grupos poderosos
en su propio beneficio»**, porque la codificación original del Latinobarómetro es
`1 = Grupos poderosos en su propio beneficio` y `2 = Para el bien de todo el pueblo`
—verificado en las olas 2018, 2020 y 2023—. Con esa dirección, el signo positivo
significa «gobernar para los poderosos va con más insatisfacción», que es el
resultado intuitivo. El comentario del código sugiere lo contrario y también está
invertido: véase la sección 5.

### 3.4 Los pipelines entrenados

Los 16 archivos `models/pipeline_*.pkl` almacenan el diccionario invertido en el campo
de metadatos `artefacto["etiquetas_target"]`. Se verificó que **ningún módulo ni
notebook lee ese campo**: es metadato inerte que no interviene en ningún cálculo ni
en ningún rótulo. Por tanto no es necesario reentrenar.

---

## 4. Qué SÍ cambia

### 4.1 El caso de Venezuela (sección 4.5.1 y sección 5.1) — cambio sustantivo

La tesis describe un aumento de la satisfacción declarada mientras el índice de
poliarquía se desploma, y atribuye esa divergencia a autocensura o expresión
estratégica, con apoyo en la literatura sobre autocracias informacionales (Guriev y
Treisman).

Con las etiquetas correctas el patrón es el inverso: la proporción que declara
**«Nada satisfecho»** sube del 21,9 % (2013) al 61,3 % (2018) mientras la poliarquía
cae de 0,327 a 0,196. Es el patrón esperado y no constituye una anomalía, de modo que
**el argumento de expresión estratégica no se sigue de esta evidencia y debe
retirarse**.

Asimismo, los «148 registros (12,5 %) en las dos categorías de insatisfacción,
distribuidas entre 57 y 91» son en realidad los códigos 1 y 2, es decir las dos
categorías de **satisfacción**.

**La decisión de excluir Venezuela sigue siendo defendible**, pero sobre dos
fundamentos distintos que permanecen intactos:

1. La divergencia distributiva frente al resto de la región, medida por el estadístico
   de Kolmogorov–Smirnov de 0,2552 en 2017 (p < 0,001), que es invariante a la
   etiqueta.
2. La coincidencia temporal del quiebre con la instalación de la Asamblea Nacional
   Constituyente en agosto de 2017, documentada por la Secretaría General de la OEA.

### 4.2 Los signos de los indicadores de V-Dem (sección 5.2) — cambio sustantivo

Los índices de V-Dem están orientados de forma inequívoca: un valor más alto indica
más democracia. Como no están invertidos, al corregir el target su lectura cambia de
signo.

Valores de `results/tables/eda_correlaciones_features.csv`, Spearman a nivel de
registro sobre el conjunto de entrenamiento:

| Indicador | ρ individual | ρ país-año | Lectura correcta |
|---|---|---|---|
| `v2x_egal` | −0,1703 | −0,4652 | más componente igualitario ⇒ **más** satisfacción |
| `v2xcl_rol` | −0,1320 | −0,3527 | más igualdad ante la ley ⇒ **más** satisfacción |
| `v2x_polyarchy` | −0,0854 | −0,2034 | más poliarquía ⇒ **más** satisfacción |
| `v2x_corr` | +0,0821 | +0,1888 | más corrupción ⇒ **menos** satisfacción |

La columna de ρ individual que reporta la sección 5.2 difiere de la del artefacto en
la segunda cifra decimal, presumiblemente por el subconjunto sobre el que se calculó;
no he podido reverificarlo porque el documento no forma parte del repositorio. Eso no
afecta a la corrección, que es de **signo**: sobre el conjunto de entrenamiento y
sobre el total modelado (430.895 registros) los cuatro signos son los mismos
—−0,1610, −0,1225, −0,0843 y +0,0849 respectivamente en el total—, de modo que la
lectura corregida no depende del subconjunto.

La sección 5.2 presenta estos cuatro signos como «asociación negativa de los
indicadores de calidad democrática con la satisfacción» y los ofrece como coherentes
con la tesis de los ciudadanos críticos de Norris. Con las etiquetas correctas la
asociación es **positiva** y el resultado es el intuitivo: más calidad democrática va
con más satisfacción, y más corrupción con menos.

**Consecuencia para la sección 5.6 y el objetivo específico 5.** La convergencia
parcial con Norris no puede apoyarse en los signos de los índices contextuales. Sigue
apoyada, en cambio, en el hecho de que el bloque de percepción política es el dominante
de la agregación SHAP (0,4818) y en que el contexto democrático queda en cuarto lugar
con tres de sus cuatro indicadores por debajo de 0,008, que es la observación que la
sección 5.6 ya reporta y que no depende de la dirección del target.

**Punto de código afectado.** El notebook 02 imprimía este bloque como conclusión de
la celda de correlaciones:

```text
HALLAZGO V-DEM: correlaciones negativas con calidad democrática
  Interpretación: ciudadanos más exigentes en democracias más avanzadas
  (Norris, 2011 — Critical Citizens)
  → Nuevo H5 de la tesis
```

Estaba mal por dos motivos independientes. El signo se leía al revés, y el puntero a
H5 era obsoleto: en la versión final H5 es la estabilidad de los rankings SHAP entre
subregiones, que se contrasta en el apartado 7 del NB05 y no tiene relación con estas
correlaciones. El bloque se ha sustituido por una lectura que deriva la dirección de
cada indicador del propio `df_corr` y comprueba contra la dirección esperable,
declarada antes del resultado (`r < 0` para los índices de calidad democrática,
`r > 0` para el índice de corrupción, cuya orientación es inversa). Sobre los datos
actuales los cuatro la cumplen.

### 4.3 El grupo C de LIME (sección 4.11.3 y sección 5.5.3) — cambio sustantivo

El grupo C se define en la tesis como «casos institucionalmente discordantes» y se
selecciona con este criterio, que opera sobre la **clase predicha**:

```python
((df_aux["v2x_polyarchy"] > 0.6) & (df_aux["y_pred"] <= 1)) |
((df_aux["v2x_polyarchy"] < 0.3) & (df_aux["y_pred"] >= 2))
```

La condición `y_pred <= 1` se concibió como «satisfacción predicha baja». Con las
etiquetas correctas, `{0, 1}` son **Muy satisfecho** y **Más bien satisfecho**, es
decir satisfacción predicha **alta**. Por tanto la primera rama selecciona casos de
poliarquía alta con satisfacción predicha alta, y la segunda, de poliarquía baja con
insatisfacción predicha: en ambos casos, **concordancia** institucional y no
discordancia.

Se verificó sobre `results/tables/lime_CatBoost_smotenc.csv`: los 50 casos del grupo
tienen `clase_pred` ∈ {0: 15, 1: 35}, es decir **los 50 son satisfechos** según el
codebook.

Hay una precisión adicional. La segunda rama del criterio está **vacía por
construcción**: la poliarquía mínima del conjunto de prueba es 0,335, de modo que
ningún registro cumple `< 0,3` —consecuencia directa de haber excluido a Venezuela y
Nicaragua, los dos países con índices por debajo de ese umbral—. Los 50 casos
provienen íntegramente de la primera rama, así que el grupo no es bidireccional en la
práctica aunque el criterio lo sea en el código.

**Qué no cambia.** Las cifras de LIME se mantienen: los pesos medios del grupo
(0,0243 para confianza en el Gobierno, 0,0243 para distribución del ingreso y 0,0225
para situación económica del país) y el hallazgo de que las tres primeras variables
son las mismas y en el mismo orden en los tres grupos.

**Qué cambia.** El nombre del grupo y lo que demuestra. No ilustra cómo explica el
modelo casos en tensión entre contexto institucional y percepción individual, sino
cómo los explica en casos donde ambos coinciden. El grupo C debe describirse como
«casos institucionalmente concordantes: poliarquía alta con satisfacción predicha
alta».

**Qué se cambió en el repositorio.** El renombrado se aplicó de forma completa, porque
dejar el identificador antiguo junto a la descripción nueva equivalía a mantener la
afirmación errónea en el código:

| Antes | Ahora |
|---|---|
| `PARAMETERS["CASOS_LIME_DISCORDANTES"]` | `PARAMETERS["CASOS_LIME_CONCORDANTES"]` (los dos perfiles de `config.py`) |
| `n_discordantes`, `idx_disc`, `mask_disc`, `cands_disc`, `idxs_disc`, `df_lime_disc` | `n_concordantes`, `idx_conc`, `mask_conc`, `cands_conc`, `idxs_conc`, `df_lime_conc` |
| `Casos discordantes inst.`, `Grupo C: discordantes inst.`, `Discordante` | equivalentes con «concordantes» |
| valor `"discordante"` de la columna `grupo` | `"concordante"` |

La última fila es la única que altera un artefacto de datos: al regenerar
`results/tables/lime_{modelo}_{estrategia}.csv`, la columna `grupo` contendrá
`concordante` donde el archivo actual y la tabla de la tesis dicen `discordante`. Son
las mismas 50 filas de casos con los mismos pesos; solo cambia la cadena.

El markdown de la sección 8 del NB04 también estaba desactualizado por un motivo
ajeno a las etiquetas: anunciaba «dos criterios» y «100 casos discordantes» cuando el
código usa tres grupos de 100, 50 y 50. Quedó reescrito con los tres.

### 4.4 La salida sobre la que se calculan las curvas ALE (sección 4.11.2 y sección 5.5.2)

Ni la sección 4.11.2, que escribe la fórmula con un `∂f/∂z_j` genérico, ni el rótulo
del eje vertical de las figuras («Cambio en la predicción del modelo») declaran sobre
qué salida del modelo se calcula el efecto acumulado. El NB04 lo fija así:

```python
def _pred_fn_ale(X):
    ...
    return clf_plot.predict_proba(df)
...
# Clase 2 (Más bien satisfecho) como referencia
ale_vals = ale_result.ale_values[idx_var_ale][:, 2]
```

El predictor devuelve la matriz de probabilidades por clase y `[:, 2]` selecciona la
columna 2. Las cinco curvas ALE miden, por tanto, el efecto local acumulado sobre
**la probabilidad predicha de la clase 2**, que con las etiquetas correctas es
**«No muy satisfecho»** —la categoría modal, con el 42,0 % del conjunto de prueba—.
No es la clase esperada ni un índice agregado de satisfacción. El comentario del
código la nombra «Más bien satisfecho», que es la etiqueta invertida.

**Consecuencia interpretativa.** Un tramo ascendente de la curva no significa «menos
satisfacción» en general: significa que aumenta la probabilidad de esa categoría
intermedia concreta. Esa precisión explica, además, la forma de U invertida que la
sección 5.5.2 describe como «patrón no monótono» sin justificarla. Que la
probabilidad de una categoría **intermedia** alcance su máximo en valores intermedios
del predictor y caiga en ambos extremos es exactamente lo esperable, porque en los
extremos la masa se desplaza a las categorías vecinas. En la Figura 5.9, el descenso
a −0,029 en «Ninguna confianza en el Gobierno» refleja que esos casos se desplazan a
«Nada satisfecho», no que su insatisfacción disminuya. Leída así, la no monotonía
deja de ser una anomalía y pasa a ser evidencia de que el modelo respeta la
estructura ordinal del target.

### 4.5 Rótulos que deben intercambiarse

Todos los siguientes pasajes mantienen sus cifras y solo requieren permutar el nombre
de las categorías:

| Ubicación | Corrección |
|---|---|
| Sección 1.2 | La clase mayoritaria (≈43 %) es **No muy satisfecho**, no *Más bien satisfecho* |
| Sección 4.3.1 | Las cuatro categorías en orden descendente: 1 = Muy satisfecho … 4 = Nada satisfecho |
| Tabla 5.2 y Figura 5.1 | Encabezados de clase permutados; las proporciones no cambian |
| Sección 5.3.2 y Tabla 5.14 | El conjunto de prueba binario es 32,4 % **satisfechos** y 67,6 % **insatisfechos** |
| Sección 5.3.3 | En la clase 0 (**Muy satisfecho**) la precisión 0,4489 supera al recall 0,3500; la categoría mejor recuperada (F1 = 0,5796) es **No muy satisfecho**; el 29,1 % de los **muy satisfechos** se clasifica como **No muy satisfecho** |
| Tabla 5.21 | «Sobreestimación» y «subestimación» describen el valor del código, no el nivel de satisfacción: predecir un código mayor equivale a predecir **menos** satisfacción. El caso más frecuente (1.154, 3,29 %) es clasificar como *No muy satisfecho* a quien declaró *Muy satisfecho* |
| Sección 6.3, limitación 8 | Los F1 de 0,3933 y 0,4003 corresponden a las dos clases de **satisfacción**, no de insatisfacción |
| Sección 4.9.4 y Tabla 4.14 | La agrupación binaria conserva el umbral, pero `{0,1}` = **Satisfecho** y `{2,3}` = **Insatisfecho** |
| Sección 2.4.4 | El ejemplo de error grave sigue siendo válido como ilustración; los extremos de la escala se nombran al revés |

### 4.6 La formulación binaria del Experimento 2

El umbral de corte no cambia: la partición sigue siendo entre las dos categorías
superiores y las dos inferiores de la escala, que es la misma que emplea Rosa et al.
Lo que se invierte es el nombre de cada grupo: `{0,1}` agrupa a los satisfechos y
`{2,3}` a los insatisfechos. Todas las métricas de la Tabla 5.14 se mantienen; el F1
macro de 0,7377 y el AUROC de 0,8261 son simétricos respecto del intercambio de
nombres de clase.

Hay sin embargo una consecuencia que no es de nomenclatura: la **clase positiva** de
la variante binaria es `1`, y `y_bin = (y >= 2)`, de modo que la clase positiva es la
de los **insatisfechos**. El AUROC de 0,8261 mide por tanto la capacidad de ordenar
correctamente a los insatisfechos, no a los satisfechos. Bajo la lectura invertida se
habría descrito al revés.

### 4.7 La identidad de la clase minoritaria (Tabla 5.11 y OE3) — cambio sustantivo

`CLASE_MINORITARIA = 0` es numéricamente correcto: la clase 0 es en efecto la de menor
prevalencia en los tres conjuntos —9,4 % en entrenamiento, 9,3 % en validación y
11,3 % en prueba—, y también lo era bajo la lectura invertida. Lo que cambia es **de
quién** se trata: la categoría minoritaria es la de los **más satisfechos**
(«Muy satisfecho»), no la de los más insatisfechos.

Esto reencuadra todo el apartado 7.1 del NB03 y el OE3. Las cifras no se mueven: las
nueve mejoras distinguibles del F1 de la clase 0 siguen yendo de **+0,0823**
(XGBoost [smotenc]) a **+0,1826** (OLO [pesos_clase]), y el único caso no
distinguible sigue siendo LightGBM [smotenc] (−0,0134, IC 95 % [−0,0422; 0,0220]).
Lo que cambia es la frase que las acompaña: el balanceo mejora la detección del
extremo **satisfecho** de la escala, que es el minoritario en Latinoamérica, y no la
del extremo insatisfecho. La conclusión metodológica del OE3 —que la ponderación por
frecuencia inversa domina al sobremuestreo sintético— es independiente de qué
categoría ocupe el extremo minoritario y se mantiene sin cambios.

La misma corrección se aplica a la limitación 8 de la sección 6.3, ya recogida en la
tabla del apartado 4.5: los F1 de 0,3933 y 0,4003 son los de las dos categorías de
satisfacción.

---

## 5. Puntos del código afectados

| Ubicación | Naturaleza | Efecto |
|---|---|---|
| `utils/config.py`, `ETIQUETAS` | diccionario invertido | raíz del problema; alimenta rótulos de NB02, NB03 y NB04 |
| `utils/plots.py`, `plot_matrices_confusion()` | lista **literal** invertida: `["Nada\nsat.", "No muy\nsat.", "Más bien\nsat.", "Muy\nsat."]` | no se corrige al arreglar `config.py`; requiere edición propia |
| `notebooks/04_explicabilidad_xai.ipynb`, `etiq_c` | lista **literal** invertida: `["Para nada\nsat.", "No muy\nsat.", "Mas bien\nsat.", "Muy\nsat."]` | ídem: requiere edición propia |
| `notebooks/02_preprocesamiento_entrenamiento.ipynb`, `etiq_cortas` (figura de desbalance) | lista **literal** invertida: `["Para\nnada", "No\nmuy", "Más\nbien", "Muy"]` | ídem: requiere edición propia |
| `notebooks/02_preprocesamiento_entrenamiento.ipynb`, `neta` (histograma por subperiodo) | el cálculo `(clases 2+3) − (clases 0+1)` es la **insatisfacción** neta, rotulada solo como «neta» | el signo se leía al revés; se renombra a `insat_neta` |
| `notebooks/04_explicabilidad_xai.ipynb` | `class_names=list(ETIQUETAS.values())` pasado a LIME y `target_names` al reporte por clase | los nombres de clase de las explicaciones LIME salen invertidos |
| `notebooks/04_explicabilidad_xai.ipynb`, celda del ALE | comentario `# Clase 2 (Más bien satisfecho) como referencia` | la clase 2 es «No muy satisfecho»; el cálculo es correcto, el comentario no |
| `notebooks/02_preprocesamiento_entrenamiento.ipynb`, recodificaciones binarias | comentario `# País para todos=1, para poderosos=0` | **invertido**: el original es `1 = Grupos poderosos`, de modo que el valor 1 es «para los poderosos» |
| `notebooks/02_preprocesamiento_entrenamiento.ipynb`, celdas del E2 | markdown y comentario: `{0,1} → Insatisfecho, {2,3} → Satisfecho` | **invertido**: los nombres de los dos grupos binarios |
| `notebooks/03_evaluacion_comparativa.ipynb`, apartado 7.1 | prosa: la clase 0 descrita como «Para nada satisfecho» | el cálculo es correcto; el texto identifica mal a la categoría minoritaria |
| `utils/metrics.py`, `_f1_clase_0()` | docstring: «la minoritaria del target» sin nombrarla | correcto pero ambiguo; se presta a leer «minoritaria» como «insatisfecha» |
| `notebooks/02_preprocesamiento_entrenamiento.ipynb`, cierre de la celda de correlaciones | bloque `HALLAZGO V-DEM` que leía el signo al revés y apuntaba a un H5 inexistente | **invertido**: era la única interpretación de signo escrita en el código (apartado 4.2) |
| `utils/config.py` y `notebooks/04_explicabilidad_xai.ipynb`, grupo C de LIME | clave, identificadores y rótulos con el nombre `discordante` | **invertido**: el grupo es concordante (apartado 4.3); renombrado por completo |
| `notebooks/04_explicabilidad_xai.ipynb`, `tipo_error` | valores `Sobreestimacion` / `Subestimacion` | califican el **código** de clase, no el nivel de satisfacción; se mantienen los valores y se documenta el sentido |
| `models/pipeline_*.pkl`, campo `etiquetas_target` | metadato | inerte: ningún módulo lo lee; no exige reentrenar |

Hay **cuatro** puntos de rotulación que deben corregirse a mano para que las figuras
salgan bien: el diccionario de `config.py` y **tres** listas literales que no lo
consultan (la de `plots.py` y una en cada uno de los notebooks 02 y 04). Corregir
solo el diccionario deja invertidas la figura de matrices de confusión comparadas, la
de errores graves y la de desbalance de clases.

En cambio, las funciones `metricas_por_clase()` y `matriz_confusion_df()` de
`utils/metrics.py`, y `plot_matriz_confusion_modelo()` y `plot_metricas_por_clase()`
de `utils/plots.py`, sí derivan de `ETIQUETAS`, de modo que se corrigen solas al
arreglar el diccionario.

Los comentarios invertidos —el de la clase de referencia del ALE, el de `B_001_101` y
los de la variante binaria— no afectan ningún cálculo, pero inducen a error a quien
lea el código para derivar la dirección de un signo o el nombre de una clase.

**Estado del repositorio.** Todos los puntos de esta tabla están ya corregidos en el
código fuente: `ETIQUETAS`, las dos listas literales, los comentarios, la prosa del
NB03, el docstring de `metrics.py` y el `README.md`. El único artefacto que conserva
la rotulación antigua es el documento de tesis, que está cerrado; de ahí la existencia
de este archivo. Las tablas y figuras guardadas en `results/` se regeneran con las
etiquetas correctas al volver a ejecutar los notebooks, y sus **valores numéricos no
cambian** por las razones del apartado 3.

---

## 6. Resumen

El error es de rotulación y no de cálculo. Ninguna métrica, ningún intervalo de
confianza, ninguna importancia SHAP y ninguno de los cinco veredictos de hipótesis se
modifica. Lo que cambia es el nombre de las categorías en las tablas y figuras, y dos
interpretaciones sustantivas: el argumento de expresión estratégica en el caso
venezolano, que debe retirarse, y el signo con que se leen las asociaciones de los
indicadores de V-Dem, que pasa a ser el intuitivo y deja de respaldar por esa vía la
tesis de los ciudadanos críticos.
