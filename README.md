# DemocraticSatisfactionLatam
Satisfacción de la democracia en Latinoamérica

### Estructura de directorios esperada

```
proyecto/
├── data/
│   ├── F00018833-Latinobarometro_Serie_de_Tiempo_1995_2024.xlsx   ← diccionario de mapeo
│   ├── raw_latinobarometro/
│   │   ├── Latinobarometro_1995.dta
│   │   ├── Latinobarometro_1996.dta
│   │   │   ... (25 archivos)
│   │   └── Latinobarometro_2024.dta
│   ├── raw_v-dem/
│   │   └── V-Dem-CY-Core-v16.csv
│   └── variables/
│       └── latinobarometro_variables.csv
└── notebooks/
    └── 01_carga_datos.ipynb   ← este archivo
```