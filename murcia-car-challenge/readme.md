# My solution to [murcia car challenge](https://www.kaggle.com/c/murcia-car-challenge)


## Data
- Datos del **coche**: `Marca`, `Modelo`, `Año`, `Combust`, `Cv`, `Puertas`:
  - [x] Sacar modelo base (GOLF, IBIZA, ...)
  - [ ] Sacar submodelo (GOLF GTI, ...)
  - [ ] Sacar precio nuevo de fábrica
- Datos del uso **coche**: `Kms`:
  - [ ] ITV pasada o no (Información no disponible)
- Datos del **anuncio**: `Provincia`, `Localidad`, `Vendedor`, `Tiempo`:
  - [x] Sacar Latitud y Longitud de la Provincia
  - [ ] Sacar Latitud y Longitud de la Localidad
  - [ ] Sacar Renta per Cápita de la Provincia
  - [x] Sacar variable numérica días de Tiempo
  - [x] Sacar anuncio destacado (sí/no) de Tiempo


## Models
- GBM
  - [x] LightGBM
  - [ ] Catboost with/without [text column](https://upura.hatenablog.com/entry/2020/03/03/195929)
- NN
  - [x] Simple with Fast.ai
  - [ ] Simple with Keras
  - [ ] With custom embedding by car. by seller
- FM
- Ensemble

## Evaluación
- Ver mayores fallos y entender pq (tipo de coche?)
- Ver histograma de precios reales % 1000
