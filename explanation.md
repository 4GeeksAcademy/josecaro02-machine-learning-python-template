# Cómo se resolvió el proyecto, paso a paso

Este documento cuenta **qué se hizo, en qué orden, por qué y cómo se ejecutó**, para que puedas explicarlo con tus palabras. Va de lo general a lo concreto.

---

## 1. La pregunta que hay que responder

Harbor House Café tiene un promedio de **4.5 estrellas**. La account manager duda: *¿lo que la gente escribió suena tan positivo como ese 4.5?*

Tenemos 500 reseñas en `data/raw/reviews.csv` con tres columnas: `review_id`, `rating` (estrellas humanas) y `review_text` (el texto).

**Idea central de la solución:** hacer que un modelo de IA lea solo el texto, estime cuántas estrellas le daría, y comparar eso con las estrellas reales. Si coinciden, el texto confirma el 4.5. Si no, hay que explicar por qué.

---

## 2. Por qué se usa un modelo ya entrenado

Entrenar un modelo desde cero necesita miles de ejemplos etiquetados, mucho tiempo y computadoras potentes. Aquí no hay nada de eso, y además no hace falta: existen modelos **preentrenados** que ya saben leer sentimiento. El proyecto pide usar uno concreto:

> `nlptown/bert-base-multilingual-uncased-sentiment` (de Hugging Face)

Es una **red neuronal** (tipo BERT) que recibe un texto y devuelve una puntuación de 1 a 5 estrellas. Nosotros **no la entrenamos**: solo hacemos **inferencia** (usarla para predecir sin modificarla).

---

## 3. Cómo obtuve y ejecuté el modelo (lo que más confunde)

El modelo **no está en el repositorio**. Vive en un servidor, Hugging Face Hub, y se descarga automáticamente.

### 3.1 Cómo "llegó" el modelo a tu computadora
1. Instalé las librerías que saben usarlo: `transformers` (de Hugging Face) y `torch` (PyTorch, que hace los cálculos de la red neuronal). Instalé `torch` en versión **CPU** porque tu computadora no tiene GPU y así ocupa mucho menos disco.
2. En el código se escribe una sola línea:

```python
classifier = pipeline("text-classification",
                      model="nlptown/bert-base-multilingual-uncased-sentiment",
                      revision="8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d")
```

3. La **primera vez** que se ejecuta, `transformers` se conecta a Hugging Face y descarga los pesos del modelo a una carpeta de caché (`~/.cache/huggingface`, unos 640 MB en tu equipo). Las siguientes veces ya no descarga nada, lo lee de ahí.
4. Por eso en el repositorio no hay pesos del modelo, tal como exige la rúbrica.

### 3.2 Qué es el `revision` (y cómo lo obtuve)
Los modelos de Hugging Face son como repositorios de Git: el autor puede cambiarlos. El `revision` es el identificador exacto (*commit*) de una versión. Lo consulté a la API pública de Hugging Face (`https://huggingface.co/api/models/nlptown/bert-base-multilingual-uncased-sentiment`, campo `sha`) y lo puse en el código. A esto se le llama **fijar (pin) la versión**: si alguien clona tu repositorio dentro de seis meses, obtiene exactamente el mismo modelo y los mismos resultados.

### 3.3 Qué hace `pipeline()` por dentro
Cuando le pasas una reseña, encadena tres pasos:
1. **Tokenizador:** convierte el texto en números.
2. **Modelo:** la red neuronal procesa esos números.
3. **Salida:** devuelve una etiqueta como `"4 stars"` y una confianza (0 a 1).

### 3.4 Cómo se ejecutó sobre las 500 reseñas
- Se carga el modelo **una sola vez** (cargarlo es lento).
- Se le pasa la lista completa de 500 textos de una sola vez, en lotes de 32 (`batch_size=32`), que es más rápido que reseña por reseña.
- Con la etiqueta `"4 stars"` me quedo con el número `4`.

---

## 4. Orden en que se hizo el trabajo

### Paso 0 · Preparar el entorno
- Creé un **entorno virtual** `.venv` (una carpeta con su propio Python y librerías, aislada del resto de tu computadora) y lo añadí a `.gitignore` para no subirlo a GitHub.
- Comprobé el disco antes de instalar (había 13 GB libres) porque `torch` y el modelo pesan bastante.
- Instalé torch (CPU), transformers, pandas, matplotlib, seaborn y jupyter.

### Paso 1 · Mirar los datos (EDA) antes de modelar
Con un análisis rápido encontré:
- 500 filas, sin valores faltantes.
- Estrellas: 360 de 5, 72 de 4, 38 de 3, 18 de 2, 12 de 1. Promedio exacto 4.5.
- Según las estrellas, las bandas son **86.4 % positivo, 7.6 % neutral y 6.0 % negativo**.
- Textos cortos (~29 palabras), en inglés y generados con plantillas (493 textos únicos de 500).

**Por qué importa:** el 4.5 no significa que todos puntuaron 4.5; casi tres de cada cuatro son 5 estrellas. Y para comparar con el modelo necesitaba convertir las estrellas humanas a las mismas bandas que usará el modelo.

### Paso 2 · Decidir el mapeo a bandas
El modelo da 1–5 estrellas, pero la pregunta es positivo / neutral / negativo. Se usa la regla del enunciado, igual para humanos y modelo, para poder compararlos:

| Estrellas | Banda |
|---|---|
| 1–2 | Negativo |
| 3 | Neutral |
| 4–5 | Positivo |

### Paso 3 · Escribir `src/app.py` (la versión de producción)
Lee el CSV, carga el modelo una vez, predice las 500 reseñas, añade tres columnas (`pred_stars`, `pred_score`, `sentiment_band`) y guarda `data/processed/reviews_with_sentiment.csv`. Lo ejecuté con `python src/app.py` y generó 500 filas sin nulos.

### Paso 4 · Analizar los resultados
Con las predicciones en la mano comparé modelo contra estrellas humanas:

| | Humano | Modelo |
|---|---|---|
| Positivo | 86.4 % | 82.4 % |
| Neutral | 7.6 % | 8.2 % |
| Negativo | 6.0 % | 9.4 % |

- Coinciden en la banda en el **95 %** de las reseñas.
- Estrellas medias: **4.40** (modelo) frente a **4.50** (humano).
- **Conclusión:** sí, lo escrito confirma el 4.5. La pequeña diferencia es que el modelo ve más negativas de las que hay.

### Paso 5 · Entender de dónde viene la diferencia (falsos negativos)
El enunciado avisaba de un posible problema: el modelo se entrenó con reseñas de **productos** y nosotros le damos reseñas de **servicio**. Busqué los **falsos negativos**: reseñas con 4–5 estrellas que el modelo puntúa con 1–2. Hay **16**. Al leerlas vi que se repetían dos frases:

- «Every dish was **bursting** with flavor» (12 casos)
- «They went out of their way to accommodate my **allergy**» (4 casos)

Para comprobar que la causa eran esas palabras hice un **experimento**: la misma frase cambiando una sola palabra.

| Frase | Predicción |
|---|---|
| Every dish was **bursting** with flavor. | 1 estrella |
| Every dish was **full of** flavor. | 5 estrellas |
| …accommodate my **allergy**. | 1 estrella |
| …accommodate my **dietary needs**. | 2 estrellas |

Con «bursting» el efecto es clarísimo. Con «allergy» es más débil (sube de 1 a 2). Mi hipótesis: en reseñas de productos "bursting" (reventando) suele describir un daño y "allergy" aparece en quejas; en un café ambas son elogios. **Esa explicación sobre los datos de entrenamiento es una hipótesis razonable, no un hecho comprobado** (no tengo acceso a los datos con que se entrenó). Lo que sí está comprobado es que esas palabras empujan la predicción hacia 1 estrella.

Dato clave: las otras 353 reseñas positivas no tuvieron ningún falso negativo, así que **esas dos expresiones explican todos los errores**. Si se corrigieran, el modelo daría 85.6 % de positivas, casi igual al 86.4 % humano.

### Paso 6 · Contar la historia en el notebook
`src/explore.ipynb` reúne todo lo anterior en orden: objetivos → EDA → insights → plan y elección del modelo → inferencia → comparación con 4.5 → falsos negativos → muestra manual de 18 reseñas → conclusiones. Está **ejecutado**, con las tablas y gráficos guardados dentro. Lo generé con un script y lo ejecuté con `jupyter nbconvert --execute`. También hay una celda que comprueba que las predicciones del notebook coinciden con las de `app.py`.

### Paso 7 · Fijar dependencias
`requirements.txt` quedó con versiones exactas de lo que usa el proyecto (`torch==2.15.0`, `transformers==5.23.0`, `pandas==3.1.1`, …) tomadas de lo que realmente instalé.

---

## 5. Por qué hay dos "formas" de ejecutar: notebook y script

| | `src/explore.ipynb` | `src/app.py` |
|---|---|---|
| Para qué | Contar la historia y analizar | Producción: repetir el proceso limpio |
| Contiene | Texto, código, gráficos y resultados | Solo la lógica de inferencia |
| Salida | Los resultados dentro del documento | `data/processed/reviews_with_sentiment.csv` |

Es el patrón del template: **el notebook para la historia, el script para producción**.

---

## 6. Cómo ejecutarlo tú

```bash
cd ~/Escritorio/4Geeks/exercises/josecaro02-machine-learning-python-template
source .venv/bin/activate
python src/app.py
jupyter nbconvert --to notebook --execute --inplace src/explore.ipynb
```

Si lo ejecutas en otra computadora, instala PyTorch CPU antes de `pip install -r requirements.txt`:
`pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`

---

## 7. Resumen en cinco frases (para explicarlo en clase)

1. Teníamos 500 reseñas con estrellas y queríamos saber si el **texto** coincide con el 4.5.
2. En vez de entrenar nada, usamos un modelo preentrenado de Hugging Face, descargado automáticamente y **fijado por versión**.
3. Convertimos las estrellas del modelo y las humanas a las mismas bandas y las comparamos: **sí coinciden (95 %)**.
4. La pequeña diferencia viene de 16 falsos negativos causados por dos expresiones («bursting», «allergy»), porque el modelo aprendió con productos y no con servicios.
5. Lo dejamos documentado en un notebook ejecutado, un script de producción y un CSV con las predicciones.

## 8. Limitaciones (para ser honestos)

- Los textos del dataset siguen plantillas muy repetitivas y los casos negativos son pocos (30): con reseñas reales el patrón podría cambiar.
- La verificación manual cubre 18 reseñas, no las 500.
- No hice la extensión opcional (comparar con `tabularisai/multilingual-sentiment-analysis`).
- Las notas de la muestra manual son un borrador mío: conviene que las revises y las reescribas con tus palabras.

---

## 9. `src/app.py` explicado bloque por bloque

El archivo hace **una sola cosa**: leer el CSV, pedirle al modelo una predicción por reseña y guardar el resultado. Tiene cinco partes.

### 9.1 Imports y constantes (arriba del archivo)

```python
from pathlib import Path
import pandas as pd
from transformers import pipeline

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH  = ROOT / "data" / "raw" / "reviews.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "reviews_with_sentiment.csv"
MODEL_NAME     = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"
BATCH_SIZE = 32
```

- `pandas` maneja la tabla (el CSV). `pipeline` es la función de Hugging Face que carga el modelo.
- `ROOT` calcula la carpeta del proyecto a partir de la ubicación del propio archivo. Así el script funciona aunque lo ejecutes desde otra carpeta.
- `MODEL_NAME` y `MODEL_REVISION` son el modelo y su versión fijada (el *pin*).
- `BATCH_SIZE` = cuántas reseñas procesa el modelo a la vez.

### 9.2 `load_model()`: traer el modelo

```python
def load_model():
    return pipeline("text-classification", model=MODEL_NAME, revision=MODEL_REVISION)
```

Aquí ocurre la descarga (la primera vez) y la carga del modelo en memoria. Devuelve un objeto, `classifier`, que **se comporta como una función**: le das texto y te devuelve una etiqueta.

### 9.3 `star_to_band()`: convertir estrellas en banda

```python
def star_to_band(stars):
    if stars <= 2: return "Negativo"
    if stars == 3: return "Neutral"
    return "Positivo"
```

Es una regla escrita a mano (la del enunciado), no inteligencia artificial. Se aplica por igual a las estrellas del modelo y a las humanas.

### 9.4 `predict()`: el corazón del script

```python
results = classifier(df["review_text"].tolist(), batch_size=BATCH_SIZE, truncation=True)
out["pred_stars"]     = [int(r["label"].split()[0]) for r in results]
out["pred_score"]     = [round(r["score"], 4) for r in results]
out["sentiment_band"] = out["pred_stars"].map(star_to_band)
```

- Primera línea: le pasa al modelo la lista de 500 textos. **Aquí es donde trabaja el modelo.** Devuelve una lista de 500 resultados con este aspecto: `{"label": "4 stars", "score": 0.72}`.
- `r["label"].split()[0]` toma `"4 stars"`, lo parte por espacios y se queda con `"4"`; `int()` lo vuelve número.
- `pred_score` es la confianza del modelo.
- `sentiment_band` se calcula con la regla de arriba.
- `truncation=True` corta textos muy largos (no es el caso aquí, pero evita errores).

### 9.5 `main()` y el cierre

```python
df = pd.read_csv(INPUT_PATH)         # 1. leer
classifier = load_model()            # 2. cargar modelo (una vez)
enriched = predict(df, classifier)   # 3. predecir
enriched.to_csv(OUTPUT_PATH, ...)    # 4. guardar
```

`if __name__ == "__main__": main()` significa «ejecuta `main()` solo cuando el archivo se corre directamente (`python src/app.py`)».

**Resultado:** el CSV original más tres columnas nuevas. Nada más. `app.py` **no escribe ningún informe**: solo produce números.

---

## 10. ¿Quién hizo el "informe" del notebook? Tú preguntas: ¿yo, el modelo o el código?

Respuesta honesta: **las tres cosas contribuyeron, pero cada una hizo algo distinto, y lo que parece "informe" lo escribí yo.**

| Quién | Qué hizo | Qué NO hizo |
|---|---|---|
| **El modelo** (nlptown BERT) | Por cada reseña devolvió `"N stars"` y una confianza. Nada más. | No analizó, no comparó, no explicó, no escribió una sola palabra del informe. |
| **El código** (Python, pandas, matplotlib) | Calculó los porcentajes, la matriz de confusión, los conteos de falsos negativos y dibujó los gráficos. Son operaciones mecánicas y repetibles. | No interpreta. Un `value_counts()` te da números, no conclusiones. |
| **Yo (Claude)** | Decidí la estructura del notebook, **escribí el código** (de `app.py` y de cada celda) y **escribí todo el texto**: los insights, la hipótesis de «bursting» y «allergy», las conclusiones, las notas de la muestra manual. | No «descubrí» nada por magia: vi los números que el código produjo y los interpreté. |
| **Tú** | Hasta ahora, pedir y revisar. | Falta que entiendas y hagas tuyas la hipótesis y las notas manuales. |

### Cómo ocurrió realmente, en orden

1. **Escribí `app.py` y lo ejecuté.** El código llamó al modelo y se generó el CSV.
2. **Exploré los resultados en la terminal** con pequeños scripts de pandas: tabla de aciertos, lista de falsos negativos, conteo de frases repetidas.
3. **Vi** que 12 de los 16 falsos negativos tenían «bursting» y que los otros 4 tenían «allergy». Esa **observación la hice yo leyendo la salida del código**; el código solo mostró las reseñas y los conteos.
4. **Probé una idea** (cambiar una palabra) con el propio modelo y comprobé que «bursting» → 1 estrella pero «full of flavor» → 5 estrellas.
5. **Con los números ya conocidos, escribí el notebook:** celdas de código que vuelven a calcular todo y celdas de texto con mis explicaciones.
6. **Lo ejecuté** con `jupyter nbconvert --execute` para que los resultados quedaran guardados dentro del archivo, y **comprobé** que los números que yo había escrito en el texto (82.4 %, 95 %, 16 falsos negativos, etc.) coincidieran con las salidas reales.

### Un detalle importante que debes saber

**El texto en markdown del notebook no se genera solo.** Los porcentajes que aparecen dentro de las explicaciones («82.4 % positivo», «16 falsos negativos»…) los escribí a mano después de ver los resultados. Las celdas de código sí se recalculan si cambias los datos, pero los párrafos no se actualizarían. Por eso las verifiqué contra la ejecución real.

### Por qué esto importa para tu clase

- El proyecto permite usar un agente de código (hay un `PROMPT.es.md` justamente para eso), pero dice que lo que viene después del EDA es **responsabilidad tuya**.
- Si te preguntan «¿qué hace el modelo?», la respuesta es: *solo predice estrellas por reseña*. Todo lo demás (comparar, encontrar falsos negativos, explicarlos) es análisis que se hace **con el resultado del modelo**.
- Para poder defenderlo, lo mínimo es: (1) entender las secciones 9 y 3 de este documento, (2) leer las 18 reseñas de la sección 8 del notebook y reescribir las notas con tus palabras, (3) poder explicar la hipótesis de «bursting» y por qué es una hipótesis y no un hecho comprobado.
