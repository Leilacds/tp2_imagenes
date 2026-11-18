# Registro de pruebas

## Objetivo

Registrar pruebas del MVP de reducción de ruido con difusión. Cada prueba incluye la imagen de entrada, la configuración, el estado de ejecución y una interpretación. La aplicación muestra la entrada frente a SDXL, una línea base de filtro bilateral, un mapa de diferencias y el análisis de objetos RT-DETR.

> Una ejecución exitosa confirma que el flujo puede procesar la imagen; no demuestra por sí sola que la salida sea más fiel o de mejor calidad. SDXL puede alterar o inventar detalles.

## Configuración común

- **Entorno:** Google Colab con GPU T4.
- **Intensidad de difusión (`strength`):** 0.25.
- **Semilla:** 42.
- **Tamaño de las imágenes:** 640 × 480 píxeles.
- **Estado del flujo:** la autora informó que ambas pruebas se ejecutaron correctamente.

La intensidad 0.25 y la semilla 42 corresponden a los valores usados en ambas pruebas según lo informado. La semilla fija el inicio aleatorio de la generación para facilitar la comparación; no garantiza que el resultado sea idéntico entre distintos entornos.
La cantidad de pasos de inferencia no se configura desde la interfaz; queda fijada en [`src/tp2_imagenes/diffusion.py`](../src/tp2_imagenes/diffusion.py).

## Validación adicional — Ejecución local

- **Estado:** la autora informó que el proyecto se ejecutó correctamente en otra computadora.
- **Detalles del entorno:** sistema operativo, GPU, versión de Python e imagen usada no fueron registrados.
- **Interpretación:** esto aporta una validación adicional de ejecución local. Para que la prueba sea reproducible, conviene anotar esas especificaciones si están disponibles.

---

## Prueba 1 — Caso favorable

### Imagen de entrada

[`assets/ejemplos/prueba_favorable_ruidosa.png`](../assets/ejemplos/prueba_favorable_ruidosa.png)

Ilustración sintética de una taza y una planta en una escena oscura, con ruido añadido. Tiene objetos reconocibles y bordes relativamente claros, por lo que permite observar si el procesamiento reduce el grano sin cambiar la estructura general.

### Parámetros

| Parámetro | Valor |
|---|---:|
| Intensidad de difusión | 0.25 |
| Semilla | 42 |

### Resultado y estado

- **Ejecución:** correcta, según lo informado por la autora.
- **Salida visible en la captura:** la imagen de referencia del filtro bilateral se ve más suave y con menos grano; la taza y la planta siguen siendo reconocibles. El mapa de diferencias resalta principalmente contornos y zonas modificadas.
- **RT-DETR antes:** `pottedplant` (1) y `vase` (2).
- **RT-DETR después:** `vase` (1).
- **Etiquetas que dejaron de detectarse:** `pottedplant` (1) y `vase` (1).

### Interpretación

Esta imagen funciona como caso favorable por tener una escena sencilla y objetos reconocibles. La salida del filtro bilateral parece reducir el grano manteniendo las formas principales, mientras que el mapa confirma que hubo cambios de píxeles, no si esos cambios son una mejora.

RT-DETR dejó de detectar la planta y uno de los dos objetos clasificados como `vase`. Esto no demuestra que hayan desaparecido de la imagen: el detector puede equivocarse, especialmente con una ilustración sintética. Es una señal para inspeccionar visualmente el contenido.

La captura compartida no permite evaluar directamente el comparador original/SDXL; por eso todavía no se concluye si la salida de difusión conservó mejor la escena o inventó detalles.

**Captura antes/después:** compartida en la conversación; no incluida en el repositorio.

---

## Prueba 2 — Texto pequeño y patrones repetidos

### Imagen de entrada

[`assets/ejemplos/prueba_limite_texto_y_patrones.png`](../assets/ejemplos/prueba_limite_texto_y_patrones.png)

Ilustración sintética con una etiqueta, texto pequeño, barras y patrones repetidos. Es un caso desafiante porque los detalles finos pueden ser difíciles de conservar para un modelo generativo.

### Parámetros

| Parámetro | Valor |
|---|---:|
| Intensidad de difusión | 0.25 |
| Semilla | 42 |

### Resultado y estado

- **Ejecución:** correcta, según lo informado por la autora.
- **Salida visible en la captura:** la imagen de referencia del filtro bilateral se ve más limpia; el texto de la etiqueta, el código de barras y los patrones siguen siendo visualmente reconocibles. El mapa resalta numerosos cambios de píxeles y contornos.
- **RT-DETR antes:** `laptop` (1).
- **RT-DETR después:** ninguno con confianza suficiente.
- **Etiquetas que dejaron de detectarse:** `laptop` (1).

### Interpretación

La referencia bilateral reduce visualmente el ruido y, en la captura, los textos y patrones principales siguen siendo legibles. El mapa de diferencias confirma cambios de píxeles, pero no permite juzgar por sí solo si la restauración es fiel.

RT-DETR detectó `laptop` en la imagen original y ningún objeto con confianza suficiente en la imagen procesada. Como la entrada es una ilustración sintética de una etiqueta, la etiqueta `laptop` puede ser una clasificación errónea; su ausencia después no demuestra que se haya eliminado un objeto real.

La captura compartida no muestra directamente el comparador original/SDXL. Por eso, esta observación describe la referencia bilateral y las herramientas de análisis visibles, pero no basta para decidir si SDXL conservó las letras o inventó detalles.

**Captura antes/después:** compartida en la conversación; no incluida en el repositorio.

---

## Comparación y conclusión

Las dos pruebas utilizaron la misma intensidad y semilla para mantener constante la configuración. La primera explora la reducción de ruido en una escena sencilla; la segunda permite buscar cambios en texto y patrones finos.

La autora informó que ambas se procesaron correctamente en Colab y compartió capturas de las dos pruebas. Las referencias bilaterales se ven más limpias en las capturas, mientras que RT-DETR deja de detectar algunas etiquetas. Esto motiva una inspección visual, pero no permite afirmar que la salida de difusión conserve fielmente el contenido, porque las capturas no muestran directamente el comparador original/SDXL. No se asigna una puntuación de calidad a partir del estado de ejecución solamente.
