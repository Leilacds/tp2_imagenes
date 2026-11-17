# Limpiador de ruido en fotos con difusión

MVP para reducir el ruido de fotos tomadas con poca luz. Usa un modelo de difusión de Hugging Face y muestra también una línea base clásica, las detecciones de objetos antes y después, y un mapa de los píxeles que cambiaron.

> **Aviso:** la difusión puede modificar o inventar detalles. El resultado es una mejora visual generada, no una recuperación fiel ni evidencia de lo que había en la escena.

## Caso de uso

**Persona usuaria inicial propuesta:** alguien que fotografía objetos con su celular en interiores con poca luz y quiere reducir el granulado sin perder de vista qué partes de la imagen cambió el sistema.

El MVP se limita a la reducción de ruido; no intenta corregir desenfoque de movimiento, reconstruir detalles faltantes ni procesar lotes.

## Flujo

1. La persona carga una imagen.
2. SDXL image-to-image procesa una versión de hasta 768 píxeles en el lado mayor. La intensidad se puede ajustar.
3. La aplicación calcula una línea base con un filtro bilateral.
4. RT-DETR detecta objetos en la imagen original y en la procesada, y muestra las etiquetas que cambiaron.
5. Se presentan la comparación antes/después, la línea base y un mapa de diferencias.

```text
             +----------------------+
             |  Persona usuaria     |
             |  carga una imagen    |
             +----------+-----------+
                        |
                        v
             +----------------------+
             | Interfaz web Gradio  |
             | intensidad + semilla|
             +----------+-----------+
                        |
            +-----------+-----------+
            |                       |
            v                       v
 +----------------------+  +----------------------+
 | SDXL image-to-image  |  | Filtro bilateral    |
 | resultado generado  |  | línea base clásica  |
 +----------+-----------+  +----------------------+
            |
            v
 +----------------------+
 | RT-DETR analiza      |
 | original y resultado |
 +----------+-----------+
            |
            v
 +----------------------------------------------+
 | Comparación antes/después + línea base       |
 | mapa de diferencias + etiquetas detectadas   |
 +----------------------------------------------+
            |
            v
 +----------------------------------------------+
 | Revisar cambios: la difusión puede inventar  |
 | detalles; el análisis no prueba autenticidad |
 +----------------------------------------------+
```

Las detecciones no prueban que el contenido sea idéntico: el detector puede equivocarse y no compara la identidad exacta de cada objeto. El mapa muestra cambios de píxeles, no su significado. La línea base es una referencia visual, no una métrica de calidad.

## Uso de la interfaz

1. Abrí la app y cargá una imagen en **Imagen de entrada**. La imagen debe tener al menos 64 × 64 píxeles; se procesa en color RGB.
2. Elegí la **intensidad de difusión** y la **semilla** (ver abajo). Para la primera prueba, podés dejar los valores iniciales.
3. Presioná **Reducir ruido** y esperá a que terminen la difusión y el análisis visual. La primera ejecución puede tardar porque se descargan y cargan los modelos.
4. Revisá todas las salidas. En el comparador **Original / difusión**, mové el divisor para inspeccionar distintas zonas antes y después.

### Intensidad de difusión

- Rango disponible: **0.10–0.45**, con pasos de **0.05**. Valor inicial: **0.25**.
- Es el parámetro `strength` de SDXL image-to-image; **no es un porcentaje de ruido eliminado**.
- Un valor bajo limita cuánto se aparta la generación de la imagen de entrada. Un valor más alto permite cambios mayores, y con ellos aumenta el riesgo de modificar o inventar objetos, texturas, bordes y letras.
- Para comparar valores, mantené la misma imagen y semilla. Empezá por 0.25 y revisá el resultado antes de probar una intensidad mayor.

### Semilla

- Rango disponible: **0–999999**. Valor inicial: **42**.
- La semilla inicializa la generación aleatoria del modelo. Con la misma imagen, modelo, intensidad y entorno, reutilizarla permite repetir la configuración y facilita la comparación.
- Cambiar la semilla puede producir otra versión de la imagen; **no cambia directamente la cantidad de ruido ni garantiza una mejora**. La reproducibilidad exacta puede variar entre versiones de software y hardware.

### Cómo interpretar las salidas

- **Original / difusión:** comparación interactiva de la entrada con el resultado generado por SDXL.
- **Línea base: filtro bilateral:** resultado de un filtro clásico aplicado a la entrada. Sirve de referencia visual; no utiliza difusión ni determina por sí solo cuál resultado es mejor.
- **Mapa de diferencias:** resalta los cambios de píxeles entre la entrada y la salida de difusión. Más brillo significa una diferencia mayor tras amplificarla para verla; no indica si el cambio es correcto, real o una mejora.
- **Análisis de objetos (RT-DETR):** compara las etiquetas y cantidades de objetos detectados antes y después. Solo incluye detecciones con confianza de al menos **0.40**. Una etiqueta que aparece o desaparece es una señal para revisar la imagen, no una prueba concluyente: el detector puede equivocarse y no verifica identidad ni autenticidad.

## Imágenes de prueba

En [`assets/ejemplos/`](assets/ejemplos/) hay dos imágenes sintéticas para probar el flujo:

- `prueba_favorable_ruidosa.png`: escena sencilla y ruidosa, pensada como caso favorable.
- `prueba_limite_texto_y_patrones.png`: texto pequeño y patrones repetidos para observar posibles alteraciones.

Son ilustraciones generadas por código, no fotografías reales. Sirven para una prueba inicial de la aplicación, pero no reemplazan la validación con fotos propias.

También están disponibles dos copias con ruido de fotos de prueba:

- `foto_cafe_ruido_favorable.png`: caso positivo inicial, con objetos reconocibles.
- `foto_edificio_ruido_negativa.png`: caso desafiante por sus líneas finas, ventanas y texto.

Estas pruebas ayudan a observar el comportamiento del sistema, pero no garantizan que el resultado vaya a ser favorable o negativo antes de ejecutar el modelo.

## Modelos y licencias

- Difusión: [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), licencia **CreativeML Open RAIL++-M**. Se usa con `diffusers` en modo image-to-image y baja intensidad.
- Análisis visual: [`PekingU/rtdetr_r50vd`](https://huggingface.co/PekingU/rtdetr_r50vd), licencia **Apache-2.0**, para detección de objetos.

Revisá las fichas y condiciones de uso de los modelos antes de utilizarlos. La aplicación no requiere una API paga ni un token de Hugging Face para estos modelos públicos; si el Hub solicita aceptar condiciones, hacelo desde la página del modelo.

## Ejecución local

Se requiere Python 3.12 y `uv`. La difusión necesita una GPU NVIDIA compatible para una ejecución práctica; en esta máquina de desarrollo se detectó una GPU integrada Intel, por lo que la prueba con difusión debe hacerse en Colab con GPU.

```powershell
uv sync
uv run app.py
```

La primera ejecución descarga los modelos. Para una instalación local con GPU, instalá PyTorch según la guía oficial correspondiente a tu sistema antes de probar la aplicación. La app muestra un error explícito si PyTorch no detecta CUDA.

## Ejecución en Google Colab

1. Abrí [`notebooks/ejecutar_en_colab.ipynb`](https://colab.research.google.com/github/Leilacds/tp2_imagenes/blob/desarrollo/notebooks/ejecutar_en_colab.ipynb) en Colab. El notebook debe estar subido a GitHub en la rama `desarrollo`.
2. Elegí **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4**.
3. Ejecutá las celdas en orden. El notebook verifica CUDA, clona la rama `desarrollo`, instala el proyecto y lanza la aplicación.
4. Abrí el enlace temporal de Gradio que imprime la última celda. La primera instalación y descarga de los modelos puede demorar.

El uso de la GPU gratuita depende de la disponibilidad de Colab; que aparezca la opción T4 no garantiza que siempre haya recursos disponibles.

## Pruebas

Las pruebas unitarias de las transformaciones de imagen no descargan modelos:

```powershell
uv run --group dev pytest
```

## Limitaciones conocidas

- SDXL es un modelo generativo general, no un denoiser especializado. Incluso con baja intensidad puede cambiar texturas, bordes u objetos.
- Se procesa como máximo a 768 píxeles en el lado mayor y el resultado se redimensiona al tamaño original para mostrarlo.
- RT-DETR detecta clases de objetos, pero no verifica identidad ni autenticidad.
- El mapa de diferencias señala cambios de píxeles; no distingue mejoras de alteraciones.
- El rendimiento, la memoria disponible y los tiempos dependen de la GPU asignada por Colab.

## Uso de asistentes de IA

La persona autora debe completar esta sección antes de entregar: qué partes implementó o decidió, qué sugerencias recibió de asistentes de IA y cómo verificó cada resultado. Hay que poder explicar y modificar el código del proyecto.

## Demo

El video de cinco minutos debería mostrar: el problema y la persona usuaria, una imagen de entrada, el procesamiento, la comparación con la línea base y el análisis visual, las limitaciones y la reflexión sobre qué puede ser real o inventado.
