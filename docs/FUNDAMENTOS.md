# Fundamentos

Este documento describe la teoría en la que se basa el proyecto, los supuestos de los que depende y dónde fallan esos supuestos en la práctica.

## 1. Complejidad de Kolmogorov

Sea $U$ una máquina de Turing universal de prefijos. La complejidad de Kolmogorov de una cadena $x$ es la longitud del programa más corto con el que $U$ produce $x$:

$$K(x) = \min \{\, |p| : U(p) = x \,\}$$

La complejidad condicional $K(x \mid y)$ es la misma cantidad cuando $U$ recibe $y$ como entrada auxiliar.

Propiedades que se usan aquí:

- **Invariancia.** Para dos máquinas universales $U, V$ existe una constante $c_{UV}$ tal que $|K_U(x) - K_V(x)| \le c_{UV}$ para todo $x$. La elección de la máquina solo desplaza $K$ en una constante.
- **No computabilidad.** Ningún algoritmo calcula $K(x)$. Es semicomputable por arriba: se puede aproximar desde arriba, pero nunca certificar desde abajo.
- **Simetría de la información.** $K(x, y) = K(x) + K(y \mid x) + O(\log K(x, y))$. La información necesaria para describir un par es la de un objeto más lo que el otro le añade.
- **Incompresibilidad.** La mayoría de las cadenas de longitud $n$ cumplen $K(x) \ge n - c$. Los datos aleatorios no tienen una descripción más corta.

$K$ mide el contenido de información de un objeto individual, sin modelo probabilístico. Su relación con la entropía de Shannon es que, para una fuente computable $P$, la complejidad esperada $\sum_x P(x) K(x)$ coincide con $H(P)$ salvo un término aditivo del orden de $K(P)$.

## 2. Distancia de información

Bennett, Gács, Li, Vitányi y Zurek (1998) definen la distancia de información como la longitud del programa más corto que convierte cualquiera de los dos objetos en el otro:

$$E(x, y) = \max \{ K(x \mid y),\, K(y \mid x) \}$$

salvo términos logarítmicos. $E$ es una métrica salvo esos términos y es mínima entre las distancias admisibles: toda distancia $D$ semicomputable por arriba que cumpla una condición de densidad satisface $E(x, y) \le D(x, y) + O(1)$. Si dos objetos son cercanos bajo cualquier noción computable de similitud, también lo son bajo $E$.

$E$ es absoluta, así que los objetos largos obtienen distancias grandes. Li, Chen, Li, Ma y Vitányi (2004) la normalizan:

$$\mathrm{NID}(x, y) = \frac{\max \{ K(x \mid y),\, K(y \mid x) \}}{\max \{ K(x),\, K(y) \}}$$

La NID toma valores en $[0, 1]$ salvo términos que se anulan, cumple los axiomas de métrica salvo $O(1/K)$ y es universal entre las distancias normalizadas admisibles. No exige elegir características de antemano, y por eso resulta atractiva para malware: no hay una lista de características en la que deba caber la similitud de cada familia.

## 3. La compresión como aproximación de K

Un compresor sin pérdida $C$ transforma $x$ en una cadena decodificable, así que $|C(x)|$ es la longitud de una descripción válida de $x$ y una cota superior de $K(x)$ salvo el tamaño del descompresor. Escribiendo $C(x)$ para esa longitud y $xy$ para la concatenación, la simetría de la información sugiere

$$K(y \mid x) \approx C(xy) - C(x)$$

Sustituyendo en la NID se obtiene la Normalized Compression Distance (Cilibrasi y Vitányi, 2005):

$$\mathrm{NCD}(x, y) = \frac{C(xy) - \min \{ C(x), C(y) \}}{\max \{ C(x), C(y) \}}$$

Esto funciona porque un compresor es un modelo de predicción. Por el teorema de codificación de fuente, un código asigna unos $-\log_2 \hat p(x)$ bits según el modelo $\hat p$ que implementa, de modo que un buen compresor es un buen modelo de los datos que ha visto. Cuando el compresor procesa $y$ después de $x$, todo lo que aprendió de $x$ (subcadenas repetidas, estadísticas de símbolos, contextos) reduce el coste de $y$. Esa reducción es la información compartida.

### Compresores normales

Cilibrasi y Vitányi llaman normal a $C$ si, salvo términos logarítmicos:

| axioma | enunciado |
|---|---|
| idempotencia | $C(xx) = C(x)$ y $C(\varepsilon) = 0$ |
| monotonía | $C(xy) \ge C(x)$ |
| simetría | $C(xy) = C(yx)$ |
| distributividad | $C(xy) + C(z) \le C(xz) + C(yz)$ |

Con un compresor normal, la NCD es una distancia normalizada admisible con valores en $[0, 1 + \epsilon]$ que cumple las desigualdades de métrica salvo $\epsilon$. Los compresores reales solo aproximan estos axiomas, y las diferencias se pueden medir.

## 4. Cómo se desvían los compresores reales

Los compresores de este proyecto construyen modelos distintos:

| códec | modelo | límite relevante para la NCD |
|---|---|---|
| DEFLATE | LZ77 con ventana deslizante, codificación Huffman | ventana de 32 KiB |
| bzip2 | transformada de Burrows-Wheeler, move-to-front, Huffman | bloques independientes de 100 a 900 kB |
| xz | LZ77 más codificación de rango con modelos de contexto (LZMA2) | diccionario de 256 KiB a 64 MiB |
| zstd | LZ77 más Huffman y codificación de entropía de estados finitos | ventana de hasta 128 MiB en este proyecto |

Las desviaciones que importan en la práctica:

- **Ventana finita.** Un compresor LZ solo puede referenciar datos dentro de su ventana. Si $|x| + |y|$ la supera, $y$ no puede reutilizar $x$, así que $C(xy) \approx C(x) + C(y)$ y la NCD tiende a 1 sin importar la similitud (Cebrián, Alfonseca y Ortega, 2005). En este proyecto todos los pares del corpus sintético superan la ventana de DEFLATE, y `NCD_deflate(x, x)` pasa de 0.9 con 64 KiB de datos aleatorios. La CLI cuenta esos pares y avisa.
- **Fallo de idempotencia.** La BWT ordena rotaciones del bloque; una segunda copia literal no se codifica como una referencia hacia atrás barata. bzip2 da $\mathrm{NCD}(x, x) \approx 0.22$ a $0.27$ con datos incompresibles, frente a unos $0.001$ de zstd.
- **Coste fijo.** Cabeceras, sumas de verificación y tablas de codificación de entropía añaden un coste casi constante por llamada. En objetos pequeños ese coste domina $C(x)$ y las diferencias entre pares se pierden. El proyecto desactiva en zstd la suma de verificación, el tamaño de contenido y el identificador de diccionario, y los experimentos con flujos muestran que las ventanas de 16 flujos (176 bytes) son demasiado pequeñas.
- **Asimetría.** En general $C(xy) \ne C(yx)$. El estimador `concat-sym` promedia ambos órdenes al doble de coste.
- **Saturación por aleatoriedad.** Si $x$ ya es casi aleatorio (cifrado o empaquetado), $C(x) \approx |x|$ y su NCD con cualquier objeto es cercana a 1. La compresión no puede encontrar estructura que no sea visible en los bytes.

Por eso pueden aparecer valores de NCD fuera de $[0, 1]$, y se conservan sin recortar como diagnóstico.

## 5. Estimadores

**Concatenación.** `concat` calcula $C(xy)$ para cada par; es el estimador estándar. Con $n$ objetos necesita $n$ compresiones individuales, que se guardan en caché, y $n(n-1)/2$ compresiones conjuntas, cada una de las cuales vuelve a comprimir $x$.

**Compresión condicional.** `conditional` estima $K(y \mid x)$ directamente. zstd puede cargar $x$ como diccionario de contenido crudo, un prefijo al que las coincidencias pueden referirse sin emitirlo. El objeto más simple de cada par ($C$ menor) hace de diccionario, así que la longitud comprimida del otro aproxima $\max \{ K(x \mid y), K(y \mid x) \}$, el numerador de la NID:

$$\mathrm{NCD}_{\mathrm{cond}}(x, y) = \frac{C(y \mid x)}{C(y)}, \quad C(x) \le C(y)$$

El diccionario se prepara una vez por objeto y se reutiliza en toda su fila, lo que evita volver a comprimir $x$ y dio una aceleración de 3.4x con zstd nivel 19. Un diccionario crudo transfiere coincidencias LZ pero no estadísticas del codificador de entropía, lo que es una explicación plausible de por qué ordena peor a los vecinos que `concat` en ventanas de flujos medianas y repetitivas.

## 6. Clasificación con una distancia

La NCD produce distancias, no vectores de características, así que los clasificadores deben trabajar solo con distancias.

**k vecinos más cercanos.** Una muestra toma la etiqueta mayoritaria de sus $k$ referencias más cercanas; los empates se resuelven a favor de la clase del miembro más cercano. Con una métrica verdadera y datos suficientes, el error del 1-NN es asintóticamente como mucho el doble del error de Bayes (Cover y Hart, 1967). La NCD es solo aproximadamente una métrica, así que esto es una orientación, no una garantía. El k-NN no necesita entrenamiento ni embedding, y aísla la calidad de la distancia en sí.

**Puntuación de anomalía.** La puntuación de una consulta es su NCD media a las $k$ referencias benignas más cercanas. Es detección de novedad: no se usan datos de ataque para ajustarla, y todo lo que comparte poca información con el comportamiento benigno conocido obtiene una puntuación alta.

**Aislar la representación.** Las líneas base incluyen k-NN con distancia euclídea sobre características clásicas estandarizadas, junto a árboles de gradient boosting. Comparar el k-NN con NCD contra el k-NN euclídeo mide la representación (bytes crudos más compresor frente a características diseñadas) con el clasificador fijo; comparar contra GBDT mide el método completo frente a una referencia fuerte.

## 7. Aplicación a malware

Las variantes de una familia comparten código: las mismas funciones recompiladas, los mismos cargadores, las mismas rutinas de cifrado. El código compartido implica subcadenas comunes largas y estadísticas de instrucciones parecidas, que un compresor LZ convierte directamente en un $C(xy)$ menor.

Decisiones que se derivan de esto:

- **Solo secciones de código por defecto.** Recursos, iconos, relleno y overlays son grandes y a menudo se comparten con software no relacionado, y pueden ocultar la señal del código. La vista `code` conserva las secciones ejecutables; `no-overlay` elimina los datos añadidos tras la última sección.
- **Ventana suficiente para binarios completos.** zstd se configura con una ventana de 128 MiB para que muestras de varios megabytes sigan viéndose entre sí dentro de $xy$.
- **Puntos ciegos conocidos.** El empaquetado y el cifrado vuelven los bytes casi aleatorios, así que la NCD refleja el empaquetador y no la carga útil. Añadir código benigno a una muestra la acerca a vecinos benignos, una evasión que aún no se ha medido. Trabajo previo relacionado: Wehner (2007) usó la NCD para analizar gusanos y tráfico, Bailey et al. (2007) para agrupar comportamiento de malware, y Raff y Nicholas (2017) propusieron LZJD como alternativa más rápida para secuencias largas.

## 8. Aplicación a tráfico de red

Un registro de flujo es un vector de unos 80 números, demasiado corto para comprimirse de forma útil. El proyecto convierte el tráfico en cadenas:

1. Cada característica numérica se transforma en su intervalo de cuantil, un byte por característica, con los intervalos ajustados solo sobre flujos de entrenamiento.
2. Los flujos consecutivos de un mismo host de origen forman una ventana, con `0xFF` entre flujos.

Un escaneo, una inundación o un bucle de fuerza bruta repiten el mismo patrón de flujo, que se convierte en un patrón de bytes repetido y se comprime bien junto a otras ventanas con el mismo comportamiento. La ventana debe ser lo bastante larga para superar el coste fijo del compresor: en tráfico generado, el macro-F1 del k-NN con `concat` pasa de 0.11 con 16 flujos a 1.0 con 64.

## 9. Principios de evaluación

- **Las mismas muestras para todos los modelos.** La NCD y las líneas base leen el mismo archivo de partición, de modo que las diferencias vienen del método y no de los datos.
- **Particiones temporales.** El malware evoluciona. Las particiones aleatorias ponen variantes de una misma campaña a ambos lados y sobrestiman la precisión; entrenar con muestras antiguas y evaluar con muestras nuevas sigue a Pendlebury et al. (2019). Las familias de prueba que no aparecen en entrenamiento se descartan, así que la tarea sigue siendo de mundo cerrado.
- **Sin fugas de información.** Los intervalos de cuantil se ajustan sobre flujos de entrenamiento, las referencias de anomalía son solo ventanas benignas de entrenamiento y las ventanas de prueba nunca se usan como referencia.
- **Métricas acordes a la pregunta.** El macro-F1 pondera igual las familias raras y las comunes. El ROC AUC mide qué tan bien una puntuación ordena los ataques por encima de las ventanas benignas sin fijar un umbral.
- **Calidad de los datos.** CIC-IDS2017 tiene errores documentados en características y etiquetas (Engelen et al., 2021; Rosay et al., 2022), así que para publicar resultados es preferible una versión corregida.

## 10. Principios de ingeniería

- **Entrada no confiable.** Las muestras son adversarias. El análisis comprueba límites, se ejecuta bajo `catch_unwind`, recurre a los bytes crudos si falla y rechaza archivos de más de 256 MiB. Las muestras se leen como bytes y nunca se ejecutan ni se mapean.
- **Coste.** Una matriz por pares necesita $O(n^2)$ compresiones. $C(x)$ se guarda en caché, las filas se procesan en paralelo con un contexto de compresión por hilo, la salida comprimida se cuenta en lugar de almacenarse y el estimador condicional reutiliza cada diccionario en toda una fila.
- **Reproducibilidad.** El corpus sintético usa un generador SplitMix64 con semilla, liblzma se enlaza estáticamente y Nix fija la cadena de herramientas.
- **Verificación.** Los fallos conocidos son pruebas unitarias (ventana de DEFLATE, autodistancia de bzip2), las matrices se contrastan con cálculos de pares individuales, y la CLI en Rust y los bindings de Python deben dar el mismo AUC.

## Referencias

- C. H. Bennett, P. Gács, M. Li, P. Vitányi, W. Zurek. Information distance. IEEE Trans. Inf. Theory 44(4), 1998.
- M. Li, X. Chen, X. Li, B. Ma, P. Vitányi. The similarity metric. IEEE Trans. Inf. Theory 50(12), 2004.
- R. Cilibrasi, P. Vitányi. Clustering by compression. IEEE Trans. Inf. Theory 51(4), 2005.
- M. Cebrián, M. Alfonseca, A. Ortega. Common pitfalls using the normalized compression distance. Commun. Inf. Syst. 5(4), 2005.
- M. Li, P. Vitányi. An Introduction to Kolmogorov Complexity and Its Applications. Springer, 4.ª ed., 2019.
- T. Cover, P. Hart. Nearest neighbor pattern classification. IEEE Trans. Inf. Theory 13(1), 1967.
- S. Wehner. Analyzing worms and network traffic using compression. Journal of Computer Security 15(3), 2007.
- M. Bailey et al. Automated classification and analysis of internet malware. RAID 2007.
- E. Raff, C. Nicholas. An alternative to NCD for large sequences, Lempel-Ziv Jaccard Distance. KDD 2017.
- F. Pendlebury et al. TESSERACT: Eliminating experimental bias in malware classification across space and time. USENIX Security 2019.
- G. Engelen, V. Rimmer, W. Joosen. Troubleshooting an intrusion detection dataset: the CICIDS2017 case study. IEEE SPW 2021.
- A. Rosay, E. Cheval, F. Carlier, P. Leroux. Network intrusion detection: a comprehensive analysis of CIC-IDS2017. ICISSP 2022.
