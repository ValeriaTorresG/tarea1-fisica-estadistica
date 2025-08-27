# Simulación númerica de la marcha aleatoria

### Valeria Torres Gomez - 202110363
---

Simula una **marcha aleatoria 1D** y verifica dos resultados:

1. Para \(N\) muy grande y \(p=1/2\), la **posición final** \(X_N\) se aproxima a una **gaussiana** (Teorema Central del Límite).
2. El **crecimiento del segundo momento** \(\langle x^2\rangle\) es **lineal** en \(N\), lo que permite estimar la **constante de difusión** \(D\) a partir de la pendiente.

## Estructura
- `src/random_walk.py`: contiene las funciones principales.
- `main.py`: ejecuta el modulo principal para realizar los cálculos y generar gráficas.

## Ejecución

Parametros:

- `N`: número de pasos por caminata.
- `M` (def: 20000): número de corridas independientes.
- `a` (def: 1.0): tamaño de paso.
- `p` (def: 0.5): probabilidad de paso a la derecha (+a).
- `dt` (def: 1.0): tiempo por paso.
- `bins` (def: 30): bins del histograma.
- `n-tray` (def: 5): cantidad de trayectorias a graficar.
- `save-prefix` (def: ./plots/): prefijo de salida para los archivos.
- `seed` (def: 42): semilla para Numpy.

```
python main.py --N 500 --M 20000 --a 1.0 --p 0.5 \
               --dt 1.0 --bins 30 --n-tray 5
```
