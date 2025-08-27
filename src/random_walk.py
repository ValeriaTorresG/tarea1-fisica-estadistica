import numpy as np

def muestrear_puntos_finales(N, M, a, p, rng):
    """
    Genera las posiciones finales de una marcha aleatoria 1D después de N pasos.

    Cada paso tiene longitud fija a y probabilidad p de ser hacia la derecha (+a),
    y (1-p) hacia la izquierda (-a).

    Args:
        N (int): Número de pasos en cada marcha aleatoria.
        M (int): Número de corridas independientes (tamaño de la muestra).
        a (float): Longitud de cada paso.
        p (float): Probabilidad de dar un paso hacia la derecha (+a).
        rng (np.random.Generator): Generador de números aleatorios de NumPy.

    Returns:
        np.ndarray: Arreglo de tamaño (M,) con las posiciones finales x_j.
    """
    K = rng.binomial(N, p, size=M)
    x = a * (2 * K - N)
    return x

def estimadores_muestrales(x):
    """
    Calcula los estimadores muestrales de una muestra de posiciones.

    Args:
        x (array_like): Arreglo unidimensional con las posiciones finales de las corridas.

    Returns:
        tuple:
            - media (float): Valor medio <x>.
            - momento2 (float): Segundo momento <x^2>.
            - varianza (float): Varianza de la muestra.
    """
    x = np.array(x)
    media = x.mean()
    momento2 = (x**2).mean()
    varianza = x.var(ddof=0)
    return media, momento2, varianza

def gaussiana_tcl(xgrid, N, a):
    """
    Calcula la densidad gaussiana aproximada (Teorema Central del Límite)
    para la distribución de la marcha aleatoria 1D no sesgada.

    Args:
        xgrid (array_like): Puntos del eje x donde evaluar la densidad.
        N (int): Número de pasos de la caminata.
        a (float): Longitud de cada paso.

    Returns:
        np.ndarray: Valores de la densidad gaussiana evaluada en xgrid.
    """
    sigma2 = (a * a) * N
    return (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-xgrid**2 / (2 * sigma2))

def ajuste_lineal(Ns, ys):
    """
    Ajusta una recta y ~ m*N + b a los datos (OLS, mínimos cuadrados).

    Args:
        Ns (array_like): Valores de N (eje x).
        ys (array_like): Valores de <x^2>(N) (eje y).

    Returns:
        tuple:
            - m (float): Pendiente del ajuste lineal.
            - b (float): Intercepto del ajuste lineal.
    """
    Ns = np.array(Ns)
    ys = np.array(ys)
    Nbar = Ns.mean()
    ybar = ys.mean()
    m = ((Ns - Nbar) * (ys - ybar)).sum() / ((Ns - Nbar) ** 2).sum()
    b = ybar - m * Nbar
    return m, b

def difusion_desde_pendiente(m, dt):
    """
    Calcula la constante de difusión D a partir de la pendiente m del ajuste.

    Relación: d<x^2>/dN = 2 D dt → D = m / (2 dt).

    Args:
        m (float): Pendiente de la recta <x^2>(N).
        dt (float): Duración temporal de cada paso.

    Returns:
        float: Estimación de la constante de difusión D.
    """
    return m / (2.0 * dt)

def histograma(N, M, a, p, rng):
    """
    Genera un conjunto de posiciones finales y sus estadísticos para un N fijo.

    Args:
        N (int): Número de pasos de cada caminata.
        M (int): Número de corridas independientes.
        a (float): Longitud de cada paso.
        p (float): Probabilidad de dar un paso hacia la derecha.
        rng (np.random.Generator): Generador aleatorio.

    Returns:
        tuple:
            - x (np.ndarray): Muestra de posiciones finales (M,).
            - media (float): Estimador <x>.
            - momento2 (float): Estimador <x^2>.
    """
    x = muestrear_puntos_finales(N, M, a, p, rng)
    media, momento2, _ = estimadores_muestrales(x)
    return x, media, momento2

def linealidad(N_lista, M, a, p, dt, rng):
    """
    Calcula <x^2>(N) para varios valores de N, ajusta la relación lineal y estima D.

    Args:
        N_lista (list[int]): Lista de valores de N a simular.
        M (int): Número de corridas independientes para cada N.
        a (float): Longitud de cada paso.
        p (float): Probabilidad de dar un paso hacia la derecha.
        dt (float): Duración temporal de cada paso.
        rng (np.random.Generator): Generador aleatorio.

    Returns:
        tuple:
            - Ns (np.ndarray): Valores de N simulados.
            - x2_medios (np.ndarray): Valores medios <x^2> para cada N.
            - m (float): Pendiente del ajuste lineal.
            - b (float): Intercepto del ajuste lineal.
            - D_hat (float): Estimación de la constante de difusión.
    """
    Ns = []
    x2_medios = []

    for N in N_lista:
        x = muestrear_puntos_finales(N, M, a, p, rng)
        _, momento2, _ = estimadores_muestrales(x)
        Ns.append(N)
        x2_medios.append(momento2)

    m, b = ajuste_lineal(Ns, x2_medios)
    D_hat = difusion_desde_pendiente(m, dt=dt)
    return np.array(Ns), np.array(x2_medios), m, b, D_hat

def tray_completas(M, N, a, p, rng):
    """
    Genera trayectorias completas de la marcha aleatoria 1D.

    Args:
        M (int): Número de trayectorias a simular.
        N (int): Número de pasos en cada trayectoria.
        a (float): Longitud de cada paso.
        p (float): Probabilidad de paso hacia la derecha (+a).
        rng (np.random.Generator): Generador aleatorio.

    Returns:
        tuple:
            - paths (np.ndarray): Arreglo (M, N) con las posiciones intermedias
              en cada paso de cada trayectoria.
            - x_final (np.ndarray): Arreglo (M,) con las posiciones finales
              tras N pasos.
    """
    pasos = np.where(rng.random((M, N)) < p, 1, -1).astype(np.int8)
    paths = (a * pasos).cumsum(axis=1)
    x_final = paths[:, -1]
    return paths, x_final