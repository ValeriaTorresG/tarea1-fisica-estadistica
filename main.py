import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.random_walk import *

def construir_lista(N, k=7):
    """
    Construye una lista de valores de N a partir de N
    """
    return np.linspace(10, N, k, dtype=int).tolist()

def grafico_histograma(x, N, a, bins, savepath):
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins, density=True, alpha=0.5)

    grilla = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ax.plot(grilla, gaussiana_tcl(grilla, N, a), color='navy', label='Fit Gaussiano')

    ax.set_xlabel('Pos final')
    ax.set_ylabel('Densidad de probabilidad')
    ax.set_title(f'Marcha aleatoria con $N$={N}')
    ax.legend()

    fig.savefig(savepath, dpi=360)
    plt.close(fig)


def grafico_linealidad(Ns, x2_medios, m, b, D_hat, a, dt, savepath):
    fig, ax = plt.subplots()
    ax.plot(Ns, x2_medios, 'o', label=r'Datos $\langle x^2 \rangle$')
    ax.plot(Ns, m*np.array(Ns) + b, '-', label=f'Ajuste (m={m:.3f}, D={D_hat:.3f})')

    ax.set_xlabel('N (número de pasos)')
    ax.set_ylabel(r'$\langle x^2 \rangle$')
    ax.set_title(r'Crecimiento de $\langle x^2 \rangle$ con N')
    ax.legend()

    fig.savefig(savepath, dpi=360)
    plt.close(fig)

def grafico_trayectorias(paths, N, a, p, savepath, max_plot):
    fig, ax = plt.subplots()
    M = paths.shape[0]
    for j in range(min(M, max_plot)):
        ax.plot(range(1, N + 1), paths[j], alpha=0.8)
    ax.set_xlabel('Número de pasos')
    ax.set_ylabel('Posición')
    ax.set_title(f'{M} trayectorias de {N} pasos (a={a}, p={p})')

    fig.savefig(savepath, dpi=360)
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, required=True, help='Número de pasos por caminata')
    parser.add_argument('--M', type=int, default=20000, help='Número de corridas independientes')
    parser.add_argument('--a', type=float, default=1.0, help='Tamaño de paso')
    parser.add_argument('--p', type=float, default=0.5, help='Probabilidad de paso a la derecha')
    parser.add_argument('--dt', type=float, default=1.0, help='Tiempo por paso delta t')
    parser.add_argument('--bins', type=int, default=30, help='Bins del histograma')
    parser.add_argument('--n-tray', type=int, default=5, help='Cantidad de trayectorias a graficar')
    parser.add_argument('--save-prefix', type=str, default='./plots/', help='Prefijo de ruta para guardar las graficas')
    parser.add_argument('--seed', type=int, default=42, help='Semilla para el generador aleatorio de Numpy')
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x = muestrear_puntos_finales(args.N, args.M, args.a, args.p, rng)
    media, momento2, _ = estimadores_muestrales(x)
    print(f'---> Histograma: <x> = {media:.6f}, <x^2> ≈ {momento2:.6f}, teórico (p=1/2): a^2 N = {args.a*args.a*args.N:.6f}')

    save_hist = f'{args.save_prefix}_hist.png'
    grafico_histograma(x, args.N, args.a, args.bins, savepath=save_hist)

    N_list = construir_lista(args.N)
    x2_medios = []
    for N in N_list:
        xs = muestrear_puntos_finales(N, args.M, args.a, args.p, rng)
        _, mu2, _ = estimadores_muestrales(xs)
        x2_medios.append(mu2)

    m, b = ajuste_lineal(N_list, x2_medios)
    D_hat = difusion_desde_pendiente(m, args.dt)
    print(f'---> linealidad: N_list = {N_list}')
    print(f'm ≈ {m:.6f}, b ≈ {b:.6f}, D = {D_hat:.6f}' f'(teórico p=1/2: {args.a*args.a/(2*args.dt):.6f})')

    save_lin = f'{args.save_prefix}_linealidad.png'
    grafico_linealidad(N_list, x2_medios, m, b, D_hat, args.a, args.dt, save_lin)

    paths, finales = tray_completas(args.n_tray, args.N, args.a, args.p, rng)
    save_tray = f'{args.save_prefix}_tray.png'
    grafico_trayectorias(paths, args.N, args.a, args.p, save_tray, args.n_tray)

if __name__ == '__main__':
    main()