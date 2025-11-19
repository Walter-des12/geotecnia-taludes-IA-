import numpy as np

def peso_especifico_medio(γ):
    return γ

def fs_fellenius(xc, yc, R, Hslice, γ, c, phi_deg, ru=0.0, n_slices=30):
    phi = np.radians(phi_deg)
    Δx = (2*R) / n_slices

    sum_resisting = 0.0
    sum_driving = 0.0

    for i in range(n_slices):
        x = xc - R + Δx/2 + i*Δx
        y = yc + np.sqrt(max(R**2 - (x-xc)**2, 0))

        area_slice = Hslice * Δx
        W = γ * area_slice

        alpha = np.arctan2((x - xc), np.sqrt(max(R**2 - (x-xc)**2, 0)))

        N = W * np.cos(alpha)
        U = ru * N
        N_eff = N - U

        S_res = c + N_eff * np.tan(phi)

        sum_resisting += S_res
        sum_driving += W * np.sin(alpha)

    FS = sum_resisting / sum_driving if sum_driving != 0 else np.inf
    return FS


def fs_bishop(xc, yc, R, Hslice, γ, c, phi_deg, ru=0.0, n_slices=30, max_iter=100, tol=1e-5):
    phi = np.radians(phi_deg)
    Δx = (2*R) / n_slices
    FS = 1.0

    for _ in range(max_iter):
        sum_num = 0.0
        sum_den = 0.0

        for i in range(n_slices):
            x = xc - R + Δx/2 + i*Δx
            alpha = np.arctan2((x-xc), np.sqrt(max(R**2 - (x-xc)**2, 0)))
            area_slice = Hslice * Δx
            W = γ * area_slice
            m = 1 - ru

            num_i = c + m * W * np.tan(phi)
            num_i *= Δx

            den_i = W * np.sin(alpha) + (num_i * np.tan(phi)) / FS

            sum_num += num_i
            sum_den += den_i

        FS_new = sum_num / sum_den if sum_den != 0 else np.inf
        if abs(FS_new - FS) < tol:
            return FS_new

        FS = FS_new

    return FS


def calcular_fs(metodo, xc, yc, R, H, γ, c, phi, ru=0.0, n=30):
    metodo = metodo.lower()

    if metodo == "fellenius":
        return fs_fellenius(xc, yc, R, H, γ, c, phi, ru, n)

    elif metodo == "bishop":
        return fs_bishop(xc, yc, R, H, γ, c, phi, ru, n)

    else:
        raise ValueError("Método desconocido. Usa 'fellenius' o 'bishop'.")
