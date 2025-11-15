from pyslope import Slope, Material, Udl, LineLoad

def calcular_fs_pyslope(
    H, beta, peso, cohesion, phi, ru,
    udl_magnitude=None,
    ll_magnitude=None
):
    """
    Wrapper simple para ejecutar un análisis de pySlope.
    Devuelve:
        - FS crítico
        - figura crítica
    """

    # 1. Crear la pendiente
    s = Slope(height=H, angle=beta, length=None)

    # 2. Crear material único (puedes luego agregar estratos)
    m1 = Material(
        unit_weight=peso,
        friction_angle=phi,
        cohesion=cohesion,
        depth_to_bottom=H  # capa completa
    )
    s.set_materials(m1)

    # 3. Definir agua (ru)
    # pySlope usa profundidad desde la CRESTA
    if ru > 0:
        # profundidad equivalente según ru
        water_depth = H * (1 - ru)
        s.set_water_table(water_depth)

    # 4. Cargas opcionales
    if udl_magnitude:
        s.set_udls(Udl(magnitude=udl_magnitude))

    if ll_magnitude:
        s.set_lls(LineLoad(magnitude=ll_magnitude))

    # 5. Límites de análisis automáticos
    left = s.get_top_coordinates()[0] - H
    right = s.get_bottom_coordinates()[0] + H
    s.set_analysis_limits(left, right)

    # 6. Opciones de iteración
    s.update_analysis_options(slices=50, iterations=2000)

    # 7. Ejecutar análisis
    s.analyse_slope()

    # 8. Obtener FS crítico
    fs = s.get_min_FOS()

    # 9. Obtener figura
    fig = s.plot_critical()

    return fs, fig
