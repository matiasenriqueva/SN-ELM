"""Figuras del análisis, con las reglas de artwork de Soft Computing (Springer).

Formato: plantilla Springer Nature a una columna, con 131 mm de ancho de texto y 194 mm de
alto. Reglas de artwork de la revista: Arial o Helvetica de 8 a 12 pt al tamaño final; sin
títulos ni leyendas de figura dentro del archivo; partes rotuladas con letras minúsculas;
líneas de al menos 0.3 pt, sin trazos ni letras tenues, y patrones además del color.

Codificación fija, para que una variante se lea igual en todas las figuras y también
en blanco y negro: modelo = color y forma del marcador (círculo ELM, cuadrado RBF,
triángulo Kriging); bucle = estilo de línea y relleno (SN continuo y relleno, SSA
discontinuo y hueco). Colores: primeros tres de la paleta de dataviz, validados.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
import pandas as pd  # noqa: E402

MM = 1 / 25.4
COLUMN_WIDTH = 84 * MM
FULL_WIDTH = 130 * MM       # ancho de texto de la plantilla a una columna (372 pt)

TEXT, GRAY, GRID = "#0b0b0b", "#52514e", "#b4b3ae"
REFERENCE_GRAY = "#8f8e88"
MODEL_COLOR = {"ELM": "#2a78d6", "RBF": "#eb6834", "KRG": "#1baf7a", None: GRAY}
MODEL_MARKER = {"ELM": "o", "RBF": "s", "KRG": "^"}
LOOP_LINE = {"ELM": "-", "SSA": "--"}
# Nombres en las figuras: el bucle ELM-NSGA-II se llama SN (surrogate NSGA-II) para no
# confundirlo con el modelo ELM. Los datos conservan los nombres internos.
DISPLAY = {"ELM-ELM": "SN-ELM", "ELM-RBF": "SN-RBF", "ELM-KRG": "SN-KRG", "NSGAII": "NSGA-II"}
LOOP_DISPLAY = {"ELM": "SN", "SSA": "SSA"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "text.color": TEXT, "axes.edgecolor": TEXT,
    "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.4, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.titlelocation": "left",
    "lines.linewidth": 1.2, "lines.markersize": 4.5, "legend.frameon": False, "legend.handlelength": 2.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})


def display(algo: str) -> str:
    variant, _, rest = algo.partition("|")
    name = DISPLAY.get(variant, variant)
    return f"{name}|{rest}" if rest else name


def part(index: int, text: str = "") -> str:
    """Rótulo de una parte de la figura: (a), (b), ..."""
    return f"({chr(97 + index)}) {text}".rstrip()


def style(algo: str) -> dict:
    variant = algo.split("|")[0]
    if variant == "NSGAII":
        return dict(color=GRAY, linestyle="-", marker="D", markerfacecolor=GRAY)
    if variant == "LHS":
        return dict(color=GRAY, linestyle=":", marker="x", markerfacecolor=GRAY)
    color = MODEL_COLOR["KRG"]
    if variant == "KRVEA":
        return dict(color=color, linestyle=":", marker="^", markerfacecolor=color)
    if variant == "KRVEA-pysamoo":
        return dict(color=color, linestyle="-.", marker="v", markerfacecolor=color)
    loop, model = variant.split("-")
    color = MODEL_COLOR[model]
    return dict(color=color, linestyle=LOOP_LINE[loop], marker=MODEL_MARKER[model],
                markerfacecolor=color if loop == "ELM" else "white", markeredgecolor=color)


def _save(fig, path: Path) -> None:
    fig.savefig(path)          # sin bbox_inches: el archivo conserva el ancho exigido
    plt.close(fig)


# ======================================================================
# Diagrama de diferencia crítica
# ======================================================================

def _cd_panel(ax, avg_ranks: pd.Series, cd: float, label: str) -> None:
    """Coordenadas en milímetros reales para que las etiquetas de 8 pt no se solapen."""
    ranks = avg_ranks.sort_values()
    names, values = [display(n) for n in ranks.index], ranks.to_numpy()
    k = len(values)
    lo, hi = 1, max(k, int(np.ceil(values.max())))
    margin, scale_mm = 25.0, COLUMN_WIDTH / MM - 2 * 25.0 - 2.0
    unit = scale_mm / (hi - lo)                      # mm por unidad de rango
    to_x = lambda r: margin + 1.0 + (r - lo) * unit  # noqa: E731
    half = int(np.ceil(k / 2))

    groups = []
    for i in range(k):
        j = max(j for j in range(i, k) if values[j] - values[i] <= cd)
        if j > i and not any(a <= i and j <= b for a, b in groups):
            groups.append((i, j))

    ax.axis("off")
    ax.set_xlim(0, COLUMN_WIDTH / MM)
    bottom = -(len(groups) * 1.0 + 2.0 + half * 3.7)
    ax.set_ylim(bottom, 11.0)
    ax.text(0, 10.6, label, ha="left", va="top")

    ax.hlines(0, to_x(lo), to_x(hi), color=TEXT, linewidth=0.8)
    for r in range(lo, hi + 1):
        ax.vlines(to_x(r), 0, 1.0, color=TEXT, linewidth=0.8)
        ax.text(to_x(r), 1.3, str(r), ha="center", va="bottom")
    ax.hlines(7.2, to_x(lo), to_x(lo) + cd * unit, color=TEXT, linewidth=1.0)
    ax.text(to_x(lo) + cd * unit + 1.5, 7.2, f"CD = {cd:.2f}", ha="left", va="center")

    y0 = -(len(groups) * 1.0 + 1.5)
    for i, (name, value) in enumerate(zip(names, values)):
        left = i < half
        row = i if left else k - 1 - i
        y = y0 - row * 3.7
        x_end = margin if left else COLUMN_WIDTH / MM - margin
        ax.plot([to_x(value), to_x(value), x_end], [0, y, y], color=GRAY, linewidth=0.6, solid_joinstyle="miter")
        ax.text(x_end + (-0.8 if left else 0.8), y, name, ha="right" if left else "left", va="center")
    for n, (i, j) in enumerate(groups):
        ax.hlines(-0.8 - n * 1.0, to_x(values[i]) - 0.5, to_x(values[j]) + 0.5, color=TEXT, linewidth=1.6)


def cd_figure(panels: dict, path: Path) -> None:
    """Un diagrama de diferencia crítica por M (Demšar, 2006). panels: {M: (rangos, CD)}."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(COLUMN_WIDTH, 40 * MM * len(panels)), squeeze=False)
    for i, (ax, (M, (ranks, cd))) in enumerate(zip(axes[:, 0], sorted(panels.items()))):
        _cd_panel(ax, ranks, cd, part(i, f"$M = {M}$"))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.02)
    _save(fig, path)


# ======================================================================
# Ablación, costo, bucle, curvas
# ======================================================================

def ablation_plot(tables: dict, metric_name: str, path: Path) -> None:
    """Rango medio por modelo, una línea por bucle; un panel por M.

    Un bucle al que le falta algún modelo se dibuja sólo con marcadores, unidos por una línea
    vertical punteada al otro bucle con el mismo modelo: una línea que saltara el modelo
    ausente sugeriría un valor que no se midió.
    """
    fig, axes = plt.subplots(1, len(tables), figsize=(COLUMN_WIDTH, 58 * MM), sharey=True, squeeze=False,
                             layout="constrained")
    for i, (ax, (M, table)) in enumerate(zip(axes[0], sorted(tables.items()))):
        x = np.arange(len(table.columns))
        values = table.to_numpy(dtype=float)
        partial = np.isnan(values).any(axis=1)
        for j in np.flatnonzero(~np.isnan(values).any(axis=0)) if partial.any() else []:
            ax.vlines(x[j], values[:, j].min(), values[:, j].max(), color=GRAY, linestyle=":", linewidth=0.8)
        for row, loop in enumerate(table.index):
            ax.plot(x, values[row], linestyle="none" if partial[row] else LOOP_LINE[loop], color=TEXT,
                    marker="o" if loop == "ELM" else "s", markerfacecolor=TEXT if loop == "ELM" else "white",
                    label=f"{LOOP_DISPLAY[loop]} loop")
        ax.set_xticks(x, ["Kriging" if c == "KRG" else c for c in table.columns])
        ax.set_xlim(-0.3, len(x) - 0.7)
        ax.set_title(part(i, f"$M = {M}$"))
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel(f"Mean rank ({metric_name}, 1 = best)")
    axes[0][0].invert_yaxis()
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    _save(fig, path)


def _place_labels(ax, points) -> None:
    """Etiqueta cada punto en la primera posición (derecha, izquierda, abajo, arriba) libre de choques."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    candidates = [((5, 0), "left", "center"), ((-5, 0), "right", "center"), ((0, -8), "center", "center"),
                  ((0, 8), "center", "center"), ((5, -7), "left", "center"), ((5, 7), "left", "center"),
                  ((-5, -7), "right", "center"), ((-5, 7), "right", "center")]
    # los marcadores también cuentan como obstáculos para las etiquetas
    centers = [ax.transData.transform((x, y)) for _, x, y in points]
    placed = [Bbox.from_extents(cx - 6, cy - 6, cx + 6, cy + 6) for cx, cy in centers]
    frame = ax.get_window_extent(renderer)
    for name, x, y in sorted(points, key=lambda p: p[2]):
        for (dx, dy), ha, va in candidates:
            text = ax.annotate(display(name), (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va)
            box = text.get_window_extent(renderer).expanded(1.02, 1.05)
            inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
            if inside and not any(box.overlaps(other) for other in placed):
                placed.append(box)
                break
            text.remove()
        else:
            ax.annotate(display(name), (x, y), xytext=(5, 0), textcoords="offset points", va="center")


def cost_quality_plot(tables: dict, metric_name: str, path: Path) -> None:
    """Rango medio frente a CPU medio por corrida (escala log); un panel por M."""
    fig, axes = plt.subplots(1, len(tables), figsize=(FULL_WIDTH, 62 * MM), sharey=True, squeeze=False,
                             layout="constrained")
    k = max(len(table) for table in tables.values())
    for i, (ax, (M, table)) in enumerate(zip(axes[0], sorted(tables.items()))):
        for algo, row in table.iterrows():
            s = style(algo)
            ax.plot(row["cpu_mean_s"], row["avg_rank"], linestyle="none", marker=s["marker"], color=s["color"],
                    markerfacecolor=s["markerfacecolor"], markersize=6, markeredgewidth=1.0)
        ax.set_xscale("log")
        lo, _ = ax.get_xlim()
        ax.set_xlim(lo, table["cpu_mean_s"].max() * 15)   # espacio para las etiquetas de los puntos más a la derecha
        ax.set_ylim(k + 0.8, 0.4)                          # eje compartido: 1 = mejor arriba
        ax.set_xlabel("Mean CPU time per run (s, log scale)")
        ax.set_title(part(i, f"$M = {M}$"))
    axes[0][0].set_ylabel(f"Mean rank ({metric_name}, 1 = best)")
    fig.canvas.draw()
    for ax, (M, table) in zip(axes[0], sorted(tables.items())):
        _place_labels(ax, [(algo, row["cpu_mean_s"], row["avg_rank"]) for algo, row in table.iterrows()])
    _save(fig, path)


def loop_effect_plot(effects: pd.DataFrame, path: Path) -> None:
    """Rango relativo de cada G dentro de cada variante (0 = mejor); un panel por M."""
    g_eff = effects[effects["factor"] == "G"]
    Ms = sorted(g_eff["M"].unique())
    fig, axes = plt.subplots(1, len(Ms), figsize=(COLUMN_WIDTH, 82 * MM), sharey=True, squeeze=False,
                             layout="constrained")
    for i, (ax, M) in enumerate(zip(axes[0], Ms)):
        h = g_eff[g_eff["M"] == M]
        levels = sorted(h["level"].unique())
        for variant, v in h.groupby("variant"):
            v = v.set_index("level").reindex(levels)
            ax.plot(range(len(levels)), v["relative_rank"], label=display(variant), **style(variant))
        ax.set_xticks(range(len(levels)), [str(int(lv)) for lv in levels])
        ax.set_xlim(-0.25, len(levels) - 0.75)
        ax.set_xlabel("$G$")
        ax.set_title(part(i, f"$M = {int(M)}$"))
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Relative rank within variant (0 = best)")
    axes[0][0].invert_yaxis()
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside upper center", ncol=3, columnspacing=1.0)
    _save(fig, path)


def anytime_plot(curves: pd.DataFrame, metric_name: str, path: Path) -> None:
    """Valor relativo al mejor final por problema, promedio sobre problemas; un panel por M."""
    Ms = sorted(curves["M"].unique())
    fig, axes = plt.subplots(1, len(Ms), figsize=(FULL_WIDTH, 80 * MM), sharey=True, squeeze=False,
                             layout="constrained")
    for i, (ax, M) in enumerate(zip(axes[0], Ms)):
        g = curves[curves["M"] == M]
        for algo, h in g.groupby("algo"):
            ax.plot(h["evals"], h["relative"], label=display(algo), markevery=4, **style(algo))
        ax.set_xlabel("True evaluations")
        ax.set_title(part(i, f"$M = {M}$"))
    axes[0][0].set_ylabel(f"Relative {metric_name} (1 = best final)")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside upper center", ncol=4, columnspacing=1.0)
    _save(fig, path)


# ======================================================================
# Esquema del método
# ======================================================================

SURROGATE_FILL = "#dce9f8"      # tinte del azul del ELM: pasos sobre el surrogado


def sn_elm_diagram(path: Path) -> None:
    """Diagrama de flujo de SN-ELM en coordenadas de milímetros (84 mm de ancho).

    Los pasos que sólo usan el ELM llevan relleno y borde discontinuo; los que evalúan el
    problema real, fondo blanco y borde continuo, para que se distingan también en gris.
    """
    width, height = COLUMN_WIDTH / MM, 112.0
    fig = plt.figure(figsize=(COLUMN_WIDTH, height * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    cx, box_w, box_h = 38.0, 60.0, 10.5

    def box(y, text, surrogate=False):
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (cx - box_w / 2, y - box_h / 2), box_w, box_h, boxstyle="round,pad=0,rounding_size=1.2",
            facecolor=SURROGATE_FILL if surrogate else "white", edgecolor=TEXT, linewidth=0.8,
            linestyle=(0, (3, 1.5)) if surrogate else "-"))
        ax.text(cx, y, text, ha="center", va="center", linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=TEXT, linewidth=0.8,
                                                            mutation_scale=7, shrinkA=0, shrinkB=0))

    steps = [
        (104.0, "Evaluate an LHS design of $N_T$ solutions\ninitialize the archives $P_E$ and $P_T$", False),
        (89.0, "Train the ELM on $P_T$\nrandom hidden layer, ridge output weights", True),
        (74.0, "Run NSGA-II on the ELM for $G$ generations\nfrom the non-dominated solutions of $P_E$", True),
        (59.0, "Select up to $N_S$ candidates of the first front\nby crowding distance, at distance $\\geq\\varepsilon$ from $P_E$", True),
        (44.0, "Evaluate the candidates on the true problem\nadd them to $P_E$ and $P_T$", False),
    ]
    for y, text, surrogate in steps:
        box(y, text, surrogate)
    for (y0, _, _), (y1, _, _) in zip(steps, steps[1:]):
        arrow(cx, y0 - box_h / 2, cx, y1 + box_h / 2)

    # decisión del presupuesto
    yd, dw, dh = 27.0, 30.0, 11.0
    ax.add_patch(matplotlib.patches.Polygon([(cx, yd + dh / 2), (cx + dw / 2, yd), (cx, yd - dh / 2), (cx - dw / 2, yd)],
                                            closed=True, facecolor="white", edgecolor=TEXT, linewidth=0.8))
    ax.text(cx, yd, "$|P_E| < N_E$ ?", ha="center", va="center")
    arrow(cx, 44.0 - box_h / 2, cx, yd + dh / 2)
    # sí: vuelve a entrenar
    x_loop = cx + box_w / 2 + 7.0
    ax.plot([cx + dw / 2, x_loop, x_loop], [yd, yd, 89.0], color=TEXT, linewidth=0.8)
    arrow(x_loop, 89.0, cx + box_w / 2, 89.0)
    ax.text(cx + dw / 2 + 1.5, yd + 1.2, "yes", ha="left", va="bottom")
    # no: termina
    yr = 10.5
    box(yr, "Return the non-dominated solutions of $P_E$")
    arrow(cx, yd - dh / 2, cx, yr + box_h / 2)
    ax.text(cx + 1.5, (yd - dh / 2 + yr + box_h / 2) / 2, "no", ha="left", va="center")

    # leyenda: tipo de paso
    for i, (label, surrogate) in enumerate((("On the ELM surrogate", True), ("True evaluations", False))):
        x = 2.0 + i * 41.0
        ax.add_patch(matplotlib.patches.Rectangle((x, 0.8), 5.0, 3.2, facecolor=SURROGATE_FILL if surrogate else "white",
                                                  edgecolor=TEXT, linewidth=0.8,
                                                  linestyle=(0, (3, 1.5)) if surrogate else "-"))
        ax.text(x + 6.5, 2.4, label, ha="left", va="center")
    _save(fig, path)


# ======================================================================
# Frentes
# ======================================================================

def fronts_plot(fronts: dict, reference: dict, problems, algos, M: int, path: Path) -> None:
    """Frente de la corrida mediana de cada variante frente al frente de referencia (M = 2)."""
    fig, axes = plt.subplots(1, len(problems), figsize=(FULL_WIDTH, 56 * MM), squeeze=False, layout="constrained")
    # los marcadores se achican en el orden de dibujo para que ninguno tape por completo a otro
    sizes = np.linspace(5.0, 2.6, max(len(algos), 2))
    for i, (ax, p) in enumerate(zip(axes[0], problems)):
        ref = reference[p]
        for algo, size in zip(algos, sizes):
            F = fronts.get((p, algo))
            if F is None:
                continue
            s = style(algo)
            ax.plot(F[:, 0], F[:, 1], linestyle="none", marker=s["marker"], color=s["color"], markersize=size,
                    markerfacecolor=s["markerfacecolor"], markeredgewidth=0.5, label=display(algo))
        ax.plot(ref[:, 0], ref[:, 1], linestyle="none", marker=".", markersize=1.0, color=TEXT,
                label="Reference front", zorder=5)
        ax.set_xlim(-0.05, ref[:, 0].max() * 1.25)
        ax.set_ylim(-0.2, ref[:, 1].max() * 1.25)
        ax.set_title(part(i, f"ZCAT{p}"))
        ax.set_xlabel("$f_1$")
    axes[0][0].set_ylabel("$f_2$")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside upper center", ncol=len(algos) + 1,
               markerscale=1.5)
    _save(fig, path)


def _fronts_figure_3d(fronts: dict, reference: dict, problems, algos, hv: dict, max_ratio: float = 1.25):
    """Una fila por algoritmo y una columna por problema; devuelve (figura, proporción fuera de los ejes)."""
    fig = plt.figure(figsize=(FULL_WIDTH, 40 * MM * len(algos) + 6 * MM))
    shown, total = 0, 0
    for r, algo in enumerate(algos):
        s = style(algo)
        for c, p in enumerate(problems):
            ax = fig.add_subplot(len(algos), len(problems), r * len(problems) + c + 1, projection="3d")
            ref = reference[p]
            limits = ref.max(axis=0) * max_ratio
            ax.scatter(*ref.T, s=0.5, color=REFERENCE_GRAY, depthshade=False, linewidths=0)
            F = fronts.get((p, algo))
            if F is not None:
                inside = F[(F <= limits).all(axis=1)]    # matplotlib no recorta en 3D
                shown, total = shown + len(inside), total + len(F)
                ax.scatter(*inside.T, s=5, color=s["color"], marker=s["marker"], depthshade=False, linewidths=0)
                ax.text2D(0.0, 0.98, f"{part(r * len(problems) + c)} HV = {hv[(p, algo)]:.3f}",
                          transform=ax.transAxes, ha="left", va="top")
            ax.set_xlim(0, limits[0]); ax.set_ylim(0, limits[1]); ax.set_zlim(0, limits[2])
            ax.view_init(elev=25, azim=45)
            ax.set_box_aspect((1, 1, 1), zoom=0.85)
            ax.set_xticks([1]); ax.set_yticks([4]); ax.set_zticks([9])
            ax.tick_params(pad=-4, length=0)
            ax.set_xlabel("$f_1$", labelpad=-10)
            ax.set_ylabel("$f_2$", labelpad=-10)
            ax.set_zlabel("$f_3$", labelpad=-10)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_rotate_label(False)
                axis.set_pane_color((1, 1, 1, 0))
                axis._axinfo["grid"].update(color=GRID, linewidth=0.4)
            if r == 0:
                ax.set_title(f"ZCAT{p}", loc="center", pad=4)
        fig.text(0.004, 1 - (r + 0.5) * 0.965 / len(algos) - 0.035, display(algo), rotation=90, va="center", ha="left")
    fig.subplots_adjust(left=0.035, right=1, top=0.965, bottom=0.0, wspace=0.02, hspace=0.02)
    return fig, (1 - shown / total if total else 0.0)


def fronts_plot_3d(fronts: dict, reference: dict, problems, algos, hv: dict, path: Path) -> float:
    fig, outside = _fronts_figure_3d(fronts, reference, problems, algos, hv)
    _save(fig, path)
    return outside


def fronts_pages_3d(fronts: dict, reference: dict, problems, algos, hv: dict, path: Path,
                    per_page: int = 4, front_matter: str = "") -> None:
    """PDF de varias páginas con todos los problemas (Online Resource)."""
    with PdfPages(path) as pdf:
        if front_matter:
            fig = plt.figure(figsize=(FULL_WIDTH, 60 * MM))
            fig.text(0.02, 0.95, front_matter, va="top", ha="left", wrap=True)
            pdf.savefig(fig)
            plt.close(fig)
        for start in range(0, len(problems), per_page):
            fig, _ = _fronts_figure_3d(fronts, reference, problems[start:start + per_page], algos, hv)
            pdf.savefig(fig)
            plt.close(fig)
