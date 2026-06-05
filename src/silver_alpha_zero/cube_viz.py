from pathlib import Path

import magiccube
from magiccube import Color, Face

from mcst import Action

# Standard Rubik's Cube sticker colors.
_FACE_COLORS = {
    Color.R: "#B71234",
    Color.O: "#FF5800",
    Color.W: "#FFFFFF",
    Color.Y: "#FFD500",
    Color.B: "#0046AD",
    Color.G: "#009B48",
}

# Position of each face on the unfolded "cross" net, in units of `size` cells:
#       U
#    L  F  R  B
#       D
_NET_LAYOUT = {
    Face.U: (1, 0),
    Face.L: (0, 1),
    Face.F: (1, 1),
    Face.R: (2, 1),
    Face.B: (3, 1),
    Face.D: (1, 2),
}


def _draw_cube(ax, cube: magiccube.Cube, title: str = "") -> None:
    from matplotlib.patches import Rectangle

    ax.clear()
    size = cube.size
    width, height = 4 * size, 3 * size
    faces = cube.get_all_faces()
    for face, (face_col, face_row) in _NET_LAYOUT.items():
        for row, colors in enumerate(faces[face]):
            for col, color in enumerate(colors):
                x = face_col * size + col
                y = height - (face_row * size + row) - 1  # flip so row 0 is on top
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        1,
                        1,
                        facecolor=_FACE_COLORS[color],
                        edgecolor="black",
                        linewidth=1.5,
                    )
                )
    ax.set_xlim(-0.3, width + 0.3)
    ax.set_ylim(-0.3, height + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14)


def _cube_quads(cube: magiccube.Cube) -> tuple[list, list]:
    """Build (polygons, facecolors) for every exposed sticker in 3D.

    magiccube indexes pieces by (x, y, z) with per-axis sticker colors:
    axis 0 = x (L/R), axis 1 = y (D/U), axis 2 = z (F/B). We map cube coords to
    plot coords as (x, z, y) so that y is the vertical axis.
    """
    n = cube.size
    polys: list = []
    colors: list = []

    def p(x: int, y: int, z: int) -> tuple[int, int, int]:
        return (x, z, y)  # vertical axis = y

    for (x, y, z), piece in cube.get_all_pieces().items():
        if x in (0, n - 1) and (color := piece.get_piece_color(0)) is not None:
            px = 0 if x == 0 else n
            polys.append([p(px, y, z), p(px, y + 1, z), p(px, y + 1, z + 1), p(px, y, z + 1)])
            colors.append(_FACE_COLORS[color])
        if y in (0, n - 1) and (color := piece.get_piece_color(1)) is not None:
            py = 0 if y == 0 else n
            polys.append([p(x, py, z), p(x + 1, py, z), p(x + 1, py, z + 1), p(x, py, z + 1)])
            colors.append(_FACE_COLORS[color])
        if z in (0, n - 1) and (color := piece.get_piece_color(2)) is not None:
            pz = 0 if z == 0 else n
            polys.append([p(x, y, pz), p(x + 1, y, pz), p(x + 1, y + 1, pz), p(x, y + 1, pz)])
            colors.append(_FACE_COLORS[color])
    return polys, colors


def _draw_cube_3d(ax, cube: magiccube.Cube, title: str = "") -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax.clear()
    n = cube.size
    polys, colors = _cube_quads(cube)
    ax.add_collection3d(
        Poly3DCollection(polys, facecolors=colors, edgecolor="black", linewidths=1.2)
    )
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-52)
    ax.set_axis_off()
    ax.set_title(title, fontsize=14)


def save_trajectory_gif(
    cube: magiccube.Cube,
    actions: list[Action],
    path: str | Path,
    fps: int = 2,
    mode: str = "2d",
) -> Path:
    """Apply `actions` to a copy of `cube` and render each step into a GIF.

    `mode` is "2d" (unfolded net) or "3d" (rotatable cube projection).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    work = magiccube.Cube(cube.size, state=cube.get())
    frames = [(work.get(), "scrambled")]
    for i, action in enumerate(actions, start=1):
        work.rotate(str(action))
        solved = " (solved)" if work.is_done() else ""
        frames.append((work.get(), f"{i}/{len(actions)}: {action}{solved}"))

    if mode == "3d":
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        draw = _draw_cube_3d
    else:
        fig, ax = plt.subplots(figsize=(2 * cube.size, 1.7 * cube.size))
        draw = _draw_cube

    def update(frame_idx: int) -> None:
        state_str, title = frames[frame_idx]
        draw(ax, magiccube.Cube(cube.size, state=state_str), title)

    # Hold the final frame for a moment by repeating it.
    order = list(range(len(frames))) + [len(frames) - 1] * fps
    anim = FuncAnimation(fig, update, frames=order, interval=1000 / fps)
    path = Path(path)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
