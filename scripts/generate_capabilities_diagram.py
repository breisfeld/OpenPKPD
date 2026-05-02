from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import cairosvg
from cli_anything.excalidraw.core import Scene
from cli_anything.excalidraw.render import export_svg


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "temp"
SCENE_PATH = OUT_DIR / "openpkpd_capabilities.excalidraw"
SVG_PATH = OUT_DIR / "openpkpd_capabilities.svg"
PNG_PATH = OUT_DIR / "openpkpd_capabilities.png"
SVG_NS = {"svg": "http://www.w3.org/2000/svg"}
FONT_FAMILY = "'Helvetica Neue', Helvetica, Arial, sans-serif"

TITLE_TEXTS = {
    "OpenPKPD",
    "Model setup and execution",
    "Estimation and inference",
    "PK/PD model families",
    "Simulation and analysis",
    "Outputs and integrations",
    "Examples and validation",
}


def add_card(
    scene: Scene,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: list[str],
    stroke: str,
    bg: str,
    roughness: float = 1.4,
) -> str:
    card_id = scene.add_shape(
        "rectangle",
        x=x,
        y=y,
        width=width,
        height=height,
        label=None,
        strokeColor=stroke,
        backgroundColor=bg,
        fillStyle="solid",
        strokeWidth=2.2,
        roughness=roughness,
        radius=18,
    )
    title_y = y + 26
    scene.add_text(
        title,
        x=x + 18,
        y=title_y,
        width=width - 36,
        font_size=20,
        textAlign="center",
        strokeColor="#1e1e1e",
        roughness=1.1,
        fontFamily=1,
    )
    body_y = y + 80
    for idx, line in enumerate(lines):
        scene.add_text(
            line,
            x=x + 18,
            y=body_y + idx * 32,
            width=width - 36,
            font_size=18,
            textAlign="center",
            strokeColor="#1e1e1e",
            roughness=1.1,
            fontFamily=1,
        )
    return card_id


def add_annotation(
    scene: Scene,
    *,
    text: str,
    x: float,
    y: float,
    width: float = 260,
) -> None:
    scene.add_text(
        text,
        x=x,
        y=y,
        width=width,
        font_size=18,
        strokeColor="#4b5563",
        roughness=1.2,
        fontFamily=1,
    )


def add_connector(
    scene: Scene,
    *,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    color: str = "#64748b",
) -> str:
    return scene.add_arrow(
        x=start_x,
        y=start_y,
        width=end_x - start_x,
        height=end_y - start_y,
        strokeColor=color,
        strokeWidth=2.0,
        roughness=1.1,
        end_arrowhead="triangle",
    )


def build_scene() -> Scene:
    scene = Scene()
    scene.set_background("#f7f4ee")

    title_id = scene.add_flowchart_node(
        "OpenPKPD",
        x=480,
        y=34,
        width=320,
        height=90,
        shape="rectangle",
        strokeColor="#1f2937",
        backgroundColor="#fff7d6",
        fillStyle="solid",
        strokeWidth=2.8,
        roughness=1.0,
        radius=22,
        padding=22,
        fontSize=32,
        fontFamily=1,
    )
    add_annotation(
        scene,
        text="Population PK/PD analysis from model setup to validation and reporting",
        x=332,
        y=6,
        width=560,
    )

    cards = {
        "setup": add_card(
            scene,
            x=70,
            y=230,
            width=350,
            height=210,
            title="Model setup and execution",
            lines=[
                "ModelBuilder API and CLI",
                "NONMEM-style control streams",
                "Desktop GUI workflows",
                "PK, DES, ERROR compilation",
            ],
            stroke="#9a3412",
            bg="#ffedd5",
        ),
        "estimation": add_card(
            scene,
            x=445,
            y=230,
            width=350,
            height=210,
            title="Estimation and inference",
            lines=[
                "FO, FOCE, FOCEI, Laplacian",
                "SAEM, IMP, IMPMAP",
                "Bayes(Laplace) and NUTS",
                "Nonparametric estimation",
            ],
            stroke="#7c2d12",
            bg="#fde7df",
        ),
        "families": add_card(
            scene,
            x=820,
            y=230,
            width=350,
            height=210,
            title="PK/PD model families",
            lines=[
                "Analytical ADVAN1-5, 7, 11, 12",
                "ODE ADVAN6, 8, 10, 13",
                "DDE via ADVAN16",
                "PK/PD, TTE, count, TMDD, PBPK",
            ],
            stroke="#1d4ed8",
            bg="#dbeafe",
        ),
        "analysis": add_card(
            scene,
            x=70,
            y=505,
            width=350,
            height=210,
            title="Simulation and analysis",
            lines=[
                "Replicate simulation",
                "VPC, pcVPC, NPC, NPDE",
                "NCA and sparse NCA",
                "Bootstrap and design",
            ],
            stroke="#065f46",
            bg="#d1fae5",
        ),
        "outputs": add_card(
            scene,
            x=445,
            y=505,
            width=350,
            height=210,
            title="Outputs and integrations",
            lines=[
                "NONMEM-compatible outputs",
                "HTML reports and PDF export",
                "SBML import",
                "Parallel execution",
            ],
            stroke="#6b21a8",
            bg="#f3e8ff",
        ),
        "validation": add_card(
            scene,
            x=820,
            y=505,
            width=350,
            height=210,
            title="Examples and validation",
            lines=[
                "34 shipped examples",
                "Marimo notebooks",
                "Unit, integration, regression tests",
                "Cross-tool validation",
            ],
            stroke="#92400e",
            bg="#fef3c7",
        ),
    }

    top_centers = {
        "setup": (245, 230),
        "estimation": (620, 230),
        "families": (995, 230),
    }
    bottom_centers = {
        "setup": (245, 505),
        "estimation": (620, 505),
        "families": (995, 505),
    }

    add_connector(scene, start_x=600, start_y=124, end_x=245, end_y=230)
    add_connector(scene, start_x=640, start_y=124, end_x=620, end_y=230)
    add_connector(scene, start_x=680, start_y=124, end_x=995, end_y=230)

    add_connector(
        scene,
        start_x=top_centers["setup"][0],
        start_y=440,
        end_x=bottom_centers["setup"][0],
        end_y=505,
        color="#94a3b8",
    )
    add_connector(
        scene,
        start_x=top_centers["estimation"][0],
        start_y=440,
        end_x=bottom_centers["estimation"][0],
        end_y=505,
        color="#94a3b8",
    )
    add_connector(
        scene,
        start_x=top_centers["families"][0],
        start_y=440,
        end_x=bottom_centers["families"][0],
        end_y=505,
        color="#94a3b8",
    )

    add_annotation(
        scene,
        text="Built for regular PK/PD workflows, advanced methods, and transparent validation.",
        x=335,
        y=760,
        width=540,
    )
    return scene


def postprocess_svg(path: Path) -> None:
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(path)
    root = tree.getroot()
    for text in root.findall(".//svg:text", SVG_NS):
        text.set("font-family", FONT_FAMILY)
        if (text.text or "").strip() in TITLE_TEXTS:
            text.set("font-weight", "700")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def render_png_from_svg(svg_path: Path, png_path: Path, *, scale: float = 2.5) -> None:
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        scale=scale,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    scene.save(SCENE_PATH)
    export_svg(scene, SVG_PATH)
    postprocess_svg(SVG_PATH)
    render_png_from_svg(SVG_PATH, PNG_PATH, scale=2.5)
    print(f"Wrote {SCENE_PATH}")
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {PNG_PATH}")


if __name__ == "__main__":
    main()
