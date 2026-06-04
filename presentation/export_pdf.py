from __future__ import annotations

import argparse
import functools
import socket
import sys
import threading
import zlib
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


SLIDE_WIDTH = 1920
SLIDE_HEIGHT = 1080
DEFAULT_SCALE = 2.0
PDF_WIDTH = 1920
PDF_HEIGHT = 1080


@dataclass
class PdfLink:
    href: str
    x: float
    y: float
    width: float
    height: float


@dataclass
class SlideExport:
    path: Path
    links: list[PdfLink]


class QuietStaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_static_server(root: Path) -> ThreadingHTTPServer:
    handler = functools.partial(QuietStaticHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", find_free_port()), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_out = project_root / "presentation" / "output" / "race-strategist-deck.pdf"
    default_slides_dir = project_root / "presentation" / "output" / "slides"

    parser = argparse.ArgumentParser(
        description="Export the HTML presentation by screenshotting each slide and combining them into a PDF."
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=project_root / "presentation" / "index.html",
        help="Path to the deck HTML file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--slides-dir",
        type=Path,
        default=default_slides_dir,
        help="Directory for per-slide PNG screenshots.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help="Screenshot pixel scale. Use 2 for 3840x2160 output or 3 for 5760x3240 output.",
    )
    return parser.parse_args()


def import_dependencies():
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Missing dependency: pillow. Run `pip install -r requirements.txt`.") from exc

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("Missing dependency: playwright. Run `pip install -r requirements.txt`.") from exc

    return Image, PlaywrightError, sync_playwright


def clean_old_screenshots(slides_dir: Path) -> None:
    slides_dir.mkdir(parents=True, exist_ok=True)
    for path in slides_dir.glob("slide-*.png"):
        path.unlink()


def validate_export_options(scale: float) -> None:
    if scale < 1:
        raise SystemExit("--scale must be at least 1.")


def export_slide_screenshots(html_path: Path, slides_dir: Path, scale: float) -> list[SlideExport]:
    _image, PlaywrightError, sync_playwright = import_dependencies()
    project_root = Path(__file__).resolve().parents[1]
    html_path = html_path.resolve()

    if not html_path.exists():
        raise SystemExit(f"Deck HTML not found: {html_path}")

    clean_old_screenshots(slides_dir)
    server: ThreadingHTTPServer | None = None
    try:
        server = start_static_server(project_root)
        host, port = server.server_address
        html_url = f"http://{host}:{port}/{html_path.relative_to(project_root).as_posix()}"
    except PermissionError:
        html_url = html_path.as_uri()

    export_css = f"""
      :root {{ --slide-scale: 1 !important; }}
      html,
      body {{
        width: {SLIDE_WIDTH}px !important;
        margin: 0 !important;
        padding: 0 !important;
        background: #111 !important;
        overflow: visible !important;
      }}
      .deck {{
        margin: 0 !important;
        padding: 0 !important;
      }}
      .slide-outer {{
        width: {SLIDE_WIDTH}px !important;
        height: {SLIDE_HEIGHT}px !important;
        margin: 0 !important;
      }}
      .slide {{
        width: {SLIDE_WIDTH}px !important;
        height: {SLIDE_HEIGHT}px !important;
        transform: none !important;
      }}
      .print-button {{
        display: none !important;
      }}
    """

    screenshots: list[SlideExport] = []
    try:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except PlaywrightError as exc:
                raise SystemExit(
                    "Playwright Chromium is not installed. Run `python -m playwright install chromium`."
                ) from exc

            with browser:
                page = browser.new_page(
                    viewport={"width": SLIDE_WIDTH, "height": SLIDE_HEIGHT},
                    device_scale_factor=scale,
                )
                page.goto(html_url, wait_until="networkidle")
                page.add_style_tag(content=export_css)
                page.evaluate("() => document.fonts ? document.fonts.ready : true")
                page.wait_for_function(
                    "() => Array.from(document.images).every((img) => img.complete && img.naturalWidth > 0)",
                    timeout=15000,
                )

                slide_count = page.locator(".slide").count()
                if slide_count == 0:
                    raise SystemExit("No `.slide` elements found in the deck.")

                for index in range(slide_count):
                    slide = page.locator(".slide").nth(index)
                    slide.scroll_into_view_if_needed()
                    path = slides_dir / f"slide-{index + 1:02d}.png"
                    raw_links = slide.evaluate(
                        """(slide) => {
                          const slideRect = slide.getBoundingClientRect();
                          return Array.from(slide.querySelectorAll('a[href]'))
                            .map((anchor) => {
                              const rect = anchor.getBoundingClientRect();
                              return {
                                href: anchor.href,
                                x: rect.left - slideRect.left,
                                y: rect.top - slideRect.top,
                                width: rect.width,
                                height: rect.height
                              };
                            })
                            .filter((link) => link.href && link.width > 0 && link.height > 0);
                        }"""
                    )
                    slide.screenshot(path=str(path), animations="disabled")
                    links = [
                        PdfLink(
                            href=str(link["href"]),
                            x=float(link["x"]),
                            y=float(link["y"]),
                            width=float(link["width"]),
                            height=float(link["height"]),
                        )
                        for link in raw_links
                    ]
                    screenshots.append(SlideExport(path=path, links=links))
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()

    return screenshots


def pdf_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def pdf_hex_string(value: str) -> str:
    return f"<{value.encode('utf-8').hex().upper()}>"


def link_rect(link: PdfLink) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(PDF_WIDTH, link.x))
    x2 = max(0.0, min(PDF_WIDTH, link.x + link.width))
    y1 = max(0.0, min(PDF_HEIGHT, PDF_HEIGHT - link.y - link.height))
    y2 = max(0.0, min(PDF_HEIGHT, PDF_HEIGHT - link.y))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def link_annotation_object(link: PdfLink) -> bytes | None:
    rect = link_rect(link)
    if rect is None:
        return None

    x1, y1, x2, y2 = rect
    rect_text = " ".join(pdf_number(value) for value in (x1, y1, x2, y2))
    return (
        b"<<\n"
        b"/Type /Annot\n"
        b"/Subtype /Link\n"
        + f"/Rect [{rect_text}]\n".encode("ascii")
        + b"/Border [0 0 0]\n"
        + f"/A << /S /URI /URI {pdf_hex_string(link.href)} >>\n".encode("ascii")
        + b">>"
    )


def write_pdf(slides: list[SlideExport], out_path: Path) -> None:
    Image, _playwright_error, _sync_playwright = import_dependencies()
    if not slides:
        raise SystemExit("No screenshots were generated; PDF export aborted.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    objects: list[bytes] = []
    page_object_ids: list[int] = []

    for slide in slides:
        with Image.open(slide.path) as image:
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            image_data = zlib.compress(rgb_image.tobytes(), level=9)

        image_object_id = len(objects) + 1
        objects.append(
            b"<<\n"
            b"/Type /XObject\n"
            b"/Subtype /Image\n"
            + f"/Width {width}\n/Height {height}\n".encode("ascii")
            + b"/ColorSpace /DeviceRGB\n"
            b"/BitsPerComponent 8\n"
            b"/Filter /FlateDecode\n"
            + f"/Length {len(image_data)}\n".encode("ascii")
            + b">>\nstream\n"
            + image_data
            + b"\nendstream"
        )

        content = f"q\n{PDF_WIDTH} 0 0 {PDF_HEIGHT} 0 0 cm\n/SlideImage Do\nQ\n".encode("ascii")
        content_object_id = len(objects) + 1
        objects.append(
            b"<<\n"
            + f"/Length {len(content)}\n".encode("ascii")
            + b">>\nstream\n"
            + content
            + b"endstream"
        )

        annotation_ids = []
        for link in slide.links:
            annotation = link_annotation_object(link)
            if annotation is None:
                continue
            annotation_ids.append(len(objects) + 1)
            objects.append(annotation)

        annots = b""
        if annotation_ids:
            annot_refs = " ".join(f"{annot_id} 0 R" for annot_id in annotation_ids).encode("ascii")
            annots = b"/Annots [" + annot_refs + b"]\n"

        page_object_id = len(objects) + 1
        page_object_ids.append(page_object_id)
        objects.append(
            b"<<\n"
            b"/Type /Page\n"
            b"/Parent 0 0 R\n"
            + f"/MediaBox [0 0 {PDF_WIDTH} {PDF_HEIGHT}]\n".encode("ascii")
            + b"/Resources << /XObject << /SlideImage "
            + f"{image_object_id} 0 R".encode("ascii")
            + b" >> >>\n"
            + f"/Contents {content_object_id} 0 R\n".encode("ascii")
            + annots
            + b">>"
        )

    pages_object_id = len(objects) + 1
    page_refs = " ".join(f"{page_id} 0 R" for page_id in page_object_ids).encode("ascii")
    objects.append(
        b"<<\n"
        b"/Type /Pages\n"
        + f"/Count {len(page_object_ids)}\n".encode("ascii")
        + b"/Kids ["
        + page_refs
        + b"]\n"
        + b">>"
    )

    catalog_object_id = len(objects) + 1
    objects.append(
        b"<<\n"
        b"/Type /Catalog\n"
        + f"/Pages {pages_object_id} 0 R\n".encode("ascii")
        + b">>"
    )

    objects = [
        obj.replace(b"/Parent 0 0 R", f"/Parent {pages_object_id} 0 R".encode("ascii"))
        for obj in objects
    ]

    offsets: list[int] = []
    with out_path.open("wb") as pdf:
        pdf.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        for index, obj in enumerate(objects, start=1):
            offsets.append(pdf.tell())
            pdf.write(f"{index} 0 obj\n".encode("ascii"))
            pdf.write(obj)
            pdf.write(b"\nendobj\n")

        xref_offset = pdf.tell()
        pdf.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        pdf.write(b"0000000000 65535 f \n")
        for offset in offsets:
            pdf.write(f"{offset:010d} 00000 n \n".encode("ascii"))

        pdf.write(
            b"trailer\n"
            + b"<<\n"
            + f"/Size {len(objects) + 1}\n".encode("ascii")
            + f"/Root {catalog_object_id} 0 R\n".encode("ascii")
            + b">>\n"
            b"startxref\n"
            + f"{xref_offset}\n".encode("ascii")
            + b"%%EOF\n"
        )


def main() -> None:
    args = parse_args()
    validate_export_options(args.scale)
    slides = export_slide_screenshots(args.html, args.slides_dir, args.scale)
    write_pdf(slides, args.out)
    link_count = sum(len(slide.links) for slide in slides)
    print(f"Wrote {len(slides)} screenshots to {args.slides_dir}")
    print(f"Wrote PDF to {args.out}")
    print(f"Added {link_count} PDF link annotations")
    print(f"Rendered at {args.scale:g}x screenshot scale with lossless PDF image streams")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
