# Screenshot PDF Export

Use this exporter when browser print output is unreliable. It renders every HTML slide as a high-density PNG, then combines those screenshots into a lossless image-based PDF.

Visible HTML links are preserved as clickable PDF link annotations.

## Setup

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

## Export

```bash
python presentation/export_pdf.py
```

Default outputs:

- `presentation/output/slides/slide-01.png`, `slide-02.png`, ...
- `presentation/output/race-strategist-deck.pdf`

By default the exporter captures each slide at `2x` pixel density, so every screenshot is `3840x2160` while the PDF keeps the same 16:9 page size.

Custom PDF output:

```bash
python presentation/export_pdf.py --out presentation/output/custom.pdf
```

Custom screenshots directory:

```bash
python presentation/export_pdf.py --slides-dir presentation/output/custom-slides
```

Sharper export:

```bash
python presentation/export_pdf.py --scale 3
```

This produces `5760x3240` screenshots and a larger PDF.
