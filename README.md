# Magnetron Design Calculator (Carter + Collins/Clogston)

Physics-based magnetron design and exploration tool with both:

- a **CLI** (interactive prompts and preset-driven runs), and
- a **Flask web UI** (single-page interface served from a Jinja template).

It implements a practical design workflow combining:

- Carter (2018) magnetron design equations (operating point, geometry, mode separation, coupling/Q heuristics, etc.)
- Collins & Clogston (1948, MIT Rad Lab Vol. 6) **universal reduced parameters** (Collins Eq. 25–26), including the digitized Fig. 10‑9 curve used for the characteristic current scale
- Reference magnetrons/datasheets to provide example-based “typical” ranges (p10–p90) for the Collins reduced parameters

## Features

- End-to-end operating-point and geometry calculations with vane-count sweeps.
- Collins/Clogston reduced parameters: \(b, v, i, g, p\) computed using Collins Eq. (25)–(26).
- Web UI served at `/ui` with:
  - Local persistence (`localStorage`) for type/cathode and key overrides (\(\eta\), \(Z_{dc}\)).
  - A collapsible formulas section rendered in **LaTeX-style** using **KaTeX**.
  - Typical ranges (10th–90th percentile) derived from the built-in reference database.
- JSON API endpoints for automation (`/calculate`, `/types`, `/cathodes`, etc.).

## Project structure

- `magnetron_design.py` — all physics, CLI, reference database, and Flask server.
- `templates/magnetron_design.html` — the web UI template (HTML + JS + KaTeX auto-render).
- `OpenFlask.bat` — convenience launcher (Windows).

## Requirements

- Python 3.9+ (3.10/3.11 recommended)
- Optional but recommended (for the UI/API):
  - `flask`

Install Flask:

```bash
pip install flask
```

## Run (CLI)

Interactive mode:

```bash
python magnetron_design.py
```

Preset examples:

```bash
python magnetron_design.py --preset radar_x
python magnetron_design.py --preset linac13
```

Load a reference tube from the internal DB:

```bash
python magnetron_design.py --list-db
python magnetron_design.py --load-db mg5193
```

## Run (Web UI + API)

Start the Flask server:

```bash
python magnetron_design.py --flask
```

Then open:

- UI: http://127.0.0.1:5000/ui
- API root: http://127.0.0.1:5000/
- Health: http://127.0.0.1:5000/health

### API usage example

GET:

```bash
curl "http://127.0.0.1:5000/calculate?freq=9.375&power=250&type=s_pls&cath=disp"
```

POST:

```bash
curl -X POST http://127.0.0.1:5000/calculate \
  -H "Content-Type: application/json" \
  -d '{"freq": 9.375, "power": 250, "type": "s_pls", "cath": "disp"}'
```

## Notes

- The KaTeX assets are loaded from a CDN inside `templates/magnetron_design.html`. If you need **offline** rendering, vendor KaTeX locally and update the `<link>`/`<script>` tags.
- The reduced parameter `g` depends on an estimated load conductance \(G_L\); the UI formulas section documents the estimation approach used.

## References

- Carter, R. G. (2018). *Microwave and RF Vacuum Electronic Power Sources*. Cambridge University Press.
- Collins, G. B. et al. (1948). *Microwave Magnetrons*, MIT Radiation Laboratory Series Vol. 6. McGraw-Hill.
- Clogston, A. M. (1948). “Principles of Design”, in Collins Vol. 6.
