# ICLR 2027 submission build

    latexmk -pdf main.tex

`main.tex` inputs `sections/*.tex`. Style files are the official ICLR 2027
package from https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip
(downloaded 2026-09-10) and are committed so the build is self-contained.

## Page discipline

The 9-page limit covers the main text only: references, the mandatory AI-use
statement, and appendices are all off-budget. To measure the main text alone:

    bash measure.sh

Sections not yet written are present as placeholders sized to their word
budget, so the reported page count already includes them. That is deliberate:
every compression decision is measured against the finished paper's length,
not against a partial one.

## Anonymity

`\iclrfinalcopy` stays commented until camera-ready. A submission that reveals
author identity is desk rejected.
