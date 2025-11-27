# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and Building

### Installing LaTeX on macOS with Brew

This project requires a LaTeX distribution to compile. Install MacTeX (the macOS equivalent of TeXLive):

```bash
brew install --cask mactex
```

After installation, you may need to restart your terminal or add `/Library/TeX/texbin` to your PATH if LaTeX commands aren't recognized.

### Compiling to PDF

To compile the main document to PDF, use `latexmk` (included with MacTeX):

```bash
# Recommended: latexmk automatically handles multiple passes for citations
latexmk -pdf main.tex

# Clean up auxiliary files
latexmk -c main.tex
```

`latexmk` is included with MacTeX and automatically handles the necessary multiple compilation passes for citations and cross-references.

Alternatively, if you prefer direct `pdflatex` (may need multiple runs):
```bash
pdflatex -interaction=nonstopmode main.tex
```

### Compiling the Rebuttal Document

Similarly, compile the rebuttal document:

```bash
pdflatex -interaction=nonstopmode rebuttal.tex
# or
latexmk -pdf rebuttal.tex
```

## Project Structure

### Main Files

- **main.tex**: The primary conference paper document. Uses the CVPR style by default (uncomment `\usepackage{cvpr}` for camera-ready version).
- **rebuttal.tex**: Rebuttal document for conference responses.
- **preamble.tex**: Shared package imports and configuration.
- **main.bib**: Bibliography file with references.

### Sections

Papers are organized in the `sec/` directory:
- `0_abstract.tex`: Abstract
- `1_intro.tex`: Introduction
- `2_formatting.tex`: Formatting guidelines and examples
- `3_finalcopy.tex`: Main content
- `X_suppl.tex`: Supplementary material (commented out by default)

### Styling

- **cvpr.sty**: Main CVPR template style file. Supports three modes:
  - `\usepackage[review]{cvpr}` - Review version with line numbers and anonymous mode
  - `\usepackage{cvpr}` - Camera-ready final version
  - `\usepackage[pagenumbers]{cvpr}` - Adds page numbers (e.g., for arXiv)
- **ieeenat_fullname.bst**: Bibliography style file (IEEE with full author names)

## Compilation Notes

- The template uses hyperref for cross-references. If you encounter validation issues with camera-ready submission, you can comment out the hyperref package (line 21 in main.tex), but you'll need to delete `*.aux` files before re-running.
- Multiple LaTeX passes are required for bibliography and cross-references to resolve correctly.
- The GitHub Actions workflow (`.github/workflows/latex-build.yml`) validates the build on every push.

## Paper Modes

Edit main.tex line 7 to switch modes:
- `\usepackage[review]{cvpr}` - For peer review (anonymous, line numbers)
- `\usepackage{cvpr}` - For camera-ready submission
- `\usepackage[pagenumbers]{cvpr}` - For arXiv with page numbers
