# Felix-DGSVY-2025

Data analysis and visualisation pipeline for Felix's 2025 DGSVY (Drinks, Gender, Sex, Vaping, You) survey — an anonymous student survey run by [Felix](https://felixonline.co.uk), Imperial College London's student newspaper.

The pipeline ingests raw Qualtrics survey exports, cleans and standardises the data, and produces publication-ready SVG charts across six thematic sections.

---

## Survey Sections

The survey covers 57 questions across six topics:

| Section | Questions | Topics |
|---|---|---|
| **About You** | Q2–Q8 | Age, student type, department, year of study, gender identity, gender expression, sexual orientation |
| **Drinking** | Q9–Q17 | Alcohol consumption frequency, drink types, shot-drinking, family awareness, reasons for drinking, illness from alcohol, weekly unit estimates |
| **Nicotine** | Q18–Q34 | Cigarette and vape use, snus habits, brand preferences, quit attempts, reasons for use |
| **Drugs** | Q35–Q46 | Drugs taken, crossfading, drugs refused, frequency, family awareness, medical attention required, age of first use |
| **Location** | Q47–Q49 | Substances and drugs taken on campus, mapped locations within campus |
| **Weed** | Q50–Q57 | Cannabis consumption methods, greening out, edibles, synthetic cannabinoids (spice), drug sources, dealer proximity, contact methods |

---

## Project Structure

```
Felix-DGSVY-2025/
├── main.py                      # Entry point — runs all six section modules
├── preprocessing/
│   ├── data_cleaner.py          # Cleans raw Qualtrics CSV export
│   ├── data_loader.py           # Loads cleaned data and headers
│   ├── visualise.py             # EDAVisualiser class (all chart types)
│   ├── overview.py              # Generates summary statistics (overview.json)
│   └── questions.txt            # Human-readable question index (Q2–Q57)
├── sections/
│   ├── about_you/about_you.py
│   ├── drinking/drinking.py
│   ├── nicotine/nicotine.py
│   ├── drugs/drugs.py
│   ├── location/location.py
│   └── weed/weed.py
└── data/                        # Not tracked (GDPR) — see Data section below
    ├── data.csv                 # Raw Qualtrics export
    ├── cleaned_data.csv         # Output of DataCleaner
    └── headers.json             # Question code → question text mapping
```

Each section module saves its SVG plots to `sections/<section>/plots/`. Open-text responses that cannot be charted are written to `.txt` files in the same directory.

---

## How It Works

**1. Data cleaning** (`preprocessing/data_cleaner.py`)

`DataCleaner` reads the raw Qualtrics CSV, strips metadata columns (IP addresses, timestamps, distribution info), removes the Qualtrics header rows, de-duplicates responses, fills blank cells with `"None"`, and saves `cleaned_data.csv`. It also extracts the question-text headers and saves them to `headers.json`.

**2. Data loading** (`preprocessing/data_loader.py`)

`DataLoader` provides a single access point for the cleaned DataFrame and headers dictionary, used by every section module.

**3. Visualisation** (`preprocessing/visualise.py`)

`EDAVisualiser` wraps matplotlib and seaborn with a consistent newspaper-style theme (dark navy, burgundy, and slate tones; DIN Alternate font; clean spines). It exposes:

- `bar_chart` — vertical bar chart with optional custom ordering
- `horizontal_bar_chart` — horizontal bar chart; optionally parses comma-separated multi-select responses
- `pie_chart` — donut-style pie chart
- `histogram` — distribution histogram with optional KDE overlay
- `combination_heatmap` — crosstab heatmap for two columns

All plots are saved as high-resolution SVGs.

**4. Section modules** (`sections/*/`)

Each section class instantiates `DataLoader` and `EDAVisualiser`, then defines one method per question. Calling `plot_all()` generates the full set of charts for that section. `main.py` runs all six sections in sequence via subprocess.

---

## To Run

**Prerequisites**

```bash
pip install pandas matplotlib seaborn numpy
```

**Run the full pipeline**

Ensure `data/cleaned_data.csv` and `data/headers.json` exist (produced by `DataCleaner`). If starting from a raw Qualtrics export at `data/data.csv`, run the cleaner first:

```bash
python3 preprocessing/data_cleaner.py
```

Then run all sections:

```bash
python3 main.py
```

Output SVGs are written to `sections/<section>/plots/`.

To run a single section:

```bash
python3 sections/drinking/drinking.py
```

---

## Data

The raw survey data is not publicly available. Responses are held under GDPR and cannot be shared or provided on request.

The `data/` directory is excluded from version control via `.gitignore`.
