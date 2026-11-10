# 📚 Reading Atlas

An interactive dashboard that analyses every book I've read: 109 reads in Uzbek, English and Russian since school. It goes past counting books. It detects the **eras** of my reading with change-point segmentation, maps books by theme, recommends what to read next with a reason for each pick, and reports honestly what my reading leaves out.

Built with Python, pandas, scikit-learn, Plotly and Streamlit.

![Reading eras](docs/eras.png)

## What's inside

| Tab | What it shows | Method |
|---|---|---|
| **Overview** | Books per year by category, language mix, original vs translation, authors I return to | Aggregation |
| **Timeline** | Every finished book on a timeline; busiest month, longest streak, longest gap | Gap & streak detection |
| **Reading eras** | Automatically detected phases of reading, with a robustness check for every boundary | Optimal change-point segmentation (dynamic programming), penalty sweep |
| **Book galaxy** | All books on a 2-D map where neighbours share themes. Click a book to see its nearest neighbours | TF-IDF → truncated SVD → t-SNE, cosine similarity |
| **Time travel** | How old each book was when I read it, from Marcus Aurelius (~180 AD) to 2023 | Log-scale scatter |
| **World map** | Where my authors come from | Choropleth |
| **Blind spots** | Under-represented authors, regions and languages, computed from the data | Rule-based findings |
| **What next** | Explainable recommendations ("because you read …") with a mode that targets blind spots | Recency-weighted taste profile, cosine ranking, novelty bonus |
| **Data** | Data-quality report and the full enriched table, downloadable | |

## Findings from my own list

- **106 distinct books, 86 authors, 21 countries.** I read them in 3 languages, but they were written in 11. **40%** I read in translation.
- **Three re-reads, each in a different language from the first read:** *1984* (English → Uzbek), *The Subtle Art…* (Uzbek → English) and Otabek Mahkamov's book (Uzbek → English).
- **Five reading eras** at the default penalty:
  1. Fiction & self-help (Mar 2021 – Aug 2022)
  2. A business burst (Aug – Oct 2022)
  3. Self-help & faith (Oct 2022 – Jul 2023)
  4. A run of classics (Jan 2024)
  5. Fiction & economics/society (Feb 2024 – now)
- **Longest gap: 575 days** (Feb 2025 → Aug 2026). That's the period I worked as a data scientist and research intern, and the gap ended with Tolstoy and Pushkin in Russian.
- **Blind spots:** only 9% of my books are by women (counting co-authors). I've read no Korean literature despite living in Korea, and no African authors.

## How the era detection works

Dated books are sorted by finish date, and each becomes a one-hot vector over 7 categories. The segmentation minimises

$$\sum_{\text{segments}} \sum_{i \in s} \lVert x_i - \bar{x}_s \rVert^2 \;+\; \beta \cdot (\text{number of segments})$$

This is solved exactly with an O(n²) dynamic programme, the same optimal-partitioning problem that PELT speeds up, so the project doesn't need `ruptures`. Each era must contain at least a minimum number of books. Results depend on β, so the dashboard re-runs the segmentation for 26 values of β and reports what share of runs keep each boundary. A boundary that survives most of the sweep is a real shift in reading, not noise.

## How the recommender works

- Each book becomes a TF-IDF vector built from its tags (double weight), its category and a short description. The books I've read and the candidates are fitted in the same vector space.
- My taste profile is the average of the books I've read, weighted by recency with an adjustable half-life. Undated books get a small fixed weight.
- Candidates are ranked by cosine similarity to that profile. The explanation for each pick is the two books on my list closest to it.
- **Stretch mode** adds a novelty bonus for women authors, countries I haven't read yet, and languages no book on my list was originally written in.

TF-IDF runs offline with no model download. To use dense sentence embeddings instead, swap `semantic.vectorize`.

## Run it

```bash
git clone <this repo> && cd reading-dashboard
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
pytest                                               # unit tests + full app render
```

**Deploy for free:** push to GitHub, then go to [share.streamlit.io](https://share.streamlit.io) → *New app* → choose the repo and `app.py`.

## Adding a book

The easiest way is the **Add a book** button at the top of the dashboard. It writes to the CSV files below, and also handles re-reads (details copied from the earlier read) and the currently-reading list. To edit the files by hand instead:

1. Add a row to `data/reading_list.csv`: `id,title,author,finish_date (dd.mm.yyyy),read_language`.
2. Add a row with the same `id` to `data/book_metadata.csv`. Optionally, run `python scripts/enrich_openlibrary.py --ids <id>` to get suggested publication year, page count and subjects from Open Library.

If you skip step 2, the book still appears in the dashboard with "Unknown" fields, and the Data tab lists it as missing metadata.

**Currently reading:** list books in progress in `data/currently_reading.csv` (`title,author,started (dd.mm.yyyy),read_language,progress (0-100)`; only `title` is required). They show as cards above the tabs and don't count in any statistic. When you finish one, delete its row there and add it to `reading_list.csv` as above.

## Automatic metadata (experimental)

`reading_dash/enrich.py` asks Claude to fill in a book's metadata from just its title, author and read language, in the `book_metadata.csv` format (rows are marked `verify=True`). To measure how good that is, `scripts/eval_enrichment.py` hides the hand-made metadata for a sample of books and scores Claude's answers field by field:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/eval_enrichment.py --dry-run   # show the sample, no API calls
python scripts/eval_enrichment.py             # results in reports/
```

## Data notes

- `reading_list.csv` is my original log, unchanged. `book_metadata.csv` was compiled by hand. 37 rows are marked `verify=True`, mostly Uzbek and Islamic titles whose publication year or author origin I couldn't confirm.
- 19 books (the earliest, read before I logged dates) have no finish date. They're excluded from time-based charts and included everywhere else.
- Three dates are shared by three or more books, most notably six on 28 Dec 2024. Those were probably logged in batches, which the Data tab flags.
- "Author country" is the country each author is chiefly associated with. For co-authored books it's the first author's.

## Project structure

```
app.py                    Streamlit dashboard
reading_dash/
  data.py                 loading, merging, derived fields, data-quality report
  eras.py                 change-point segmentation + boundary robustness
  semantic.py             TF-IDF space, t-SNE layout, nearest neighbours
  recommend.py            recency-weighted, explainable recommender
  insights.py             headline stats, pace, blind-spot findings
data/                     reading log, book metadata, recommendation pool
scripts/enrich_openlibrary.py
tests/test_analysis.py
```
