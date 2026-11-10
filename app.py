"""Reading Atlas — an interactive analysis of everything I've read.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from reading_dash.data import (
    CATEGORIES,
    CATEGORY_COLORS,
    COUNTRIES,
    LANGUAGE_COLORS,
    add_book,
    load_books,
    load_candidates,
    load_current,
    remove_current,
    save_current,
    unique_works,
)
from reading_dash.eras import SHORT_NAMES
from reading_dash.insights import blind_spots, headline, pace
from reading_dash.recommend import recommend
from reading_dash.semantic import build_space, galaxy_layout, neighbours

st.set_page_config(page_title="Reading Atlas", page_icon="📚", layout="wide")

# --------------------------------------------------------------------------- look & feel
INK = "#1C1917"
MUTED = "#78716C"
GRID = "rgba(120,113,108,0.12)"
ACCENT = "#2F4B7C"  # the one colour for single-series charts
BAND = "rgba(120,113,108,0.07)"
PLOT_CONFIG = {"displayModeBar": False}

# Hover text: show only what the chart doesn't already say.
BOOKS_Y = "%{y} books<extra></extra>"
BOOKS_X = "%{x} books<extra></extra>"
TIP = "%{hovertext}<extra></extra>"  # for point charts, the text is prebuilt in a "tip" column

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1180px;}
    [data-testid="stMetricLabel"] p {color: #78716C; font-size: 0.82rem;}
    [data-testid="stMetricValue"] {font-size: 1.7rem; font-weight: 600;}
    h1 {font-weight: 650; letter-spacing: -0.02em;}
    </style>
    """,
    unsafe_allow_html=True,
)


def style(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    """One quiet look for every chart: no titles inside the plot, light grid, short category names."""
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=36 if legend else 8, b=0),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=MUTED),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left", title=None,
                    font=dict(color=INK), itemsizing="constant"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", bordercolor="#E7E5E4", font=dict(color=INK, size=12)),
        bargap=0.3,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, title=None)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, title=None, labelalias=SHORT_NAMES)
    fig.for_each_trace(lambda t: t.update(name=SHORT_NAMES.get(t.name, t.name)))
    return fig


def chart(fig: go.Figure, title: str | None = None, note: str | None = None, *, height: int = 360,
          legend: bool = True, key: str, **kwargs):
    """Chart heading as page text (consistent with the rest of the page), then the plot."""
    if title:
        st.markdown(f"**{title}**")
    if note:
        st.caption(note)
    return st.plotly_chart(style(fig, height, legend), key=key, config=PLOT_CONFIG, **kwargs)


def short(cat: str) -> str:
    return SHORT_NAMES.get(cat, cat)


def month_label(value) -> str:
    """'2023-03' -> 'Mar 2023'; anything unparseable is shown as-is."""
    try:
        return pd.Period(str(value), freq="M").strftime("%b %Y")
    except (ValueError, TypeError):
        return str(value)


# --------------------------------------------------------------------------- data
@st.cache_data
def get_data():
    return load_books()


@st.cache_data
def get_candidates():
    return load_candidates()


@st.cache_resource
def get_space(_works: pd.DataFrame, _cand: pd.DataFrame, key: int):
    space = build_space(_works, _cand)
    space["xy"] = galaxy_layout(space["read"])
    return space


df, quality = get_data()
works = unique_works(df)
current = load_current()  # not cached: a tiny file you edit often, so changes show on refresh
candidates = get_candidates()
# Don't recommend a book that's already on the nightstand.
candidates = candidates[~candidates["title"].str.lower().isin(current["title"].str.lower())].reset_index(drop=True)
space = get_space(works, candidates, key=len(works))
h = headline(df)

# --------------------------------------------------------------------------- add a book
NEW_BOOK = "A new book"


@st.dialog("Add a book", width="large")
def add_book_dialog():
    reading_now = current["title"].tolist()
    read_before = sorted(works["original_title"].unique(), key=str.lower)
    options = ([NEW_BOOK] + [f"{t}  · currently reading" for t in reading_now]
               + [f"{t}  · re-read" for t in read_before])
    pick = st.selectbox("Book", options,
                        help="Pick a book you're reading to mark it finished or update your progress, "
                             "or one you've read before to log a re-read.")
    k = str(options.index(pick))  # widget keys per choice, so fields refill when the choice changes
    cur_row = current.iloc[reading_now.index(pick.rsplit("  · ", 1)[0])] if pick.endswith("currently reading") else None
    old = works[works["original_title"] == pick.rsplit("  · ", 1)[0]].iloc[-1] if pick.endswith("re-read") else None
    status = st.segmented_control("Status", ["Finished", "Currently reading"],
                                  default="Currently reading" if cur_row is not None else "Finished",
                                  key=f"s{k}") or "Finished"

    src = cur_row if cur_row is not None else old
    c = st.columns(2)
    title = c[0].text_input("Title *", value="" if src is None else src["title"], key=f"t{k}",
                            placeholder="As written on your copy")
    author = c[1].text_input("Author *", key=f"a{k}", value="" if src is None else
                             (src["author"] if cur_row is not None else src["canonical_author"]))
    langs = sorted(set(df["read_language"]) | {"Uzbek", "English", "Russian"})
    lang0 = None if src is None else src["read_language"]
    c = st.columns(2)
    read_language = c[0].selectbox("Read in *", langs, index=langs.index(lang0) if lang0 in langs else None,
                                   accept_new_options=True, placeholder="Choose or type", key=f"l{k}")
    if status == "Finished":
        when = c[1].date_input("Finished on", value=date.today(), max_value=date.today(), format="DD.MM.YYYY",
                               key=f"d{k}")
    else:
        start0 = cur_row["started"] if cur_row is not None and pd.notna(cur_row["started"]) else date.today()
        when = c[1].date_input("Started on", value=start0, max_value=date.today(), format="DD.MM.YYYY", key=f"d{k}")
        prog0 = int(cur_row["progress"]) if cur_row is not None and pd.notna(cur_row["progress"]) else 0
        progress = st.slider("Progress", 0, 100, prog0, 5, format="%d%%", key=f"p{k}")

    meta = None
    if status == "Finished" and old is not None:
        st.caption("Details (category, country, year…) are copied from your earlier read.")
        meta = {"original_title": old["original_title"], "canonical_author": old["canonical_author"],
                "author_country": old["author_country"], "author_gender": old["author_gender"],
                "original_language": old["original_language"],
                "pub_year": int(old["pub_year"]) if pd.notna(old["pub_year"]) else "",
                "category": old["category"], "fiction": bool(old["fiction"]), "tags": old["tags"],
                "description": old["description"], "verify": bool(old["verify"])}
    elif status == "Finished":
        names = {v[0]: code for code, v in COUNTRIES.items()}
        orig_langs = sorted(set(df["original_language"]) - {"Unknown"})
        with st.expander("More details (optional)"):
            c = st.columns(2)
            category = c[0].selectbox("Category", CATEGORIES, index=None, format_func=short, placeholder="Choose",
                                      key=f"c{k}")
            pub_year = c[1].number_input("Year published", min_value=-800, max_value=date.today().year, value=None,
                                         step=1, placeholder="e.g. 1880", key=f"y{k}")
            c = st.columns(2)
            country = c[0].selectbox("Author's country", sorted(names), index=None, placeholder="Choose",
                                     key=f"co{k}")
            original_language = c[1].selectbox("Originally written in", orig_langs, index=None,
                                               accept_new_options=True, placeholder="Same as read in", key=f"ol{k}")
        meta = {"original_title": title.strip(), "canonical_author": author.strip(),
                "author_country": names.get(country, ""), "author_gender": "Unknown",
                "original_language": original_language or read_language or "",
                "pub_year": int(pub_year) if pub_year is not None else "", "category": category or "",
                "fiction": category == "Literature", "tags": "", "description": "",
                # Flag for review unless the details the charts rely on are all filled in.
                "verify": not (category and pub_year is not None and country)}

    if st.button("Save", type="primary", width="stretch"):
        if not title.strip() or not author.strip() or not read_language:
            st.error("Title, author and the language you read it in are required.")
            return
        if status == "Currently reading":
            save_current({"title": title.strip(), "author": author.strip(), "started": when,
                          "read_language": read_language, "progress": progress})
            msg = f"Saved to currently reading: {title.strip()}"
        else:
            add_book({"title": title.strip(), "author": author.strip(), "finish_date": when,
                      "read_language": read_language}, meta)
            remove_current(cur_row["title"] if cur_row is not None else title)
            msg = f"Added to your reading list: {title.strip()}"
        get_data.clear()
        st.session_state["flash"] = msg
        st.rerun()


# --------------------------------------------------------------------------- header
head, button = st.columns([5, 1], vertical_alignment="center")
head.title("Reading Atlas")
if button.button("Add a book", icon=":material/add:", width="stretch"):
    add_book_dialog()
if msg := st.session_state.pop("flash", None):
    st.toast(msg, icon=":material/check:")
st.caption(
    f"{h['reads']} reads · {h['works']} distinct books · {h['authors']} authors · {h['countries']} countries · "
    f"read in {h['read_languages']} languages, written in {h['original_languages']}"
)

if not current.empty:
    st.markdown("**Currently reading**")
    cols = st.columns(3)
    today = pd.Timestamp.today().normalize()
    for k, r in enumerate(current.itertuples()):
        with cols[k % 3], st.container(border=True):
            st.markdown(f"**{r.title}**")
            bits = [b for b in (r.author, f"in {r.read_language}" if r.read_language else "") if b]
            if pd.notna(r.started):
                bits.append(f"started {r.started:%d %b %Y} (day {max((today - r.started).days, 0) + 1})")
            if bits:
                st.caption(" · ".join(bits))
            if pd.notna(r.progress):
                st.progress(r.progress / 100, text=f"{r.progress:.0f}%")

tabs = st.tabs(["Overview", "Timeline", "Themes", "Breadth", "Next reads", "Data"])

# --------------------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Books read", h["reads"], help=f"{h['works']} distinct works, {h['rereads']} re-reads")
    c[1].metric("Authors", h["authors"])
    c[2].metric("Fiction", f"{h['fiction_share']:.0%}")
    c[3].metric("Read in translation", f"{h['translated_share']:.0%}")

    by_year = df[df["dated"]].groupby(["year", "category"]).size().reset_index(name="books")
    fig = px.bar(by_year, x="year", y="books", color="category", color_discrete_map=CATEGORY_COLORS,
                 category_orders={"category": CATEGORIES})
    fig.update_traces(hovertemplate=BOOKS_Y)
    fig.update_xaxes(type="category")
    chart(fig, "Books finished per year", height=340, key="ov_year")

    left, right = st.columns(2, gap="large")
    with left:
        lang_year = df[df["dated"]].groupby(["year", "read_language"]).size().reset_index(name="books")
        lang_year["share"] = lang_year["books"] / lang_year.groupby("year")["books"].transform("sum")
        fig = px.bar(lang_year, x="year", y="share", color="read_language", color_discrete_map=LANGUAGE_COLORS,
                     custom_data=["books"])
        fig.update_traces(hovertemplate="%{customdata[0]} books (%{y:.0%})<extra></extra>")
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(type="category")
        chart(fig, "Language I read in", height=260, key="ov_lang")
    with right:
        top_authors = df.groupby("canonical_author").size().sort_values(ascending=False)
        top_authors = top_authors[top_authors > 1].head(8).reset_index(name="books")
        fig = px.bar(top_authors, x="books", y="canonical_author", orientation="h")
        fig.update_traces(marker_color=ACCENT, hovertemplate=BOOKS_X)
        fig.update_yaxes(categoryorder="total ascending", showgrid=False)
        fig.update_xaxes(dtick=1, showgrid=True, gridcolor=GRID)
        chart(fig, "Authors I return to", height=260, legend=False, key="ov_auth")

# --------------------------------------------------------------------------- timeline
with tabs[1]:
    p = pace(df)
    if not p:
        st.info("No dated books yet.")
    else:
        c = st.columns(3)
        c[0].metric("Busiest month", month_label(p["busiest_month"]))
        c[0].caption(f"{p['busiest_count']} books")
        c[1].metric("Longest streak", f"{p['streak_months']} months")
        c[1].caption(f"ending {month_label(p['streak_end'])}")
        c[2].metric("Longest gap", f"{p['longest_gap_days']} days")
        if p["gap_from"] is not None:
            c[2].caption(f"{p['gap_from']:%b %Y} → {p['gap_to']:%b %Y}")

        dated = df[df["dated"]].copy()
        dated["tip"] = ("<b>" + dated["original_title"] + "</b><br>" + dated["canonical_author"]
                        + " · " + dated["finish_date"].dt.strftime("%d %b %Y"))
        fig = px.strip(dated, x="finish_date", y="category", color="category", color_discrete_map=CATEGORY_COLORS,
                       hover_name="tip", category_orders={"category": CATEGORIES})
        fig.update_traces(marker=dict(size=9, line=dict(width=1, color="white")), jitter=0.3, hovertemplate=TIP)
        if p["gap_from"] is not None:
            fig.add_vrect(x0=p["gap_from"], x1=p["gap_to"], fillcolor=BAND, line_width=0,
                          annotation_text="longest gap", annotation_position="top left",
                          annotation_font=dict(color=MUTED, size=11))
        chart(fig, "Every finished book", height=380, legend=False, key="tl_strip")

        monthly = p["monthly"].reset_index()
        monthly.columns = ["month", "books"]
        monthly["month"] = monthly["month"].dt.to_timestamp()
        fig = px.bar(monthly, x="month", y="books")
        fig.update_traces(marker_color=ACCENT, hovertemplate=BOOKS_Y)
        chart(fig, "Books per month", f"Active in {p['active_months']} of {p['total_months']} months, "
              f"{p['per_active_month']:.1f} books per active month", height=200, legend=False, key="tl_month")

# --------------------------------------------------------------------------- themes (galaxy)
with tabs[2]:
    st.caption(
        "Each dot is a book, placed by the similarity of its themes (TF-IDF over tags and descriptions, "
        "reduced with SVD and t-SNE). Click a dot to see its nearest neighbours."
    )
    g = works.copy()
    g["x"], g["y"] = space["xy"][:, 0], space["xy"][:, 1]
    g["size"] = g["times_read"].astype(float) * 10
    g["idx"] = range(len(g))
    g["tip"] = "<b>" + g["original_title"] + "</b><br>" + g["canonical_author"]

    color_by = st.segmented_control(
        "Colour by", ["category", "read_language"], default="category",
        format_func=lambda s: {"category": "Category", "read_language": "Language"}[s],
    ) or "category"
    cmap = CATEGORY_COLORS if color_by == "category" else LANGUAGE_COLORS
    fig = px.scatter(
        g, x="x", y="y", color=color_by, color_discrete_map=cmap, size="size", size_max=13,
        hover_name="tip", custom_data=["idx"], category_orders={"category": CATEGORIES},
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.95), hovertemplate=TIP)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    event = chart(fig, height=540, key="galaxy", on_select="rerun", selection_mode="points")

    picked = None
    try:
        pts = event.selection.points if event else []
        if pts:
            picked = int(pts[0]["customdata"][0])
    except (AttributeError, KeyError, IndexError, TypeError):
        picked = None
    titles = g["original_title"].tolist()
    default_idx = titles.index("Anna Karenina") if "Anna Karenina" in titles else 0
    choice = st.selectbox("Book", titles, index=picked if picked is not None else default_idx)
    i = titles.index(choice)
    row = g.iloc[i]
    a, b = st.columns([1, 1], gap="large")
    with a:
        st.markdown(f"#### {row['original_title']}")
        year = int(row["pub_year"]) if pd.notna(row["pub_year"]) else "year unknown"
        st.caption(f"{row['canonical_author']} · {short(row['category'])} · {year}")
        st.write(row["description"])
    with b:
        st.markdown("**Closest books on my list**")
        for j, s in neighbours(space["read"], i, k=6):
            nb = g.iloc[j]
            st.markdown(f"{nb['original_title']} <span style='color:{MUTED}'>· {nb['canonical_author']} · "
                        f"{s:.2f}</span>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- breadth
with tabs[3]:
    st.caption("What my reading covers, and what it leaves out. Computed from the data.")
    notes = blind_spots(df)
    flag = {"high": ":red[gap]", "medium": ":orange[thin]", "low": ":gray[fine]"}
    cols = st.columns(4)
    for k, n in enumerate(notes):
        with cols[k % 4]:
            with st.container(border=True):
                st.metric(n["title"], n["value"])
                st.caption(f"{flag[n['severity']]} · {n['detail']}")

    geo = df[df["author_country"] != ""].drop_duplicates("original_title")
    by_c = (
        geo.groupby(["author_country", "country_name"])
        .agg(books=("original_title", "size"), titles=("original_title", lambda s: "<br>".join(s.head(8))))
        .reset_index()
    )
    by_c["tip"] = "<b>" + by_c["country_name"] + "</b> · " + by_c["books"].astype(str) + " books<br>" + by_c["titles"]
    fig = px.choropleth(by_c, locations="author_country", color="books", hover_name="tip",
                        color_continuous_scale=[[0, "#DCE3EE"], [1, ACCENT]])
    fig.update_traces(hovertemplate=TIP)
    fig.update_geos(showframe=False, showcoastlines=False, showcountries=True, countrycolor="#E7E5E4",
                    showland=True, landcolor="#F5F5F4", projection_type="natural earth",
                    projection_rotation_lon=20, lataxis_range=[-56, 84], fitbounds=False,
                    bgcolor="rgba(0,0,0,0)")
    fig.update_layout(coloraxis_colorbar=dict(thickness=8, len=0.5, title=None, outlinewidth=0))
    chart(fig, "Where my authors come from",
          "Country = where each author is chiefly associated with; co-authored books use the first author.",
          height=440, legend=False, key="map")

    a, b = st.columns([3, 2], gap="large")
    with a:
        tt = df[df["dated"] & df["pub_year"].notna()].copy()
        tt["age"] = (tt["finish_date"].dt.year - tt["pub_year"]).clip(lower=1)
        tt["tip"] = ("<b>" + tt["original_title"] + "</b><br>" + tt["canonical_author"]
                     + " · published " + tt["pub_year"].astype(int).astype(str))
        fig = px.scatter(tt, x="finish_date", y="age", color="category", color_discrete_map=CATEGORY_COLORS,
                         hover_name="tip", log_y=True, category_orders={"category": CATEGORIES})
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")), hovertemplate=TIP)
        fig.update_yaxes(tickvals=[1, 10, 100, 1000])
        allpub = df[df["pub_year"].notna()].drop_duplicates("original_title")
        oldest = allpub.nsmallest(3, "pub_year")
        chart(fig, "How old each book was when I read it",
              "Years, log scale. Oldest: " + "; ".join(f"{r.original_title} (~{int(r.pub_year)})"
                                                         for r in oldest.itertuples()),
              height=340, legend=False, key="tt_scatter")
    with b:
        ol = df.drop_duplicates("original_title")["original_language"].value_counts().reset_index()
        ol.columns = ["language", "books"]
        fig = px.bar(ol, x="books", y="language", orientation="h")
        fig.update_traces(marker_color=ACCENT, hovertemplate=BOOKS_X)
        fig.update_yaxes(categoryorder="total ascending", showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor=GRID)
        chart(fig, "Language the book was written in", height=340, legend=False, key="bs_lang")

# --------------------------------------------------------------------------- next reads
with tabs[4]:
    st.caption(
        "Candidates scored against a recency-weighted taste profile. Each pick names the books on my list "
        "it is closest to."
    )
    a, b = st.columns([2, 1], gap="large")
    mode = a.segmented_control("Mode", ["More of what I love", "Stretch my blind spots"],
                               default="More of what I love") or "More of what I love"
    half_life = b.slider("Taste memory (half-life, days)", 90, 1500, 540, 30,
                         help="Lower = only recent reading matters")
    recs = recommend(works, candidates, space["read"], space["cand"],
                     mode="stretch" if mode.startswith("Stretch") else "similar", half_life_days=half_life, n=8)
    cols = st.columns(2, gap="medium")
    for k, r in enumerate(recs.itertuples()):
        with cols[k % 2]:
            with st.container(border=True):
                st.markdown(f"**{r.title}**")
                year = int(r.pub_year) if pd.notna(r.pub_year) else ""
                st.caption(f"{r.author} · {short(r.category)} · {r.country_name} · {year}")
                st.write(r.description)
                extra = []
                if r.because:
                    extra.append("Because you read " + ", ".join(f"*{t}*" for t in r.because))
                if r.new_for_you:
                    extra.append("New for you: " + ", ".join(r.new_for_you))
                if extra:
                    st.caption("  \n".join(extra))

# --------------------------------------------------------------------------- data
with tabs[5]:
    c = st.columns(4)
    c[0].metric("Undated books", len(quality.undated), help="Read before I started logging dates")
    c[1].metric("Flagged to verify", len(quality.needs_verification), help="Metadata compiled by hand with low confidence")
    c[2].metric("Batch-logged dates", sum(quality.batch_dates.values()),
                help="3+ books sharing one finish date usually means they were logged together")
    c[3].metric("Rows out of date order", quality.out_of_order)
    if quality.batch_dates:
        st.caption("Dates shared by 3+ books: " + ", ".join(f"{d} ({n})" for d, n in quality.batch_dates.items()))
    if quality.missing_metadata:
        st.warning(f"Books with no metadata row yet: {quality.missing_metadata}. Add them to data/book_metadata.csv.")

    show = df[["id", "title", "original_title", "canonical_author", "finish_date", "read_language",
               "original_language", "translated", "category", "pub_year", "country_name", "times_read", "verify"]]
    st.dataframe(show, hide_index=True, column_config={
        "finish_date": st.column_config.DateColumn("finished", format="DD.MM.YYYY"),
        "pub_year": st.column_config.NumberColumn("published", format="%d"),
    })
    st.download_button("Download enriched data (CSV)", show.to_csv(index=False).encode("utf-8"),
                       file_name="reading_atlas.csv", mime="text/csv")
