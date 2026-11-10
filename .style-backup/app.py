"""Reading Atlas — an interactive analysis of everything I've read.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from reading_dash.data import (
    CATEGORIES,
    CATEGORY_COLORS,
    LANGUAGE_COLORS,
    load_books,
    load_candidates,
    unique_works,
)
from reading_dash.eras import SHORT_NAMES, boundary_stability, detect_eras
from reading_dash.insights import blind_spots, era_story, headline, pace
from reading_dash.recommend import recommend
from reading_dash.semantic import build_space, galaxy_layout, neighbours

st.set_page_config(page_title="Reading Atlas", page_icon="📚", layout="wide")

WRITING_ERAS = ["Before 1500", "1500–1799", "1800s", "1900–1949", "1950–1999", "2000s", "2010s+"]
PLOT_FONT = dict(family="Inter, system-ui, sans-serif", size=13)


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        font=PLOT_FONT,
        title=dict(x=0, xanchor="left", y=0.98, yanchor="top"),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, title=None),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


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


df_all, quality = get_data()
works_all = unique_works(df_all)
candidates = get_candidates()
space = get_space(works_all, candidates, key=len(works_all))

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📚 Reading Atlas")
    st.caption("Every book I've read since school, analysed.")
    langs = st.multiselect("Read in", sorted(df_all["read_language"].unique()), default=sorted(df_all["read_language"].unique()))
    cats = st.multiselect("Categories", CATEGORIES, default=CATEGORIES)
    years = sorted(df_all["year"].dropna().astype(int).unique())
    yr = st.select_slider("Finish year", options=years, value=(years[0], years[-1]))
    include_undated = st.checkbox("Include undated books (read before 2021)", value=True)
    st.divider()
    st.caption(
        "Filters apply to Overview, Timeline, Time travel, Map and Blind spots. "
        "Eras, Galaxy and recommendations always use the full history."
    )

mask = df_all["read_language"].isin(langs) & df_all["category"].isin(cats + ["Unknown"])
year_ok = df_all["year"].between(yr[0], yr[1]).fillna(False).astype(bool)
mask &= year_ok | (~df_all["dated"] & include_undated)
df = df_all[mask]

if df.empty:
    st.warning("No books match these filters.")
    st.stop()

h = headline(df)

# --------------------------------------------------------------------------- header
st.title("Reading Atlas")
st.markdown(
    f"**{h['reads']} reads · {h['works']} distinct books · {h['authors']} authors · "
    f"{h['countries']} countries · read in {h['read_languages']} languages, written in {h['original_languages']}.**"
)

tabs = st.tabs(
    ["Overview", "Timeline", "Reading eras", "Book galaxy", "Time travel", "World map", "Blind spots", "What next", "Data"]
)

# --------------------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(6)
    c[0].metric("Books read", h["reads"])
    c[1].metric("Distinct works", h["works"], help="Re-reads counted once")
    c[2].metric("Re-reads", h["rereads"])
    c[3].metric("Read in translation", f"{h['translated_share']:.0%}")
    c[4].metric("Fiction", f"{h['fiction_share']:.0%}")
    c[5].metric("Author countries", h["countries"])

    left, right = st.columns([3, 2])
    with left:
        by_year = df[df["dated"]].groupby(["year", "category"]).size().reset_index(name="books")
        fig = px.bar(
            by_year, x="year", y="books", color="category", color_discrete_map=CATEGORY_COLORS,
            category_orders={"category": CATEGORIES}, title="What I read each year (dated books)",
        )
        fig.update_xaxes(type="category", title=None)
        st.plotly_chart(style(fig), key="ov_year")
    with right:
        mix = df["category"].value_counts().reset_index()
        mix.columns = ["category", "books"]
        fig = px.pie(mix, names="category", values="books", hole=0.55, color="category",
                     color_discrete_map=CATEGORY_COLORS, title="Category mix")
        fig.update_traces(textinfo="percent", sort=False)
        st.plotly_chart(style(fig), key="ov_pie")

    left, right = st.columns(2)
    with left:
        lang_year = df[df["dated"]].groupby(["year", "read_language"]).size().reset_index(name="books")
        lang_year["share"] = lang_year["books"] / lang_year.groupby("year")["books"].transform("sum")
        fig = px.bar(lang_year, x="year", y="share", color="read_language", color_discrete_map=LANGUAGE_COLORS,
                     title="Language I read in, share per year")
        fig.update_yaxes(tickformat=".0%", title=None)
        fig.update_xaxes(type="category", title=None)
        st.plotly_chart(style(fig, 360), key="ov_lang")
    with right:
        tr = df.groupby(["read_language", "translated"]).size().reset_index(name="books")
        tr["how"] = tr["translated"].map({True: "In translation", False: "In the original"})
        fig = px.bar(tr, x="books", y="read_language", color="how", orientation="h",
                     color_discrete_map={"In translation": "#94A3B8", "In the original": "#4C6EF5"},
                     title="Original or translation?")
        fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, 360), key="ov_trans")

    top_authors = df.groupby("canonical_author").size().sort_values(ascending=False)
    top_authors = top_authors[top_authors > 1]
    st.subheader("Authors I keep coming back to")
    st.write(" · ".join(f"**{a}** ({n})" for a, n in top_authors.items()))

# --------------------------------------------------------------------------- timeline
with tabs[1]:
    p = pace(df)
    if not p:
        st.info("No dated books in the current filter.")
    else:
        c = st.columns(4)
        c[0].metric("Busiest month", p["busiest_month"], f"{p['busiest_count']} books", delta_color="off")
        c[1].metric("Longest monthly streak", f"{p['streak_months']} months", f"ending {p['streak_end']}", delta_color="off")
        gap_span = f"{p['gap_from']:%b %Y} → {p['gap_to']:%b %Y}" if p["gap_from"] is not None else ""
        c[2].metric("Longest gap", f"{p['longest_gap_days']} days", gap_span, delta_color="off")
        c[3].metric("Books per active month", f"{p['per_active_month']:.1f}",
                    f"{p['active_months']}/{p['total_months']} months active", delta_color="off")

        dated = df[df["dated"]].copy()
        dated["hover"] = dated["original_title"] + " — " + dated["canonical_author"]
        fig = px.strip(
            dated, x="finish_date", y="category", color="read_language", color_discrete_map=LANGUAGE_COLORS,
            hover_name="hover", hover_data={"finish_date": "|%d %b %Y", "category": False, "read_language": True},
            category_orders={"category": CATEGORIES}, title="Every finished book, by category",
        )
        fig.update_traces(marker=dict(size=11, opacity=0.85, line=dict(width=1, color="white")), jitter=0.35)
        if p["gap_from"] is not None:
            fig.add_vrect(x0=p["gap_from"], x1=p["gap_to"], fillcolor="#94A3B8", opacity=0.12, line_width=0,
                          annotation_text="longest gap", annotation_position="top left")
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, 460), key="tl_strip")

        monthly = p["monthly"].reset_index()
        monthly.columns = ["month", "books"]
        monthly["month"] = monthly["month"].dt.to_timestamp()
        fig = px.bar(monthly, x="month", y="books", title="Books finished per month")
        fig.update_traces(marker_color="#4C6EF5")
        fig.update_xaxes(title=None)
        st.plotly_chart(style(fig, 300), key="tl_month")

        cum = dated.sort_values("finish_date").assign(n=lambda d: range(1, len(d) + 1))
        fig = px.line(cum, x="finish_date", y="n", title="Cumulative books (dated only)", line_shape="hv")
        fig.update_traces(line_color="#0F9D58")
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, 300), key="tl_cum")

# --------------------------------------------------------------------------- eras
with tabs[2]:
    st.markdown(
        "Books are ordered by finish date and each becomes a vector over categories. An **optimal change-point "
        "segmentation** (dynamic programming, L2 cost plus a penalty per era) finds where the mix of what I "
        "read shifts. Raise the penalty for fewer, broader eras."
    )
    a, b = st.columns(2)
    penalty = a.slider("Penalty per era", 1.0, 4.0, 2.0, 0.25)
    min_size = b.slider("Minimum books per era", 3, 10, 4)
    eras, assign = detect_eras(df_all, penalty=penalty, min_size=min_size)
    stab = boundary_stability(df_all, min_size=min_size)

    st.markdown(era_story(eras))

    dated_all = df_all[df_all["dated"]].copy()
    dated_all["era"] = assign
    palette = ["rgba(76,110,245,0.08)", "rgba(232,163,23,0.10)"]
    fig = px.strip(
        dated_all, x="finish_date", y="category", color="category", color_discrete_map=CATEGORY_COLORS,
        hover_name="original_title", category_orders={"category": CATEGORIES},
    )
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")), jitter=0.3)
    for r in eras.itertuples():
        fig.add_vrect(x0=r.start, x1=r.end + pd.Timedelta(days=1), fillcolor=palette[r.era % 2], line_width=0,
                      annotation_text=f"<b>{r.era}</b>", annotation_position="top left",
                      annotation_font_size=13)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    st.plotly_chart(style(fig, 480), key="era_strip")

    cols = st.columns(min(len(eras), 4) or 1)
    for i, r in enumerate(eras.itertuples()):
        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"**Era {r.era}: {r.label}**")
                st.caption(f"{r.start:%b %Y} – {r.end:%b %Y} · {r.books} books")
                for cat, share in sorted(r.mix.items(), key=lambda kv: -kv[1])[:3]:
                    st.progress(float(share), text=f"{SHORT_NAMES.get(cat, cat)} {share:.0%}")
                if r.era > 1:
                    s = stab.get(r.start_pos)
                    if s is None:
                        near = [v for k, v in stab.items() if abs(k - r.start_pos) <= 2]
                        s = max(near) if near else 0.0
                    st.caption(f"Boundary robustness: {s:.0%} of penalty settings agree")

    with st.expander("How robust are the boundaries?"):
        st.write(
            "The segmentation is re-run for 26 penalty values between 1.0 and 3.5. A boundary that appears for "
            "most of them is a real shift in reading; one that appears only at low penalties is noise."
        )
        order = dated_all.sort_values(["finish_date", "id"]).reset_index(drop=True)
        sdf = pd.DataFrame({"pos": stab.index, "support": stab.values})
        sdf = sdf[sdf["pos"] < len(order)]
        sdf["date"] = order.loc[sdf["pos"], "finish_date"].to_numpy()
        sdf["first book"] = order.loc[sdf["pos"], "original_title"].to_numpy()
        fig = px.bar(sdf.sort_values("date"), x="date", y="support", hover_name="first book",
                     title="Share of penalty settings that place a boundary here")
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        fig.update_traces(marker_color="#9333EA", width=1000 * 3600 * 24 * 12)
        st.plotly_chart(style(fig, 300), key="era_stab")

# --------------------------------------------------------------------------- galaxy
with tabs[3]:
    st.markdown(
        "Each dot is a book, placed by the similarity of its themes (TF-IDF over tags and descriptions, "
        "reduced with SVD and t-SNE). Books that sit close together are about similar things. "
        "**Click a dot** to see its nearest neighbours."
    )
    g = works_all.copy()
    g["x"], g["y"] = space["xy"][:, 0], space["xy"][:, 1]
    _, assign_default = detect_eras(df_all)
    era_map = dict(zip(df_all.loc[assign_default.index, "original_title"], assign_default.astype(str)))
    g["era"] = g["original_title"].map(era_map).fillna("undated")
    g["fiction_label"] = g["fiction"].map({True: "Fiction", False: "Non-fiction"})
    g["size"] = g["times_read"].astype(float) * 10
    g["idx"] = range(len(g))

    color_by = st.radio("Colour by", ["category", "read_language", "era", "fiction_label", "continent"],
                        horizontal=True, format_func=lambda s: s.replace("_label", "").replace("_", " ").title())
    cmap = CATEGORY_COLORS if color_by == "category" else LANGUAGE_COLORS if color_by == "read_language" else None
    fig = px.scatter(
        g, x="x", y="y", color=color_by, color_discrete_map=cmap, size="size", size_max=16,
        hover_name="original_title", hover_data={"canonical_author": True, "x": False, "y": False, "size": False,
                                                 "idx": False, color_by: True},
        custom_data=["idx"], category_orders={"category": CATEGORIES},
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.9))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    event = st.plotly_chart(style(fig, 600), key="galaxy", on_select="rerun", selection_mode="points")

    picked = None
    try:
        pts = event.selection.points if event else []
        if pts:
            picked = int(pts[0]["customdata"][0])
    except (AttributeError, KeyError, IndexError, TypeError):
        picked = None
    titles = g["original_title"].tolist()
    default_idx = titles.index("Anna Karenina") if "Anna Karenina" in titles else 0
    choice = st.selectbox("…or pick a book", titles, index=picked if picked is not None else default_idx)
    i = titles.index(choice)
    row = g.iloc[i]
    a, b = st.columns([2, 3])
    with a:
        st.markdown(f"### {row['original_title']}")
        st.caption(f"{row['canonical_author']} · {row['category']} · {int(row['pub_year']) if pd.notna(row['pub_year']) else 'year unknown'}")
        st.write(row["description"])
    with b:
        st.markdown("**Closest books on my list**")
        for j, s in neighbours(space["read"], i, k=6):
            nb = g.iloc[j]
            st.markdown(f"- **{nb['original_title']}** — {nb['canonical_author']}  \n  <small>similarity {s:.2f} · {nb['category']}</small>",
                        unsafe_allow_html=True)

# --------------------------------------------------------------------------- time travel
with tabs[4]:
    st.markdown("How far back in time does each book take me? Publication year against the date I finished it.")
    tt = df[df["dated"] & df["pub_year"].notna()].copy()
    tt["age"] = (tt["finish_date"].dt.year - tt["pub_year"]).clip(lower=1)
    view = st.radio("Show", ["Years between writing and reading (log scale)", "Publication year"], horizontal=True)
    y = "age" if view.startswith("Years") else "pub_year"
    fig = px.scatter(
        tt, x="finish_date", y=y, color="category", color_discrete_map=CATEGORY_COLORS,
        hover_name="original_title", hover_data={"canonical_author": True, "pub_year": True, "age": True, "finish_date": "|%b %Y"},
        log_y=(y == "age"), category_orders={"category": CATEGORIES},
    )
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
    fig.update_xaxes(title="Finished")
    fig.update_yaxes(title="Years old when I read it" if y == "age" else "Published")
    if y == "age":
        fig.update_yaxes(tickvals=[1, 3, 10, 30, 100, 300, 1000, 2000])
    st.plotly_chart(style(fig, 520), key="tt_scatter")

    allpub = df[df["pub_year"].notna()].drop_duplicates("original_title").copy()
    allpub["era_of_writing"] = pd.cut(
        allpub["pub_year"], [0, 1500, 1800, 1900, 1950, 2000, 2010, 2030],
        labels=WRITING_ERAS,
    ).astype(str)
    fig = px.histogram(allpub, x="era_of_writing", color="category", color_discrete_map=CATEGORY_COLORS,
                       category_orders={"category": CATEGORIES, "era_of_writing": WRITING_ERAS},
                       title="When the books were written")
    fig.update_xaxes(title=None)
    st.plotly_chart(style(fig, 360), key="tt_hist")
    oldest = allpub.nsmallest(3, "pub_year")
    st.caption("Oldest: " + "; ".join(f"{r.original_title} (~{int(r.pub_year)})" for r in oldest.itertuples()))

# --------------------------------------------------------------------------- map
with tabs[5]:
    geo = df[df["author_country"] != ""].drop_duplicates("original_title")
    by_c = (
        geo.groupby(["author_country", "country_name"])
        .agg(books=("original_title", "size"), titles=("original_title", lambda s: "<br>".join(s.head(8))))
        .reset_index()
    )
    fig = px.choropleth(
        by_c, locations="author_country", color="books", hover_name="country_name",
        hover_data={"titles": True, "author_country": False}, color_continuous_scale="Blues",
        title="Where my authors come from",
    )
    fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth", bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(style(fig, 520), key="map")
    st.caption("Country = the country each author is chiefly associated with. Co-authored books use the first author.")

    a, b = st.columns(2)
    with a:
        cont = geo.groupby(["continent", "category"]).size().reset_index(name="books")
        fig = px.bar(cont, x="books", y="continent", color="category", orientation="h",
                     color_discrete_map=CATEGORY_COLORS, title="By continent")
        fig.update_yaxes(title=None, categoryorder="total ascending")
        st.plotly_chart(style(fig, 360), key="map_cont")
    with b:
        top = by_c.nlargest(10, "books")
        fig = px.bar(top, x="books", y="country_name", orientation="h", title="Top countries")
        fig.update_traces(marker_color="#4C6EF5")
        fig.update_yaxes(title=None, categoryorder="total ascending")
        st.plotly_chart(style(fig, 360), key="map_top")

# --------------------------------------------------------------------------- blind spots
with tabs[6]:
    st.markdown("An honest look at what my reading leaves out, computed from the data.")
    notes = blind_spots(df)
    sev_icon = {"high": "🔴", "medium": "🟠", "low": "🟢"}
    cols = st.columns(3)
    for k, n in enumerate(notes):
        with cols[k % 3]:
            with st.container(border=True):
                st.markdown(f"{sev_icon[n['severity']]} **{n['title']}**")
                st.markdown(f"<span style='font-size:2rem;font-weight:700'>{n['value']}</span>", unsafe_allow_html=True)
                st.caption(n["detail"])

    a, b = st.columns(2)
    with a:
        gdf = df.drop_duplicates("original_title")["author_gender"].map(
            {"M": "Men", "F": "Women", "Mixed": "Mixed co-authors", "Unknown": "Unknown"}).value_counts().reset_index()
        gdf.columns = ["author", "books"]
        fig = px.pie(gdf, names="author", values="books", hole=0.55, title="Author gender",
                     color="author", color_discrete_map={"Men": "#64748B", "Women": "#EC4899", "Mixed co-authors": "#A855F7", "Unknown": "#CBD5E1"})
        st.plotly_chart(style(fig, 360), key="bs_gender")
    with b:
        ol = df.drop_duplicates("original_title")["original_language"].value_counts().reset_index()
        ol.columns = ["language", "books"]
        fig = px.bar(ol, x="books", y="language", orientation="h", title="Language the book was written in")
        fig.update_traces(marker_color="#0891B2")
        fig.update_yaxes(title=None, categoryorder="total ascending")
        st.plotly_chart(style(fig, 360), key="bs_lang")

# --------------------------------------------------------------------------- what next
with tabs[7]:
    st.markdown(
        "Recommendations from a pool of candidate books, scored against a **recency-weighted taste profile** "
        "(recent books count more). Each pick says which books on my list it is closest to."
    )
    a, b = st.columns([2, 1])
    mode = a.radio("Mode", ["More of what I love", "Stretch my blind spots"], horizontal=True)
    half_life = b.slider("Taste memory (half-life, days)", 90, 1500, 540, 30,
                         help="Lower = only recent reading matters")
    recs = recommend(works_all, candidates, space["read"], space["cand"],
                     mode="stretch" if mode.startswith("Stretch") else "similar", half_life_days=half_life, n=8)
    cols = st.columns(2)
    for k, r in enumerate(recs.itertuples()):
        with cols[k % 2]:
            with st.container(border=True):
                st.markdown(f"**{k + 1}. {r.title}** — {r.author}")
                st.caption(f"{r.category} · {r.country_name} · {int(r.pub_year) if pd.notna(r.pub_year) else ''} · originally in {r.original_language}")
                st.write(r.description)
                if r.because:
                    st.markdown("Because you read: " + ", ".join(f"*{t}*" for t in r.because))
                if r.new_for_you:
                    st.markdown(" ".join(f"`{t}`" for t in r.new_for_you))

# --------------------------------------------------------------------------- data
with tabs[8]:
    st.subheader("Data quality")
    c = st.columns(4)
    c[0].metric("Undated books", len(quality.undated), help="Read before I started logging dates")
    c[1].metric("Books flagged to verify", len(quality.needs_verification), help="Metadata compiled by hand with low confidence")
    c[2].metric("Batch-logged dates", sum(quality.batch_dates.values()),
                help="3+ books sharing one finish date usually means they were logged together")
    c[3].metric("Rows out of date order", quality.out_of_order)
    if quality.batch_dates:
        st.caption("Dates shared by 3+ books: " + ", ".join(f"{d} ({n})" for d, n in quality.batch_dates.items()))
    if quality.missing_metadata:
        st.warning(f"Books with no metadata row yet: {quality.missing_metadata}. Add them to data/book_metadata.csv.")

    show = df_all[["id", "title", "original_title", "canonical_author", "finish_date", "read_language",
                   "original_language", "translated", "category", "pub_year", "country_name", "times_read", "verify"]]
    st.dataframe(show, hide_index=True, column_config={
        "finish_date": st.column_config.DateColumn("finished", format="DD.MM.YYYY"),
        "pub_year": st.column_config.NumberColumn("published", format="%d"),
    })
    st.download_button("Download enriched data (CSV)", show.to_csv(index=False).encode("utf-8"),
                       file_name="reading_atlas.csv", mime="text/csv")
