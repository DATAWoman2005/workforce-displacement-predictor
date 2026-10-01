from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="StrataWork | Workforce Transition Portal",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------
# Data paths and loading
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "features.csv"
MODEL_PATH = BASE_DIR / "models" / "experimental_random_forest.joblib"
ROLE_SKILL_PATH = BASE_DIR / "data" / "processed" / "role_skill_profiles.csv"
ROLE_SUMMARY_PATH = BASE_DIR / "data" / "processed" / "role_market_summary.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_recommender_data():
    role_skill_profile = pd.read_csv(
        ROLE_SKILL_PATH,
        index_col=0,
    )

    role_summary = pd.read_csv(ROLE_SUMMARY_PATH)

    return role_skill_profile, role_summary


# loading application artifacts
df = load_data()
model = load_model()

role_skill_profile, role_summary = load_recommender_data()

similarity_matrix = cosine_similarity(role_skill_profile)

role_similarity = pd.DataFrame(
    similarity_matrix,
    index=role_skill_profile.index,
    columns=role_skill_profile.index,
)


# ---------------------------------------------------------
# Career transition recommender
# ---------------------------------------------------------
def get_skill_candidates(current_role, top_n=5):
    """Return the most skill-similar alternative roles."""

    if current_role not in role_similarity.index:
        raise ValueError(f"Unknown role: {current_role}")

    candidates = (
        role_similarity.loc[current_role]
        .drop(current_role)
        .sort_values(ascending=False)
        .head(top_n)
        .rename("skill_similarity")
        .reset_index()
        .rename(columns={"Job_Title": "candidate_role"})
    )

    return candidates


def rank_transitions(current_role, top_n=3):
    """Rank alternative occupations using skill and market signals."""

    candidates = get_skill_candidates(
        current_role,
        top_n=len(role_similarity) - 1,
    )

    candidates = candidates.merge(
        role_summary,
        left_on="candidate_role",
        right_on="Job_Title",
        how="left",
    ).drop(columns="Job_Title")

    candidates["transition_score"] = (
        0.60 * candidates["skill_similarity"]
        + 0.25 * candidates["growth_share"]
        + 0.15 * (1 - candidates["high_risk_share"])
    )

    candidates = (
        candidates.sort_values(
            "transition_score",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return candidates


def explain_transition(current_role, candidate_role, top_n=3):
    """Identify shared strengths and destination-oriented skill gaps."""

    current = role_skill_profile.loc[current_role]
    candidate = role_skill_profile.loc[candidate_role]

    comparison = pd.DataFrame(
        {
            "current_profile": current,
            "candidate_profile": candidate,
        }
    )

    comparison["shared_strength"] = (
        comparison["current_profile"] + comparison["candidate_profile"]
    ) / 2

    comparison["skill_gap"] = (
        comparison["candidate_profile"] - comparison["current_profile"]
    )

    shared_strengths = (
        comparison.sort_values("shared_strength", ascending=False)
        .head(top_n)
        .index.tolist()
    )

    skills_to_strengthen = (
        comparison[comparison["skill_gap"] > 0]
        .sort_values("skill_gap", ascending=False)
        .head(top_n)
        .index.tolist()
    )

    return shared_strengths, skills_to_strengthen


def recommend_careers(current_role, top_n=3):
    """Return ranked career transitions with skill explanations."""

    recommendations = rank_transitions(
        current_role,
        top_n=top_n,
    ).copy()

    shared_list = []
    strengthen_list = []

    for candidate_role in recommendations["candidate_role"]:

        shared, strengthen = explain_transition(
            current_role,
            candidate_role,
        )

        shared_list.append(", ".join(shared))
        strengthen_list.append(", ".join(strengthen))

    recommendations["shared_strengths"] = shared_list
    recommendations["skills_to_strengthen"] = strengthen_list

    return recommendations


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("StrataWork")
st.subheader("Workforce Transition Portal")

st.caption(
    "Explore AI-related automation risk and skill-adjacent "
    "career transition opportunities."
)

st.divider()


# ---------------------------------------------------------
# Navigation
# ---------------------------------------------------------
worker_tab, workforce_tab = st.tabs(["👤 Worker View", "📊 Workforce View"])


# ---------------------------------------------------------
# Worker View
# ---------------------------------------------------------
with worker_tab:

    st.header("Worker Career Transition")

    st.write(
        "Explore an experimental automation-risk estimate and "
        "identify skill-adjacent career transition opportunities."
    )

    st.info(
        "Risk estimates are exploratory and are based on patterns "
        "in the supplied hackathon dataset."
    )

    st.subheader("1. Your Profile")

    # Available choices are taken directly from the supplied dataset.
    job_titles = sorted(df["Job_Title"].dropna().unique())
    industries = sorted(df["Industry"].dropna().unique())
    company_sizes = sorted(df["Company_Size"].dropna().unique())
    locations = sorted(df["Location"].dropna().unique())
    ai_levels = sorted(df["AI_Adoption_Level"].dropna().unique())
    skills = sorted(df["Required_Skills"].dropna().unique())
    remote_options = sorted(df["Remote_Friendly"].dropna().unique())
    growth_options = sorted(df["Job_Growth_Projection"].dropna().unique())

    salary_min = int(df["Salary_USD"].min())
    salary_max = int(df["Salary_USD"].max())
    salary_median = int(df["Salary_USD"].median())

    with st.form("worker_profile_form"):

        col1, col2 = st.columns(2)

        with col1:

            job_title = st.selectbox(
                "Job title",
                job_titles,
            )

            industry = st.selectbox(
                "Industry",
                industries,
            )

            company_size = st.selectbox(
                "Company size",
                company_sizes,
            )

            location = st.selectbox(
                "Location",
                locations,
            )

            ai_adoption = st.selectbox(
                "AI adoption level",
                ai_levels,
            )

        with col2:

            required_skill = st.selectbox(
                "Primary required skill",
                skills,
            )

            remote_friendly = st.selectbox(
                "Remote work",
                remote_options,
            )

            projected_growth = st.selectbox(
                "Projected job growth",
                growth_options,
            )

            salary = st.number_input(
                "Salary (USD)",
                min_value=salary_min,
                max_value=salary_max,
                value=salary_median,
                step=1000,
            )

        analyse_profile = st.form_submit_button(
            "Analyse Profile",
            type="primary",
            use_container_width=True,
        )

    if analyse_profile:

        st.success("Profile captured successfully.")

        profile = pd.DataFrame(
            {
                "Job_Title": [job_title],
                "Industry": [industry],
                "Company_Size": [company_size],
                "Location": [location],
                "AI_Adoption_Level": [ai_adoption],
                "Required_Skills": [required_skill],
                "Salary_USD": [salary],
                "Remote_Friendly": [remote_friendly],
                "Job_Growth_Projection": [projected_growth],
            }
        )

        st.subheader("Profile Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        summary_col1.metric(
            "Current role",
            job_title,
        )

        summary_col2.metric(
            "Industry",
            industry,
        )

        summary_col3.metric(
            "Primary skill",
            required_skill,
        )

        with st.expander("View profile details"):
            st.dataframe(
                profile,
                use_container_width=True,
                hide_index=True,
            )
        # ---------------------------------------------------------
        # Experimental automation-risk prediction
        # ---------------------------------------------------------
        prediction = model.predict(profile)[0]

        probabilities = model.predict_proba(profile)[0]
        classes = model.classes_

        probability_map = dict(zip(classes, probabilities))

        predicted_confidence = probability_map[prediction]

        st.divider()

        st.subheader("2. Experimental Automation-Risk Estimate")

        risk_col1, risk_col2 = st.columns(2)

        risk_col1.metric(
            "Predicted risk",
            prediction,
        )

        risk_col2.metric(
            "Model confidence",
            f"{predicted_confidence:.1%}",
        )

        st.write("**Class probabilities**")

        probability_df = pd.DataFrame(
            {
                "Risk level": classes,
                "Probability": probabilities,
            }
        ).set_index("Risk level")

        st.bar_chart(probability_df)

        st.caption(
            "This is an experimental model estimate based on patterns in the "
            "supplied hackathon dataset. Validation performance was close to "
            "chance level, so this result should not be interpreted as an "
            "individual probability of job displacement."
        )
        # ---------------------------------------------------------
        # Career transition recommendations
        # ---------------------------------------------------------
        recommendations = recommend_careers(
            job_title,
            top_n=3,
        )

        st.divider()

        st.subheader("3. Career Transition Opportunities")

        st.write(
            "These alternatives are ranked using skill similarity, "
            "projected growth and observed automation-risk patterns "
            "in the supplied dataset."
        )

        rec_cols = st.columns(3)

        for i, (_, row) in enumerate(recommendations.iterrows()):

            with rec_cols[i]:

                st.markdown(f"### {i + 1}. {row['candidate_role']}")

                st.metric(
                    "Transition score",
                    f"{row['transition_score']:.2f}",
                )

                st.write(f"**Skill similarity:** " f"{row['skill_similarity']:.0%}")

                st.write(f"**Median salary:** " f"${row['median_salary']:,.0f}")

                st.write(f"**Growth share:** " f"{row['growth_share']:.0%}")

                st.write(
                    f"**Observed high-risk share:** " f"{row['high_risk_share']:.0%}"
                )

                st.markdown("**Shared strengths**")

                st.write(
                    row["shared_strengths"]
                    if row["shared_strengths"]
                    else "No shared strengths identified."
                )

                st.markdown("**Skills to strengthen**")

                st.write(
                    row["skills_to_strengthen"]
                    if row["skills_to_strengthen"]
                    else "No additional skill gaps identified."
                )

        st.caption(
            "Career-transition scores are descriptive decision-support "
            "signals, not probabilities of career success. Skill profiles "
            "are derived from the supplied dataset, which records one "
            "required-skill category per observation."
        )
# ---------------------------------------------------------
# Workforce View
# ---------------------------------------------------------
with workforce_tab:

    st.header("Workforce Risk Overview")

    st.write(
        "Explore automation-risk patterns across the supplied job-market "
        "observations to support workforce planning and reskilling discussions."
    )

    st.info(
        "This dashboard describes the supplied hackathon dataset. "
        "The observed percentages should not be interpreted as population-level "
        "estimates of real-world job displacement."
    )

    # -----------------------------------------------------
    # 1. Workforce snapshot
    # -----------------------------------------------------
    st.subheader("1. Workforce Snapshot")

    risk_order = ["Low", "Medium", "High"]

    risk_counts = df["Automation_Risk"].value_counts().reindex(risk_order).fillna(0)

    risk_shares = risk_counts / len(df)

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Observations",
        f"{len(df):,}",
    )

    metric2.metric(
        "Low risk",
        f"{risk_shares['Low']:.1%}",
    )

    metric3.metric(
        "Medium risk",
        f"{risk_shares['Medium']:.1%}",
    )

    metric4.metric(
        "High risk",
        f"{risk_shares['High']:.1%}",
    )

    # -----------------------------------------------------
    # 2. Overall risk distribution
    # -----------------------------------------------------
    st.divider()

    st.subheader("2. Automation-Risk Distribution")

    risk_distribution = pd.DataFrame({"Share": risk_shares})

    st.bar_chart(risk_distribution)

    st.caption(
        "The target is approximately balanced across Low, Medium and High "
        "automation-risk categories."
    )

    # -----------------------------------------------------
    # 3. Risk patterns by role and industry
    # -----------------------------------------------------
    st.divider()

    st.subheader("3. Risk Patterns")

    role_tab, industry_tab = st.tabs(["By Job Title", "By Industry"])

    with role_tab:

        role_risk = pd.crosstab(
            df["Job_Title"],
            df["Automation_Risk"],
            normalize="index",
        )

        role_risk = role_risk.reindex(
            columns=risk_order,
            fill_value=0,
        )

        st.write(
            "Share of observations within each job title assigned to "
            "Low, Medium or High automation risk."
        )

        st.bar_chart(role_risk)

    with industry_tab:

        industry_risk = pd.crosstab(
            df["Industry"],
            df["Automation_Risk"],
            normalize="index",
        )

        industry_risk = industry_risk.reindex(
            columns=risk_order,
            fill_value=0,
        )

        st.write(
            "Share of observations within each industry assigned to "
            "Low, Medium or High automation risk."
        )

        st.bar_chart(industry_risk)

    # -----------------------------------------------------
    # 4. Evidence and interpretation
    # -----------------------------------------------------
    st.divider()

    st.subheader("4. Evidence & Interpretation")

    evidence_col1, evidence_col2 = st.columns(2)

    with evidence_col1:

        st.markdown("#### What the data shows")

        st.write(
            "• Automation risk is approximately balanced across the "
            "three target categories."
        )

        st.write(
            "• Descriptive differences can be observed across job titles, "
            "industries and other workforce characteristics."
        )

        st.write("• The dataset contains 500 complete job-market observations.")

    with evidence_col2:

        st.markdown("#### What the analysis does not establish")

        st.write(
            "• Stage 1 did not identify statistically robust associations "
            "between the supplied predictors and automation risk after "
            "multiple-testing correction."
        )

        st.write(
            "• The experimental classifier performed close to the "
            "three-class chance baseline."
        )

        st.write(
            "• These patterns therefore support exploration and scenario "
            "discussion rather than validated workforce forecasting."
        )

    st.warning(
        "Decision-support use only: workforce leaders should combine these "
        "signals with validated labour-market evidence, organizational context "
        "and human judgment before making employment or reskilling decisions."
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "10Alytics × JengaGlobal Q3 Students Hackathon 2026 · "
    "Track C: Data Science\n\n"
    "Developed by Adijat Adenaike"
)
