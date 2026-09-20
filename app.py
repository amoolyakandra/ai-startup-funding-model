import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Startup Funding Assessment",
    page_icon="🚀",
    layout="wide"
)


# =========================================================
# LOAD TRAINED MODELS
# =========================================================

classifier = joblib.load("startup_classifier.pkl")
regressor = joblib.load("startup_regressor.pkl")
kmeans = joblib.load("startup_kmeans.pkl")
scaler = joblib.load("startup_scaler.pkl")

classification_features = joblib.load("classification_features.pkl")
cluster_features = joblib.load("cluster_features.pkl")


# =========================================================
# CLUSTER NAMES
# =========================================================

cluster_names = {
    0: "Market-Driven Startups",
    1: "Early-Stage / Developing Startups",
    2: "Product-Driven Startups",
    3: "Team & Network-Driven Startups"
}


# =========================================================
# BERKUS SCORE
# =========================================================

def calculate_berkus_score(data):

    idea_score = np.mean([
        data["problem_severity"],
        data["market_size"],
        data["market_growth"]
    ])

    prototype_score = np.mean([
        data["prototype_stage"],
        data["technology_readiness"],
        data["ip_strength"]
    ])

    team_score = np.mean([
        data["founder_experience"],
        data["team_expertise"]
    ])

    relationship_score = np.mean([
        data["strategic_partnerships"],
        data["advisor_strength"],
        data["incubator_support"]
    ])

    traction_score = np.mean([
        min(data["users"] / 1000, 10),
        data["pilot_customers"],
        data["letters_of_intent"]
    ])

    berkus_score = np.mean([
        idea_score,
        prototype_score,
        team_score,
        relationship_score,
        traction_score
    ])

    return round(berkus_score, 2)


# =========================================================
# STARTUP CLUSTER
# =========================================================

def get_cluster(startup_data):

    cluster_input = pd.DataFrame([startup_data])[cluster_features]

    cluster_scaled = scaler.transform(cluster_input)

    cluster_number = int(kmeans.predict(cluster_scaled)[0])

    return cluster_number, cluster_names.get(
        cluster_number,
        "Unclassified Startup"
    )


# =========================================================
# STRENGTHS AND RISKS
# =========================================================

def identify_strengths_risks(data):

    strengths = []
    risks = []

    if data["market_size"] >= 7:
        strengths.append("Large target market")
    elif data["market_size"] <= 4:
        risks.append("Limited market size")

    if data["market_growth"] >= 7:
        strengths.append("Strong market growth")
    elif data["market_growth"] <= 4:
        risks.append("Low market growth")

    if data["founder_experience"] >= 7:
        strengths.append("Experienced founder")
    elif data["founder_experience"] <= 3:
        risks.append("Limited founder experience")

    if data["team_expertise"] >= 7:
        strengths.append("Strong team expertise")
    elif data["team_expertise"] <= 4:
        risks.append("Team expertise needs strengthening")

    if data["prototype_stage"] >= 7:
        strengths.append("Advanced prototype")
    elif data["prototype_stage"] <= 4:
        risks.append("Early prototype stage")

    if data["users"] >= 2000:
        strengths.append("Strong user traction")
    elif data["users"] <= 500:
        risks.append("Limited user traction")

    if data["strategic_partnerships"] >= 7:
        strengths.append("Strong strategic partnerships")
    elif data["strategic_partnerships"] <= 3:
        risks.append("Limited strategic partnerships")

    if data["scalability"] >= 7:
        strengths.append("High scalability")
    elif data["scalability"] <= 4:
        risks.append("Limited scalability")

    if data["competition"] >= 8:
        risks.append("High competitive pressure")
    elif data["competition"] <= 4:
        strengths.append("Manageable competitive pressure")

    return strengths, risks


# =========================================================
# TITLE
# =========================================================

st.title("🚀 AI Startup Funding Assessment")

st.markdown(
    """
    **AI-powered decision-support system for evaluating pre-revenue,
    seed-stage startups.**

    The system combines machine learning with the **Berkus Method**
    to provide a structured funding-readiness assessment.
    """
)

st.divider()


# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("Startup Information")

industry = st.sidebar.selectbox(
    "Industry",
    [
        "FinTech",
        "HealthTech",
        "EdTech",
        "AgriTech",
        "CleanTech",
        "E-Commerce",
        "SaaS",
        "AI / DeepTech",
        "E-Commerce"
    ]
)

st.sidebar.subheader("Founder & Team")

founder_experience = st.sidebar.slider(
    "Founder Experience",
    0, 10, 5
)

team_size = st.sidebar.slider(
    "Team Size",
    1, 20, 5
)

team_expertise = st.sidebar.slider(
    "Team Expertise",
    0, 10, 5
)


st.sidebar.subheader("Product & Technology")

prototype_stage = st.sidebar.slider(
    "Prototype Stage",
    0, 10, 5
)

technology_readiness = st.sidebar.slider(
    "Technology Readiness",
    0, 10, 5
)

ip_strength = st.sidebar.slider(
    "IP Strength",
    0, 10, 5
)


st.sidebar.subheader("Market")

market_size = st.sidebar.slider(
    "Market Size",
    0, 10, 5
)

market_growth = st.sidebar.slider(
    "Market Growth",
    0, 10, 5
)

competition = st.sidebar.slider(
    "Competition",
    0, 10, 5
)


st.sidebar.subheader("Relationships")

strategic_partnerships = st.sidebar.slider(
    "Strategic Partnerships",
    0, 10, 5
)

advisor_strength = st.sidebar.slider(
    "Advisor Strength",
    0, 10, 5
)

incubator_support = st.sidebar.slider(
    "Incubator Support",
    0, 10, 5
)


st.sidebar.subheader("Traction")

users = st.sidebar.number_input(
    "Current Users",
    min_value=0,
    max_value=100000,
    value=1000
)

pilot_customers = st.sidebar.number_input(
    "Pilot Customers",
    min_value=0,
    max_value=100,
    value=5
)

letters_of_intent = st.sidebar.number_input(
    "Letters of Intent",
    min_value=0,
    max_value=100,
    value=3
)


st.sidebar.subheader("Business Potential")

funding_requested = st.sidebar.number_input(
    "Funding Requested (₹ Lakhs)",
    min_value=10,
    max_value=500,
    value=100
)

problem_severity = st.sidebar.slider(
    "Problem Severity",
    0, 10, 5
)

scalability = st.sidebar.slider(
    "Scalability",
    0, 10, 5
)


# =========================================================
# CREATE STARTUP DATA
# =========================================================

startup = {
    "founder_experience": founder_experience,
    "team_size": team_size,
    "team_expertise": team_expertise,
    "prototype_stage": prototype_stage,
    "technology_readiness": technology_readiness,
    "ip_strength": ip_strength,
    "market_size": market_size,
    "market_growth": market_growth,
    "competition": competition,
    "strategic_partnerships": strategic_partnerships,
    "advisor_strength": advisor_strength,
    "incubator_support": incubator_support,
    "users": users,
    "pilot_customers": pilot_customers,
    "letters_of_intent": letters_of_intent,
    "funding_requested": funding_requested,
    "problem_severity": problem_severity,
    "scalability": scalability
}


# =========================================================
# ASSESSMENT BUTTON
# =========================================================

if st.button("🔍 Assess Startup", use_container_width=True):

    # Create dataframe for ML model
    input_df = pd.DataFrame([startup])

    # Make sure feature order matches training
    X_classification = input_df[classification_features]

    # AI classification
    prediction = classifier.predict(X_classification)[0]

    probabilities = classifier.predict_proba(X_classification)[0]

    class_names = classifier.classes_

    probability_dict = dict(
        zip(class_names, probabilities)
    )

    # Regression score
    predicted_score = regressor.predict(
        X_classification
    )[0]

    predicted_score = np.clip(
        predicted_score,
        0,
        100
    )

    # Berkus score
    berkus_score = calculate_berkus_score(startup)

    # Cluster
    cluster_number, cluster_name = get_cluster(startup)

    # Strengths and risks
    strengths, risks = identify_strengths_risks(startup)


    # =====================================================
    # RESULTS
    # =====================================================

    st.divider()

    st.subheader("AI Funding Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "AI Funding Score",
            f"{predicted_score:.2f}/100"
        )

    with col2:
        st.metric(
            "Berkus Score",
            f"{berkus_score:.2f}/10"
        )

    with col3:
        st.metric(
            "AI Category",
            prediction
        )


    # =====================================================
    # PROBABILITIES
    # =====================================================

    st.subheader("Funding Category Probabilities")

    probability_cols = st.columns(len(class_names))

    for i, category in enumerate(class_names):

        with probability_cols[i]:

            st.metric(
                category,
                f"{probability_dict[category] * 100:.2f}%"
            )


    # =====================================================
    # CLUSTER
    # =====================================================

    st.subheader("Startup Cluster")

    st.info(
        f"Cluster {cluster_number}: **{cluster_name}**"
    )


    # =====================================================
    # STRENGTHS & RISKS
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Key Strengths")

        if strengths:

            for strength in strengths:
                st.success(strength)

        else:

            st.write("No major strengths identified.")


    with col2:

        st.subheader("Areas to Watch")

        if risks:

            for risk in risks:
                st.warning(risk)

        else:

            st.write("No major areas identified.")


    # =====================================================
    # BERKUS BREAKDOWN
    # =====================================================

    st.subheader("Berkus Method Breakdown")

    idea_score = np.mean([
        problem_severity,
        market_size,
        market_growth
    ])

    prototype_score = np.mean([
        prototype_stage,
        technology_readiness,
        ip_strength
    ])

    team_score = np.mean([
        founder_experience,
        team_expertise
    ])

    relationship_score = np.mean([
        strategic_partnerships,
        advisor_strength,
        incubator_support
    ])

    traction_score = np.mean([
        min(users / 1000, 10),
        pilot_customers,
        letters_of_intent
    ])

    berkus_data = pd.DataFrame({
        "Factor": [
            "Idea / Problem",
            "Prototype / Technology",
            "Team",
            "Relationships",
            "Traction"
        ],
        "Score": [
            idea_score,
            prototype_score,
            team_score,
            relationship_score,
            traction_score
        ]
    })

    st.bar_chart(
        berkus_data.set_index("Factor"))

        # ============================================
# AI VS BERKUS COMPARISON
# ============================================

st.subheader("AI vs. Berkus Method")
idea_score = np.mean([
    problem_severity,
    market_size,
    market_growth
])

prototype_score = np.mean([
    prototype_stage,
    technology_readiness,
    ip_strength
])

team_score = np.mean([
    founder_experience,
    team_expertise
])

relationship_score = np.mean([
    strategic_partnerships,
    advisor_strength,
    incubator_support
])

traction_score = np.mean([
    min(users / 1000, 10),
    pilot_customers,
    letters_of_intent
])

berkus_score = np.mean([
    idea_score,
    prototype_score,
    team_score,
    relationship_score,
    traction_score
])

comparison_data = pd.DataFrame({
    "Method": ["Berkus Method", "AI Model"],
    "Score": [
        berkus_score * 10,
        predicted_score
    ]
})

st.bar_chart(
    comparison_data.set_index("Method")
)

st.write(
    "The Berkus Method provides a structured early-stage assessment, "
    "while the AI model combines multiple startup characteristics "
    "to generate a broader funding-readiness score."
)


    # =====================================================
    # IMPORTANT NOTE
    # =====================================================

st.caption(
        "This system is a decision-support prototype trained on "
        "synthetic startup data. It does not replace human investor "
        "due diligence or guarantee funding outcomes."
    )

