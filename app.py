import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from llm_processor import (
    extract_food_information,
    generate_advice
)

from fuzzy_logic import (
    analyze_fuzzy_system,
    create_fuzzy_system
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.markdown(
    """
    <div style="
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        color: #176B3A;
        margin-top: 5px;
        margin-bottom: 25px;
    ">
        🍽️ Smart Food Waste Reduction Advisor
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CUSTOM CSS
#
# IMPORTANT:
# No HTML content is used for the actual application UI.
# This CSS only changes appearance.
# ============================================================

st.markdown(
    """
    <style>

    /* =========================================================
       MAIN APP
       ========================================================= */

    .stApp {
        background-color: #F4F7F5;
        color: #17221D;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =========================================================
       IMPORTANT:
       DO NOT USE GLOBAL color:white RULES.
       Streamlit text must remain dark on light backgrounds.
       ========================================================= */

    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown span {
        color: #17221D !important;
    }


    /* =========================================================
       HEADINGS
       ========================================================= */

    h1 {
        color: #176B3A !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #176B3A !important;
        font-weight: 800 !important;
    }

    h3 {
        color: #176B3A !important;
        font-weight: 700 !important;
    }

    h4 {
        color: #176B3A !important;
        font-weight: 700 !important;
    }


    /* =========================================================
       NORMAL TEXT
       ========================================================= */

    p {
        color: #34443B !important;
    }

    li {
        color: #34443B !important;
    }


    /* =========================================================
       HERO SECTION
       ========================================================= */

    .hero {
        background: linear-gradient(
            135deg,
            #176B3A,
            #2E8B57
        );

        border-radius: 24px;
        padding: 45px 35px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .hero-title {
        color: white !important;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        color: #F4FFF8 !important;
        font-size: 19px;
        line-height: 1.6;
    }


    /* =========================================================
       FEATURE CARDS
       ========================================================= */

    .feature-card {
        background-color: #FFFFFF;
        border: 1px solid #D8E5DD;
        border-radius: 18px;
        padding: 25px;
        min-height: 230px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.07);
    }

    .feature-card h3 {
        color: #176B3A !important;
        font-size: 22px;
        margin-bottom: 15px;
    }

    .feature-card p {
        color: #34443B !important;
        font-size: 16px;
        line-height: 1.7;
    }


    /* =========================================================
       INFO CARDS
       ========================================================= */

    .info-card {
        background-color: #FFFFFF;
        border: 1px solid #D8E5DD;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }

    .info-card-title {
        color: #176B3A !important;
        font-size: 19px;
        font-weight: 700;
    }

    .info-card-text {
        color: #34443B !important;
        font-size: 15px;
        line-height: 1.6;
    }


    /* =========================================================
       FOOD DETECTED
       ========================================================= */

    .food-detected {
        background-color: #E8F5EC;
        border-left: 6px solid #176B3A;
        border-radius: 12px;
        padding: 17px 22px;
        margin: 15px 0;
    }

    .food-detected-text {
        color: #176B3A !important;
        font-size: 20px;
        font-weight: 700;
    }


    /* =========================================================
       RISK BOX
       ========================================================= */

    .risk-box {
        background-color: #FFFFFF;
        border: 2px solid #2E8B57;
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }

    .risk-title {
        color: #34443B !important;
        font-size: 18px;
        font-weight: 700;
    }

    .risk-score {
        color: #176B3A !important;
        font-size: 42px;
        font-weight: 800;
        margin: 10px 0;
    }

    .risk-level {
        color: #176B3A !important;
        font-size: 22px;
        font-weight: 800;
    }


    /* =========================================================
       WHY RISK SECTION
       ========================================================= */

    .why-risk {
        background-color: #FFFFFF;
        border: 1px solid #D8E5DD;
        border-radius: 15px;
        padding: 22px;
        margin: 20px 0;
    }

    .why-risk p {
        color: #34443B !important;
        font-size: 16px;
        line-height: 1.7;
    }


    /* =========================================================
       ACTIVATED RULES
       ========================================================= */

    .rule-card {
        background-color: #FFFFFF;
        border-left: 6px solid #176B3A;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 12px 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.06);
    }

    .rule-card h3 {
        color: #176B3A !important;
        margin-bottom: 8px;
    }

    .rule-card p {
        color: #34443B !important;
        line-height: 1.6;
    }

    .rule-activation {
        color: #176B3A !important;
        font-weight: 700;
    }


    /* =========================================================
       FUZZIFICATION / DEFUZZIFICATION
       ========================================================= */

    .step-description {
        color: #34443B !important;
        font-size: 16px;
        line-height: 1.6;
    }


    /* =========================================================
       AI RECOMMENDATION
       ========================================================= */

    .recommendation-box {
        background-color: #E8F5EC;
        border: 1px solid #B9DCC5;
        border-left: 6px solid #176B3A;
        border-radius: 14px;
        padding: 22px;
        margin: 15px 0;
    }

    .recommendation-box p {
        color: #26352D !important;
        font-size: 16px;
        line-height: 1.7;
    }


    /* =========================================================
       PREVENTION TIPS
       ========================================================= */

    .tips-box {
        background-color: #FFFFFF;
        border: 1px solid #D8E5DD;
        border-radius: 15px;
        padding: 22px;
    }

    .tips-box p {
        color: #34443B !important;
        font-size: 16px;
        line-height: 1.8;
    }


    /* =========================================================
       PROJECT HIGHLIGHTS
       ========================================================= */

    .highlight-card {
        background-color: #FFFFFF;
        border: 1px solid #D8E5DD;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        min-height: 150px;
    }

    .highlight-title {
        color: #176B3A !important;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .highlight-text {
        color: #34443B !important;
        font-size: 15px;
        line-height: 1.6;
    }


    /* =========================================================
       FOOD SAFETY NOTICE
       ========================================================= */

    .safety-notice {
        background-color: #FFF4D6;
        border: 1px solid #E5B84B;
        border-left: 6px solid #D99A00;
        border-radius: 12px;
        padding: 18px 20px;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .safety-title {
        color: #7A4F00 !important;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .safety-text {
        color: #4A3A1A !important;
        font-size: 15px;
        line-height: 1.6;
    }


    /* =========================================================
       FOOTER
       ========================================================= */

    .footer {
        background-color: #E8F0EB;
        border-radius: 15px;
        padding: 20px;
        margin-top: 35px;
        text-align: center;
    }

    .footer p {
        color: #52635A !important;
        font-size: 14px;
    }


    /* =========================================================
       STREAMLIT INPUTS
       ========================================================= */

    label {
        color: #17221D !important;
        font-weight: 600 !important;
    }

    .stTextArea textarea {
        color: #17221D !important;
        background-color: #FFFFFF !important;
    }

    .stTextInput input {
        color: #17221D !important;
        background-color: #FFFFFF !important;
    }


    /* =========================================================
       BUTTONS
       ========================================================= */

    .stButton button {
        border-radius: 10px;
        font-weight: 700;
    }


    /* =========================================================
       DATAFRAME / TABLE TEXT
       ========================================================= */

    [data-testid="stDataFrame"] {
        color: #17221D !important;
    }


    /* =========================================================
       ALERTS
       ========================================================= */

    [data-testid="stAlert"] {
        color: #17221D !important;
    }



    /* =========================================================
       SIDEBAR - DARK BACKGROUND / LIGHT TEXT
       ========================================================= */

    section[data-testid="stSidebar"] {
        background-color: #1F2937 !important;
    }

    section[data-testid="stSidebar"] > div {
        background-color: #1F2937 !important;
    }

    /* Override the global markdown text colors inside sidebar */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #F3F4F6 !important;
    }

    /* Sidebar headings */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #22C55E !important;
        font-weight: 700 !important;
    }

    /* Sidebar divider */
    section[data-testid="stSidebar"] hr {
        border-color: #4B5563 !important;
    }

    /* Sidebar links, if any */
    section[data-testid="stSidebar"] a {
        color: #F3F4F6 !important;
    }

    /* Keep sidebar icons/emoji visible */
    section[data-testid="stSidebar"] svg {
        color: #F3F4F6 !important;
    }


    /* =========================================================
       SIDEBAR TEXT FIX
       ========================================================= */

    section[data-testid="stSidebar"] {
        background-color: #1F2937 !important;
    }

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #F3F4F6 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #22C55E !important;
    }

    section[data-testid="stSidebar"] strong,
    section[data-testid="stSidebar"] b {
        color: #F3F4F6 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #4B5563 !important;
    }

    section[data-testid="stSidebar"] a {
        color: #F3F4F6 !important;
    }

    section[data-testid="stSidebar"] button {
        color: #F3F4F6 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🍽️ Smart Food Waste Advisor")

    st.write(
        "This application combines "
        "**LangChain + LLM + Fuzzy Logic** "
        "to analyze food-waste risk."
    )

    st.divider()

    st.subheader("System Components")

    st.write("🤖 LangChain + Groq LLM")
    st.write("🧠 Fuzzy Inference System")
    st.write("📊 Membership Functions")
    st.write("📐 Fuzzy Rules")
    st.write("🎯 Defuzzification")
    st.write("💡 AI Recommendation")

    st.divider()

    st.subheader("Project Purpose")

    st.write(
        "Help users understand leftover-food "
        "waste risk and receive practical "
        "waste-reduction suggestions."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="main-title">🍽️ Smart Food Waste Reduction Advisor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI + Fuzzy Intelligence Decision Support System '
    'for Smart Food Waste Management'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PROJECT DESCRIPTION
# ============================================================

st.info(
    "Enter a natural-language description of ANY leftover food. "
    "The AI extracts the food information automatically, "
    "then the fuzzy inference system calculates the waste risk."
)


# ============================================================
# FEATURE OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">⭐ System Features</div>',
    unsafe_allow_html=True
)

feature1, feature2, feature3, feature4 = st.columns(4)

with feature1:

    st.success(
        "🤖 AI Extraction\n\n"
        "LangChain understands natural-language "
        "food descriptions and extracts food "
        "information automatically."
    )

with feature2:

    st.success(
        "🧠 Fuzzy Intelligence\n\n"
        "Membership functions and fuzzy rules "
        "handle uncertain real-world conditions."
    )

with feature3:

    st.success(
        "📊 Risk Analysis\n\n"
        "Calculates a continuous food-waste "
        "risk score from 0 to 100."
    )

with feature4:

    st.success(
        "💡 AI Advice\n\n"
        "Generates a personalized food-waste "
        "reduction recommendation."
    )


# ============================================================
# USER INPUT
# ============================================================

st.markdown(
    '<div class="section-title">📝 Food Waste Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "Describe the leftover food naturally."
)

user_input = st.text_area(
    "Food description",
    value="10% leftover rice for 2 days",
    height=130,
    placeholder=(
        "Example: 50% leftover chicken, "
        "30% expected demand, stored for 2 days"
    )
)


# ============================================================
# EXAMPLE INPUTS
# ============================================================

st.subheader("💬 Example Inputs")

example1, example2, example3 = st.columns(3)

with example1:

    if st.button(
        "🍚 Rice Example",
        use_container_width=True
    ):

        st.session_state.food_input = (
            "10% leftover rice for 2 days"
        )

        st.rerun()


with example2:

    if st.button(
        "🍗 Chicken Example",
        use_container_width=True
    ):

        st.session_state.food_input = (
            "40% leftover chicken, "
            "30% demand, stored for 2 days"
        )

        st.rerun()


with example3:

    if st.button(
        "🍝 Pasta Example",
        use_container_width=True
    ):

        st.session_state.food_input = (
            "60% leftover pasta, "
            "20% demand, stored for 4 days"
        )

        st.rerun()


# ============================================================
# USE SESSION STATE INPUT
# ============================================================

if "food_input" in st.session_state:

    user_input = st.session_state.food_input


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🔍 Analyze Food Waste Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if not user_input.strip():

        st.error(
            "Please enter a food description first."
        )

        st.stop()

    try:

        # ====================================================
        # STEP 1
        # LANGCHAIN FOOD EXTRACTION
        # ====================================================

        with st.spinner(
            "🤖 LangChain is understanding your food description..."
        ):

            food_data = extract_food_information(
                user_input
            )

        food_type = food_data["food_type"]

        leftover_quantity = food_data[
            "leftover_quantity"
        ]

        consumption_demand = food_data[
            "consumption_demand"
        ]

        storage_time = food_data[
            "storage_time"
        ]

        # ====================================================
        # DISPLAY EXTRACTED INFORMATION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🤖 AI-Extracted Food Information'
            '</div>',
            unsafe_allow_html=True
        )

        st.success(
            f"🍽️ Food Detected: **{food_type}**"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🍚 Leftover Quantity",
                f"{leftover_quantity:.0f}%"
            )

        with col2:

            st.metric(
                "👥 Consumption Demand",
                f"{consumption_demand:.0f}%"
            )

        with col3:

            st.metric(
                "📅 Storage Time",
                f"{storage_time:.1f} days"
            )

        # ====================================================
        # STEP 2
        # FUZZY ANALYSIS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🧠 Step 1 — Fuzzification'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            "The crisp input values are converted into "
            "fuzzy membership degrees between 0 and 1."
        )

        with st.spinner(
            "🧠 Running fuzzy inference..."
        ):

            fuzzy_result = analyze_fuzzy_system(
                leftover_quantity,
                consumption_demand,
                storage_time
            )

        risk_score = fuzzy_result[
            "risk_score"
        ]

        risk_level = fuzzy_result[
            "risk_level"
        ]

        membership_values = fuzzy_result[
            "membership_values"
        ]

        rule_evaluations = fuzzy_result[
            "rule_evaluations"
        ]

        defuzzified_score = fuzzy_result[
            "defuzzified_score"
        ]

        # ====================================================
        # MEMBERSHIP TABLE
        # ====================================================

        membership_table = pd.DataFrame({

            "Fuzzy Variable": [
                "Leftover Quantity",
                "Leftover Quantity",
                "Leftover Quantity",
                "Consumption Demand",
                "Consumption Demand",
                "Consumption Demand",
                "Storage Time",
                "Storage Time",
                "Storage Time"
            ],

            "Category": [
                "Low",
                "Medium",
                "High",
                "Low",
                "Medium",
                "High",
                "Short",
                "Medium",
                "Long"
            ],

            "Membership Degree": [

                membership_values[
                    "leftover"
                ]["Low"],

                membership_values[
                    "leftover"
                ]["Medium"],

                membership_values[
                    "leftover"
                ]["High"],

                membership_values[
                    "demand"
                ]["Low"],

                membership_values[
                    "demand"
                ]["Medium"],

                membership_values[
                    "demand"
                ]["High"],

                membership_values[
                    "storage"
                ]["Short"],

                membership_values[
                    "storage"
                ]["Medium"],

                membership_values[
                    "storage"
                ]["Long"]
            ]
        })

        st.dataframe(
            membership_table,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # MEMBERSHIP FUNCTION GRAPHS
        # ====================================================

        st.subheader(
            "📈 Membership Functions"
        )

        (
            leftover_var,
            demand_var,
            storage_var,
            waste_risk_var,
            _
        ) = create_fuzzy_system()

        graph1, graph2, graph3 = st.columns(3)

        # ----------------------------------------------------
        # LEFTOVER GRAPH
        # ----------------------------------------------------

        with graph1:

            fig1, ax1 = plt.subplots(
                figsize=(5, 3)
            )

            ax1.plot(
                leftover_var.universe,
                leftover_var["low"].mf,
                label="Low"
            )

            ax1.plot(
                leftover_var.universe,
                leftover_var["medium"].mf,
                label="Medium"
            )

            ax1.plot(
                leftover_var.universe,
                leftover_var["high"].mf,
                label="High"
            )

            ax1.axvline(
                leftover_quantity,
                linestyle="--",
                label="Input"
            )

            ax1.set_title(
                "Leftover Quantity"
            )

            ax1.set_xlabel(
                "Percentage"
            )

            ax1.set_ylabel(
                "Membership"
            )

            ax1.legend()

            ax1.grid(
                alpha=0.2
            )

            st.pyplot(
                fig1,
                clear_figure=True
            )

        # ----------------------------------------------------
        # DEMAND GRAPH
        # ----------------------------------------------------

        with graph2:

            fig2, ax2 = plt.subplots(
                figsize=(5, 3)
            )

            ax2.plot(
                demand_var.universe,
                demand_var["low"].mf,
                label="Low"
            )

            ax2.plot(
                demand_var.universe,
                demand_var["medium"].mf,
                label="Medium"
            )

            ax2.plot(
                demand_var.universe,
                demand_var["high"].mf,
                label="High"
            )

            ax2.axvline(
                consumption_demand,
                linestyle="--",
                label="Input"
            )

            ax2.set_title(
                "Consumption Demand"
            )

            ax2.set_xlabel(
                "Percentage"
            )

            ax2.set_ylabel(
                "Membership"
            )

            ax2.legend()

            ax2.grid(
                alpha=0.2
            )

            st.pyplot(
                fig2,
                clear_figure=True
            )

        # ----------------------------------------------------
        # STORAGE GRAPH
        # ----------------------------------------------------

        with graph3:

            fig3, ax3 = plt.subplots(
                figsize=(5, 3)
            )

            ax3.plot(
                storage_var.universe,
                storage_var["short"].mf,
                label="Short"
            )

            ax3.plot(
                storage_var.universe,
                storage_var["medium"].mf,
                label="Medium"
            )

            ax3.plot(
                storage_var.universe,
                storage_var["long"].mf,
                label="Long"
            )

            ax3.axvline(
                storage_time,
                linestyle="--",
                label="Input"
            )

            ax3.set_title(
                "Storage Time"
            )

            ax3.set_xlabel(
                "Days"
            )

            ax3.set_ylabel(
                "Membership"
            )

            ax3.legend()

            ax3.grid(
                alpha=0.2
            )

            st.pyplot(
                fig3,
                clear_figure=True
            )

        # ====================================================
        # STEP 2 — FUZZY RULE EVALUATION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🔥 Step 2 — Fuzzy Rule Evaluation'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            "The fuzzy rules evaluate the membership degrees "
            "and determine the strength of each rule."
        )

        rule_table = pd.DataFrame(
            rule_evaluations
        )

        rule_table = rule_table.rename(
            columns={
                "rule": "Rule",
                "condition": "Condition",
                "output": "Output",
                "activation": "Activation Strength"
            }
        )

        st.dataframe(
            rule_table,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # ACTIVATED RULES
        # ====================================================

        st.subheader(
            "🔥 Activated Rules"
        )

        activated_rules = [
            rule
            for rule in rule_evaluations
            if rule["activation"] > 0
        ]

        if activated_rules:

            for rule in activated_rules:

                st.info(
                    f"**{rule['rule']}**\n\n"
                    f"Condition: {rule['condition']}\n\n"
                    f"Output: {rule['output']}\n\n"
                    f"Activation Strength: "
                    f"{rule['activation']:.3f}"
                )

        else:

            st.warning(
                "No manually displayed rule had a "
                "positive activation strength."
            )

        # ====================================================
        # STEP 3 — DEFUZZIFICATION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🎯 Step 3 — Defuzzification'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            "The fuzzy inference system converts the "
            "fuzzy output into one crisp waste-risk score."
        )

        score_col1, score_col2 = st.columns(2)

        with score_col1:

            st.metric(
                "🎯 Fuzzy Waste Risk Score",
                f"{defuzzified_score:.2f} / 100"
            )

        with score_col2:

            if risk_level == "Low":

                st.success(
                    f"🟢 Risk Level: {risk_level}"
                )

            elif risk_level == "Medium":

                st.warning(
                    f"🟡 Risk Level: {risk_level}"
                )

            else:

                st.error(
                    f"🔴 Risk Level: {risk_level}"
                )

        # ====================================================
        # RISK PROGRESS
        # ====================================================

        st.progress(
            min(
                max(
                    int(risk_score),
                    0
                ),
                100
            )
        )

        # ====================================================
        # WHY THIS RISK?
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🔎 Why Did I Get This Risk?'
            '</div>',
            unsafe_allow_html=True
        )

        if leftover_quantity >= 60:

            st.write(
                f"• Leftover quantity is high "
                f"({leftover_quantity:.0f}%)."
            )

        elif leftover_quantity >= 20:

            st.write(
                f"• Leftover quantity is moderate "
                f"({leftover_quantity:.0f}%)."
            )

        else:

            st.write(
                f"• Leftover quantity is low "
                f"({leftover_quantity:.0f}%)."
            )

        if consumption_demand <= 40:

            st.write(
                f"• Consumption demand is low "
                f"({consumption_demand:.0f}%)."
            )

        elif consumption_demand <= 70:

            st.write(
                f"• Consumption demand is moderate "
                f"({consumption_demand:.0f}%)."
            )

        else:

            st.write(
                f"• Consumption demand is high "
                f"({consumption_demand:.0f}%)."
            )

        if storage_time <= 2:

            st.write(
                f"• Storage time is short "
                f"({storage_time:.1f} days)."
            )

        elif storage_time <= 4:

            st.write(
                f"• Storage time is moderate "
                f"({storage_time:.1f} days)."
            )

        else:

            st.write(
                f"• Storage time is relatively long "
                f"({storage_time:.1f} days)."
            )

        # ====================================================
        # AI RECOMMENDATION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '💡 AI Food Waste Recommendation'
            '</div>',
            unsafe_allow_html=True
        )

        with st.spinner(
            "💡 Generating personalized recommendation..."
        ):

            advice = generate_advice(
                food_type,
                leftover_quantity,
                consumption_demand,
                storage_time,
                risk_score,
                risk_level
            )

        st.success(
            advice
        )

        # ====================================================
        # FOOD WASTE PREVENTION TIPS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🌱 Food Waste Prevention Tips'
            '</div>',
            unsafe_allow_html=True
        )

        tips_col1, tips_col2 = st.columns(2)

        with tips_col1:

            st.write(
                "✅ Prepare portions according to expected demand."
            )

            st.write(
                "✅ Store leftovers in clearly labelled containers."
            )

            st.write(
                "✅ Keep track of stored leftovers."
            )

        with tips_col2:

            st.write(
                "✅ Plan meals before preparing large quantities."
            )

            st.write(
                "✅ Use older stored food appropriately before newer portions."
            )

            st.write(
                "✅ Freeze suitable leftovers when appropriate."
            )

        # ====================================================
        # SYSTEM ARCHITECTURE
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🏗️ System Architecture'
            '</div>',
            unsafe_allow_html=True
        )

        architecture = pd.DataFrame({

            "Stage": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6"
            ],

            "Component": [
                "User Input",
                "LangChain + LLM",
                "Fuzzification",
                "Fuzzy Rule Evaluation",
                "Defuzzification",
                "AI Recommendation"
            ],

            "Purpose": [
                "Natural-language food description",
                "Extract food type and numerical information",
                "Convert crisp values into membership degrees",
                "Apply fuzzy IF-THEN rules",
                "Generate a single risk score",
                "Explain the result in natural language"
            ]
        })

        st.dataframe(
            architecture,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # TECHNOLOGY STACK
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🛠️ Technology Stack'
            '</div>',
            unsafe_allow_html=True
        )

        technology = pd.DataFrame({

            "Technology": [
                "Python",
                "Streamlit",
                "LangChain",
                "Groq",
                "OpenAI GPT-OSS-20B",
                "scikit-fuzzy",
                "NumPy",
                "Pandas",
                "Matplotlib"
            ],

            "Purpose": [
                "Core programming language",
                "Web application interface",
                "LLM application framework",
                "LLM API provider",
                "Language model",
                "Fuzzy inference system",
                "Numerical computation",
                "Data tables",
                "Membership-function visualization"
            ]
        })

        st.dataframe(
            technology,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # PROJECT HIGHLIGHTS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '✨ Project Highlights'
            '</div>',
            unsafe_allow_html=True
        )

        highlight1, highlight2 = st.columns(2)

        with highlight1:

            st.write(
                "🤖 **Natural-Language AI**"
            )

            st.write(
                "The user can describe food naturally "
                "instead of filling multiple forms."
            )

            st.write(
                "🧠 **Genuine Fuzzy Inference**"
            )

            st.write(
                "The system uses membership functions, "
                "fuzzy rules and defuzzification."
            )

            st.write(
                "📊 **Explainable Analysis**"
            )

            st.write(
                "Membership values and activated rules "
                "are displayed to the user."
            )

            st.write(
                "🌱 **Food-Waste Reduction**"
            )

            st.write(
                "The system provides practical suggestions "
                "for reducing food waste."
            )

        with highlight2:

            st.write(
                "🍽️ **Any Food Type**"
            )

            st.write(
                "The AI can identify rice, pasta, chicken, "
                "vegetables, bread, fish and other foods."
            )

            st.write(
                "🎯 **Continuous Risk Score**"
            )

            st.write(
                "The fuzzy system produces a score from "
                "0 to 100."
            )

            st.write(
                "💡 **AI Explanation**"
            )

            st.write(
                "LangChain generates a human-readable "
                "recommendation."
            )

            st.write(
                "🌐 **Interactive Web Application**"
            )

            st.write(
                "The entire system runs through a "
                "Streamlit web interface."
            )

        # ====================================================
        # FOOD SAFETY NOTICE
        # ====================================================

        st.warning(
            "⚠️ Food Safety Notice: This application is an "
            "educational decision-support system for food-waste "
            "reduction. Risk scores are not a substitute for "
            "official food-safety guidance."
        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except json.JSONDecodeError:

        st.error(
            "The AI returned an invalid response. "
            "Please try the analysis again."
        )

    except Exception as e:

        st.error(
            "An error occurred while analyzing the food."
        )

        st.exception(e)