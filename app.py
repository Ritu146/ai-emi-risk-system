import streamlit as st
from predict import predict

# Optional imports (safe handling)
try:
    from rag import retrieve
    from llm import generate
    CHAT_ENABLED = True
except:
    CHAT_ENABLED = False

# --------------------------
# TITLE
# --------------------------
st.title("💳 AI Credit Risk App")

# --------------------------
# PREDICTION SECTION
# --------------------------
st.header("📊 Credit Risk Prediction")

# age = st.number_input("Age", key="age_input")
# income = st.number_input("Monthly Income", key="income_input")
age = st.number_input("Age", key="age")
NumberOfTime30_59DaysPastDueNotWorse = st.number_input("30-59 Days Past Due", key="dpd_30")
DebtRatio = st.number_input("Debt Ratio", key="debt_ratio")
income = st.number_input("Monthly Income", key="income")
NumberOfOpenCreditLinesAndLoans = st.number_input("Open Credit Lines", key="open_credit")
NumberOfTimes90DaysLate = st.number_input("90 Days Late", key="late_90")
NumberRealEstateLoansOrLines = st.number_input("Real Estate Loans", key="real_estate")
NumberOfTime60_89DaysPastDueNotWorse = st.number_input("60-89 Days Past Due", key="dpd_60")
NumberOfDependents = st.number_input("Dependents", key="dependents")

if st.button("Predict", key="predict_btn"):
    data = [0, 0, age, NumberOfTime30_59DaysPastDueNotWorse, DebtRatio, income, NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate, NumberRealEstateLoansOrLines, NumberOfTime60_89DaysPastDueNotWorse, NumberOfDependents]
    result = predict(data)
    st.success(f"Default Probability: {result}")

# --------------------------
# CHAT SECTION
# --------------------------
st.header("💬 Ask AI About Risk")

if CHAT_ENABLED:
    query = st.text_input("Ask something about risk:", key="chat_input")

    if query:
        try:
            context = retrieve(query)
            answer = generate(query, context)

            st.write("### Answer:")
            st.write(answer)

        except Exception as e:
            st.error("Chat system error. Check model loading.")
else:
    st.warning("Chat feature not available (rag/llm not loaded)")
