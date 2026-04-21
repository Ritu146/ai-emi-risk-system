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

age = st.number_input("Age", key="age_input")
income = st.number_input("Monthly Income", key="income_input")

if st.button("Predict", key="predict_btn"):
    data = [0, 0, age, 0, 0, income, 0, 0, 0, 0, 0]
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
