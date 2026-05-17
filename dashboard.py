import streamlit as st
import pandas as pd

st.title("🎓 Classroom AI Dashboard")

data = {
    "Student": ["1", "2", "3"],
    "Attention %": [92, 45, 81]
}

df = pd.DataFrame(data)

st.dataframe(df)

st.bar_chart(df.set_index("Student"))