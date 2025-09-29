import streamlit as st
import pandas as pd
import altair as alt
from transformers import AutoTokenizer, AutoModelForCausalLM

# Title

mock_sequences = [
    "ACGTGCTAGCTAGCTAGGCTAAGCTAGCTAGGCTA",
    "CGTAGCTAGCTAGCTACGATCGTAGCTAGCTAGC",
    "GCTAGCTAGGCTAGCTAGCTAGCTAGGCTAGCTA"
]
mock_species = ["Protist A", "Cnidarian B", "Unknown Eukaryote"]
mock_abundance = [35, 50, 15]

st.title("Deep-Sea eDNA AI Analysis Dashboard")

# 1️⃣ Mock eDNA Input
st.header("Input eDNA Sequences")
sequence_input = st.text_area("Paste eDNA sequences (FASTA or raw DNA):", "\n".join(mock_sequences))

# 2️⃣ Load Pretrained Model (HyenaDNA)
st.header("Processing")
model_name = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

st.text("Sequences processed through AI model (HyenaDNA)")

# 3️⃣ Mock Prediction & Classification
st.header("Predicted Taxa & Abundance")
df = pd.DataFrame({
    "Taxa": mock_species,
    "Abundance": mock_abundance
})
st.table(df)

# 4️⃣ Visualization
st.header("Community Composition")
chart = alt.Chart(df).mark_bar().encode(
    x='Taxa',
    y='Abundance',
    color='Taxa'
)
st.altair_chart(chart, use_container_width=True)

st.header("Pie Chart View")
pie_chart = alt.Chart(df).mark_arc().encode(
    theta="Abundance",
    color="Taxa"
)
st.altair_chart(pie_chart, use_container_width=True)
