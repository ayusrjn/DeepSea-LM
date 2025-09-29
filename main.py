import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from helical.models.hyena_dna import HyenaDNAConfig, HyenaDNAFineTuningModel
import plotly.express as px

st.set_page_config(page_title="EDNA Prototype Dashboard", layout="wide")

st.markdown("<h1 style='text-align:center;color:teal;'>EDNA eDNA Classifier Prototype</h1>", unsafe_allow_html=True)
st.markdown("AI-driven analysis of DNA sequences to detect promoters as a proof-of-concept for deep-sea biodiversity applications.")

# Sidebar
st.sidebar.header("Instructions")
st.sidebar.info("""
1. Enter DNA sequences manually or upload a CSV/FASTA file.
2. Click 'Predict' to get predictions.
3. Explore probability plots and tables.
4. Future: classify eukaryotic/non-eukaryotic sequences from deep-sea eDNA.
""")

tab1, tab2 = st.tabs(["TATA Promoter Prediction", "Future: Eukaryotic/Non-eukaryotic"])

with tab1:
    st.subheader("Input DNA Sequences")
    seq_input = st.text_area("Enter sequences (one per line) or upload CSV", height=150)
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        sequences = df.iloc[:,0].astype(str).tolist()
    else:
        sequences = [seq.strip().upper() for seq in seq_input.split("\n") if seq.strip()]

    if st.button("Predict", key="predict_tab1"):
        if not sequences:
            st.warning("Please provide sequences!")
        else:
            # --- Load Model ---
            device = "cuda" if torch.cuda.is_available() else "cpu"
            hyena_config = HyenaDNAConfig(
                model_name="hyenadna-tiny-1k-seqlen-d256",
                batch_size=10,
                device=device
            )
            output_size = 2
            loaded_model = HyenaDNAFineTuningModel(
                hyena_config,
                fine_tuning_head="classification",
                output_size=output_size
            )
            state_dict = torch.load("hyenadna_promoter_tata_fine_tuned.pt", map_location=device)
            loaded_model.model.load_state_dict(state_dict)
            loaded_model.model.eval()

            processed_data = loaded_model.process_data(sequences)
            predictions = loaded_model.get_outputs(processed_data)

            # --- Apply Softmax for probabilities ---
            probs = F.softmax(torch.tensor(predictions), dim=1).numpy()

            # --- Argmax for predicted class (like classification_report) ---
            pred_class = predictions.argmax(axis=1)

            # Table
            result_df = pd.DataFrame({
                "Sequence": sequences,
                "Predicted Class": pred_class,
                "Prob 0": probs[:,0],
                "Prob 1": probs[:,1]
            })
            st.markdown("### Prediction Table")
            st.dataframe(result_df.style.highlight_max(subset=["Prob 0","Prob 1"], color="lightgreen"))

            # Probability Plot
            st.markdown("### Probability Plot")
            prob_df = pd.DataFrame(probs, columns=["Class 0","Class 1"])
            prob_df["Sequence"] = sequences
            prob_df_long = prob_df.melt(id_vars="Sequence", var_name="Class", value_name="Probability")
            fig = px.bar(prob_df_long, x="Sequence", y="Probability", color="Class", barmode="group",
                         title="Class Probabilities")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.info("Future functionality: classify eukaryotic vs non-eukaryotic sequences from deep-sea eDNA samples. This will scale to large datasets for biodiversity assessment.")
