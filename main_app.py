import streamlit as st
import numpy as np
import torch
from helical.models.hyena_dna import HyenaDNAFineTuningModel, HyenaDNAConfig

# ---------------------------
# Device setup
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Load saved model
# ---------------------------
hyena_config = HyenaDNAConfig(
    model_name="hyenadna-tiny-1k-seqlen-d256",
    batch_size=1,
    device=device
)

# Load your fine-tuned model
hyena_fine_tune = HyenaDNAFineTuningModel.load_model(
    "hyenadna_tiny_finetuned", 
    hyena_config
)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Deep-Sea eDNA Classifier")
st.markdown("Enter a DNA sequence and the model will predict its class.")

sequence = st.text_area("DNA Sequence:", "ACGTACGTACGTACGT")

if st.button("Classify"):
    if not sequence:
        st.warning("Please enter a DNA sequence.")
    else:
        # Process the sequence
        dataset = hyena_fine_tune.process_data([sequence])
        
        # Get model outputs (logits)
        outputs = hyena_fine_tune.get_outputs(dataset)
        
        # Predicted class
        pred_class = np.argmax(outputs, axis=1)[0]
        
        # Confidence (softmax)
        probs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
        confidence = probs[0][pred_class] * 100
        
        st.success(f"Predicted class: {pred_class}")
        st.info(f"Confidence: {confidence:.2f}%")
