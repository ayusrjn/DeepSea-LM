import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel

# Title
st.title("Deep-Sea eDNA Prototype (HyenaDNA)")

# 1️⃣ Input DNA sequence
st.subheader("Input eDNA Sequence")
default_seq = "ACGTGCTAGCTAGCTAGGCTAAGCTAGCTAGGCTA"
sequence = st.text_area("Paste DNA sequence:", default_seq)

# 2️⃣ Load model (real HyenaDNA)
model_name = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# 3️⃣ Tokenize
tokens = tokenizer(sequence, return_tensors="pt").input_ids

# 4️⃣ Get embeddings instead of generate()
if st.button("Run AI Analysis"):
    with st.spinner("Processing with HyenaDNA..."):
        with torch.no_grad():
            outputs = model(tokens)
            # Take mean pooling across sequence
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    st.success("✅ Embedding generated!")

    # Show first few numbers of embedding
    st.subheader("Sequence Embedding (truncated)")
    st.code(embedding[:10], language="python")

    # Mock example: similarity score (self-similarity for now)
    similarity = float(torch.cosine_similarity(
        torch.tensor(embedding), torch.tensor(embedding), dim=0
    ))
    st.metric("Self-similarity", f"{similarity:.2f}")
