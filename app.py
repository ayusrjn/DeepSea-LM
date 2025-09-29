import torch
import torch.nn.functional as F
from helical.models.hyena_dna import HyenaDNA, HyenaDNAConfig, HyenaDNAFineTuningModel

# --- 3A: Recreate the Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
hyena_config = HyenaDNAConfig(
    model_name="hyenadna-tiny-1k-seqlen-d256",
    batch_size=10,
    device=device
)

# --- 3B: Recreate the Model Architecture ---
output_size = 2 # Binary classification (0 and 1)
loaded_hyena_fine_tune = HyenaDNAFineTuningModel(
    hyena_config,
    fine_tuning_head="classification",
    output_size=output_size
)

# --- 3C: Load the Saved Weights ---
load_path = "hyenadna_promoter_tata_fine_tuned.pt"
state_dict = torch.load(load_path, map_location=device)
loaded_hyena_fine_tune.model.load_state_dict(state_dict)

# --- 3D: Set to Evaluation Mode ---
loaded_hyena_fine_tune.model.eval()
print("Model loaded successfully and ready for inference.")

# --- 3E: Make a Prediction (Inference) ---
new_sequence = ["TCCTATTGATTATGGGTTCGAATAGTACCAGATGTTTTGCCAATCCTAAATCGGTAGGAAAGTGGCTTGTCGTCGTCAGGCTTATTATCAACTCTTATGCACAAGAAAGGTACTCATCTTCTATAAACTACATAAGACCTGAATCTAATCAAAGGGAGAAAGCGCAGAACATCAGATTTAAAGCGGTTTTGCTTGATACACTCAGCCTTGTCTCTTTGTAAGGATTTTGGGGTACCTATGAATAATACATCTAGTAGTGTTAGTAAACCAACGTATGGGATTTTGGGATACATAGTTTTCCAGTGTTTCTTATCCGTGATAGTTTAATGGTCAGAATGGGCGCTTGTCGCGTGCCAGATCGGGGTTCAATTCCCCGTCGCGGAGAATTTTTTTAAGCTTCTATTAAAGAAGCTTTTTTTCACTTATATCTGATGGATGATGAATAGCTAGTTCAAACGGAAATCTTTGATAAAGCTATATCAAAATTCAAAGCCCAATAA"]

processed_data = loaded_hyena_fine_tune.process_data(new_sequence)
logits = loaded_hyena_fine_tune.get_outputs(processed_data)

# --- Apply Softmax to Get Probabilities ---
probabilities = F.softmax(torch.tensor(logits), dim=1)

# --- Get Predicted Class ---
predicted_class = probabilities.argmax(dim=1)

print(f"Probabilities: {probabilities}")
print(f"Predicted class: {predicted_class}")
