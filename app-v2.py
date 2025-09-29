# --- IMPORTS ---
from helical.models.hyena_dna import HyenaDNA, HyenaDNAConfig, HyenaDNAFineTuningModel
import torch
import pandas as pd

# --- 1️⃣ SET DEVICE ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2️⃣ CONFIGURATION ---
hyena_config = HyenaDNAConfig(
    model_name="hyenadna-tiny-1k-seqlen-d256",
    batch_size=10,
    device=device
)

# --- 3️⃣ RECREATE MODEL ARCHITECTURE ---
output_size = 2  # Binary classification
loaded_hyena_fine_tune = HyenaDNAFineTuningModel(
    hyena_config,
    fine_tuning_head="classification",
    output_size=output_size
)

# --- 4️⃣ LOAD FINE-TUNED WEIGHTS ---
load_path = "hyenadna_promoter_tata_fine_tuned.pt"
state_dict = torch.load(load_path, map_location=device)
loaded_hyena_fine_tune.model.load_state_dict(state_dict)
loaded_hyena_fine_tune.model.eval()
print("Model loaded successfully and ready for inference.")

# --- 5️⃣ LOAD SEQUENCES ---
# Option 1: from CSV
# CSV should have a column 'sequence'
# df = pd.read_csv("all_sequences.csv")
# all_sequences = df['sequence'].tolist()

# Option 2: hardcoded sequences
all_sequences = [
    "TATAAA", "TATATA", "TATAGA", "TATAAT", "TATGAA",
    "TATTTA", "TATGTA", "TACAAA", "TATCTA"
]

# --- 6️⃣ PROCESS AND PREDICT ---
processed_data = loaded_hyena_fine_tune.process_data(all_sequences)
predictions = loaded_hyena_fine_tune.get_outputs(processed_data)
predicted_classes = predictions.argmax(axis=1)

# --- 7️⃣ OPTIONAL: GET PROBABILITIES ---
probabilities = torch.softmax(torch.tensor(predictions), dim=1)

# --- 8️⃣ DISPLAY RESULTS ---
results = []
for seq, pred, prob in zip(all_sequences, predicted_classes, probabilities):
    results.append({
        "Sequence": seq,
        "Predicted Class": int(pred),
        "Class 0 Prob": float(prob[0]),
        "Class 1 Prob": float(prob[1])
    })

results_df = pd.DataFrame(results)
print(results_df)

# --- 9️⃣ SAVE RESULTS (optional) ---
results_df.to_csv("tata_promoter_predictions.csv", index=False)
print("Predictions saved to 'tata_promoter_predictions.csv'.")
