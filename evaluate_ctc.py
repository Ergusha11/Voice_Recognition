import torch
from torch.utils.data import DataLoader
from dataset_ctc import CTCDataset, ctc_collate_fn
from model_ctc import CTCSpeechModel
from vocabulary import Vocabulary
import os
import numpy as np

# Función para calcular la Distancia de Levenshtein (necesaria para WER)
def wer(reference, hypothesis):
    ref = reference.split()
    hyp = hypothesis.split()
    d = np.zeros((len(ref) + 1) * (len(hyp) + 1), dtype=np.uint32)
    d = d.reshape((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(ref)][len(hyp)] / float(len(ref))

# Configuración
INPUT_SIZE = 13
HIDDEN_SIZE = 30
NUM_LAYERS = 2
BATCH_SIZE = 1
AUDIO_DIR = 'synthetic_phrases_test' # Usaremos el set de prueba independiente
MODEL_PATH = 'ctc_phrase_model.pth'

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = Vocabulary()
    
    # 1. Cargar Datos
    dataset = CTCDataset(AUDIO_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ctc_collate_fn)
    
    # 2. Cargar Modelo
    model = CTCSpeechModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, len(vocab)).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Modelo {MODEL_PATH} no encontrado.")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    total_samples = 0
    exact_matches = 0
    total_wer = 0
    
    print(f"Evaluando {len(dataset)} muestras...")
    
    with torch.no_grad():
        for i, (inputs, targets, input_lengths, target_lengths) in enumerate(loader):
            inputs = inputs.to(device)
            
            # Obtener el texto real (Ground Truth)
            # Reconstruimos la frase real a partir de los targets
            reference_text = vocab.int_to_text(targets.tolist())
            
            # Predicción
            outputs = model(inputs) # (Batch, Time, Classes)
            _, predicted_indices = torch.max(outputs, dim=2)
            hypothesis_text = vocab.int_to_text(predicted_indices.squeeze().tolist())
            
            # Limpieza básica para comparación (eliminar espacios extra)
            ref = " ".join(reference_text.split())
            hyp = " ".join(hypothesis_text.split())
            
            # Métricas
            if ref == hyp:
                exact_matches += 1
            
            error_rate = wer(ref, hyp)
            total_wer += error_rate
            total_samples += 1
            
            if i < 10: # Mostrar los primeros 10 ejemplos
                print(f"Ref: '{ref}' | Hyp: '{hyp}' | WER: {error_rate:.2f}")

    accuracy = (exact_matches / total_samples) * 100
    avg_wer = (total_wer / total_samples) * 100
    
    print("-" * 30)
    print(f"RESULTADOS DE LA EVALUACIÓN:")
    print(f"Precisión (Exact Match): {accuracy:.2f}%")
    print(f"Word Error Rate (WER): {avg_wer:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    evaluate()
