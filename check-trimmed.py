import torch

from src.model_loader import load_nmt_model


def run_diagnostic(model_path):
    print(f"--- Diagnostic for: {model_path} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load the model and tokenizer using your custom loader
    # This automatically applies your RemappedTokenizer if vocab_id_map.json exists
    try:
        model, tokenizer = load_nmt_model(model_path, device)
        print("✓ Model and Tokenizer loaded successfully.")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return

    # 2. Check Vocabulary vs. Embedding Size
    vocab_size = len(tokenizer)
    embed_size = model.get_input_embeddings().weight.shape[0]

    print(f"Tokenizer vocab size: {vocab_size}")
    print(f"Model embedding size: {embed_size}")

    if vocab_size != embed_size:
        print(
            f"!!! CRITICAL MISMATCH: Vocab ({vocab_size}) != Embeddings ({embed_size})"
        )
        print("This is likely causing your CUDA index out-of-bounds error.")
    else:
        print("✓ Vocab and Embeddings are synchronized.")

    # 3. Test Encoding (Checking for the BatchEncoding sequence length error)
    test_text = "This is a test sentence."
    try:
        inputs = tokenizer(test_text, return_tensors="pt").to(device)
        print(
            f"✓ Tokenizer encoding test passed. Input IDs shape: {inputs.input_ids.shape}"
        )

        # 4. Check for Out-of-Bounds IDs
        max_id = inputs.input_ids.max().item()
        if max_id >= embed_size:
            print(
                f"!!! ERROR: Tokenizer produced ID {max_id}, which is >= Embed size {embed_size}"
            )
        else:
            print(f"✓ No out-of-bounds IDs detected in test sample (Max ID: {max_id})")

        # 5. Simple Forward Pass
        with torch.no_grad():
            # Dummy decoder_input_ids for encoder-decoder models
            decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(device)
            outputs = model(
                input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids
            )
            print("✓ Model forward pass successful.")

    except Exception as e:
        print(f"✗ Test execution failed: {e}")


if __name__ == "__main__":
    MODEL_DIR = "models/m2m100-trimmed"
    run_diagnostic(MODEL_DIR)
