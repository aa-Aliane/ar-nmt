# run_sanity.py
from src.model_loader import load_nmt_model

for path in ["models/m2m100-trimmed", "models/m2m100_finetuned_20260417_153716"]:
    model, tokenizer = load_nmt_model(path, device="cuda")
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    
    inner.src_lang = "en"
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
    
    # Check what forced_bos_token_id resolves to
    ar_token_id = inner.lang_code_to_id["ar"]
    print(f"\n[{path}]")
    print(f"  ar lang token id : {ar_token_id}")
    print(f"  pad token id     : {inner.pad_token_id}")
    
    out = model.generate(**inputs, forced_bos_token_id=ar_token_id, max_new_tokens=50)
    print(f"  output           : {inner.decode(out[0], skip_special_tokens=True)}")