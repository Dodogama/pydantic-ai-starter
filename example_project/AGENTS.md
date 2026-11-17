# Conventions
- Pure Python stdlib only. No numpy/torch.
- Keep public signatures in model.py unchanged; train.py and eval.py depend on them.
- Never fit anything on validation data.
