import importlib

mods = ["torch", "transformers", "accelerate", "modelscope"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(m, "OK", getattr(mod, "__version__", "no_version"))
    except Exception as e:
        print(m, "FAIL", repr(e))
