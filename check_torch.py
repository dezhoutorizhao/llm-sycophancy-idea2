try:
    import torch
    print("torch_ok", torch.__version__)
    print("cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch_fail", repr(e))
