import torch
import logging

def init_device(device_str: str) -> torch.device:
    # cuda
    if device_str == "cuda":
        assert torch.cuda.is_available(), "No CUDA device available!"
        logging.info(
            f"CUDA version: {torch.version.cuda}, CUDA device: {torch.cuda.current_device()}"
        )
        torch.cuda.init()
        return torch.device("cuda")
    # mps
    if device_str == "mps":
        assert torch.backends.mps.is_available(), "No MPS backend is available!"
        logging.info("Using MPS GPU acceleration")
        return torch.device("mps")
    # cpu
    if device_str == "cpu":
        logging.info("Using CPU")
        return torch.device("cpu")
    raise NotImplementedError
