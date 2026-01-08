import subprocess
import sys


def install_cpu_torch():
    print(f"Folosesc Python de aici: {sys.executable}")

    print("--- 1. Dezinstalez vechiul Torch... ---")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])

    print("--- 2. Instalez Torch CPU (fara erori DLL)... ---")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])

    print("\n Instalarea s-a terminat cu succes.")


if __name__ == "__main__":
    try:
        install_cpu_torch()

        print("--- 3. Testez daca merge... ---")
        import torch

        print(f"Versiune Torch instalata: {torch.__version__}")
        x = torch.rand(2, 2)
        print("Tensor de test:\n", x)
        print("  Poti rula train.py acum.")

    except Exception as e:
        print(f" Ceva nu a mers: {e}")
