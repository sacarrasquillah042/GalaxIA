#!/usr/bin/env python3
"""
Verificación de la GPU antes de empezar. Correr esto ANTES que cualquier otra cosa.

    python scripts/check_gpu.py

Si la GPU no aparece, el plan de 4 días no funciona: hay que arreglarlo primero.
"""
import os
import sys

# Silencia los avisos informativos de TF (no son errores).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main() -> int:
    print(f"Python  : {sys.version.split()[0]}")
    print(f"Entorno : {sys.prefix}")
    if sys.prefix == sys.base_prefix:
        print("  AVISO: no parece que el entorno virtual esté activado.")

    try:
        import tensorflow as tf
    except ImportError:
        print("\nTensorFlow no está instalado en este entorno.")
        print("  pip install -r requirements-gpu.txt")
        return 1

    print(f"TensorFlow: {tf.__version__}")
    print(f"Compilado con CUDA: {tf.test.is_built_with_cuda()}")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"\nGPUs detectadas: {gpus}")

    if not gpus:
        print("\n" + "=" * 62)
        print("NO SE DETECTÓ GPU. Revisar, en este orden:")
        print("  1. nvidia-smi          -> ¿aparece la 4060 Ti?")
        print("  2. ¿está activado el .venv? (source .venv/bin/activate)")
        print("  3. ¿se instaló tensorflow[and-cuda] y no tensorflow-cpu?")
        print("     pip list | grep -E 'tensorflow|nvidia-cudnn'")
        print("  4. NO instalar el CUDA Toolkit del sistema: entra en")
        print("     conflicto con las librerías pip del entorno.")
        print("=" * 62)
        return 1

    # Sin esto TF reserva los 16 GB al arrancar y no se puede tener
    # un notebook y un script corriendo a la vez.
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
        print(f"  memory_growth activado en {g.name}")

    for d in tf.config.experimental.get_device_details(gpus[0]).items():
        print(f"  {d[0]}: {d[1]}")

    # --- Prueba real de cómputo ---
    with tf.device("/GPU:0"):
        x = tf.random.normal([1000, 1000])
        print(f"\nSuma de prueba: {tf.reduce_sum(x).numpy():.4f}")

        import time
        a = tf.random.normal([4096, 4096])
        tf.matmul(a, a).numpy()          # descarta la primera (incluye warm-up)
        t0 = time.time()
        for _ in range(10):
            tf.matmul(a, a)
        dt = (time.time() - t0) / 10
        tflops = 2 * 4096 ** 3 / dt / 1e12
        print(f"matmul 4096x4096: {dt*1000:.1f} ms  (~{tflops:.1f} TFLOPS fp32)")

    # --- mixed_float16: 1.5-2x en la CNN gracias a los tensor cores ---
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"\nPolítica de precisión: {tf.keras.mixed_precision.global_policy().name}")
    print("  Recordatorio: la última capa debe ir en float32 ->")
    print("  Dense(n_clases, activation='softmax', dtype='float32')")

    print("\nTodo listo. Se puede seguir con build_cache.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
