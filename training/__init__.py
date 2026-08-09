"""The learning half of luna-shougi: HalfKP data, training, quantization.

The engine plays with integers on the CPU; this package trains the floats on
the GPU and turns them into those integers. training/verify.py is what keeps
the two honest.
"""
