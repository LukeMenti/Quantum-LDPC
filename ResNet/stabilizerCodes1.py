import os
from enum import Enum
from pathlib import Path
import numpy as np

class StabilizerCodesType(Enum):
    GeneralizedBicycle = 0
    HypergraphProduct = 1
    Toric = 3

class FileReader:
    def __init__(self, n, k, m, code_type, trained=False):
        self.mycodetype = code_type
        self.N = n
        self.K = k
        self.M = m
        self.G_rows = 0  # Initialize G_rows, you might need to set it later
        self.mTrained = trained
        self.trained_iter = 0  # Initialize trained_iter, you might need to set it later
        self.maxDc = 0  # Initialize maxDc, you might need to set it later
        self.maxDv = 0  # Initialize maxDv, you might need to set it later
        self.dv = []  # Initialize dv as an empty list
        self.dc = []  # Initialize dc as an empty list
        self.Nv = []  # Initialize Nv as an empty list
        self.Mc = []  # Initialize Mc as an empty list
        self.checkVal = []  # Initialize checkVal as an empty list
        self.varVal = []  # Initialize varVal as an empty list
        self.Nvk = []  # Initialize Nvk as an empty list
        self.Mck = []  # Initialize Mck as an empty list
        self.G = []  # Initialize G as an empty list
        self.weights_cn = []  # Initialize weights_cn as an empty list
        self.weights_vn = []  # Initialize weights_vn as an empty list
        self.weights_ri = []  # Initialize weights_ri as an empty list
        self.weights_llr = []  # Initialize weights_llr as an empty list

    def code_type_string(self):
        return str(self.mycodetype.name)

    def construct_weights_path(self, filename):
        return Path(filename)

    def load_cn_weights(self):
        pass  # Implement loading weights from file for cn weights

    def load_vn_weights(self):
        pass  # Implement loading weights from file for vn weights

    def load_llr_weights(self):
        pass  # Implement loading weights from file for llr weights

    def load_ri_weights(self):
        pass  # Implement loading weights from file for ri weights

    def read_H(self):
        pass  # Implement reading H matrix from file

    def read_G(self):
        pass  # Implement reading G matrix from file

    def check_symplectic(self):
        pass  # Implement checking symplectic condition

    @staticmethod
    def trace_inner_product(a, b):
        return bool(a & b)  # Implement trace inner product calculation



