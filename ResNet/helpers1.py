from fileReader import FileReader
import numpy as np

class StabilizerCodes:
    def __init__(self, n, k, m, codeType, fr, trained=False):
        self.mycodetype = codeType
        self.N = n
        self.K = k
        self.M = m
        self.G_rows = self.N + self.K
        self.mTrained = trained
        self.error = []
        self.error_hat = []
        self.syn = []
        self.errorString = []
        self.maxDc = 0
        self.maxDv = 0
        self.dv = []
        self.dc = []
        self.Nv = []
        self.Mc = []
        self.checkVal = []
        self.varVal = []
        self.Nvk = []
        self.Mck = []
        self.G = []
        self.weights_cn = []
        self.weights_vn = []
        self.weights_llr = []
        self.trained_iter = 0
        self.print_msg = False
        self.matrix_supplier = fr

    def decode(self, L, epsilon):
        # Implementation of the decode method
        pass

    def flooding_decode(self, L, epsilon):
        # Implementation of the flooding_decode method
        pass

    def check_success(self, Taux, Tauy, Tauz):
        # Implementation of the check_success method
        pass

    @staticmethod
    def trace_inner_product(a, b):
        # Implementation of the trace_inner_product method
        return not (a == 0 or b == 0 or a == b)

    def add_error_given_epsilon(self, epsilon):
        # Implementation of the add_error_given_epsilon method
        pass

    def calculate_syndrome(self):
        # Implementation of the calculate_syndrome method
        pass

    @staticmethod
    def quantize_belief(Taux, Tauy, Tauz):
        # Implementation of the quantize_belief method
        pass
