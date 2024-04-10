import os
import numpy as np
import os
import numpy as np

class FileReader:
    def __init__(self, n, k, m, code_type, trained):
        self.mycodetype = code_type
        self.N = n
        self.K = k
        self.M = m
        self.G_rows = n + k
        self.mTrained = trained
        self.maxDv = 0
        self.maxDc = 0
        self.dv = []
        self.dc = []
        self.Nvk = []
        self.Mck = []
        self.Nv = []
        self.Mc = []
        self.checkVal = []
        self.varVal = []
        self.G = []
        self.weights_cn = []
        self.weights_ri = []
        self.weights_llr = []
        self.weights_vn = []
        self.trained_iter = 0
        self.read_H()
        self.read_G()
        if trained:
            self.load_cn_weights()
            self.load_llr_weights()
            self.load_vn_weights()
            self.load_ri_weights()
    print("ciao")
    def read_H(self):
        code_type_string = self.code_type_string()
        filename = f"./PCMs/{code_type_string}_{self.N}_{self.K}/{code_type_string}_{self.N}_{self.K}_H_{self.M}.alist"
        checkValues = []
        VariableValues = []
        with open(filename, 'r') as matrix_file:
            n, m = map(int, matrix_file.readline().split())
            assert n == self.N, "read_H: file-specified N not as expected"
            assert m == self.M, "read_H: file-specified M not as expected"
            self.maxDv, self.maxDc = map(int, matrix_file.readline().split())

            self.dv = list(map(int, matrix_file.readline().split()))
            self.Nvk = [[] for _ in range(n)]
            self.dv.extend(map(int, matrix_file.readline().split()))  # Last value
            self.dc = list(map(int, matrix_file.readline().split()))
            self.Mck = [[] for _ in range(m)]
            self.dc.extend(map(int, matrix_file.readline().split()))  # Last value

            for i in range(n):
                self.Nv.append([0] * self.dv[i])
                line = list(map(int, matrix_file.readline().split()))
                print("Line:", line)  # Add this line for debugging
                
                for j in range(self.dv[i]):
                    print("i:", i, "j:", j, "len(line):", len(line))
                    self.Nv[i][j] = line[j] - 1
                    self.Mck[self.Nv[i][j]].append(j)

            for i in range(m):
                self.Mc.append([0] * self.dc[i])
                line = list(map(int, matrix_file.readline().split()))
                for j in range(self.dc[i]):
                    self.Mc[i][j] = line[j] - 1
                    self.Nvk[self.Mc[i][j]].append(j)

            for i in range(m):
                checkValues.append(list(map(int, matrix_file.readline().split())))

            for i in range(n):
                VariableValues.append(list(map(int, matrix_file.readline().split())))

        self.checkVal = checkValues
        self.varVal = VariableValues

    def read_G(self):
        code_type_string = self.code_type_string()
        filename = f"./PCMs/{code_type_string}_{self.N}_{self.K}/{code_type_string}_{self.N}_{self.K}_G.txt"
        with open(filename, 'r') as matrix_file:
            for _ in range(self.G_rows):
                row = list(map(int, matrix_file.readline().split()))
                self.G.append(row)

    def load_cn_weights(self):
        path = self.construct_weights_path("weight_cn.txt")
        weight_cn = []
        with open(path, 'r') as weights_file:
            dec_iter = int(weights_file.readline())
            self.trained_iter = dec_iter
            n, m = map(int, weights_file.readline().split())
            assert n == self.N, "load_cn_weights: file-specified N not as expected"
            assert m == self.M, "load_cn_weights: file-specified M not as expected"
            for _ in range(dec_iter):
                weight_cn_tmp = []
                for _ in range(m):
                    weight_cn_row = list(map(float, weights_file.readline().split()))
                    weight_cn_tmp.append(weight_cn_row)
                weight_cn.append(weight_cn_tmp)
        self.weights_cn = weight_cn

    def load_ri_weights(self):
        path = self.construct_weights_path("weight_ri.txt")
        weight_ri = []
        with open(path, 'r') as weights_file:
            dec_iter = int(weights_file.readline())
            self.trained_iter = dec_iter
            n, m = map(int, weights_file.readline().split())
            assert n == self.N, "load_ri_weights: file-specified N not as expected"
            assert m == self.M, "load_ri_weights: file-specified M not as expected"
            for _ in range(dec_iter):
                weight_ri_tmp = []
                for _ in range(m):
                    weight_ri_row = list(map(float, weights_file.readline().split()))
                    weight_ri_tmp.append(weight_ri_row)
                weight_ri.append(weight_ri_tmp)
        self.weights_ri = weight_ri

    def load_llr_weights(self):
        path = self.construct_weights_path("weight_llr.txt")
        weight_llr = []
        with open(path, 'r') as weights_file:
            dec_iter = int(weights_file.readline())
            for _ in range(dec_iter):
                weight_llr.append(list(map(float, weights_file.readline().split())))
        self.weights_llr = weight_llr

    def load_vn_weights(self):
        path = self.construct_weights_path("weight_vn.txt")
        weight_vn = []
        with open(path, 'r') as weights_file:
            dec_iter = int(weights_file.readline())
            n, m = map(int, weights_file.readline().split())
            assert n == self.N, "load_vn_weights: file-specified N not as expected"
            assert m == self.M, "load_vn_weights: file-specified M not as expected"
            for _ in range(dec_iter):
                weight_vn_tmp = []
                for _ in range(n):
                    weight_vn_row = list(map(float, weights_file.readline().split()))
                    weight_vn_tmp.append(weight_vn_row)
                weight_vn.append(weight_vn_tmp)
        self.weights_vn = weight_vn

    def construct_weights_path(self, filename):
        code_type_string = self.code_type_string()
        directory_name_builder = f"{code_type_string}_{self.N}_{self.K}_{self.M}"
        path = os.path.join("training_results", directory_name_builder, filename)
        return path

    def code_type_string(self):
        if self.mycodetype == 'GB':
            return "GB"
        elif self.mycodetype == 'HP':
            return "HP"
        elif self.mycodetype == 'toric':
            return "toric"
        else:
            raise ValueError("Unimplemented codetype")

    def check_symplectic(self):
        for rowid in range(self.M):
            vec1 = np.zeros(self.N, dtype=int)
            vec2 = np.zeros(self.N, dtype=int)

            for j in range(self.dc[rowid]):
                vec1[self.Mc[rowid][j]] = self.checkVal[rowid][j]

            for i in range(self.M):
                for j in range(self.dc[i]):
                    vec2[self.Mc[i][j]] = self.checkVal[i][j]

                syn_check = False
                for j in range(self.N):
                    syn_check = self.trace_inner_product(vec1[j], vec2[j]) != 0
                    if syn_check:
                        raise ValueError("check_symplectic: syn_check % 2 != 0")

            for i in range(self.G_rows):
                syn_check = False
                for j in range(self.N):
                    syn_check = self.trace_inner_product(vec1[j], self.G[i][j]) != 0
                    if syn_check:
                        raise ValueError("check_symplectic / GH^T: syn_check % 2 != 0")

        print("Check symplectic ok")

    @staticmethod
    def trace_inner_product(a, b):
        return not (a == 0 or b == 0 or a == b)

# Enum definition for stabilizerCodesType
class StabilizerCodesType:
    GeneralizedBicycle = "GB"
    HypergraphProduct = "HP"
    Toric = "toric"

class FileReader:
    def __init__(self, n, k, m, code_type, trained):
        self.mycodetype = code_type
        self.N = n
        self.K = k
        self.M = m
        self.G_rows = n + k
        self.mTrained = trained
        self.maxDv = 0
        self.maxDc = 0
        self.dv = []
        self.dc = []
        self.Nvk = []
        self.Mck = []
        self.Nv = []
        self.Mc = []
        self.checkVal = []
        self.varVal = []
        self.G = []
        self.weights_cn = []
        self.weights_ri = []
        self.weights_llr = []
        self.weights_vn = []
        self.trained_iter = 0
        self.read_H()
        self.read_G()
        if trained:
            self.load_cn_weights()
            self.load_llr_weights()
            self.load_vn_weights()
            self.load_ri_weights()

    def read_H(self):
        code_type_string = self.code_type_string()
        filename = f"./PCMs/{code_type_string}_{self.N}_{self.K}/{code_type_string}_{self.N}_{self.K}_H_{self.M}.alist"
        checkValues = []
        VariableValues = []
        with open(filename, 'r') as matrix_file:
            n, m = map(int, matrix_file.readline().split())
            assert n == self.N, "read_H: file-specified N not as expected"
            assert m == self.M, "read_H: file-specified M not as expected"
            self.maxDv, self.maxDc = map(int, matrix_file.readline().split())

            self.dv = list(map(int, matrix_file.readline().split()))
            self.Nvk = [[] for _ in range(n)]
            self.dv.extend(map(int, matrix_file.readline().split()))  # Last value
            self.dc = list(map(int, matrix_file.readline().split()))
            self.Mck = [[] for _ in range(m)]
            self.dc.extend(map(int, matrix_file.readline().split()))  # Last value

            for i in range(n):
                self.Nv.append([0] * self.dv[i])
                line = list(map(int, matrix_file.readline().split()))
                print("Line:", line)  # Add this line for debugging
                
                for j in range(self.dv[i]):
                    print("i:", i, "j:", j, "len(line):", len(line))
                    self.Nv[i][j] = line[j] - 1
                    self.Mck[self.Nv[i][j]].append(j)

            for i in range(m):
                self.Mc.append([0] * self.dc[i])
                line = list(map(int, matrix_file.readline().split()))
                for j in range(self.dc[i]):
                    self.Mc[i][j] = line[j] - 1
                    self.Nvk[self.Mc[i][j]].append(j)

            for i in range(m):
                checkValues.append(list(map(int, matrix_file.readline().split())))

            for i in range(n):
                VariableValues.append(list(map(int, matrix_file.readline().split())))

        self.checkVal = checkValues
        self.varVal = VariableValues

    def read_G(self):
        code_type_string = self.code_type_string()
        filename = f"./PCMs/{code_type_string}_{self.N}_{self.K}/{code_type_string}_{self.N}_{self.K}_G.txt"
        with open(filename, 'r') as matrix_file:
            for _ in range(self.G_rows):
                row = list(map(int, matrix_file.readline().split()))
                self.G.append(row)

    def load_cn_weights(self):
        path = self.construct_weights_path("weight_cn.txt")
        weight_cn = []
        with open(path, 'r') as weights_file:
            dec_iter = int(weights_file.readline())
            self.trained_iter = dec_iter
            n, m = map(int, weights_file.readline().split())
            assert n == self.N, "load_cn_weights: file-specified N not as expected"
            assert m == self.M, "load_cn_weights: file-specified M not as expected"
            for _ in range(dec_iter):
                weight_cn_tmp = []
                for _ in range(m):
                    weight_cn_row = list(map(float, weights_file.readline().split()))
                    weight_cn_tmp.append(weight_cn_row)
                weight_cn.append(weight_cn_tmp)
        self.weights_cn = weight_cn

    def load_ri_weights(self):
        path = self.construct_weights_path("weight_ri.txt")
        weight_ri = []
        with open(path, 'r') as weights_file:
            dec_iter = int(weights_file.readline())
            self.trained_iter = dec_iter
            n, m = map(int, weights_file.readline().split())
            assert n == self.N, "load_ri_weights: file-specified N not as expected"
            assert m == self.M, "load_ri_weights: file-specified M not as expected"
            for _ in range(dec_iter):
                weight_ri_tmp = []
                for _ in range(m):
                    weight_ri_row = list(map(float, weights_file.readline().split()))
                    weight_ri_tmp.append(weight_ri_row)
                weight_ri.append(weight_ri_tmp)
        self.weights_ri = weight_ri

    def load_llr_weights(self):
        path = self.construct_weights_path("weight_llr.txt")
        weight_llr = []
        with open(path, 'r') as weights_file:
            dec_iter = int(weights_file.readline())
            for _ in range(dec_iter):
                weight_llr.append(list(map(float, weights_file.readline().split())))
        self.weights_llr = weight_llr

    def load_vn_weights(self):
        path = self.construct_weights_path("weight_vn.txt")
        weight_vn = []
        with open(path, 'r') as weights_file:
            dec_iter = int(weights_file.readline())
            n, m = map(int, weights_file.readline().split())
            assert n == self.N, "load_vn_weights: file-specified N not as expected"
            assert m == self.M, "load_vn_weights: file-specified M not as expected"
            for _ in range(dec_iter):
                weight_vn_tmp = []
                for _ in range(n):
                    weight_vn_row = list(map(float, weights_file.readline().split()))
                    weight_vn_tmp.append(weight_vn_row)
                weight_vn.append(weight_vn_tmp)
        self.weights_vn = weight_vn

    def construct_weights_path(self, filename):
        code_type_string = self.code_type_string()
        directory_name_builder = f"{code_type_string}_{self.N}_{self.K}_{self.M}"
        path = os.path.join("training_results", directory_name_builder, filename)
        return path

    def code_type_string(self):
        if self.mycodetype == 'GB':
            return "GB"
        elif self.mycodetype == 'HP':
            return "HP"
        elif self.mycodetype == 'toric':
            return "toric"
        else:
            raise ValueError("Unimplemented codetype")

    def check_symplectic(self):
        for rowid in range(self.M):
            vec1 = np.zeros(self.N, dtype=int)
            vec2 = np.zeros(self.N, dtype=int)

            for j in range(self.dc[rowid]):
                vec1[self.Mc[rowid][j]] = self.checkVal[rowid][j]

            for i in range(self.M):
                for j in range(self.dc[i]):
                    vec2[self.Mc[i][j]] = self.checkVal[i][j]

                syn_check = False
                for j in range(self.N):
                    syn_check = self.trace_inner_product(vec1[j], vec2[j]) != 0
                    if syn_check:
                        raise ValueError("check_symplectic: syn_check % 2 != 0")

            for i in range(self.G_rows):
                syn_check = False
                for j in range(self.N):
                    syn_check = self.trace_inner_product(vec1[j], self.G[i][j]) != 0
                    if syn_check:
                        raise ValueError("check_symplectic / GH^T: syn_check % 2 != 0")

        print("Check symplectic ok")

    @staticmethod
    def trace_inner_product(a, b):
        return not (a == 0 or b == 0 or a == b)

# Enum definition for stabilizerCodesType
class StabilizerCodesType:
    GeneralizedBicycle = "GB"
    HypergraphProduct = "HP"
    Toric = "toric"
