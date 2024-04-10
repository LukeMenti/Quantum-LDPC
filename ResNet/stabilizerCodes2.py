import math
import random

class StabilizerCodes:
    def __init__(self, n, k, m, code_type, fr, trained=False):
        self.mycodetype = code_type
        self.N = n
        self.K = k
        self.M = m
        self.G_rows = self.N + self.K
        self.mTrained = trained
        self.dc = fr.dc
        self.dv = fr.dv
        self.maxDc = fr.maxDc
        self.maxDv = fr.maxDv
        self.Nv = fr.Nv
        self.Mc = fr.Mc
        self.checkVal = fr.checkVal
        self.varVal = fr.varVal
        self.Nvk = fr.Nvk
        self.Mck = fr.Mck
        self.G = fr.G
        if trained:
            self.weights_cn = fr.weights_cn
            self.weights_vn = fr.weights_vn
            self.weights_llr = fr.weights_llr

    def decode(self, L, epsilon):
        if not self.errorString:
            return [True, True]
        self.calculate_syndrome()
        self.error_hat = [0] * self.N
        return self.flooding_decode(L, epsilon)

    def add_error_given_epsilon(self, epsilon):
        self.error.clear()
        self.errorString.clear()

        for i in range(self.N):
            rndValue = random.random()
            if rndValue < epsilon / 3:
                self.error.append(1)
                self.errorString.append(f"X{i}")
            elif rndValue < epsilon * 2 / 3:
                self.error.append(2)
                self.errorString.append(f"Z{i}")
            elif rndValue < epsilon:
                self.error.append(3)
                self.errorString.append(f"Y{i}")
            else:
                self.error.append(0)

    def quantize_belief(self, Tau, Tau1, Tau2):
        nom = math.log1p(math.exp(-1.0 * Tau))
        denom = max(-1.0 * Tau1, -1.0 * Tau2) + math.log1p(math.exp(-1.0 * abs((Tau1 - Tau2))))
        ret_val = nom - denom
        if math.isnan(ret_val):
            raise RuntimeError("quantize_belief: Log difference is NaN")
        return ret_val

    @staticmethod
    def trace_inner_product(a, b):
        return not (a == 0 or b == 0 or a == b)

    def flooding_decode(self, L, epsilon):
        success = [False, False]
        L0 = math.log(3.0 * (1 - epsilon) / epsilon)
        lambda0 = math.log((1 + math.exp(-L0)) / (2 * math.exp(-L0)))

        mc2v = [[0 for _ in range(dc)] for dc in self.dc]
        mv2c = [[lambda0 for _ in range(dv)] for dv in self.dv]
        Taux = [L0] * self.N
        Tauz = [L0] * self.N
        Tauy = [L0] * self.N
        phi_msg = [0] * self.maxDc

        for decIter in range(L):
            for i in range(self.M):
                phi_sum = 0
                sign_prod = 1.0 if self.syn[i] == 0 else -1.0
                for j in range(len(self.dc[i])):
                    if mv2c[self.Mc[i][j]][self.Mck[i][j]] != 0.0:
                        phi_msg[j] = -1.0 * math.log(math.tanh(abs(mv2c[self.Mc[i][j]][self.Mck[i][j]]) / 2.0))
                    else:
                        phi_msg[j] = 60
                    phi_sum += phi_msg[j]
                    sign_prod *= 1.0 if mv2c[self.Mc[i][j]][self.Mck[i][j]] >= 0.0 else -1.0

                for j in range(len(self.dc[i])):
                    phi_extrinsic_phi_sum = phi_sum - phi_msg[j]
                    phi_phi_sum = 60 if phi_extrinsic_phi_sum == 0 else -1.0 * math.log(math.tanh(phi_extrinsic_phi_sum / 2.0))
                    mc2v[i][j] = phi_phi_sum * sign_prod * (1.0 if mv2c[self.Mc[i][j]][self.Mck[i][j]] >= 0.0 else -1.0)
                    if self.mTrained and decIter < self.trained_iter:
                        mc2v[i][j] *= self.weights_cn[decIter][i][j]
                    if math.isnan(mc2v[i][j]) or math.isinf(mc2v[i][j]):
                        raise RuntimeError("flooding_decode: mc2v[i][j] is NaN or infinity")

            for Vidx in range(self.N):
                for jj in range(len(self.dv[Vidx])):
                    Taux[Vidx] = L0
                    Tauz[Vidx] = L0
                    Tauy[Vidx] = L0
                    if self.mTrained and decIter < self.trained_iter:
                        Taux[Vidx] *= self.weights_llr[decIter + 1][Vidx]
                        Tauy[Vidx] *= self.weights_llr[decIter + 1][Vidx]
                        Tauz[Vidx] *= self.weights_llr[decIter + 1][Vidx]

                    for jj in range(len(self.dv[Vidx])):
                        if self.varVal[Vidx][jj] == 1:
                            Tauz[Vidx] += mc2v[self.Nv[Vidx][jj]][self.Nvk[Vidx][jj]]
                            Tauy[Vidx] += mc2v[self.Nv[Vidx][jj]][self.Nvk[Vidx][jj]]
                        elif self.varVal[Vidx][jj] == 2:
                            Taux[Vidx] += mc2v[self.Nv[Vidx][jj]][self.Nvk[Vidx][jj]]
                            Tauy[Vidx] += mc2v[self.Nv[Vidx][jj]][self.Nvk[Vidx][jj]]
                        elif self.varVal[Vidx][jj] == 3:
                            Taux[Vidx] += mc2v[self.Nv[Vidx][jj]][self.Nvk[Vidx][jj]]
                            Tauz[Vidx] += mc2v[self.Nv[Vidx][jj]][self.Nvk[Vidx][jj]]
                        else:
                            raise ValueError("Something is wrong")

                        Tauxi = Taux[Vidx]
                        Tauzi = Tauz[Vidx] - mc2v[self.Nv[Vidx][jj]][self.Nvk[Vidx][jj]]
                        Tauyi = Tauy[Vidx] - mc2v[self.Nv[Vidx][jj]][self.Nvk[Vidx][jj]]
                        temp = self.quantize_belief(Tauxi, Tauyi, Tauzi)
                        limit = 60
                        if temp > limit:
                            mv2c[Vidx][jj] = limit
                        elif temp < -limit:
                            mv2c[Vidx][jj] = -limit
                        else:
                            mv2c[Vidx][jj] = temp
                        if self.mTrained and decIter < self.trained_iter:
                            mv2c[Vidx][jj] *= self.weights_vn[decIter + 1][Vidx][jj]
                        if math.isnan(mv2c[Vidx][jj]) or math.isinf(mv2c[Vidx][jj]):
                            raise RuntimeError("flooding_decode: mv2c[Vidx][jj] is NaN or infinity")

            success = self.check_success(Taux, Tauy, Tauz)
            if success[0]:
                break

        return success

    def calculate_syndrome(self):
        for i in range(self.M):
            check = False
            for j in range(len(self.dc[i])):
                check = self.trace_inner_product(self.error[self.Mc[i][j]], self.checkVal[i][j]) != check
            self.syn.append(check % 2)

    def check_success(self, Taux, Tauy, Tauz):
        success = [False, False]
        self.error_hat = [0] * self.N
        for i in range(self.N):
            if Taux[i] > 0 and Tauy[i] > 0 and Tauz[i] > 0:
                self.error_hat[i] = 0
            elif Taux[i] < Tauy[i] and Taux[i] < Tauz[i]:
                self.error_hat[i] = 1
            elif Tauz[i] < Taux[i] and Tauz[i] < Tauy[i]:
                self.error_hat[i] = 2
            else:
                self.error_hat[i] = 3

        for i in range(self.M):
            check = False
            for j in range(len(self.dc[i])):
                check = self.trace_inner_product(self.error_hat[self.Mc[i][j]], self.checkVal[i][j]) != check
            if check != self.syn[i]:
                return success

        success[0] = True
        for i in range(self.G_rows):
            check = False
            for j in range(self.N):
                check = self.trace_inner_product(self.error[j], self.G[i][j]) != check
                check = self.trace_inner_product(self.error_hat[j], self.G[i][j]) != check
            if check:
                return success

        success[1] = True
        return success
