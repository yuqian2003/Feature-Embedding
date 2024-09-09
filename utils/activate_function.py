import torch
import numpy as np

class LeeOscillator():
    def __init__(self, a=[1, 1, 1, 1, -1, -1, -1, -1], b=[0.6, 0.6, -0.5, 0.5, -0.6, -0.6, -0.5, 0.5], K=50, N=100):
        self.a = a
        self.b = b
        self.K = K
        self.N = N

    def Tanh(self, x):
        N = np.random.randint(1, self.N + 1)
        u = torch.zeros((N, x.shape[1], x.shape[0]), dtype=torch.float32)
        v = torch.zeros((N, x.shape[1], x.shape[0]), dtype=torch.float32)
        z = torch.zeros((N, x.shape[1], x.shape[0]), dtype=torch.float32)
        w = torch.zeros((N, x.shape[1], x.shape[0]), dtype=torch.float32)
        u[0] = u[0] + 0.2
        z[0] = z[0] + 0.2
        for t in range(0, N - 1):
            u[t + 1] = torch.tanh(self.a[0] * u[t] - self.a[1] * v[t] + self.a[2] * z[t] + self.a[3] * x[t])
            v[t + 1] = torch.tanh(self.a[6] * z[t] - self.a[4] * u[t] - self.a[5] * v[t] + self.a[7] * x[t])
            w[t] = torch.tanh(x[t])
            z[t + 1] = (v[t + 1] - u[t + 1]) * torch.exp(-self.K * torch.pow(x[t], 2)) + w[t]
        return z[-1]

    def Softmax(self, x):
        N = np.random.randint(1, self.N + 1)
        u = torch.zeros((N, x.shape[1], x.shape[0]), dtype=torch.float32)
        v = torch.zeros((N, x.shape[1], x.shape[0]), dtype=torch.float32)
        z = torch.zeros((N, x.shape[1], x.shape[0]), dtype=torch.float32)
        w = torch.zeros((N, x.shape[1], x.shape[0]), dtype=torch.float32)
        u[0] = u[0] + 0.2
        z[0] = z[0] + 0.2
        for t in range(0, N - 1):
            u[t + 1] = torch.sigmoid(self.b[0] * u[t] - self.b[1] * v[t] + self.b[2] * z[t] + self.b[3] * x[t])
            v[t + 1] = torch.sigmoid(self.b[6] * z[t] - self.b[4] * u[t] - self.b[5] * v[t] + self.b[7] * x[t])
            w[t] = torch.sigmoid(x[t])
            z[t + 1] = (v[t + 1] - u[t + 1]) * torch.exp(-self.K * torch.pow(x[t], 2)) + w[t]
        exp_z = torch.exp(z[-1])
        return exp_z / torch.sum(exp_z, axis=0)


if __name__ == "__main__":
    Lee = LeeOscillator()
    x = torch.randn((32, 1, 9, 4))
    x = torch.reshape(x, (32, 9, 4, 1))
    for i in range(0, 8):
        print("Original: " + str(x[0][i]))
    x = torch.relu(x)
    for i in range(0, 8):
        print("ReLU: " + str(x[0][i]))
    for i in range(0, 8):
        print("Tanh: " + str(Lee.Tanh(x[0][i])))
    for i in range(0, 8):
        print("Softmax: " + str(Lee.Softmax(x[0][i])))
