import numpy as np

A = np.array([[5.103, 0.293, 0.336, 0.270],
              [0.179, 4.912, 0.394, 0.375],
              [0.189, 0.321, 2.875, 0.216],
              [0.317, 0.165, 0.386, 3.934]], dtype=float)
b = np.array([0.745, 0.381, 0.480, 0.552])


def DRL(Matrix):
    D = np.zeros((4, 4), dtype=float)
    R = np.zeros((4, 4), dtype=float)
    L = np.zeros((4, 4), dtype=float)

    for i in range(len(Matrix)):
        for j in range(len(Matrix)):
            if i == j:
                D[i][j] = Matrix[i][j]
            elif i > j:
                L[i][j] = Matrix[i][j]
            else:
                R[i][j] = Matrix[i][j]

    return D, R, L


def reform_system(Matrix, vector):
    D, R, L = DRL(Matrix)

    B = -(np.linalg.inv(D).dot(L + R))
    w, v = np.linalg.eigh(B)
    c = np.linalg.inv(D).dot(vector)
    return B, c


def yakobi(Matrix, vector):
    B, c = reform_system(Matrix, vector)
    x0 = [100, 100, 100, 100]
    eps = 0.00001
    q = np.linalg.norm(B)
    k = 0
    print(str(k) + ") " + str(x0))
    x_next = B.dot(x0) + c
    k += 1
    print(str(k) + ") " + str(x_next))
    print("norm: " + str(np.linalg.norm(x_next - x0)))
    while ((q * np.linalg.norm(x_next - x0)) / (1 - q)) >= eps:
        x0 = x_next
        x_next = B.dot(x0) + c
        k += 1
        print(str(k) + ") " + str(x_next))
        print("norm: " + str(np.linalg.norm(x_next - x0)))

    return x_next


def check(Matrix, vector, answer):
    return vector - Matrix.dot(answer)


def main():
    x = yakobi(A, b)
    print("roots " + str(x))
    print("check roots" + str(check(A, b, x)))
    return 0


if __name__ == '__main__':
    main()
