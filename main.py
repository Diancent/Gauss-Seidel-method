import numpy as np

def has_diagonal_advantage(matrix):
    for i in range(len(matrix)):
        if abs(matrix[i][i]) < sum([abs(matrix[i][j]) for j in range(len(matrix)) if j != i]):
            return False
    return True

def gauss_seidel(A, b, x):
    q = np.linalg.norm(A, 'fro')
    if not has_diagonal_advantage(A):
        raise Exception("Матриця не задовольняє умову діагональної переваги.")
    iter = 0
    while True:
        iter += 1
        print(iter - 1, "наближення x(", iter - 1, ") =", x.copy())
        x_old = x.copy()

        for i in range(A.shape[0]):
            if A[i, i] == 0:
                raise Exception("Зустрічається ділення на нуль. Метод може не сходитися.")

            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x_old[i + 1:])) / A[i, i]

        LnormInf = np.linalg.norm(x - x_old)
        print("Перевірка на точність в ітерації", iter, "є:", LnormInf)

        total = (1 - q) / q * 1e-3
        print(abs(total))
        if LnormInf <= abs(total) or iter == 1000:
            print(f"Зійшлось після {iter} ітерацій.")
            return x
    raise Exception("Метод не сходився протягом зазначеної точності.")

if __name__== "__main__":
    input_filename = "input_matrix.txt"
    with open(input_filename, "r") as file:
        lines = file.readlines()

    equations = [line.strip().split("=") for line in lines]

    A = []
    b = []
    for equation in equations:
        coeffs, rhs = equation
        coeffs = [float(coeff) for coeff in coeffs.split()]
        A.append(coeffs)
        b.append(float(rhs))

    A = np.array(A)
    b = np.array(b)

    x0 = np.zeros(A.shape[0])

    result = gauss_seidel(A, b, x0)
    print("Результат:", result)