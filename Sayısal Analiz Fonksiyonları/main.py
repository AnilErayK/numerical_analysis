import numpy as np
import math

print("########################################## Jacobi ve Gauss Seidal ##############################################")
def jacobi(A, b, tolerance=1e-4):
    x = np.zeros_like(b)
    while True:
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.allclose(x, x_new, atol=tolerance):
            break
        x = x_new
    return x

def gauss_seidel(A, b, tolerance=1e-4):
    x = np.zeros_like(b)
    while True:
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.allclose(x, x_new, atol=tolerance):
            break
        x = x_new
    return x

# Katsayı matrisi A ve sağ taraf vektörü b
A = np.array([[10, -1, 1, -3],
              [-2, 15, -3, 0.5],
              [2, -4, 11, 1.5],
              [0.5, -2.5, 5.5, -15]])
b = np.array([13, 42, -13, 22])

# Jacobi ve Gauss-Seidel yöntemleri kullanarak çözüm
x_jacobi = jacobi(A, b)
x_gauss_seidel = gauss_seidel(A, b)



print("Jacobi yöntemi ile çözüm:", x_jacobi)
print("Gauss-Seidel yöntemi ile çözüm:", x_gauss_seidel)


def is_diagonally_dominant(A):
    for i in range(len(A)):
        diagonal = abs(A[i][i])
        off_diagonal_sum = sum(abs(A[i][j]) for j in range(len(A)) if i != j)
        if diagonal <= off_diagonal_sum:
            return False
    return True

# Katsayı matrisi A1
A1 = np.array([[-5, -4, -1],
               [1, -3, -4],
               [2, -1, 5]])

# Diagonal dominance kontrolü
diagonal_dominant = is_diagonally_dominant(A1)
print("Diagonal Dominant: ", diagonal_dominant)


#2. soru diaoganal olarak dominant değildir. bu yöntemlerle çözülemez.



print("\n\n########################################## Vionello ##############################################")

def vionello(A, v, iterations=7, tolerance=1e-4):
    # Güç yöntemi ile en büyük özdeğer ve özvektörü bulma
    v_max = np.copy(v)
    for i in range(iterations):
        v_next = np.dot(A, v_max)
        norm_v_next = np.linalg.norm(v_next)
        if np.linalg.norm(v_next - v_max * norm_v_next) < tolerance:
            print("En büyük özdeğer için yakınsama sağlandı, iterasyon: ", i+1)
            break
        v_max = v_next / norm_v_next
    largest_eigenvalue = np.dot(v_max.T, np.dot(A, v_max)) / np.dot(v_max.T, v_max)

    # Ters güç yöntemi ile en küçük özdeğer ve özvektörü bulma
    try:
        A_inv = np.linalg.inv(A)
        v_min = np.copy(v)
        for i in range(iterations):
            v_next = np.dot(A_inv, v_min)
            norm_v_next = np.linalg.norm(v_next)
            if np.linalg.norm(v_next - v_min * norm_v_next) < tolerance:
                print("En küçük özdeğer için yakınsama sağlandı, iterasyon: ", i+1)
                break
            v_min = v_next / norm_v_next
        smallest_eigenvalue = np.dot(v_min.T, np.dot(A, v_min)) / np.dot(v_min.T, v_min)
    except np.linalg.LinAlgError:
        smallest_eigenvalue = None
        v_min = None

    return largest_eigenvalue, v_max, smallest_eigenvalue, v_min

# Katsayı matrisi A ve başlangıç vektörü v
A = np.array([[2, -1, 3],
              [4, 2, 6],
              [5, 1, 4]])
v = np.array([0, 1, 1], dtype=float)

# İterasyon sayısı ve tolerans
iterations = 7
tolerance = 1e-4

# Vionello yöntemi ile özdeğerler ve özvektörler
largest_eigenvalue, largest_eigenvector, smallest_eigenvalue, smallest_eigenvector = vionello(A, v, iterations, tolerance)

print("En büyük özdeğer:", largest_eigenvalue)
print("En büyük özvektör:", largest_eigenvector)
print("En küçük özdeğer:", smallest_eigenvalue if smallest_eigenvalue is not None else "Hesaplanamadı")
print("En küçük özvektör:", smallest_eigenvector if smallest_eigenvector is not None else "Hesaplanamadı")



print("\n\n########################################## Newton-Raphson ##############################################")
print("\t1")
def f(x):
    return x**3 - np.exp(-np.sqrt(x)) - 1

def df(x):
    return 3*x**2 + (np.exp(-np.sqrt(x)) / (2*np.sqrt(x)))

def newton_raphson(f, df, x0, tol=1e-4, max_iter=100):
    x = x0
    for i in range(max_iter):
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < tol:
            return x_new, i+1  # Kök ve iterasyon sayısı döndür
        x = x_new
    return x, max_iter  # Maksimum iterasyon sayısına ulaşıldı

# Başlangıç tahmini ve tolerans
x0 = 1/2
tolerance = 1e-4

# Newton-Raphson yöntemi ile kök hesaplama
root, iterations = newton_raphson(f, df, x0, tol=tolerance)

print("Bulunan kök:", root)
print("Toplam iterasyon sayısı:", iterations)


print("\n\t2")

def f(x):
    return x**3 - np.e - 1

def df(x):
    return 3*x**2

# Başlangıç değeri ve tolerans
x0 = 1.5
tolerance = 1e-4

# Newton-Raphson yöntemi ile kök hesaplama
root, iterations = newton_raphson(f, df, x0, tol=tolerance)

print("Bulunan kök:", root)
print("Toplam iterasyon sayısı:", iterations)

print("\n\t3")

def F(x, y):
    return np.array([x + np.exp(y) - 1, x + y**2 - 1])

def J(x, y):
    return np.array([[1, np.exp(y)], [1, 2*y]])

def newton_raphson_2d(F, J, x0, y0, tol=1e-5, max_iter=100):
    x, y = x0, y0
    for i in range(max_iter):
        F_val = F(x, y)
        J_val = J(x, y)
        delta = np.linalg.solve(J_val, -F_val)
        x, y = x + delta[0], y + delta[1]
        if np.linalg.norm(delta, ord=np.inf) < tol:
            return x, y, i+1  # Kökler ve iterasyon sayısı döndür
    return x, y, max_iter  # Maksimum iterasyon sayısına ulaşıldı

# Başlangıç değerleri
x0, y0 = 0.8, -1.7

# Newton-Raphson yöntemi ile kökleri bul
x, y, iterations = newton_raphson_2d(F, J, x0, y0)

print("Bulunan kökler: x =", x, ", y =", y)
print("Toplam iterasyon sayısı:", iterations)


print("\n\n########################################## Newton İleri Fark Tablosu ve İnterpolasyon ##############################################")

# Verilen veri noktaları
x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
f = np.array([1.2214, 1.4214, 1.5818, 1.8221, 2.2225, 2.4935])

# Newton ileri fark tablosunu oluşturma
n = len(x)
forward_diff = np.zeros((n, n))
forward_diff[:, 0] = f

for j in range(1, n):
    for i in range(n - j):
        forward_diff[i, j] = forward_diff[i + 1, j - 1] - forward_diff[i, j - 1]

print("Newton İleri Fark Tablosu:")
print(forward_diff)


# Newton interpolasyon polinomu ile f(0.3) ve f(0.7) değerlerini hesaplama
def newton_forward_interpolation(x, y, value):
    n = len(x)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = diff_table[i + 1, j - 1] - diff_table[i, j - 1]

    h = x[1] - x[0]
    u = (value - x[0]) / h
    interpolated_value = diff_table[0, 0]
    u_term = 1
    for i in range(1, n):
        u_term *= (u - (i - 1))
        interpolated_value += (u_term * diff_table[0, i]) / math.factorial(i)

    return interpolated_value


# f(0.3) ve f(0.7) değerlerini hesaplama
f_0_3 = newton_forward_interpolation(x, f, 0.3)
f_0_7 = newton_forward_interpolation(x, f, 0.7)

print(f"f(0.3) değeri: {f_0_3}")
print(f"f(0.7) değeri: {f_0_7}")

print("\n\n########################################## Kesen Kök - Regula Falsi - Bolzano ##############################################")

def f(x):
    return 1 - x + np.cos(x) - 2

# Kesen Kök Yöntemi (Secant Method)
def secant_method(f, x0, x1, tol=1e-3, max_iter=100):
    for i in range(max_iter):
        if abs(f(x1) - f(x0)) < tol:
            break
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x2 - x1) < tol:
            return x2, i+1
        x0, x1 = x1, x2
    return x1, max_iter

# Regula Falsi (False Position Method)
def regula_falsi(f, x0, x1, tol=1e-3, max_iter=100):
    for i in range(max_iter):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(f(x2)) < tol:
            return x2, i+1
        if f(x0) * f(x2) < 0:
            x1 = x2
        else:
            x0 = x2
    return x2, max_iter

# Bolzano (Bisection) Yöntemi
def bolzano_method(f, a, b, max_iter=7):
    if f(a) * f(b) >= 0:
        print("Bolzano method fails. f(a) and f(b) must have different signs.")
        return None, 0
    a_n, b_n = a, b
    for n in range(1, max_iter + 1):
        m_n = (a_n + b_n) / 2
        f_m_n = f(m_n)
        if f(a_n) * f_m_n < 0:
            b_n = m_n
        elif f(b_n) * f_m_n < 0:
            a_n = m_n
        elif f_m_n == 0:
            return m_n, n  # Tam kök bulundu
        else:
            print("Bolzano method fails.")
            return None, n
    return (a_n + b_n) / 2, max_iter


# a) Kesen Kök Yöntemi ile kök bulma
secant_root, secant_iterations = secant_method(f, 0, 0.2)
print(f"Kesen Kök Yöntemi ile bulunan kök: {secant_root}, Iterasyon sayısı: {secant_iterations}")

# b) Regula Falsi Yöntemi ile kök bulma
regula_falsi_root, regula_falsi_iterations = regula_falsi(f, 0, 1)
print(f"Regula Falsi Yöntemi ile bulunan kök: {regula_falsi_root}, Iterasyon sayısı: {regula_falsi_iterations}")

# Bolzano Yöntemi ile kök bulma
bolzano_root, bolzano_iterations = bolzano_method(f, 0, 1, 7)
print(f"Bolzano Yöntemi ile bulunan kök: {bolzano_root}, Iterasyon sayısı: {bolzano_iterations}")


print("\n\n########################################## Gaus Eleminasyon ##############################################")

def gauss_elimination(A, b):
    n = len(b)
    # Augmented matrix
    M = np.hstack((A, b.reshape(-1, 1)))

    # Forward elimination
    for i in range(n):
        # Pivoting
        for j in range(i + 1, n):
            if M[j, i] != 0:
                factor = M[j, i] / M[i, i]
                M[j, i:] -= factor * M[i, i:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:n])) / M[i, i]

    return x


# Verilen denklem sisteminin katsayı matrisi ve sağ taraf vektörü
A = np.array([[2, 3, -1],
              [-1, -4, 2],
              [3, 2, -5]], dtype=float)
b = np.array([1, -7, 0], dtype=float)

# Gauss eliminasyon yöntemiyle çözüm
solution = gauss_elimination(A, b)
print("Çözümler: x1 = {:.2f}, x2 = {:.2f}, x3 = {:.2f}".format(*solution))
