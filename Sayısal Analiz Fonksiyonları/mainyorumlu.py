import numpy as np  # Numpy kütüphanesini içe aktararak matris ve vektör işlemleri için kullanacağız.
import math  # Matematiksel işlemler için math kütüphanesini içe aktar.


print("########################################## Jacobi ve Gauss Seidal ##############################################")

# Jacobi yöntemi, lineer denklem sistemlerini iteratif bir şekilde çözer.
def jacobi(A, b, tolerance=1e-4):
    x = np.zeros_like(b)  # Başlangıçta x vektörünü sıfırlar ile başlat.
    while True:
        x_new = np.zeros_like(x)  # Her iterasyonda x'in yeni değerlerini saklayacak vektör.
        for i in range(A.shape[0]):  # Matrisin her bir satırı için döngü.
            s1 = np.dot(A[i, :i], x[:i])  # İlgili satırın, diagonalin altındaki elemanlarla olan çarpımı.
            s2 = np.dot(A[i, i + 1:], x[i + 1:])  # Diagonalin üstündeki elemanlarla olan çarpımı.
            x_new[i] = (b[i] - s1 - s2) / A[i, i]  # Güncellenmiş x değerini hesapla.
        if np.allclose(x, x_new, atol=tolerance):  # Yakınsama kontrolü, eski ve yeni x değerleri arasında belirli bir toleransla.
            break
        x = x_new  # x vektörünü yeni hesaplanan değerlerle güncelle.
    return x  # Yakınsadığında son x vektörünü döndür.

# Gauss-Seidel yöntemi, Jacobi'nin aksine daha hızlı yakınsar çünkü güncel değerleri kullanır.
def gauss_seidel(A, b, tolerance=1e-4):
    x = np.zeros_like(b)  # Başlangıçta x vektörünü sıfırlar ile başlat.
    while True:
        x_new = np.zeros_like(x)  # Yeni x değerlerini saklayacak vektör.
        for i in range(A.shape[0]):  # Matrisin her bir satırı için döngü.
            s1 = np.dot(A[i, :i], x_new[:i])  # Güncellenmiş x değerlerini kullanarak diagonalin altındaki elemanlarla çarpım.
            s2 = np.dot(A[i, i + 1:], x[i + 1:])  # Diagonalin üstündeki elemanlarla olan çarpım.
            x_new[i] = (b[i] - s1 - s2) / A[i, i]  # Güncellenmiş x değerini hesapla.
        if np.allclose(x, x_new, atol=tolerance):  # Yakınsama kontrolü.
            break
        x = x_new  # x vektörünü yeni hesaplanan değerlerle güncelle.
    return x  # Yakınsadığında son x vektörünü döndür.

# Katsayı matrisi A ve sağ taraf vektörü b
A = np.array([[10, -1, 1, -3],
              [-2, 15, -3, 0.5],
              [2, -4, 11, 1.5],
              [0.5, -2.5, 5.5, -15]])
b = np.array([13, 42, -13, 22])

# Jacobi ve Gauss-Seidel yöntemleri ile çözümler hesaplanır ve yazdırılır.
x_jacobi = jacobi(A, b)
x_gauss_seidel = gauss_seidel(A, b)

print("Jacobi yöntemi ile çözüm:", x_jacobi)  # Jacobi yöntemi ile bulunan çözümü yazdır.
print("Gauss-Seidel yöntemi ile çözüm:", x_gauss_seidel)  # Gauss-Seidel yöntemi ile bulunan çözümü yazdır.

# Diyagonal dominant matris kontrolü yapılır.
def is_diagonally_dominant(A):
    for i in range(len(A)):  # Matrisin her satırı için döngü.
        diagonal = abs(A[i][i])  # Diyagonal elemanın mutlak değeri.
        off_diagonal_sum = sum(abs(A[i][j]) for j in range(len(A)) if i != j)  # Diyagonal dışı elemanların toplamı.
        if diagonal <= off_diagonal_sum:  # Diyagonal eleman, diğerlerinden küçükse veya eşitse, dominant değildir.
            return False
    return True  # Tüm satırlar için dominantlık sağlanıyorsa True döndür.

# Katsayı matrisi A1 ve dominantlık kontrolü
A1 = np.array([[-5, -4, -1],
               [1, -3, -4],
               [2, -1, 5]])

# Diyagonal dominantlık durumunu kontrol et ve yazdır.
diagonal_dominant = is_diagonally_dominant(A1)
print("Diagonal Dominant: ", diagonal_dominant)


print("\n\n########################################## Vionello ##############################################")

# Vionello yöntemi ile en büyük ve en küçük özdeğerleri bulma
def vionello(A, v, iterations=7, tolerance=1e-4):
    # Güç yöntemi ile en büyük özdeğer ve özvektörü bulma
    v_max = np.copy(v)  # Başlangıç vektörünü kopyala
    for i in range(iterations):  # Verilen iterasyon sayısı kadar döngü
        v_next = np.dot(A, v_max)  # A matrisi ile v_max'ı çarp
        norm_v_next = np.linalg.norm(v_next)  # v_next'in normunu hesapla
        if np.linalg.norm(v_next - v_max * norm_v_next) < tolerance:  # Yakınsama kontrolü
            print("En büyük özdeğer için yakınsama sağlandı, iterasyon: ", i+1)
            break
        v_max = v_next / norm_v_next  # v_max'i normalize edilmiş v_next ile güncelle
    largest_eigenvalue = np.dot(v_max.T, np.dot(A, v_max)) / np.dot(v_max.T, v_max)  # En büyük özdeğeri hesapla

    # Ters güç yöntemi ile en küçük özdeğer ve özvektörü bulma
    try:
        A_inv = np.linalg.inv(A)  # A'nın tersini hesapla
        v_min = np.copy(v)  # Başlangıç vektörünü kopyala
        for i in range(iterations):
            v_next = np.dot(A_inv, v_min)  # A_inv ile v_min'i çarp
            norm_v_next = np.linalg.norm(v_next)  # v_next'in normunu hesapla
            if np.linalg.norm(v_next - v_min * norm_v_next) < tolerance:  # Yakınsama kontrolü
                print("En küçük özdeğer için yakınsama sağlandı, iterasyon: ", i+1)
                break
            v_min = v_next / norm_v_next  # v_min'i normalize edilmiş v_next ile güncelle
        smallest_eigenvalue = np.dot(v_min.T, np.dot(A, v_min)) / np.dot(v_min.T, v_min)  # En küçük özdeğeri hesapla
    except np.linalg.LinAlgError:
        smallest_eigenvalue = None  # Ters matris hesaplanamazsa None döndür
        v_min = None

    return largest_eigenvalue, v_max, smallest_eigenvalue, v_min  # Özdeğerler ve özvektörler döndür

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

print("En büyük özdeğer:", largest_eigenvalue)  # En büyük özdeğeri yazdır
print("En büyük özvektör:", largest_eigenvector)  # En büyük özvektörü yazdır
print("En küçük özdeğer:", smallest_eigenvalue if smallest_eigenvalue is not None else "Hesaplanamadı")  # En küçük özdeğeri yazdır
print("En küçük özvektör:", smallest_eigenvector if smallest_eigenvector is not None else "Hesaplanamadı")  # En küçük özvektörü yazdır


print("\n\n########################################## Newton-Raphson ##############################################")
print("\t1")
# Newton-Raphson kök bulma yöntemi tanımı
def newton_raphson(f, df, x0, tol=1e-4, max_iter=100):
    x = x0  # Başlangıç değeri
    for i in range(max_iter):  # Maksimum iterasyon sayısına kadar döngü
        x_new = x - f(x) / df(x)  # Newton-Raphson formülü
        if abs(x_new - x) < tol:  # Yakınsama testi, toleransın altındaysa döngüden çık
            return x_new, i+1  # Kök ve iterasyon sayısını döndür
        x = x_new  # x'i güncelle
    return x, max_iter  # Maksimum iterasyona ulaşıldıysa son değeri döndür

# Örnek fonksiyon ve türevi
def f(x):
    return x**3 - np.exp(-np.sqrt(x)) - 1  # Fonksiyon tanımı

def df(x):
    return 3*x**2 + (np.exp(-np.sqrt(x)) / (2*np.sqrt(x)))  # Fonksiyonun türevi

# Başlangıç tahmini ve tolerans
x0 = 1/2
tolerance = 1e-4

# Newton-Raphson yöntemi ile kök bulma
root, iterations = newton_raphson(f, df, x0, tol=tolerance)

print("Bulunan kök:", root)  # Bulunan kökü yazdır
print("Toplam iterasyon sayısı:", iterations)  # Kullanılan iterasyon sayısını yazdır

print("\n\t2")

# Başka bir örnek fonksiyon ve türevi
def f(x):
    return x**3 - np.e - 1  # Fonksiyon tanımı

def df(x):
    return 3*x**2  # Fonksiyonun türevi

# Başlangıç değeri ve tolerans
x0 = 1.5
tolerance = 1e-4

# Newton-Raphson yöntemi ile kök bulma
root, iterations = newton_raphson(f, df, x0, tol=tolerance)

print("Bulunan kök:", root)  # Bulunan kökü yazdır
print("Toplam iterasyon sayısı:", iterations)  # Kullanılan iterasyon sayısını yazdır

print("\n\t3")

# İki boyutlu Newton-Raphson yöntemi tanımı
def F(x, y):
    return np.array([x + np.exp(y) - 1, x + y**2 - 1])  # İki boyutlu fonksiyonlar

def J(x, y):
    return np.array([[1, np.exp(y)], [1, 2*y]])  # Jakobian matrisi

# İki boyutlu Newton-Raphson yöntemi
def newton_raphson_2d(F, J, x0, y0, tol=1e-5, max_iter=100):
    x, y = x0, y0  # Başlangıç değerleri
    for i in range(max_iter):  # Maksimum iterasyon sayısına kadar döngü
        F_val = F(x, y)  # Fonksiyon değerleri
        J_val = J(x, y)  # Jakobian değerleri
        delta = np.linalg.solve(J_val, -F_val)  # Lineer denklem çözümü
        x, y = x + delta[0], y + delta[1]  # Kök güncellemesi
        if np.linalg.norm(delta, ord=np.inf) < tol:  # Yakınsama kontrolü
            return x, y, i+1  # Kökler ve iterasyon sayısını döndür
    return x, y, max_iter  # Maksimum iterasyona ulaşıldıysa son değerleri döndür

# Başlangıç değerleri
x0, y0 = 0.8, -1.7

# İki boyutlu Newton-Raphson yöntemi ile kök bulma
x, y, iterations = newton_raphson_2d(F, J, x0, y0)

print("Bulunan kökler: x =", x, ", y =", y)  # Bulunan kökleri yazdır
print("Toplam iterasyon sayısı:", iterations)  # Kullanılan iterasyon sayısını yazdır


print("\n\n########################################## Newton İleri Fark Tablosu ve İnterpolasyon ##############################################")

# Verilen veri noktaları
x = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
f = np.array([1.2214, 1.4214, 1.5818, 1.8221, 2.2225, 2.4935])

# Newton ileri fark tablosunu oluşturma
n = len(x)
forward_diff = np.zeros((n, n))
forward_diff[:, 0] = f  # İlk sütun olarak f değerlerini yerleştir

for j in range(1, n):  # Fark tablosunun her sütunu için
    for i in range(n - j):  # Her satır için, her ilerleyişte bir önceki sütuna göre bir azaltılarak
        forward_diff[i, j] = forward_diff[i + 1, j - 1] - forward_diff[i, j - 1]  # İleri farkı hesapla

print("Newton İleri Fark Tablosu:")
print(forward_diff)  # İleri fark tablosunu yazdır

# Newton ileri fark interpolasyonu ile belirli bir değer için tahmini hesaplama
def newton_forward_interpolation(x, y, value):
    n = len(x)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = diff_table[i + 1, j - 1] - diff_table[i, j - 1]

    h = x[1] - x[0]  # x değerleri arasındaki sabit aralık
    u = (value - x[0]) / h  # İnterpolasyon noktasının, taban değere olan normalize edilmiş uzaklığı
    interpolated_value = diff_table[0, 0]
    u_term = 1
    for i in range(1, n):
        u_term *= (u - (i - 1))
        interpolated_value += (u_term * diff_table[0, i]) / math.factorial(i)  # Faktöriyel ile normalize edilmiş ileri fark katkısı

    return interpolated_value  # İnterpolasyon sonucunu döndür

# f(0.3) ve f(0.7) değerlerini hesaplama
f_0_3 = newton_forward_interpolation(x, f, 0.3)
f_0_7 = newton_forward_interpolation(x, f, 0.7)

print(f"f(0.3) değeri: {f_0_3}")  # f(0.3) değerini yazdır
print(f"f(0.7) değeri: {f_0_7}")  # f(0.7) değerini yazdır


print("\n\n########################################## Kesen Kök - Regula Falsi - Bolzano ##############################################")

# Fonksiyon tanımı
def f(x):
    return 1 - x + np.cos(x) - 2  # Kök bulunacak fonksiyon

# Kesen Kök Yöntemi (Secant Method)
def secant_method(f, x0, x1, tol=1e-3, max_iter=100):
    for i in range(max_iter):
        if abs(f(x1) - f(x0)) < tol:  # Fark tolerans altındaysa döngüden çık
            break
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))  # Kesen kök formülü
        if abs(x2 - x1) < tol:
            return x2, i+1  # Kök ve iterasyon sayısını döndür
        x0, x1 = x1, x2  # Değerleri güncelle
    return x1, max_iter  # Maksimum iterasyona ulaşıldıysa son değeri döndür

# Regula Falsi (False Position Method)
def regula_falsi(f, x0, x1, tol=1e-3, max_iter=100):
    for i in range(max_iter):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))  # False position formülü
        if abs(f(x2)) < tol:
            return x2, i+1  # Kök ve iterasyon sayısını döndür
        if f(x0) * f(x2) < 0:
            x1 = x2  # İşaret değişimi varsa x1'i güncelle
        else:
            x0 = x2  # İşaret değişimi yoksa x0'ı güncelle
    return x2, max_iter  # Maksimum iterasyona ulaşıldıysa son değeri döndür

# Bolzano (Bisection) Yöntemi
def bolzano_method(f, a, b, max_iter=7):
    if f(a) * f(b) >= 0:
        print("Bolzano method fails. f(a) and f(b) must have different signs.")
        return None, 0  # Farklı işaretler yoksa yöntem başarısız
    a_n, b_n = a, b
    for n in range(1, max_iter + 1):
        m_n = (a_n + b_n) / 2
        f_m_n = f(m_n)
        if f(a_n) * f_m_n < 0:
            b_n = m_n
        elif f(b_n) * f_m_n < 0:
            a_n = m_n
        elif f_m_n == 0:
            return m_n, n  # Kök bulundu
        else:
            print("Bolzano method fails.")
            return None, n
    return (a_n + b_n) / 2, max_iter  # Maksimum iterasyon sonucu döndür

# Örnek kullanımlar ve sonuçların yazdırılması
secant_root, secant_iterations = secant_method(f, 0, 0.2)
print(f"Kesen Kök Yöntemi ile bulunan kök: {secant_root}, Iterasyon sayısı: {secant_iterations}")

regula_falsi_root, regula_falsi_iterations = regula_falsi(f, 0, 1)
print(f"Regula Falsi Yöntemi ile bulunan kök: {regula_falsi_root}, Iterasyon sayısı: {regula_falsi_iterations}")

bolzano_root, bolzano_iterations = bolzano_method(f, 0, 1, 7)
print(f"Bolzano Yöntemi ile bulunan kök: {bolzano_root}, Iterasyon sayısı: {bolzano_iterations}")


print("\n\n########################################## Gaus Eleminasyon ##############################################")

# Gauss eleminasyon yöntemi, lineer denklem sistemlerini çözmek için kullanılır.
def gauss_elimination(A, b):
    n = len(b)  # Denklem sayısı
    # Arttırılmış matris oluştur
    M = np.hstack((A, b.reshape(-1, 1)))

    # İleri eleme
    for i in range(n):
        # Pivoting (Sütun içindeki en büyük elemanı bulup, o satırı en üste alarak sayısal kararlılığı artırma)
        max_row = np.argmax(abs(M[i:, i])) + i
        M[[i, max_row]] = M[[max_row, i]]
        for j in range(i + 1, n):
            if M[j, i] != 0:  # Sıfır olmayan elemanları kontrol et
                factor = M[j, i] / M[i, i]  # Oranı hesapla
                M[j, i:] -= factor * M[i, i:]  # Çarpan kullanarak satırı sıfırla

    # Geri yerine koyma
    x = np.zeros(n)  # Çözüm vektörü
    for i in range(n - 1, -1, -1):  # Alt satırdan üst satıra doğru ilerle
        x[i] = (M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:n])) / M[i, i]  # Çözümü hesapla

    return x  # Çözüm vektörünü döndür

# Verilen denklem sisteminin katsayı matrisi ve sağ taraf vektörü
A = np.array([[2, 3, -1],
              [-1, -4, 2],
              [3, 2, -5]], dtype=float)
b = np.array([1, -7, 0], dtype=float)

# Gauss eliminasyon yöntemiyle çözüm
solution = gauss_elimination(A, b)
print("Çözümler: x1 = {:.2f}, x2 = {:.2f}, x3 = {:.2f}".format(*solution))

