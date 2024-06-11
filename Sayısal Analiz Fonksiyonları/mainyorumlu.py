import numpy as np

print("########################################## Jacobi ve Gauss Seidal ##############################################")

# A: Katsayı matrisi, b: Sağ taraf vektörü, tolerance: Hata toleransı
def jacobi(A, b, tolerance=1e-4):
    x = np.zeros_like(b)  # Başlangıç değeri olarak b'nin boyutunda bir sıfır vektörü oluşturuluyor.
    while True:  # Belirlenen tolerans değerine ulaşılana kadar döngü devam edecek.
        x_new = np.zeros_like(x)  # Yeni çözüm vektörü oluşturuluyor.
        for i in range(A.shape[0]):  # Satırlar üzerinde döngü başlatılıyor.
            s1 = np.dot(A[i, :i], x[:i])  # İlgili satırdaki önceki çözümlerin ilgili elemanları ile iç çarpım yapılıyor.
            s2 = np.dot(A[i, i + 1:], x[i + 1:])  # İlgili satırdaki sonraki çözümlerin ilgili elemanları ile iç çarpım yapılıyor.
            x_new[i] = (b[i] - s1 - s2) / A[i, i]  # Yeni çözüm vektörünün i. elemanı hesaplanıyor.
        if np.allclose(x, x_new, atol=tolerance):  # Hesaplanan yeni çözümün eski çözüme olan yakınlığı kontrol ediliyor.
            break  # Belirtilen toleransa ulaşıldığında döngüden çıkılıyor.
        x = x_new  # Yeni çözüm eski çözüm olarak atanıyor.
    return x  # Jacobi iteratif çözümü döndürülüyor.

# Gauss-Seidel iteratif yöntemi tanımlanıyor.
# A: Katsayı matrisi, b: Sağ taraf vektörü, tolerance: Hata toleransı
def gauss_seidel(A, b, tolerance=1e-4):
    x = np.zeros_like(b)  # Başlangıç değeri olarak b'nin boyutunda bir sıfır vektörü oluşturuluyor.
    while True:  # Belirlenen tolerans değerine ulaşılana kadar döngü devam edecek.
        x_new = np.zeros_like(x)  # Yeni çözüm vektörü oluşturuluyor.
        for i in range(A.shape[0]):  # Satırlar üzerinde döngü başlatılıyor.
            s1 = np.dot(A[i, :i], x_new[:i])  # İlgili satırdaki önceki ve yeni çözümlerin ilgili elemanları ile iç çarpım yapılıyor.
            s2 = np.dot(A[i, i + 1:], x[i + 1:])  # İlgili satırdaki sonraki çözümlerin ilgili elemanları ile iç çarpım yapılıyor.
            x_new[i] = (b[i] - s1 - s2) / A[i, i]  # Yeni çözüm vektörünün i. elemanı hesaplanıyor.
        if np.allclose(x, x_new, atol=tolerance):  # Hesaplanan yeni çözümün eski çözüme olan yakınlığı kontrol ediliyor.
            break  # Belirtilen toleransa ulaşıldığında döngüden çıkılıyor.
        x = x_new  # Yeni çözüm eski çözüm olarak atanıyor.
    return x  # Gauss-Seidel iteratif çözümü döndürülüyor.

# Katsayı matrisi A ve sağ taraf vektörü b tanımlanıyor.
A = np.array([[10, -1, 1, -3],
              [-2, 15, -3, 0.5],
              [2, -4, 11, 1.5],
              [0.5, -2.5, 5.5, -15]])
b = np.array([13, 42, -13, 22])

# Jacobi ve Gauss-Seidel yöntemleri kullanılarak çözüm yapılıyor.
x_jacobi = jacobi(A, b)  # Jacobi yöntemi ile çözüm
x_gauss_seidel = gauss_seidel(A, b)  # Gauss-Seidel yöntemi ile çözüm

# Sonuçlar ekrana yazdırılıyor.
print("Jacobi yöntemi ile çözüm:", x_jacobi)
print("Gauss-Seidel yöntemi ile çözüm:", x_gauss_seidel)

# Diagonal dominance kontrolü için bir fonksiyon tanımlanıyor.
def is_diagonally_dominant(A):
    for i in range(len(A)):  # Matrisin her satırı için döngü başlatılıyor.
        diagonal = abs(A[i][i])  # Matrisin köşegen elemanı alınıyor.
        off_diagonal_sum = sum(abs(A[i][j]) for j in range(len(A)) if i != j)  # Matrisin köşegeni dışındaki elemanların toplamı hesaplanıyor.
        if diagonal <= off_diagonal_sum:  # Köşegen elemanın, köşegen dışındaki elemanların toplamından küçük veya eşit olup olmadığı kontrol ediliyor.
            return False  # Eğer bu koşul sağlanıyorsa matris diagonal dominant değildir.
    return True  # Diagonal dominant ise True döndürülüyor.

# Diagonal dominantlik kontrolü için örnek bir matris tanımlanıyor.
A1 = np.array([[-5, -4, -1],
               [1, -3, -4],
               [2, -1, 5]])

# Diagonal dominantlik kontrol ediliyor.
diagonal_dominant = is_diagonally_dominant(A1)
print("Diagonal Dominant: ", diagonal_dominant)  # Sonuç ekrana yazdırılıyor.
#2. soru diaoganal olarak dominant değildir. bu yöntemlerle çözülemez.

print("\n\n########################################## Vionello ##############################################")

# A: Katsayı matrisi, v: Başlangıç vektörü, iterations: İterasyon sayısı, tolerance: Hata toleransı
def vionello(A, v, iterations=7, tolerance=1e-4):
    # Güç yöntemi ile en büyük özdeğer ve özvektörü bulma
    v_max = np.copy(v)  # Başlangıç vektörü kopyalanıyor.
    for i in range(iterations):  # Belirtilen iterasyon sayısına kadar döngü başlatılıyor.
        v_next = np.dot(A, v_max)  # Yeni özvektör hesaplanıyor.
        norm_v_next = np.linalg.norm(v_next)  # Yeni özvektörün normu hesaplanıyor.
        if np.linalg.norm(v_next - v_max * norm_v_next) < tolerance:  # Yakınsama kontrolü yapılıyor.
            print("En büyük özdeğer için yakınsama sağlandı, iterasyon: ", i+1)  # Yakınsama sağlandığında ekrana iterasyon sayısı yazdırılıyor.
            break  # Belirtilen toleransa ulaşıldığında döngüden çıkılıyor.
        v_max = v_next / norm_v_next  # Yeni özvektör atanıyor.
    largest_eigenvalue = np.dot(v_max.T, np.dot(A, v_max)) / np.dot(v_max.T, v_max)  # En büyük özdeğer hesaplanıyor.

    # Ters güç yöntemi ile en küçük özdeğer ve özvektörü bulma
    try:
        A_inv = np.linalg.inv(A)  # Katsayı matrisinin tersi hesaplanıyor.
        v_min = np.copy(v)  # Başlangıç vektörü kopyalanıyor.
        for i in range(iterations):  # Belirtilen iterasyon sayısına kadar döngü başlatılıyor.
            v_next = np.dot(A_inv, v_min)  # Yeni özvektör hesaplanıyor.
            norm_v_next = np.linalg.norm(v_next)  # Yeni özvektörün normu hesaplanıyor.
            if np.linalg.norm(v_next - v_min * norm_v_next) < tolerance:  # Yakınsama kontrolü yapılıyor.
                print("En küçük özdeğer için yakınsama sağlandı, iterasyon: ", i+1)  # Yakınsama sağlandığında ekrana iterasyon sayısı yazdırılıyor.
                break  # Belirtilen toleransa ulaşıldığında döngüden çıkılıyor.
            v_min = v_next / norm_v_next  # Yeni özvektör atanıyor.
        smallest_eigenvalue = np.dot(v_min.T, np.dot(A, v_min)) / np.dot(v_min.T, v_min)  # En küçük özdeğer hesaplanıyor.
    except np.linalg.LinAlgError:  # Tersi alınamayan durumlar için hata yakalanıyor.
        smallest_eigenvalue = None  # Hesaplanamadığı durumda en küçük özdeğer None olarak atanıyor.
        v_min = None  # Hesaplanamadığı durumda en küçük özvektör None olarak atanıyor.

    return largest_eigenvalue, v_max, smallest_eigenvalue, v_min  # Hesaplanan özdeğerler ve özvektörler döndürülüyor.

# Katsayı matrisi A ve başlangıç vektörü v tanımlanıyor.
A = np.array([[2, -1, 3],
              [4, 2, 6],
              [5, 1, 4]])
v = np.array([0, 1, 1], dtype=float)  # Başlangıç vektörü

# İterasyon sayısı ve tolerans tanımlanıyor.
iterations = 7
tolerance = 1e-4

# Vionello yöntemi ile özdeğerler ve özvektörler hesaplanıyor.
largest_eigenvalue, largest_eigenvector, smallest_eigenvalue, smallest_eigenvector = vionello(A, v, iterations, tolerance)

# Sonuçlar ekrana yazdırılıyor.
print("En büyük özdeğer:", largest_eigenvalue)
print("En büyük özvektör:", largest_eigenvector)
print("En küçük özdeğer:", smallest_eigenvalue if smallest_eigenvalue is not None else "Hesaplanamadı")
print("En küçük özvektör:", smallest_eigenvector if smallest_eigenvector is not None else "Hesaplanamadı")

print("\n\n########################################## Newton-Raphson ##############################################")
print("\n\t1")
# İlk fonksiyon ve türevi tanımlanıyor.
def f(x):
    return x**3 - np.exp(-np.sqrt(x)) - 1

def df(x):
    return 3*x**2 + (np.exp(-np.sqrt(x)) / (2*np.sqrt(x)))

# Newton-Raphson yöntemi tanımlanıyor.
def newton_raphson(f, df, x0, tol=1e-4, max_iter=100):
    x = x0  # Başlangıç tahmini atanıyor.
    for i in range(max_iter):  # Maksimum iterasyon sayısı kadar döngü başlatılıyor.
        x_new = x - f(x) / df(x)  # Yeni tahmin hesaplanıyor.
        if abs(x_new - x) < tol:  # Yakınsama kontrolü yapılıyor.
            return x_new, i+1  # Yakınsama sağlandığında kök ve iterasyon sayısı döndürülüyor.
        x = x_new  # Yeni tahmin eski tahmin olarak atanıyor.
    return x, max_iter  # Maksimum iterasyon sayısına ulaşıldığında döndürülüyor.

# Başlangıç tahmini ve tolerans atanıyor.
x0 = 1/2
tolerance = 1e-4

# Newton-Raphson yöntemi ile kök hesaplanıyor.
root, iterations = newton_raphson(f, df, x0, tol=tolerance)

# Sonuçlar ekrana yazdırılıyor.
print("Bulunan kök:", root)
print("Toplam iterasyon sayısı:", iterations)

print("\n\t2")

# İkinci fonksiyon ve türevi tanımlanıyor.
def f(x):
    return x**3 - np.e - 1

def df(x):
    return 3*x**2

# Başlangıç değeri ve tolerans atanıyor.
x0 = 1.5
tolerance = 1e-4

# Newton-Raphson yöntemi ile kök hesaplanıyor.
root, iterations = newton_raphson(f, df, x0, tol=tolerance)

# Sonuçlar ekrana yazdırılıyor.
print("Bulunan kök:", root)
print("Toplam iterasyon sayısı:", iterations)

print("\n\t3")

# İki boyutlu fonksiyon ve Jacobian matrisi tanımlanıyor.
def F(x, y):
    return np.array([x + np.exp(y) - 1, x + y**2 - 1])

def J(x, y):
    return np.array([[1, np.exp(y)], [1, 2*y]])

# Başlangıç değerleri atanıyor.
x0, y0 = 0.8, -1.7

# Newton-Raphson yöntemi ile kökler bulunuyor.
x, y, iterations = newton_raphson_2d(F, J, x0, y0)

# Sonuçlar ekrana yazdırılıyor.
print("Bulunan kökler: x =", x, ", y =", y)
print("Toplam iterasyon sayısı:", iterations)