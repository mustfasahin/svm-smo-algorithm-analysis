# SVM (Hard-Margin) & SMO Algoritması Analizi

Bu proje, **Algoritma Analizi** dersi kapsamında geliştirilmiş, TypeScript tabanlı bir Hard-Margin Destek Vektör Makinesi (SVM) implementasyonudur. Proje, optimizasyon problemi için **Sequential Minimal Optimization (SMO)** algoritmasını kullanmakta ve modelin karmaşıklığını Big-O notasyonu ile analiz etmektedir.

## 🚀 Öne Çıkan Özellikler

- **SMO Algoritması:** Platt (1998) tarafından önerilen, karesel programlama problemlerini analitik olarak çözen verimli optimizasyon motoru.
- **Lineer Çekirdek (Linear Kernel):** Verilerin doğrusal olarak ayrıştırılması için optimize edilmiş iç çarpım hesaplaması.
- **Gauss Veri Üretimi:** Box-Muller transformasyonu kullanılarak normal dağılımlı test verilerinin oluşturulması.
- **ASCII Görselleştirme:** Eğitim sonuçlarının ve karar sınırının doğrudan terminal üzerinden grafiksel olarak gösterimi.
- **Tip Güvenliği:** TypeScript'in sunduğu güçlü tip sistemi ve SOLID prensiplerine uygun nesne odaklı mimari.

## 📊 Algoritma Karmaşıklığı

| İşlem | Zaman Karmaşıklığı | Açıklama |
| :--- | :--- | :--- |
| **Eğitim (SMO)** | $O(n^2 \cdot T)$ | $n$: veri sayısı, $T$: iterasyon sayısı. |
| **Tahmin (Predict)** | $O(\vert SV \vert \cdot d)$ | $\vert SV \vert$: destek vektörü sayısı, $d$: boyut. |
| **Bellek (Kernel Matrix)** | $O(n^2)$ | Ön hesaplanmış kernel matrisi kullanımı. |

## 🛠 Kurulum ve Çalıştırma

Bilgisayarınızda Node.js yüklü olmalıdır.

1. Bağımlılıkları yüklemeden doğrudan çalıştırmak için:
   ```powershell
   npx tsx svm.ts
   ```

2. Geliştirme araçlarıyla çalıştırmak için:
   ```powershell
   npx ts-node svm.ts
   ```

## 📐 Yazılım Mimarisi

Proje, genişletilebilir ve test edilebilir bir yapı için şu tasarım desenlerini kullanır:
- **Facade Pattern:** `SVMClassifier` sınıfı, karmaşık eğitim süreçlerini basit bir arayüzle sunar.
- **Dependency Injection:** Çözücü (`SMOSolver`) ve görselleştirici (`ConsoleVisualizer`) sınıfları ana sınıfa enjekte edilir.
- **Immutability:** `SVMModel` ve `DataPoint` gibi sınıflar verinin bütünlüğünü korumak için değişmez yapıdadır.

## 🚗 Otonom Navigasyon Analojisi

Kodun içerisinde yer alan analojiye göre; SVM tarafından bulunan **Margin**, bir otonom aracın engeller arasından geçebileceği "güvenlik koridoru" genişliğini temsil eder. Maksimum margin, aracın en güvenli rotayı çizmesini sağlar.

---
*Bu proje akademik amaçlarla geliştirilmiştir.*
