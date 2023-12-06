1. Jaki będzie rozmiar obrazu 150x150 po przejściu przez wartstwe max2pol (2x2 filter)?  
Odp: 75x75
2. Które z poniższych nie przeciwdziała overfittingowi?  
- dropout
- augmentacja danych
- zmiana epoch
- zmiana optymalizatora
- wszystkie powyższe przeciwdziałają overfittingowi
Odp: zmiana optymalizatora
3. W jaki sposób zamrozić wartstwe lub baze?  
Odp: model.trainable = False
4. Co poprawia interpretowalność modelu?  
Odp: wszystkie niżej wymienione:
- GradCAM
- GradCAM++
- Score-CAM
- Salience Map
5. Wymień 3 wytrenowane już architektury sieci neuronowych  
Odp: AlexNet, VGG16, VGG19, ResNet50, Inception, Xception, MobileNet, MobileNetV2, MobileNetV3, DenseNet, EfficientNet, ResNeXt, SENet, ShuffleNet, SqueezeNet
6. Co robi metoda .predict()?  
Odp: Funkcja predict to podstawowa funkcja służąca do wywołania, uruchomienia modelu po jego wytrenowaniu. Funkcja ta zwraca tablicę wyników, które są przewidywanymi wartościami modelu - wykonuje wnioskowanie na podstawie danych wejściowych.
7. Co oznacza termin Transfer Learning, jak działa i dlaczego działa.  
Odp: Transfer Learning to technika uczenia maszynowego, w której wiedza z jednego modelu jest wykorzystywana do poprawy efektywności nauki innego modelu. Istnieją dwa główne rodzaje transfer learningu: fine-tuning, gdzie dostosowuje się istniejący model poprzez aktualizację wag, oraz feature extraction, gdzie używa się convolutional base jako ekstraktora cech, zamrażając niższe warstwy i dodając nowe warstwy dla nowego zadania. Ta technika działa efektywnie, gdyż ogólne cechy nabyte przez modele na dużych zbiorach danych mogą być wykorzystane do rozwiązania bardziej specyficznych zadań, co pomaga w uniknięciu overfittingu, szczególnie przy ograniczonej ilości nowych danych treningowych.
8. Co to jest callback i kiedy go używamy?  
Odp: Callback to zestaw funkcji, które można przekazać do metody model.fit() w celu wykonania określonych działań w określonych momentach treningu. Używamy go w celu zatrzymania treningu w odpowiednim momencie (Early Stopping), np. gdy model przestaje się uczyć, lub gdy osiągnie on zadowalający nas poziom dokładności.
9. W jaki sposób zorganizować katalogi datasetu, jakie funkcje/biblioteki wykorzystać do ładowania datasetu.  
Odp: Katalogi datasetu powinny być zorganizowane w następujący sposób:
```
main_directory/
...train/
......class_a/
.........a_image_1.jpg
.........a_image_2.jpg
......class_b/
.........b_image_1.jpg
.........b_image_2.jpg
...validation/
......class_a/
.........a_image_1.jpg
.........a_image_2.jpg
......class_b/
.........b_image_1.jpg
.........b_image_2.jpg
```
Do samego tworzenia struktury można wykorzystać bibliotekę os i shutil.
Natomiast do ładowania datasetu można wykorzystać metodę `image_dataset_from_directory` z biblioteki tensorflow.keras.preprocessing lub w przypadku starszych wersji tensorflow `ImageDataGenerator` z biblioteki tensorflow.keras.preprocessing.image.

10. Co to jest warstwa rescaling?  
Odp: Warstwa rescaling to warstwa, która przeskalowuje wartości pikseli obrazu do zakresu [0,1]. Jest to przydatne, gdy dane wejściowe są w różnych zakresach. (np. z RGB 0-255 do 0-1)

Inne pytania:
* Wymień nazwy dwóch warstw sieci, które nie zmieniają liczby parametrów w modelu.  
Odp: Dropout, MaxPooling2D, Flatten
* Wymień nazwy dwóch warstw sieci, które zmieniają liczby parametrów w modelu.
Odp: Conv2D, Dense, SimpleRNN, LSTM, Conv2DTranspose
* Wyjaśnij w 2-3 zdaniach dlaczego w sieciach konwolucyjnych stosowane są warstwy typu pooling.  
Odp: Warstwy typu pooling pozwalają na hierarchiczną budowę sieci neuronowej, co pozwala na analizowanie różnych pozimów abstrakcji. Pozwalają one również na zmniejszenie wymiarowości danych, co przyspiesza obliczenia oraz skupia się na najważniejszych cechach obrazu.
* Kiedy powinniśmy uczyć sieć z niskim parametrem learning_rate?  
Odp: Gdy model był już wytrenowany i chcemy go dalej dopasować do nowych danych. Explore vs Exploit - chcemy wykorzystać już istniejącą wiedzę, ale nie chcemy zbytnio zmieniać wag. Przykład: Fine Tuning.
* Co to jest augmentacja danych?  
Odp: Augmentacja danych to technika, która polega na generowaniu nowych danych treningowych na podstawie już istniejących danych treningowych. Jest to przydatne, gdy mamy mało danych treningowych, a chcemy uniknąć overfittingu.
* Podaj przykłady z wyjaśnieniem typów augmentacji danych.  
Odp: Przykłady:
  - Rotacja - obracanie obrazu o losowy kąt
  - Przesunięcie - przesuwanie obrazu o losową wartość
  - Zoom - powiększanie obrazu o losową wartość
  - Odbicie lustrzane - odbicie obrazu względem osi X lub Y
  - Jasność - zmiana jasności obrazu o losową wartość
  - Przycinanie - przycinanie obrazu o losowy rozmiar
  - Zmiana kolorów - zmiana kolorów obrazu o losową wartość
* Co to jest dropout?  
Odp: Dropout to technika, która polega na losowym wyłączaniu neuronów w trakcie treningu. Pozwala to na uniknięcie overfittingu.
* Co robi funkcja to_categorical?  
Odp: Funkcja to_categorical zamienia wektor etykiet na macierz binarną w formacie tzw. one-hot encoding. Przykład: [1, 2, 3] -> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]. Ta funkcja była używana tam gdzie stosowano funkcję kosztu categorical_crossentropy.
* Opisz działanie MaxPooling2D?  
Odp: MaxPooling2D jest warstwą, która redukuje wymiarowość danych poprzez wybieranie największej wartości z określonego obszaru. Jest to przydatne, gdy chcemy zmniejszyć wymiarowość danych, a jednocześnie skupić się na najważniejszych cechach obrazu. MaxPooling2D z parametrem pool_size=(2,2) zmniejsza wymiarowość danych 4-krotnie.
Przykład:
macierz wejściowa 3x3:
```
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
```
macierz wyjściowa 2x2 przy pool_size=(2,2):
```
[[5, 6],
 [8, 9]]
```
* Do czego służy batch_size i czym się kierujemy przy wyborze jego wartości?  
Odp: Batch size to liczba próbek, która jest przetwarzana w jednej iteracji. Przy wyborze wartości batch size kierujemy się tym, aby nie przekroczyć pamięci GPU, a jednocześnie nie zmniejszyć zbytnio wydajności obliczeń.
* Krótko opisać działanie i zastosowanie algorytmu ScoreCAM.  
Odp: ScoreCAM to algorytm, który służy do wizualizacji obszarów obrazu, które wpływają na decyzję modelu. Algorytm ten wykorzystuje "raw scores" (tj. surowe dane wyjściowe modelu przed zastosowaniem funkcji aktywacji softmax) w celu wygenerowania heatmapy, która pokazuje, które części danych wejściowych miały największy wpływ na prognozę. ScoreCAM nie wymaga modyfikacji modelu, a jedynie jego wytrenowania.
* Krótko opisać działanie i zastosowanie algorytmu GradCAM.  
Odp: GradCAM działa poprzez wykorzystanie gradientów finalnej predykcji modelu względem funkcji aktywacji ostatniej wartstwy konwolucyjnej. Ten algorytm generuje heatmapę, która pokazuje, które części danych wejściowych miały największy wpływ na prognozę. Wymaga on zmiany ostatniej funkcji aktywacji (np. z softmax) na liniową.
* Krótko opisać działanie i zastosowanie algorytmu Salience Map.  
Odp: Algorytm Salience Map jest jedną z najprostszych metod analizy obrazów, prezentując surowe wartości gradientów wyjściowych sieci neuronowej. Działa poprzez przypisanie wag pikselom na podstawie ich istotności dla wyniku sieci, umożliwiając wizualizację kluczowych obszarów. Jego zastosowania obejmują analizę obrazów, wyróżnianie istotnych cech i wspomaganie systemów rozpoznawania wzorców. Również wymaga zmiany ostatniej funkcji aktywacji na liniową.
* Wymień 2 wady i 2 zalety fine tuningu względem feature extraction.  
Odp:
Wady:
  - Fine-tuning może wymagać więcej danych treningowych niż feature extraction, aby uniknąć overfittingu, ponieważ modyfikuje się większą liczbę parametrów modelu.
  - Fine tuning wymaga więcej mocy obliczeniowej niż feature extraction, szczególnie gdy pierwotny model jest bardzo skomplikowany i głęboki.
  - Czas treningu fine tuning jest dłuższy niż feature extraction, ponieważ w fine tuningu większa ilość parametrów jest trenowana.  
Zalety:
  - Fine tuning pozwala na uzyskanie lepszych wyników niż feature extraction, ponieważ pozwala na dostosowanie modelu do konkretnego zadania, poprzez aktualizację wag w ostatnich wartstwach konwolucyjnej bazy.
  - Może być stosowany nawet gdy nowy dataset znacząco różni się od pierwotnego datasetu, ponieważ w fine tuningu aktualizuje się ostatnie wartstwy konwolucyjnej bazy, które są odpowiedzialne za wykrywanie cech ogólnych i abstrakcyjnych. To pozwala na adaptację modelu do nowych cech specyficznych dla nowego zbioru danych, nawet jeśli różnią się one od cech w zbiorze, na którym był pierwotnie szkolony.

* Porównaj fine tuning i feature extraction:  
  - Fine-tuning:
    - Definicja: Fine-tuning polega na wzięciu wcześniej wytrenowanego modelu (często na dużym zbiorze danych) i dalszym trenowaniu go na nowym, mniejszym zbiorze danych specyficznym dla celowego zadania.
    - Proces: Wytrenowany wcześniej model jest zazwyczaj używany jako punkt wyjścia, a następnie model jest "fine-tunowany", aktualizując jego wagi przy użyciu nowego zbioru danych.
    - Przykład zastosowania: Fine-tuning jest często stosowany, gdy celowe zadanie jest powiązane z pierwotnym zadaniem, dla którego model był wcześniej wytrenowany. Pozwala to modelowi dostosować swoje nauczone cechy do specyfiki nowego zbioru danych.
  - Feature Extraction:
    - Definicja: Feature extraction polega na używaniu wcześniej wytrenowanego modelu jako stałego ekstraktora cech. Wagi wcześniej wytrenowanego modelu są zamrożone, a tylko ostatnia warstwa klasyfikacji (lub warstwy) zostaje zastąpiona i trenowana na nowym zbiorze danych.
    - Proces: Dolne warstwy wcześniej wytrenowanego modelu, które przechwytują ogólne cechy, są używane jako stałe ekstraktory cech. Nowy zbiór danych jest następnie przekazywany przez te warstwy, a tylko górne warstwy są trenowane dla konkretnej klasyfikacji.
    - Przykład zastosowania: Feature extraction jest przydatne, gdy dolne warstwy wcześniej wytrenowanego modelu nauczyły się ogólnych cech, które mogą być zastosowane do nowego zadania. Jest szczególnie skuteczne, gdy nowy zbiór danych jest mały, a fine-tuning całego modelu może prowadzić do nadmiernego dopasowania.
  - Podsumowując, zarówno fine-tuning, jak i feature extraction to strategie w ramach szerszego pojęcia transfer learningu, gdzie wiedza zdobyta z jednego zadania (pre-training) jest wykorzystywana do poprawy wydajności na powiązanym zadaniu. Wybór między fine-tuningiem a feature extraction zależy od czynników takich jak podobieństwo zadań i wielkość nowego zbioru danych.

* Opisz metody automatycznego doboru hiperparametrów.
  - Callbacks: Metoda ta polega na monitorowaniu procesu uczenia w trakcie treningu i dostosowywaniu hiperparametrów na podstawie pewnych zdefiniowanych warunków. Przykłady to EarlyStopping, który przerywa trening, gdy pewna metryka przestanie się poprawiać, lub LearningRateScheduler, który dostosowuje tempo uczenia w trakcie procesu.
  - Random Search: Jest to metoda przeszukiwania przestrzeni hiperparametrów, gdzie wartości są losowo wybierane z określonych zakresów. Choć bardziej efektywne niż pełne przeszukiwanie siatki, random search pozwala na eksplorację różnorodnych kombinacji hiperparametrów.
  - Grid Search: Ta metoda polega na przeszukiwaniu wszystkich możliwych kombinacji hiperparametrów zdefiniowanych w określonych zakresach. Chociaż dokładniejsza niż random search, grid search może być bardziej kosztowna obliczeniowo, zwłaszcza w przypadku dużej liczby hiperparametrów.
* Dla problemu binarnego jakiej funkcji aktywacji użyjemy w ostatniej warstwie sieci?  
Odp: Funkcja sigmoidalna (sigmoid).
* Dla problemu wieloklasowego jakiej funkcji aktywacji użyjemy w ostatniej warstwie sieci?  
Odp: Funkcja softmax.
* Ile razy jeden obraz przejdzie przez sieć podczas uczenia?  
Odp: Tyle ile wynosi liczba epok.
* Czego użyjemy, kiedy mamy mało danych, ale o podobnych cechach, co już
wyuczona sieć?  
Odp: Transfer Learning.
* Cecha charakterystyczna MobileNet?  
Odp: MobileNet jest architekturą sieci neuronowej, która została zaprojektowana specjalnie dla urządzeń mobilnych. Jest ona bardzo lekka, co pozwala na jej wykorzystanie na urządzeniach mobilnych, które mają ograniczone zasoby obliczeniowe.
* Wypisz funkcje aktywacji.  
Odp: Sigmoid, Tanh, ReLU, LeakyReLU, Softmax, Linear, ELU, Binary Step.
* Wypisz funkcje kosztu.  
Odp: MSE (Mean Squared Error), MAE (Mean Absolute Error), Binary Crossentropy, Categorical Crossentropy, Sparse Categorical Crossentropy, Hinge, Squared Hinge.
* Wypisz funkcje optymalizatorów.  
Odp: SGD (Stochastic Gradient Descent), RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl.
* Wypisz metryki.  
Odp: Accuracy, Precision, Recall, AUC, F1-score, IoU.
* Co robi warstwa Flatten?  
Odp: Warstwa Flatten służy do spłaszczenia danych wejściowych do jednowymiarowej tablicy. Jest to przydatne, gdy chcemy przekazać dane z warstwy konwolucyjnej do warstwy Dense.
* Opisz dataset Fashion MNIST.  
Odp: Fashion MNIST to zbiór danych zawierający 70 000 obrazów w skali szarości o rozmiarze 28x28 pikseli, przedstawiających klasy ubrań. Zbiór ten jest często używany do testowania modeli sieci neuronowych, ponieważ jest on podobny do zbioru MNIST (liczby od 0 do 9), ale jest bardziej wymagający, ponieważ zawiera bardziej złożone wzorce.
* Które warstwy sieci zachowasz w transfer learningu i dlaczego?  
Odp: W transfer learningu zazwyczaj zachowuje się warstwy konwolucyjne (convolutional layers), które zazwyczaj były uczone na bardzo dużym zbiorze danych. Te warstwy są odpowiedzialne za ekstrakcję ogólnych cech, które mogą być użyteczne w różnych zadaniach. Zachowując te warstwy, model może wykorzystać już nauczone reprezentacje cech, co może przyspieszyć proces uczenia się na nowym, bardziej specyficznym zadaniu. Jednak warstwy końcowe (fully connected layers) często są modyfikowane lub zastępowane, aby dostosować się do nowego zbioru danych i zadania.
* Jak dobierzesz parametry uczenia sieci neuronowej?
  1. Współczynnik uczenia (Learning Rate):
     - Dobór wartości współczynnika uczenia ma wpływ na tempo, z jakim model aktualizuje wagi podczas procesu uczenia.
     - Warto eksperymentować z różnymi wartościami, takimi jak 0.1, 0.01, czy 0.001, aby znaleźć optymalny współczynnik uczenia.
     - Można również zastosować metody adaptacyjne, takie jak Adam czy RMSprop, które automatycznie dostosowują współczynnik uczenia w trakcie uczenia.
  2. Rozmiar batcha (Batch Size):  
     - Batch size określa, ile próbek danych jest używanych do jednej aktualizacji wag modelu.
     - Dla większych batchy może być wymagana większa ilość pamięci GPU, ale może to prowadzić do bardziej stabilnego uczenia.
     - Zbyt mały batch może wprowadzić szumy w aktualizacjach wag, ale zbyt duży może spowolnić proces uczenia.
  3. Liczba epok (Number of Epochs):
     - Liczba epok określa, ile razy cały zbiór danych jest przekazywany przez sieć podczas procesu uczenia.
     - Warto monitorować krzywą uczenia i zatrzymać uczenie, gdy osiągnięty zostanie punkt, w którym model zaczyna się przeuczać (overfitting).
     - W przypadku dużych zbiorów danych lub modeli złożonych, zalecane jest również korzystanie z technik takich jak wcześniejsze zatrzymywanie uczenia (early stopping) w celu uniknięcia przeuczenia.

  Dobieranie tych parametrów wymaga eksperymentów i dostosowywania do konkretnego zadania oraz dostępnego zestawu danych. Można również korzystać z technik automatycznego doboru hiperparametrów, takich jak Random Search czy Grid Search, aby zoptymalizować parametry w bardziej efektywny sposób.

* Opisz krótko poniższe warswy i wymień ich zastosowanie: Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Softmax.
  1. Dense (Fully Connected): Warstwa ta składa się z pełni połączonych neuronów, gdzie każdy neuron w danej warstwie jest połączony z każdym neuronem warstwy poprzedniej. Zastosowanie: klasyfikacja, regresja.
  2. Conv2D (Convolutional): Warstwa konwolucyjna stosuje filtr konwolucyjny do przetwarzania lokalnych obszarów obrazu, wykrywając hierarchię cech. Zastosowanie: rozpoznawanie wzorców w obrazach, analiza obrazów.
  3. MaxPooling2D: Warstwa ta wykonuje próbkowanie danych, zachowując jedynie najważniejsze informacje z danego obszaru poprzez wybieranie maksymalnej wartości. Zastosowanie: redukcja wymiarowości danych, pozyskiwanie kluczowych cech.
  4. Flatten: Warstwa ta przekształca wielowymiarowe dane (np. z warstwy konwolucyjnej) na jednowymiarowy wektor, przygotowując dane do warstw Dense. Zastosowanie: połączenie warstw konwolucyjnych z warstwami w pełni połączonymi.
  5. Dropout: Warstwa ta losowo "wyłącza" pewne neurony podczas treningu, co pomaga w regularyzacji modelu, zapobiegając przeuczeniu. Zastosowanie: regularyzacja, redukcja przeuczenia.
  6. Softmax: Warstwa ta przekształca wartości wyjściowe na prawdopodobieństwa, umożliwiając modelowi dokonywanie klasyfikacji wieloklasowej. Zastosowanie: klasyfikacja wieloklasowa.
