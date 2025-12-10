Projekt: Klasyfikator i liczarka monet PLN
Projekt zaliczeniowy z PRIAD. Aplikacja ma za zadanie rozpoznawać polskie monety na wideo, zliczać ich wartość i odrzucać "śmieci" (inne waluty, żetony).

Zgodnie z wytycznymi prowadzącego, nie używałem tutaj Deep Learningu (sieci neuronowych). Całość opiera się na klasycznej wizji komputerowej (OpenCV) i prostym modelu uczenia maszynowego.
Aplikacja analizuje obraz klatka po klatce, wyciąga cechy fizyczne obiektu (rozmiar, kolor, nasycenie) i na tej podstawie decyduje, jaki to nominał.
