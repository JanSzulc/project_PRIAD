#!/usr/bin/env python
# coding: utf-8

# # Rozpoznawanie monet po wielkości i kolorze (v2)
# 
# Klasyfikacja polskich monet z preprocessingiem obrazów (standaryzacja 512x512)
# przed ekstrakcją cech klasycznych.

# ## 1. Import bibliotek

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')


# ## 2. Konfiguracja

TARGET_IMAGE_SIZE = (512, 512)
DATASET_PATH = "dataset"


# ## 3. Preprocessing obrazu (z sieci neuronowej)

def prepare_single_image(img_path):
    """
    Standaryzuje obraz monety:
    - Znajduje monetę na zdjęciu
    - Wycina ją i centruje na czarnym tle
    - Skaluje do TARGET_IMAGE_SIZE (512x512)
    
    Zwraca obraz w formacie BGR lub None jeśli nie znaleziono monety.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    blurred = cv2.GaussianBlur(v, (11, 11), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        cX, cY = int(x), int(y)
        radius = int(radius * 1.05)
        
        # Maska kołowa
        mask_circle = np.zeros_like(v)
        cv2.circle(mask_circle, (cX, cY), radius, 255, -1)
        coin_on_black = cv2.bitwise_and(img, img, mask=mask_circle)
        
        # Wycinanie i centrowanie
        crop_size = 260
        half_size = crop_size // 2
        square_canvas = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        
        x1 = cX - half_size
        y1 = cY - half_size
        x2 = cX + half_size
        y2 = cY + half_size
        
        img_h, img_w = coin_on_black.shape[:2]
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(img_w, x2)
        src_y2 = min(img_h, y2)
        
        dst_x1 = max(0, -x1)
        dst_y1 = max(0, -y1)
        copy_w = src_x2 - src_x1
        copy_h = src_y2 - src_y1
        
        if copy_w > 0 and copy_h > 0:
            square_canvas[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = \
                coin_on_black[src_y1:src_y2, src_x1:src_x2]
        
        resized = cv2.resize(square_canvas, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        return resized
    else:
        return None


def prepare_image_from_array(img_bgr):
    """
    Wersja prepare_single_image dla obrazu już wczytanego do pamięci.
    Przyjmuje obraz BGR, zwraca przetworzony obraz BGR.
    """
    if img_bgr is None:
        return None
    
    img = img_bgr.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    blurred = cv2.GaussianBlur(v, (11, 11), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        cX, cY = int(x), int(y)
        radius = int(radius * 1.05)
        
        mask_circle = np.zeros_like(v)
        cv2.circle(mask_circle, (cX, cY), radius, 255, -1)
        coin_on_black = cv2.bitwise_and(img, img, mask=mask_circle)
        
        crop_size = 260
        half_size = crop_size // 2
        square_canvas = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        
        x1 = cX - half_size
        y1 = cY - half_size
        x2 = cX + half_size
        y2 = cY + half_size
        
        img_h, img_w = coin_on_black.shape[:2]
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(img_w, x2)
        src_y2 = min(img_h, y2)
        
        dst_x1 = max(0, -x1)
        dst_y1 = max(0, -y1)
        copy_w = src_x2 - src_x1
        copy_h = src_y2 - src_y1
        
        if copy_w > 0 and copy_h > 0:
            square_canvas[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = \
                coin_on_black[src_y1:src_y2, src_x1:src_x2]
        
        resized = cv2.resize(square_canvas, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        return resized
    else:
        return None


# ## 4. Ekstrakcja cech (ze standaryzowanego obrazu)

FEATURE_NAMES = [
    'diameter',
    'mean_r', 'mean_g', 'mean_b',
    'std_r', 'std_g', 'std_b',
    'mean_h', 'mean_s', 'mean_v',
    'gold_ratio', 'brightness'
]


def extract_features(image_rgb):
    """
    Ekstrahuje cechy z PRZETWORZONEGO obrazu (512x512 z monetą na czarnym tle).
    Obraz wejściowy powinien być w formacie RGB.
    """
    h, w = image_rgb.shape[:2]
    
    # Znajdź monetę (powinna być wycentrowana, ale dla pewności)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        x, y, radius = int(x), int(y), int(radius)
    else:
        # Fallback - centrum obrazu
        x, y = w // 2, h // 2
        radius = min(h, w) // 3
    
    # Znormalizowana średnica (względem rozmiaru obrazu)
    diameter = (radius * 2) / max(h, w)
    
    # Maska do ekstrakcji pikseli (70% promienia żeby uniknąć krawędzi)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), int(radius * 0.7), 255, -1)
    
    pixels = image_rgb[mask > 0]
    
    if len(pixels) < 100:
        # Fallback
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        pixels = image_rgb[mask > 0]
    
    # Cechy RGB
    mean_r = np.mean(pixels[:, 0]) / 255
    mean_g = np.mean(pixels[:, 1]) / 255
    mean_b = np.mean(pixels[:, 2]) / 255
    std_r = np.std(pixels[:, 0]) / 255
    std_g = np.std(pixels[:, 1]) / 255
    std_b = np.std(pixels[:, 2]) / 255
    
    # Cechy HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv[mask > 0]
    mean_h = np.mean(hsv_pixels[:, 0]) / 180
    mean_s = np.mean(hsv_pixels[:, 1]) / 255
    mean_v = np.mean(hsv_pixels[:, 2]) / 255
    
    # Cechy pochodne
    gold_ratio = mean_r / (mean_b + 0.001)
    brightness = (mean_r + mean_g + mean_b) / 3
    
    return np.array([
        diameter,
        mean_r, mean_g, mean_b,
        std_r, std_g, std_b,
        mean_h, mean_s, mean_v,
        gold_ratio, brightness
    ])


# ## 5. Wczytywanie datasetu z preprocessingiem

def load_dataset(data_dir):
    """
    Wczytuje dataset z preprocessingiem każdego obrazu przed ekstrakcją cech.
    Obsługuje dwie struktury:
      1) dataset/train/10gr/avers/*.jpg  (z podfolderami avers/revers)
      2) dataset/train/10gr/*.jpg        (płaska struktura)
    """
    X = []
    y = []
    skipped = 0
    
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        count = 0
        
        # Zbierz wszystkie ścieżki do obrazów (rekurencyjnie)
        image_paths = []
        for root, dirs, files in os.walk(class_path):
            for img_name in files:
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_paths.append(os.path.join(root, img_name))
        
        for img_path in image_paths:
            # PREPROCESSING - standaryzacja obrazu
            processed_img = prepare_single_image(img_path)
            
            if processed_img is not None:
                # Konwersja BGR -> RGB
                processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                
                # Ekstrakcja cech z przetworzonego obrazu
                features = extract_features(processed_rgb)
                X.append(features)
                y.append(class_name)
                count += 1
            else:
                skipped += 1
                print(f"    [SKIP] Nie znaleziono monety: {img_path}")
        
        print(f"  {class_name}: {count}")
    
    if skipped > 0:
        print(f"  (Pominięto {skipped} obrazów bez wykrytej monety)")
    
    return np.array(X), y


# ## 6. Klasyfikator z detekcją anomalii

def classify_with_anomaly_detection(X_train, y_train, X_test, prob_thresh=0.5, iso_contam=0.03):
    """
    Klasyfikacja SVM + Isolation Forest do wykrywania anomalii (obcych monet).
    """
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    proba = svm.predict_proba(X_test)
    max_proba = np.max(proba, axis=1)
    svm_pred = svm.predict(X_test)
    
    iso = IsolationForest(contamination=iso_contam, random_state=42)
    iso.fit(X_train)
    iso_pred = iso.predict(X_test)
    
    predictions = []
    for i in range(len(X_test)):
        svm_uncertain = max_proba[i] < prob_thresh
        iso_anomaly = iso_pred[i] == -1
        
        if svm_uncertain and iso_anomaly:
            predictions.append('wrong')
        else:
            predictions.append(svm_pred[i])
    
    return predictions, svm, iso


# ## 7. Wizualizacja wyników

def plot_results(y_true, y_pred):
    """Wyświetla macierz pomyłek i raport klasyfikacji."""
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Macierz pomyłek (dokładność: {acc*100:.2f}%)')
    plt.ylabel('Prawdziwa klasa')
    plt.xlabel('Przewidziana klasa')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    
    print(f"\nDokładność: {acc*100:.2f}%")
    print("\nRaport klasyfikacji:")
    print(classification_report(y_true, y_pred))


# ## 8. Funkcja do klasyfikacji pojedynczego obrazu

def classify_single_image(img_path, model_bundle):
    """
    Klasyfikuje pojedynczy obraz monety.
    
    Args:
        img_path: ścieżka do obrazu
        model_bundle: słownik z modelem, skalerem itd.
    
    Returns:
        predicted_class: przewidziana klasa lub 'wrong' dla anomalii
        confidence: pewność predykcji
    """
    svm_model = model_bundle['svm_model']
    iso_model = model_bundle['iso_model']
    scaler = model_bundle['scaler']
    prob_thresh = model_bundle.get('prob_thresh', 0.6)
    
    # Preprocessing
    processed_img = prepare_single_image(img_path)
    if processed_img is None:
        return 'unknown', 0.0
    
    processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    
    # Ekstrakcja cech
    features = extract_features(processed_rgb)
    features_scaled = scaler.transform([features])
    
    # Predykcja
    proba = svm_model.predict_proba(features_scaled)
    max_proba = np.max(proba)
    svm_pred = svm_model.predict(features_scaled)[0]
    
    iso_pred = iso_model.predict(features_scaled)[0]
    
    if max_proba < prob_thresh and iso_pred == -1:
        return 'wrong', max_proba
    else:
        return svm_pred, max_proba


# ## 9. Główny pipeline

if __name__ == "__main__":
    # Wczytywanie danych treningowych
    print("=" * 50)
    print("Wczytywanie danych treningowych (z preprocessingiem)...")
    print("=" * 50)
    X_train, y_train = load_dataset(os.path.join(DATASET_PATH, "train"))
    print(f"\nRazem: {len(X_train)} obrazów, {len(FEATURE_NAMES)} cech\n")
    
    # Wczytywanie danych testowych
    print("=" * 50)
    print("Wczytywanie danych testowych (z preprocessingiem)...")
    print("=" * 50)
    X_test, y_test = load_dataset(os.path.join(DATASET_PATH, "test"))
    print(f"\nRazem: {len(X_test)} obrazów")
    
    # Skalowanie
    print("\n" + "=" * 50)
    print("Skalowanie cech...")
    print("=" * 50)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Trening i predykcja
    print("\n" + "=" * 50)
    print("Trening klasyfikatora...")
    print("=" * 50)
    PROB_THRESH = 0.6
    ISO_CONTAM = 0.03
    
    y_pred, svm_model, iso_model = classify_with_anomaly_detection(
        X_train_scaled, y_train, X_test_scaled,
        prob_thresh=PROB_THRESH,
        iso_contam=ISO_CONTAM
    )
    
    # Wizualizacja
    print("\n" + "=" * 50)
    print("Wyniki:")
    print("=" * 50)
    plot_results(y_test, y_pred)
    
    # Zapis modelu
    print("\n" + "=" * 50)
    print("Zapisywanie modelu...")
    print("=" * 50)
    
    model_bundle = {
        'svm_model': svm_model,
        'iso_model': iso_model,
        'scaler': scaler,
        'feature_names': FEATURE_NAMES,
        'prob_thresh': PROB_THRESH,
        'target_image_size': TARGET_IMAGE_SIZE
    }
    
    output_filename = 'coin_classifier_model_v2.pkl'
    joblib.dump(model_bundle, output_filename)
    
    print(f"\nModel zapisany do: {output_filename}")
    print("\nZapisane komponenty:")
    print("  - svm_model: klasyfikator SVM")
    print("  - iso_model: detektor anomalii (Isolation Forest)")
    print("  - scaler: StandardScaler")
    print("  - feature_names: nazwy cech")
    print("  - prob_thresh: próg pewności")
    print("  - target_image_size: rozmiar obrazu po preprocessingu")
