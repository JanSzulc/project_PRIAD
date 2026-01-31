import cv2
import numpy as np
import os
import shutil
import tensorflow as tf

VIDEO_PATH = 'slideshow.mp4'
MODEL_PATH = 'coin_classifier_neural_network_model.keras'
OUTPUT_DIR = 'detected_coins_final/'

TARGET_IMAGE_SIZE = (512, 512)
VALID_PROG = 0.99

MIN_COIN_DIAMETER = 60
MARGIN_X = 20

def ensure_clean_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def prepare_image(img):
    if img is None: return None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    blurred = cv2.GaussianBlur(v, (11, 11), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius * 1.05)
        
        mask_circle = np.zeros_like(v)
        cv2.circle(mask_circle, center, radius, 255, -1)
        
        coin_on_black = cv2.bitwise_and(img, img, mask=mask_circle)

        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(c)
        
        if w_rect < 10 or h_rect < 10: return None

        coin_crop = coin_on_black[y_rect:y_rect+h_rect, x_rect:x_rect+w_rect]

        target_h, target_w = TARGET_IMAGE_SIZE
        final_canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        dst_x = (target_w - w_rect) // 2
        dst_y = (target_h - h_rect) // 2
        
        if dst_x < 0 or dst_y < 0:
             return None 

        final_canvas[dst_y:dst_y+h_rect, dst_x:dst_x+w_rect] = coin_crop
        return final_canvas
    else:
        return None

def find_coin_by_shape(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 150)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_coin = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000: continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.75:
            x, y, w, h = cv2.boundingRect(cnt)
            
            aspect_ratio = float(w) / h
            if 0.8 < aspect_ratio < 1.2:
                if area > max_area and w > MIN_COIN_DIAMETER:
                    max_area = area
                    best_coin = (x, x + w)

    return best_coin

def extract_coins():
    print("\nSKANOWANIE WIDEO I WYCINANIE")
    ensure_clean_dir(OUTPUT_DIR)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"BŁĄD: Nie można otworzyć {VIDEO_PATH}")
        return 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    screen_center_x = frame_width // 2

    count = 0
    just_captured = False
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        result = find_coin_by_shape(frame)

        if result:
            start_x, end_x = result
            coin_center = (start_x + end_x) // 2
            distance = abs(coin_center - screen_center_x)

            if distance < 15:
                if not just_captured:
                    cut_x1 = max(0, start_x - MARGIN_X)
                    cut_x2 = min(frame_width, end_x + MARGIN_X)
                    
                    strip = frame[0:frame_height, cut_x1:cut_x2]
                    
                    final_img = prepare_image(strip)
                    
                    if final_img is not None:
                        filename = os.path.join(OUTPUT_DIR, f"coin_{count:04d}.jpg")
                        cv2.imwrite(filename, final_img)
                        print(f"[{frame_idx}] Zapisano: {filename}")
                        count += 1
                        just_captured = True
            else:
                if distance > 50:
                    just_captured = False
        
        if frame_idx % 50 == 0:
            print(f"Postęp wideo: {frame_idx}/{total_frames}", end='\r')

    cap.release()
    print(f"\nZakończono wycinanie. Zapisano {count} obrazów w '{OUTPUT_DIR}'")
    return count

def get_class_names(num_classes):
    base_nominals = ['10gr', '1gr', '1zl', '20gr', '2gr', '2zl', '50gr', '5gr', '5zl', 'wrong']
    if num_classes <= 11:
        return sorted(base_nominals)
    expanded = []
    for nom in base_nominals:
        if nom == 'wrong': expanded.append('wrong')
        else:
            expanded.append(f"{nom}_awers")
            expanded.append(f"{nom}_rewers")
    return sorted(expanded)

def classify_and_count():
    print("\nKLASYFIKACJA I LICZENIE")
    
    if not os.path.exists(MODEL_PATH):
        print("Brak modelu!")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = get_class_names(model.output_shape[-1])
    
    coin_values = {
        "1gr": 0.01, "2gr": 0.02, "5gr": 0.05,
        "10gr": 0.10, "20gr": 0.20, "50gr": 0.50,
        "1zl": 1.00, "2zl": 2.00, "5zl": 5.00,
        "wrong": 0.00
    }

    counts = {k: 0 for k in coin_values.keys()}
    counts["wrong"] = 0
    total_pln = 0.0

    files = sorted(os.listdir(OUTPUT_DIR))
    
    for i, fname in enumerate(files):
        path = os.path.join(OUTPUT_DIR, fname)
        img = cv2.imread(path)
        if img is None: continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_arr = np.expand_dims(rgb, 0)

        preds = model.predict(input_arr, verbose=0)
        score = preds[0]
        idx = np.argmax(score)
        conf = score[idx]

        label = "wrong"
        if conf >= VALID_PROG:
            raw_name = class_names[idx]
            label = raw_name.split('_')[0]
        
        if label in counts:
            counts[label] += 1
            total_pln += coin_values.get(label, 0.0)
        else:
            counts["wrong"] += 1
            
        print(f"Plik {fname} -> {label.ljust(8)} ({conf*100:.1f}%)")

    print("PODSUMOWANIE")
    order = ["1gr", "2gr", "5gr", "10gr", "20gr", "50gr", "1zl", "2zl", "5zl", "wrong"]
    
    for nom in order:
        c = counts.get(nom, 0)
        if c > 0:
            print(f" {nom.ljust(8)} : {c} szt.")
            
    print(f"RAZEM WARTOŚĆ: {total_pln:.2f} PLN")

if __name__ == "__main__":
    num_extracted = extract_coins()
    
    if num_extracted > 0:
        classify_and_count()
    else:
        print("Nie wykryto żadnych monet w wideo")