import cv2
import numpy as np
import os
import random
import tensorflow as tf
import pathlib
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

TRAIN_DATA_DIR = 'dataset/train/'
TEST_DATA_DIR = 'dataset/test/'
OUTPUT_TRAIN_DATA_DIR = 'prepared_data/'
MODEL_NAME = 'coin_classifier_neural_network_model.keras'

TARGET_IMAGE_SIZE = (512, 512)
AUGMENT_ROTATION = (0, 360)
AUGMENT_COUNT = 4 
EPOCHS = 50
VALID_PROG = 0.9



def prepare_single_image(img_path):
    img = cv2.imread(img_path)
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
            square_canvas[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = coin_on_black[src_y1:src_y2, src_x1:src_x2]
        resized = cv2.resize(square_canvas, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        #show_step("Przygotowany Obraz", resized)
        return resized
    else:
        return None

def prepare_image_to_train():
    for class_names in os.listdir(TRAIN_DATA_DIR):
        for img_type in ['avers', 'revers']:
            for i in os.listdir(os.path.join(TRAIN_DATA_DIR, class_names, img_type)):
                img_path = os.path.join(TRAIN_DATA_DIR, class_names, img_type, i)
                resized = prepare_single_image(img_path)
                if resized is not None:
                    base_filename = os.path.splitext(i)[0]
                    output_path = os.path.join(OUTPUT_TRAIN_DATA_DIR, f"{class_names}_{img_type}")
                    os.makedirs(output_path, exist_ok=True)
                    output_fixed_path = os.path.join(output_path, i)
                    cv2.imwrite(output_fixed_path, resized)
                    height, width = resized.shape[:2]
                    center = (width // 2, height // 2)
                    for j in range(AUGMENT_COUNT):
                        angle = random.uniform(AUGMENT_ROTATION[0], AUGMENT_ROTATION[1])
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        augmented = cv2.warpAffine(resized, M, (width, height), borderValue=(0,0,0))
                        hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV)
                        h, s, v = cv2.split(hsv)
                        brightness_factor = random.uniform(0.7, 1.3)
                        v = np.clip(v.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
                        final_hsv = cv2.merge((h, s, v))
                        augmented = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
                        augmented_name = f"{base_filename}_aug_{j}.jpg"
                        cv2.imwrite(os.path.join(output_path, augmented_name), augmented)
                else:
                    print(f"Nie znaleziono monety: {img_path}")

def show_step(title, image):
    scale_ratio = 800 / image.shape[1]
    width = int(image.shape[1] * scale_ratio)
    height = int(image.shape[0] * scale_ratio)
    preview = cv2.resize(image, (width, height))
    
    cv2.imshow(title, preview)
    key = cv2.waitKey(0) & 0xFF
    if(key == ord('q')):
        cv2.destroyAllWindows()
        input()

def TrainModel():
    DATA_DIR = pathlib.Path(OUTPUT_TRAIN_DATA_DIR)
    BATCH_SIZE = 16 
    print("Wczytywanie zbioru treningowego...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=TARGET_IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
    )

    print("\nWczytywanie zbioru walidacyjnego (testowego)...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=TARGET_IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
    )

    class_names = train_ds.class_names
    print(f"\nWykryte klasy ({len(class_names)}): {class_names}")
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(512, 512, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.6),
    layers.Dense(len(class_names), activation='softmax')
    ])


    model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1
    )

    print("Rozpoczynam trening...")
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping]
    )
    return model, history

def ShowResults(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    actual_epochs = len(acc) 
    epochs_range = range(actual_epochs)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Skuteczność (Accuracy) - {actual_epochs} epok')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Strata (Loss) - {actual_epochs} epok')
    
    plt.show()

def determine_label_from_path(file_path):
    path_parts = os.path.normpath(file_path).split(os.sep)
    if 'wrong' in path_parts:
        return 'Anomaly'
    try:
        nominal = path_parts[-3]
        return nominal
    except IndexError:
        return "Unknown"

def run_test_evaluation(model, test_dir, train_output_dir):
    print("TESTOWANIE")
    raw_class_names = sorted([d for d in os.listdir(train_output_dir) if os.path.isdir(os.path.join(train_output_dir, d))])
    unique_nominals = sorted(list(set([name.split('_')[0] for name in raw_class_names])))
    all_labels_for_plot = unique_nominals + ["Anomaly"]

    y_true = []
    y_pred = []
    
    total_files = 0
    processed_files = 0
    
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_files += 1
                full_path = os.path.join(root, file)
                true_label = determine_label_from_path(full_path)
                if true_label == "Unknown": continue
                img_processed = prepare_single_image(full_path)
                
                if img_processed is None:
                    print(f"BŁĄD: Nie znaleziono monety na zdjęciu: {file}")
                    continue
                img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
                img_array = np.expand_dims(img_processed, 0)
                predictions = model.predict(img_array, verbose=0)
                score = predictions[0] 
                max_score = np.max(score)
                predicted_index = np.argmax(score)
                if max_score < VALID_PROG:
                    predicted_label = "Anomaly"
                else:
                    raw_label = raw_class_names[predicted_index]
                    predicted_label = raw_label.split('_')[0]
                
                y_true.append(true_label)
                y_pred.append(predicted_label)
                processed_files += 1
                
                if processed_files % 10 == 0:
                    print(f"Przetworzono {processed_files} zdjęć...", end='\r')

    print(f"\nZakończono. Przetworzono poprawnie {processed_files}/{total_files} zdjęć.")
    cm = confusion_matrix(y_true, y_pred, labels=all_labels_for_plot)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels_for_plot, yticklabels=all_labels_for_plot)
    plt.xlabel('Przewidziane przez Model')
    plt.ylabel('Prawdziwe (Folder)')
    plt.title(f'Macierz Pomyłek (Próg pewności: {VALID_PROG})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    print("\nRaport Klasyfikacji:")
    print(classification_report(y_true, y_pred, labels=all_labels_for_plot, zero_division=0))


prepare_image_to_train()
trained_model, trained_history = TrainModel()
trained_model.save(MODEL_NAME)
print(f"\nModel został zapisany do pliku: {MODEL_NAME}")
ShowResults(trained_history)

if os.path.exists(MODEL_NAME):
    print(f"Wczytywanie modelu: {MODEL_NAME}...")
    loaded_model = tf.keras.models.load_model(MODEL_NAME)
    run_test_evaluation(loaded_model, TEST_DATA_DIR, OUTPUT_TRAIN_DATA_DIR)
else:
    print("Nie znaleziono zapisanego modelu! Uruchom trening najpierw.")