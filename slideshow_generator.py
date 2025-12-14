import argparse
import random
import sys
import os
import glob

try:
    import cv2
    import numpy as np
except ImportError:
    print("Błąd: Wymagane biblioteki nie są zainstalowane.")
    print("Zainstaluj je poleceniem: pip install opencv-python")
    sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generuje plik MP4 z losowo przewijanymi zdjęciami."
    )

    parser.add_argument(
        "--images", "-i",
        nargs="+",
        required=True,
        help="Ścieżki do zdjęć (minimum 2)"
    )

    parser.add_argument(
        "--speed", "-s",
        type=float,
        required=True,
        help="Prędkość przewijania (1.0 = normalna, 2.0 = szybsza)"
    )

    return parser.parse_args()


def load_and_resize_image(image_path: str, target_width: int, target_height: int) -> np.ndarray:
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Nie można wczytać zdjęcia: {image_path}")

    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def create_transition_frame(img1: np.ndarray, img2: np.ndarray, progress: float) -> np.ndarray:
    return cv2.addWeighted(img1, 1 - progress, img2, progress, 0)


def generate_slideshow(image_paths: list, speed: float):
    width, height = 1280, 720
    output_path = "slideshow.mp4"

    print(f"Wczytywanie {len(image_paths)} zdjęć...")

    images = []
    for path in image_paths:
        try:
            img = load_and_resize_image(path, width, height)
            images.append(img)
            print(f"  ✓ {path}")
        except ValueError as e:
            print(f"  ✗ {e}")

    if len(images) < 2:
        print("Błąd: Potrzeba minimum 2 poprawnych zdjęć!")
        sys.exit(1)

    print(f"\nWczytano {len(images)} zdjęć.")

    fps = 30

    base_display_time = 3.0
    display_time = base_display_time / speed
    transition_time = 0.5 / speed

    frames_per_image = int(display_time * fps)
    transition_frames = int(transition_time * fps)

    duration = len(images) * (display_time + transition_time)
    total_frames = int(duration * fps)

    print(f"\nGenerowanie slideshow (prędkość: {speed}x, czas: {duration:.1f}s)...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Błąd: Nie można utworzyć pliku wideo: {output_path}")
        sys.exit(1)

    random.shuffle(images)

    frame_count = 0
    image_index = 0

    while frame_count < total_frames:
        current_img = images[image_index % len(images)]
        next_img = images[(image_index + 1) % len(images)]

        display_frames = min(frames_per_image, total_frames - frame_count)
        for _ in range(display_frames):
            out.write(current_img)
            frame_count += 1
            if frame_count >= total_frames:
                break

        if frame_count >= total_frames:
            break

        trans_frames = min(transition_frames, total_frames - frame_count)
        for i in range(trans_frames):
            progress = (i + 1) / transition_frames
            frame = create_transition_frame(current_img, next_img, progress)
            out.write(frame)
            frame_count += 1
            if frame_count >= total_frames:
                break

        image_index += 1

        if image_index % len(images) == 0:
            random.shuffle(images)

        progress_pct = (frame_count / total_frames) * 100
        bar_length = 40
        filled = int(bar_length * frame_count / total_frames)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r  [{bar}] {progress_pct:.1f}%", end="", flush=True)

    out.release()
    print(f"\n\n✓ Zapisano: {output_path}")
    print(f"  Rozmiar: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")


def main():
    args = parse_arguments()

    if args.speed <= 0:
        print("Błąd: Prędkość musi być większa od 0!")
        sys.exit(1)

    valid_paths = []
    all_potential_paths = []

    for item in args.images:
        found_paths = glob.glob(item)
        if found_paths:
            all_potential_paths.extend(found_paths)
        else:
            all_potential_paths.append(item)

    unique_paths = set(all_potential_paths)

    for path in sorted(list(unique_paths)):
        if os.path.isfile(path):
            valid_paths.append(path)
        else:
            if not glob.escape(path) == path:
                continue
            print(f"Ostrzeżenie: Plik nie istnieje: {path}")

    if len(valid_paths) < 2:
        print("Błąd: Podaj minimum 2 istniejące pliki zdjęć!")
        sys.exit(1)

    generate_slideshow(valid_paths, args.speed)


if __name__ == "__main__":
    main()