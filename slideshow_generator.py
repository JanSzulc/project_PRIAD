import argparse
import random
import sys
import os
import glob

try:
    import cv2
    import numpy as np
except ImportError:
    print("Blad: Wymagane biblioteki nie sa zainstalowane.")
    print("Zainstaluj je poleceniem: pip install opencv-python")
    sys.exit(1)

DATASET_BASE = "dataset/test"

COIN_FOLDERS = {
    "1gr": "1gr",
    "2gr": "2gr",
    "5gr": "5gr",
    "10gr": "10gr",
    "20gr": "20gr",
    "50gr": "50gr",
    "1zl": "1zl",
    "2zl": "2zl",
    "5zl": "5zl",
    "wrong": "wrong",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generuje film MP4 z losowo wybranymi monetami przewijanymi od prawej do lewej.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykladowe uzycie:
  %(prog)s -s 1.0 -1gr 5 -2gr 3
  %(prog)s -s 2.0 -1zl 10 -wrong 5
  %(prog)s --speed 0.5 -5gr 4 -10gr 4 -20gr 4

Program wczytuje zdjecia z folderu dataset/test/ i tworzy film
gdzie monety przewijaja sie poziomo od prawej do lewej strony.
Zdjecia sa losowo wybierane z odpowiednich folderow i moga sie powtarzac.
        """
    )

    parser.add_argument(
        "-s", "--speed",
        type=float,
        required=True,
        help="Predkosc przewijania (1.0 = normalna, 2.0 = szybsza, 0.5 = wolniejsza)"
    )

    parser.add_argument("-1gr", type=int, default=0, dest="gr1", metavar="N",
                        help="Liczba monet 1 grosz")
    parser.add_argument("-2gr", type=int, default=0, dest="gr2", metavar="N",
                        help="Liczba monet 2 grosze")
    parser.add_argument("-5gr", type=int, default=0, dest="gr5", metavar="N",
                        help="Liczba monet 5 groszy")
    parser.add_argument("-10gr", type=int, default=0, dest="gr10", metavar="N",
                        help="Liczba monet 10 groszy")
    parser.add_argument("-20gr", type=int, default=0, dest="gr20", metavar="N",
                        help="Liczba monet 20 groszy")
    parser.add_argument("-50gr", type=int, default=0, dest="gr50", metavar="N",
                        help="Liczba monet 50 groszy")
    parser.add_argument("-1zl", type=int, default=0, dest="zl1", metavar="N",
                        help="Liczba monet 1 zloty")
    parser.add_argument("-2zl", type=int, default=0, dest="zl2", metavar="N",
                        help="Liczba monet 2 zlote")
    parser.add_argument("-5zl", type=int, default=0, dest="zl5", metavar="N",
                        help="Liczba monet 5 zlotych")
    parser.add_argument("-wrong", type=int, default=0, dest="wrong", metavar="N",
                        help="Liczba niepoprawnych monet")

    return parser.parse_args()


def get_images_from_folder(folder_name: str, count: int) -> list:
    folder_path = os.path.join(DATASET_BASE, folder_name)

    if not os.path.isdir(folder_path):
        print(f"Ostrzezenie: Folder nie istnieje: {folder_path}")
        return []

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    all_images = []
    for ext in extensions:
        all_images.extend(glob.glob(os.path.join(folder_path, ext)))

    if not all_images:
        print(f"Ostrzezenie: Brak zdjec w folderze: {folder_path}")
        return []

    return random.choices(all_images, k=count)


def load_and_resize_image(image_path: str, target_height: int) -> np.ndarray:
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Nie mozna wczytac zdjecia: {image_path}")

    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)

    resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4)
    return resized


def generate_slideshow(image_paths: list, speed: float):
    frame_size = 720
    output_path = "slideshow.mp4"

    print(f"Wczytywanie {len(image_paths)} zdjec...")

    images = []
    for path in image_paths:
        try:
            img = load_and_resize_image(path, frame_size)
            images.append(img)
        except ValueError as e:
            print(f"  Blad: {e}")

    if len(images) < 1:
        print("Blad: Brak poprawnych zdjec!")
        sys.exit(1)

    print(f"Wczytano {len(images)} zdjec.")

    random.shuffle(images)

    widths = [img.shape[1] for img in images]
    positions = []
    current_pos = 0
    for w in widths:
        positions.append(current_pos)
        current_pos += w
    total_width = current_pos

    fps = 30

    base_pixels_per_second = 200
    pixels_per_frame = max(1, int((base_pixels_per_second * speed) / fps))

    total_scroll = total_width - frame_size
    if total_scroll <= 0:
        total_scroll = 1

    total_frames = max(1, total_scroll // pixels_per_frame)
    duration = total_frames / fps

    print(f"Generowanie slideshow (predkosc: {speed}x, czas: {duration:.1f}s)...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_size, frame_size))

    if not out.isOpened():
        print(f"Blad: Nie mozna utworzyc pliku wideo: {output_path}")
        sys.exit(1)

    for frame_idx in range(total_frames):
        x_offset = frame_idx * pixels_per_frame

        if x_offset + frame_size > total_width:
            x_offset = total_width - frame_size

        frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)

        for i, img in enumerate(images):
            img_start = positions[i]
            img_end = img_start + widths[i]

            if img_end <= x_offset or img_start >= x_offset + frame_size:
                continue

            src_start = max(0, x_offset - img_start)
            src_end = min(widths[i], x_offset + frame_size - img_start)

            dst_start = max(0, img_start - x_offset)
            dst_end = dst_start + (src_end - src_start)

            frame[:, dst_start:dst_end] = img[:, src_start:src_end]

        out.write(frame)

    out.release()
    print(f"Zapisano: {output_path}")
    print(f"  Rozmiar: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")


def main():
    args = parse_arguments()

    if args.speed <= 0:
        print("Blad: Predkosc musi byc wieksza od 0!")
        sys.exit(1)

    coin_counts = {
        "1gr": args.gr1,
        "2gr": args.gr2,
        "5gr": args.gr5,
        "10gr": args.gr10,
        "20gr": args.gr20,
        "50gr": args.gr50,
        "1zl": args.zl1,
        "2zl": args.zl2,
        "5zl": args.zl5,
        "wrong": args.wrong,
    }

    all_images = []
    for coin_type, count in coin_counts.items():
        if count > 0:
            folder = COIN_FOLDERS[coin_type]
            images = get_images_from_folder(folder, count)
            all_images.extend(images)
            if images:
                print(f"  {coin_type}: {len(images)} zdjec")

    if not all_images:
        print("Blad: Nie wybrano zadnych monet!")
        print("Uzyj argumentow takich jak -1gr 5 -2gr 3 aby wybrac monety.")
        sys.exit(1)

    generate_slideshow(all_images, args.speed)


if __name__ == "__main__":
    main()