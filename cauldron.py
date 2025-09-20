import numpy as np
from stl import mesh
from PIL import Image, ImageDraw
import math
import os

def stl_to_bmp_slices(stl_file_path, output_dir, num_slices=100, 
                     wall_thickness=1, infill_percent=20, infill_type="grid",
                     image_size=(240, 320)):
    """
    Конвертирует STL-файл в серию BMP-изображений (срезы)
    
    Параметры:
    stl_file_path: путь к STL-файлу
    output_dir: директория для сохранения BMP-файлов
    num_slices: количество срезов
    wall_thickness: толщина стенки в пикселях
    infill_percent: процент заполнения (0-100)
    infill_type: тип заполнения ("grid", "lines", "hex", "triangles")
    image_size: размер выходных изображений (ширина, высота)
    """
    
    # Создаем директорию для выходных файлов, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Загрузка STL-файла
    stl_mesh = mesh.Mesh.from_file(stl_file_path)
    
    # Определение границ модели
    min_z, max_z = np.min(stl_mesh.vectors[:, :, 2]), np.max(stl_mesh.vectors[:, :, 2])
    z_height = max_z - min_z
    slice_thickness = z_height / num_slices

    # Создание изображений для каждого среза
    for slice_index in range(num_slices):
        current_z = min_z + (slice_index * slice_thickness)
        
        # Создаем изображение с черным фоном
        img = Image.new('1', image_size, 0)
        draw = ImageDraw.Draw(img)
        
        # Получаем все сегменты для текущего среза
        segments = []
        for triangle in stl_mesh.vectors:
            z_coords = triangle[:, 2]
            if np.max(z_coords) < current_z or np.min(z_coords) > current_z:
                continue  # Пропустить треугольники вне среза
            
            # Найти пересечение треугольника с плоскостью Z
            intersections = []
            for i in range(3):
                p1, p2 = triangle[i], triangle[(i + 1) % 3]
                if (p1[2] >= current_z and p2[2] < current_z) or (p1[2] < current_z and p2[2] >= current_z):
                    t = (current_z - p1[2]) / (p2[2] - p1[2])
                    x = p1[0] + t * (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    intersections.append((x, y))
            
            if len(intersections) == 2:
                segments.append((intersections[0], intersections[1]))
        
        # Если нет сегментов, пропускаем этот срез
        if not segments:
            continue
        
        # Масштабирование и смещение для перевода в пиксели
        all_points = [point for segment in segments for point in segment]
        xs, ys = zip(*all_points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Коэффициенты масштабирования
        scale_x = (image_size[0] * 0.9) / (max_x - min_x) if max_x != min_x else 1
        scale_y = (image_size[1] * 0.9) / (max_y - min_y) if max_y != min_y else 1
        scale = min(scale_x, scale_y)
        
        # Смещение для центрирования
        offset_x = (image_size[0] - (max_x - min_x) * scale) / 2 - min_x * scale
        offset_y = (image_size[1] - (max_y - min_y) * scale) / 2 - min_y * scale
        
        # Функция для преобразования координат в пиксели
        def to_pixel(x, y):
            px = int(offset_x + x * scale)
            py = int(offset_y + y * scale)
            return (px, py)
        
        # Рисуем контур с заданной толщиной стенки
        for seg in segments:
            p1, p2 = seg
            draw.line([to_pixel(*p1), to_pixel(*p2)], fill=1, width=wall_thickness)
        
        # Если нужно заполнение, создаем маску и добавляем заполнение
        if infill_percent > 0:
            # Создаем маску для определения внутренней области
            mask_img = Image.new('1', image_size, 0)
            mask_draw = ImageDraw.Draw(mask_img)
            
            # Рисуем контур на маске
            for seg in segments:
                p1, p2 = seg
                mask_draw.line([to_pixel(*p1), to_pixel(*p2)], fill=1, width=1)
            
            # Заполняем контур чтобы получить маску
            # Находим точку внутри контура (приблизительный центр)
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            center_px = to_pixel(center_x, center_y)
            
            # Заполняем маску
            ImageDraw.floodfill(mask_img, center_px, 1)
            
            # Добавляем заполнение в соответствии с типом и процентом
            add_infill(draw, mask_img, infill_type, infill_percent, image_size)
        
        # Сохранение среза
        img.save(os.path.join(output_dir, f"slice_{slice_index:04d}.bmp"))
        print(f"Создан срез {slice_index+1}/{num_slices}")

def add_infill(draw, mask, infill_type, infill_percent, image_size):
    """
    Добавляет заполнение к изображению на основе маски
    """
    width, height = image_size
    
    # Определяем шаг заполнения на основе процента
    # Максимальный шаг (при 0%) - бесконечность, минимальный (при 100%) - 1 пиксель
    max_step = 20
    step = max(1, int(max_step * (100 - infill_percent) / 100))
    
    if infill_type == "lines":
        # Линейное заполнение под углом 45 градусов
        for i in range(-height, width, step):
            start = (i, 0)
            end = (i + height, height)
            draw_line_with_mask(draw, mask, start, end)
    
    elif infill_type == "grid":
        # Сетка (два направления линий)
        for i in range(-height, width, step):
            start = (i, 0)
            end = (i + height, height)
            draw_line_with_mask(draw, mask, start, end)
        
        for i in range(0, width + height, step):
            start = (i, 0)
            end = (i - height, height)
            draw_line_with_mask(draw, mask, start, end)
    
    elif infill_type == "hex":
        # Шестиугольное заполнение
        for i in range(-height, width, step):
            start = (i, 0)
            end = (i + height, height)
            draw_line_with_mask(draw, mask, start, end)
        
        for i in range(-height, width, step * 2):
            start = (i + step//2, 0)
            end = (i + step//2 + height, height)
            draw_line_with_mask(draw, mask, start, end)
        
        for i in range(0, width + height, step):
            start = (i, 0)
            end = (i - height, height)
            draw_line_with_mask(draw, mask, start, end)
    
    elif infill_type == "triangles":
        # Треугольное заполнение
        for i in range(-height, width, step):
            start = (i, 0)
            end = (i + height, height)
            draw_line_with_mask(draw, mask, start, end)
        
        for i in range(0, width + height, step):
            start = (i, 0)
            end = (i - height, height)
            draw_line_with_mask(draw, mask, start, end)
        
        for i in range(0, width, step):
            draw_line_with_mask(draw, mask, (i, 0), (i, height))

def draw_line_with_mask(draw, mask, start, end):
    """
    Рисует линию только в тех местах, где маска имеет значение 1
    """
    width, height = mask.size
    x0, y0 = start
    x1, y1 = end
    
    # Алгоритм Брезенхема для рисования линии с проверкой маски
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        # Проверяем, находится ли точка в пределах изображения и маски
        if 0 <= x0 < width and 0 <= y0 < height:
            if mask.getpixel((x0, y0)) == 1:
                draw.point((x0, y0), fill=1)
        
        if x0 == x1 and y0 == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def main():
    """
    Основная функция для взаимодействия с пользователем
    """
    print("Конвертер STL в BMP срезы")
    print("========================")
    
    # Запрос пути к STL-файлу
    stl_path = input("Введите путь к STL-файлу: ").strip('"')
    
    # Запрос параметров
    output_dir = input("Введите директорию для сохранения BMP файлов (по умолчанию: output): ").strip('"')
    if not output_dir:
        output_dir = "output"
    
    num_slices = int(input("Введите количество срезов (по умолчанию: 100): ") or "100")
    wall_thickness = int(input("Введите толщину стенки в пикселях (по умолчанию: 1): ") or "1")
    infill_percent = int(input("Введите процент заполнения (0-100, по умолчанию: 20): ") or "20")
    
    print("Доступные типы заполнения: grid, lines, hex, triangles")
    infill_type = input("Введите тип заполнения (по умолчанию: grid): ") or "grid"
    
    # Проверка корректности введенных данных
    if not os.path.isfile(stl_path):
        print("Ошибка: STL-файл не найден!")
        return
    
    if infill_percent < 0 or infill_percent > 100:
        print("Ошибка: процент заполнения должен быть между 0 и 100!")
        return
    
    if infill_type not in ["grid", "lines", "hex", "triangles"]:
        print("Ошибка: неверный тип заполнения!")
        return
    
    # Запуск конвертации
    print("\nНачинаю конвертацию...")
    stl_to_bmp_slices(
        stl_path, 
        output_dir, 
        num_slices=num_slices,
        wall_thickness=wall_thickness,
        infill_percent=infill_percent,
        infill_type=infill_type
    )
    
    print(f"\nКонвертация завершена! Файлы сохранены в директории '{output_dir}'")

if __name__ == "__main__":
    main()
