import numpy as np
from stl import mesh
from PIL import Image, ImageDraw
import os
import math
from collections import defaultdict

def stl_to_bmp_slices(stl_file_path, output_dir, layer_height=0.1, 
                     wall_thickness=1, infill_percent=20, infill_type="grid",
                     image_size=(320, 240), solid_layers=5):
    """
    Конвертирует STL-файл в серию BMP-изображений (срезы)
    
    Параметры:
    stl_file_path: путь к STL-файлу
    output_dir: директория для сохранения BMP-файлов
    layer_height: толщина слоя в единицах модели
    wall_thickness: толщина стенки в пикселях
    infill_percent: процент заполнения (0-100)
    infill_type: тип заполнения
    image_size: размер выходных изображений (ширина, высота)
    solid_layers: количество слоев с полной заливкой в начале и конце
    """
    
    # Создаем директорию для выходных файлов, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Загрузка STL-файла
    stl_mesh = mesh.Mesh.from_file(stl_file_path)
    
    # Определение границ модели по всем осям
    min_coords = np.min(stl_mesh.vectors, axis=(0, 1))
    max_coords = np.max(stl_mesh.vectors, axis=(0, 1))
    
    min_x, min_y, min_z = min_coords
    max_x, max_y, max_z = max_coords
    
    # Вычисляем высоту модели по оси Z
    z_height = max_z - min_z
    
    # Вычисляем количество слоев на основе высоты модели и толщины слоя
    num_slices = math.ceil(z_height / layer_height)
    
    # Вычисляем фактическую толщину слоя для равномерного распределения
    actual_layer_height = z_height / num_slices
    
    # Вычисляем центр и размер модели в XY-плоскости
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    size_x = max_x - min_x
    size_y = max_y - min_y
    
    # Определяем масштаб для сохранения пропорций
    scale = min((image_size[0] * 0.9) / size_x, 
                (image_size[1] * 0.9) / size_y)
    
    # Смещение для центрирования
    offset_x = image_size[0] / 2 - center_x * scale
    offset_y = image_size[1] / 2 - center_y * scale

    # Функция для преобразования координат в пиксели
    def to_pixel(x, y):
        px = int(offset_x + x * scale)
        py = int(offset_y - y * scale)  # Инвертируем Y для правильной ориентации
        return (px, py)

    print(f"Высота модели: {z_height:.2f} единиц")
    print(f"Толщина слоя: {layer_height:.2f} единиц")
    print(f"Количество слоев: {num_slices}")
    print(f"Фактическая толщина слоя: {actual_layer_height:.4f} единиц")
    
    # Создаем словарь для хранения горизонтальных плоскостей
    horizontal_planes = defaultdict(list)
    
    # Идентифицируем горизонтальные плоскости (крышки и дно)
    for triangle in stl_mesh.vectors:
        # Проверяем, является ли треугольник горизонтальным
        z_values = triangle[:, 2]
        if np.allclose(z_values, z_values[0], atol=1e-5):
            # Это горизонтальная плоскость
            z_level = z_values[0]
            horizontal_planes[z_level].append(triangle)
    
    # Создание изображений для каждого среза
    for slice_index in range(num_slices):
        current_z = min_z + (slice_index * actual_layer_height)
        
        # Создаем изображение с черным фоном
        img = Image.new('1', image_size, 0)
        draw = ImageDraw.Draw(img)
        
        # Создаем маску для заполнения
        mask_img = Image.new('1', image_size, 0)
        mask_draw = ImageDraw.Draw(mask_img)
        
        # Обрабатываем каждый треугольник
        for triangle in stl_mesh.vectors:
            # Проверяем, пересекает ли треугольник текущую плоскость Z
            z_values = triangle[:, 2]
            
            # Особый случай: горизонтальная плоскость точно на уровне среза
            if np.any(np.isclose(z_values, current_z, atol=1e-5)):
                # Это горизонтальная плоскость на уровне среза
                # Проецируем весь треугольник на плоскость
                vertices_2d = triangle[:, :2]
                
                # Преобразуем вершины в пиксели
                pixels = [to_pixel(v[0], v[1]) for v in vertices_2d]
                
                # Рисуем заполненный треугольник
                draw.polygon(pixels, fill=1)
                mask_draw.polygon(pixels, fill=1)
                continue
            
            # Обычный случай: треугольник пересекает плоскость
            if np.max(z_values) < current_z or np.min(z_values) > current_z:
                continue
            
            # Находим точки пересечения треугольника с плоскостью Z
            intersections = []
            for i in range(3):
                p1 = triangle[i]
                p2 = triangle[(i + 1) % 3]
                
                # Проверяем, пересекает ли ребро плоскость Z
                if (p1[2] >= current_z and p2[2] < current_z) or (p1[2] < current_z and p2[2] >= current_z):
                    # Вычисляем точку пересечения
                    t = (current_z - p1[2]) / (p2[2] - p1[2])
                    x = p1[0] + t * (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    intersections.append((x, y))
            
            # Если найдены две точки пересечения, рисуем отрезок
            if len(intersections) == 2:
                p1, p2 = intersections
                pixel1 = to_pixel(p1[0], p1[1])
                pixel2 = to_pixel(p2[0], p2[1])
                
                # Рисуем контур на основном изображении и маске
                draw.line([pixel1, pixel2], fill=1, width=wall_thickness)
                mask_draw.line([pixel1, pixel2], fill=1, width=1)
        
        # Если есть контур, заполняем его
        if infill_percent > 0:
            # Находим точку внутри контура для заливки
            # Используем центр ограничивающего прямоугольника маски
            bbox = mask_img.getbbox()
            if bbox:
                center_x_mask = (bbox[0] + bbox[2]) // 2
                center_y_mask = (bbox[1] + bbox[3]) // 2
                
                # Заполняем маску
                ImageDraw.floodfill(mask_img, (center_x_mask, center_y_mask), 1)
                
                # Определяем, нужно ли использовать полную заливку
                current_infill_type = infill_type
                current_infill_percent = infill_percent
                
                # Для первых и последних solid_layers слоев используем полную заливку
                if slice_index < solid_layers or slice_index >= num_slices - solid_layers:
                    current_infill_type = "full"
                    current_infill_percent = 100
                    print(f"Слой {slice_index+1}/{num_slices} (полная заливка)")
                else:
                    print(f"Слой {slice_index+1}/{num_slices} ({infill_type} заполнение)")
                
                # Добавляем заполнение
                add_infill(draw, mask_img, current_infill_type, current_infill_percent, image_size)
        
        # Сохранение среза
        img.save(os.path.join(output_dir, f"slice_{slice_index:04d}.bmp"))
    
    print(f"\nКонвертация завершена! Создано {num_slices} слоев.")

def add_infill(draw, mask, infill_type, infill_percent, image_size):
    """
    Добавляет заполнение к изображению на основе маски
    """
    width, height = image_size
    
    # Определяем шаг заполнения на основе процента
    max_step = 20
    step = max(1, int(max_step * (100 - infill_percent) / 100))
    
    if infill_type == "lines" or infill_type == "grid45":
        # Линейное заполнение под углом 45 градусов
        for i in range(-height, width, step):
            draw_line_with_mask(draw, mask, (i, 0), (i + height, height))
    
    elif infill_type == "grid":
        # Сетка (два направления линий)
        for i in range(-height, width, step):
            draw_line_with_mask(draw, mask, (i, 0), (i + height, height))
        
        for i in range(0, width + height, step):
            draw_line_with_mask(draw, mask, (i, 0), (i - height, height))
    
    elif infill_type == "hex":
        # Шестиугольное заполнение
        for i in range(-height, width, step):
            draw_line_with_mask(draw, mask, (i, 0), (i + height, height))
        
        for i in range(-height, width, step * 2):
            draw_line_with_mask(draw, mask, (i + step//2, 0), (i + step//2 + height, height))
        
        for i in range(0, width + height, step):
            draw_line_with_mask(draw, mask, (i, 0), (i - height, height))
    
    elif infill_type == "triangles":
        # Треугольное заполнение
        for i in range(-height, width, step):
            draw_line_with_mask(draw, mask, (i, 0), (i + height, height))
        
        for i in range(0, width + height, step):
            draw_line_with_mask(draw, mask, (i, 0), (i - height, height))
        
        for i in range(0, width, step):
            draw_line_with_mask(draw, mask, (i, 0), (i, height))
    
    elif infill_type == "octagons":
        # Восьмиугольники
        spacing = max(3, int(10 * (100 - infill_percent) / 100))
        for i in range(0, height, spacing):
            for j in range(0, width, spacing):
                if i + spacing < height and j + spacing < width:
                    # Рисуем восьмиугольник
                    center_x, center_y = j + spacing//2, i + spacing//2
                    size = spacing // 2
                    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                        x = int(center_x + size * np.cos(angle))
                        y = int(center_y + size * np.sin(angle))
                        if 0 <= x < width and 0 <= y < height and mask.getpixel((x, y)) == 1:
                            draw.point((x, y), fill=1)
    
    elif infill_type == "squares":
        # Квадраты
        spacing = max(2, int(10 * (100 - infill_percent) / 100))
        for i in range(0, height, spacing):
            for j in range(0, width, spacing):
                if i + spacing < height and j + spacing < width and mask.getpixel((j, i)) == 1:
                    draw.rectangle([j, i, j+spacing, i+spacing], fill=1)
    
    elif infill_type == "octet":
        # Октет
        spacing = max(2, int(10 * (100 - infill_percent) / 100))
        for i in range(0, height, spacing):
            for j in range(0, width, spacing):
                if i + spacing < height and j + spacing < width:
                    # Рисуем октетный паттерн
                    center_x, center_y = j + spacing//2, i + spacing//2
                    radius = spacing // 3
                    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                        x = int(center_x + radius * np.cos(angle))
                        y = int(center_y + radius * np.sin(angle))
                        if 0 <= x < width and 0 <= y < height and mask.getpixel((x, y)) == 1:
                            draw.point((x, y), fill=1)
    
    elif infill_type == "gyroid":
        # Гироидный паттерн
        spacing = max(2, int(10 * (100 - infill_percent) / 100))
        for i in range(0, height, spacing):
            for j in range(0, width, spacing):
                if mask.getpixel((j, i)) == 1:
                    # Упрощенная версия гироидного паттерна
                    if (i // spacing + j // spacing) % 2 == 0:
                        draw.point((j, i), fill=1)
    
    elif infill_type == "full":
        # Полное заполнение
        for i in range(height):
            for j in range(width):
                if mask.getpixel((j, i)) == 1:
                    draw.point((j, i), fill=1)

def draw_line_with_mask(draw, mask, start, end):
    """
    Рисует линию только в тех местах, где маска имеет значение 1
    """
    width, height = mask.size
    x0, y0 = start
    x1, y1 = end
    
    # Алгоритм Брезенхема для рисования линии
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

def display_menu(current_settings):
    """
    Отображает меню с текущими настройками
    """
    print("\n" + "="*50)
    print("      КОНВЕРТЕР STL В BMP СРЕЗЫ")
    print("="*50)
    print(f"1. STL файл: {current_settings['stl_path']}")
    print(f"2. Выходная директория: {current_settings['output_dir']}")
    print(f"3. Толщина слоя: {current_settings['layer_height']}")
    print(f"4. Толщина стенки: {current_settings['wall_thickness']}")
    print(f"5. Процент заполнения: {current_settings['infill_percent']}%")
    print(f"6. Тип заполнения: {current_settings['infill_type']}")
    print(f"7. Разрешение изображения: {current_settings['image_size'][0]}x{current_settings['image_size'][1]}")
    print(f"8. Слои с полной заливкой: {current_settings['solid_layers']}")
    print("9. Начать конвертацию")
    print("10. Выход")
    print("="*50)

def main():
    """
    Основная функция с текстовым меню
    """
    # Настройки по умолчанию
    settings = {
        'stl_path': '',
        'output_dir': 'output',
        'layer_height': 0.1,  # Толщина слоя по умолчанию
        'wall_thickness': 1,
        'infill_percent': 20,
        'infill_type': 'grid',
        'image_size': (320, 240),  # Ширина, высота
        'solid_layers': 5  # Количество слоев с полной заливкой
    }
    
    # Доступные типы заполнения
    infill_types = [
        "full", "grid", "grid45", "hex", "triangles", 
        "octagons", "squares", "octet", "gyroid"
    ]
    
    while True:
        display_menu(settings)
        choice = input("Выберите пункт меню: ")
        
        if choice == '1':
            path = input("Введите путь к STL-файлу: ").strip('"')
            if os.path.isfile(path) and path.lower().endswith('.stl'):
                settings['stl_path'] = path
                print("Файл успешно выбран!")
            else:
                print("Ошибка: файл не существует или не является STL файлом!")
        
        elif choice == '2':
            dir_path = input("Введите путь к выходной директории: ").strip('"')
            if dir_path:
                settings['output_dir'] = dir_path
                print("Выходная директория установлена!")
            else:
                print("Ошибка: путь не может быть пустым!")
        
        elif choice == '3':
            try:
                height = float(input("Введите толщину слоя: ") or "0.1")
                if height > 0:
                    settings['layer_height'] = height
                    print("Толщина слоя установлена!")
                else:
                    print("Ошибка: толщина слоя должна быть положительной!")
            except ValueError:
                print("Ошибка: введите число!")
        
        elif choice == '4':
            try:
                thickness = int(input("Введите толщину стенки в пикселях: ") or "1")
                if thickness > 0:
                    settings['wall_thickness'] = thickness
                    print("Толщина стенки установлена!")
                else:
                    print("Ошибка: толщина должна быть положительной!")
            except ValueError:
                print("Ошибка: введите целое число!")
        
        elif choice == '5':
            try:
                percent = int(input("Введите процент заполнения (0-100): ") or "20")
                if 0 <= percent <= 100:
                    settings['infill_percent'] = percent
                    print("Процент заполнения установлен!")
                else:
                    print("Ошибка: процент должен быть между 0 и 100!")
            except ValueError:
                print("Ошибка: введите число!")
        
        elif choice == '6':
            print("Доступные типы заполнения:")
            for i, t in enumerate(infill_types, 1):
                print(f"{i}. {t}")
            
            try:
                type_choice = int(input("Выберите тип заполнения: "))
                if 1 <= type_choice <= len(infill_types):
                    settings['infill_type'] = infill_types[type_choice-1]
                    print("Тип заполнения установлен!")
                else:
                    print("Ошибка: неверный выбор!")
            except ValueError:
                print("Ошибка: введите число!")
        
        elif choice == '7':
            try:
                width = int(input("Введите ширину изображения: ") or "320")
                height = int(input("Введите высоту изображения: ") or "240")
                if width > 0 and height > 0:
                    settings['image_size'] = (width, height)
                    print("Разрешение установлено!")
                else:
                    print("Ошибка: разрешение должно быть положительным!")
            except ValueError:
                print("Ошибка: введите числа!")
        
        elif choice == '8':
            try:
                solid_layers = int(input("Введите количество слоев с полной заливкой: ") or "5")
                if solid_layers >= 0:
                    settings['solid_layers'] = solid_layers
                    print("Количество слоев с полной заливкой установлено!")
                else:
                    print("Ошибка: количество слоев должно быть неотрицательным!")
            except ValueError:
                print("Ошибка: введите целое число!")
        
        elif choice == '9':
            if not settings['stl_path']:
                print("Ошибка: сначала выберите STL файл!")
                continue
            
            print("\nНачинаю конвертацию...")
            stl_to_bmp_slices(
                settings['stl_path'], 
                settings['output_dir'], 
                layer_height=settings['layer_height'],
                wall_thickness=settings['wall_thickness'],
                infill_percent=settings['infill_percent'],
                infill_type=settings['infill_type'],
                image_size=settings['image_size'],
                solid_layers=settings['solid_layers']
            )
            
            print(f"Файлы сохранены в директории '{settings['output_dir']}'")
            input("Нажмите Enter для продолжения...")
        
        elif choice == '10':
            print("Выход из программы...")
            break
        
        else:
            print("Неверный выбор! Попробуйте снова.")

if __name__ == "__main__":
    main()
