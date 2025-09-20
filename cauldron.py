import numpy as np
from stl import mesh
from PIL import Image, ImageDraw

def stl_to_bmp_slices(stl_file_path, output_prefix, num_slices, image_size=100):
    # Загрузка STL-файла
    stl_mesh = mesh.Mesh.from_file(stl_file_path)
    
    # Определение границ модели
    min_z, max_z = np.min(stl_mesh.vectors[:, :, 2]), np.max(stl_mesh.vectors[:, :, 2])
    z_height = max_z - min_z
    slice_thickness = z_height / num_slices

    # Создание изображений для каждого среза
    for slice_index in range(num_slices):
        current_z = min_z + (slice_index * slice_thickness)
        img = Image.new('1', (image_size, image_size), 0)  # Чёрный фон
        draw = ImageDraw.Draw(img)
        
        # Обработка каждого треугольника
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
                # Масштабирование координат в пиксели
                x1, y1 = intersections[0]
                x2, y2 = intersections[1]
                scale = 0.8 * image_size / max(np.ptp(stl_mesh.vectors[:, :, 0]), np.ptp(stl_mesh.vectors[:, :, 1]))
                offset = image_size / 2
                x1_pix = int(offset + x1 * scale)
                y1_pix = int(offset - y1 * scale)  # Инвертируем Y
                x2_pix = int(offset + x2 * scale)
                y2_pix = int(offset - y2 * scale)
                draw.line([x1_pix, y1_pix, x2_pix, y2_pix], fill=1)  # Белая линия

        # Сохранение среза
        img.save(f"{output_prefix}_slice_{slice_index:04d}.bmp")

# Пример использования
stl_to_bmp_slices("model.stl", "output", num_slices=50, image_size=200)
