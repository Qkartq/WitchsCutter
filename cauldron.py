import sys
import os
import numpy as np
from stl import mesh
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
                             QCheckBox, QFileDialog, QGroupBox, QTabWidget, QMessageBox, QLineEdit,
                             QSplitter, QSizePolicy, QTextEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import cv2
from scipy.spatial import ConvexHull
from skimage import morphology
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import tempfile
import math

class ModelViewer(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.figure(figsize=(5, 5))
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111, projection='3d')
        self.axes.set_xlabel('X')
        self.axes.set_ylabel('Y')
        self.axes.set_zlabel('Z')
        self.mesh_data = None
        
    def plot_mesh(self, stl_mesh, scale=1.0, rotation=(0, 0, 0), translation=(0, 0, 0)):
        self.axes.clear()
        
        if stl_mesh is None:
            self.draw()
            return
            
        # Apply transformations
        vectors = stl_mesh.vectors.copy()
        
        # Scale
        vectors *= scale
        
        # Rotation
        rx, ry, rz = rotation
        rx = math.radians(rx)
        ry = math.radians(ry)
        rz = math.radians(rz)
        
        # Rotation matrices
        rot_x = np.array([[1, 0, 0],
                          [0, math.cos(rx), -math.sin(rx)],
                          [0, math.sin(rx), math.cos(rx)]])
        
        rot_y = np.array([[math.cos(ry), 0, math.sin(ry)],
                          [0, 1, 0],
                          [-math.sin(ry), 0, math.cos(ry)]])
        
        rot_z = np.array([[math.cos(rz), -math.sin(rz), 0],
                          [math.sin(rz), math.cos(rz), 0],
                          [0, 0, 1]])
        
        # Apply rotation
        for i in range(len(vectors)):
            for j in range(3):
                vectors[i, j] = np.dot(rot_z, np.dot(rot_y, np.dot(rot_x, vectors[i, j])))
        
        # Translation
        vectors += np.array(translation)
        
        # Plot the mesh
        self.mesh_data = mplot3d.art3d.Poly3DCollection(vectors)
        self.mesh_data.set_edgecolor('k')
        self.mesh_data.set_facecolor('cyan')
        self.mesh_data.set_alpha(0.5)
        self.axes.add_collection3d(self.mesh_data)
        
        # Auto scale to the mesh size
        scale = vectors.flatten()
        self.axes.auto_scale_xyz(scale, scale, scale)
        
        self.draw()

class TransformSettings(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Масштаб
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Масштаб:"))
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.1, 10.0)
        self.scale.setValue(1.0)
        self.scale.setSingleStep(0.1)
        scale_layout.addWidget(self.scale)
        layout.addLayout(scale_layout)

        # Положение
        position_group = QGroupBox("Положение")
        position_layout = QVBoxLayout()
        
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.x_pos = QDoubleSpinBox()
        self.x_pos.setRange(-100.0, 100.0)
        self.x_pos.setValue(0.0)
        x_layout.addWidget(self.x_pos)
        position_layout.addLayout(x_layout)
        
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.y_pos = QDoubleSpinBox()
        self.y_pos.setRange(-100.0, 100.0)
        self.y_pos.setValue(0.0)
        y_layout.addWidget(self.y_pos)
        position_layout.addLayout(y_layout)
        
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z:"))
        self.z_pos = QDoubleSpinBox()
        self.z_pos.setRange(-100.0, 100.0)
        self.z_pos.setValue(0.0)
        z_layout.addWidget(self.z_pos)
        position_layout.addLayout(z_layout)
        
        position_group.setLayout(position_layout)
        layout.addWidget(position_group)

        # Вращение
        rotation_group = QGroupBox("Вращение (градусы)")
        rotation_layout = QVBoxLayout()
        
        x_rot_layout = QHBoxLayout()
        x_rot_layout.addWidget(QLabel("X:"))
        self.x_rot = QDoubleSpinBox()
        self.x_rot.setRange(-180.0, 180.0)
        self.x_rot.setValue(0.0)
        x_rot_layout.addWidget(self.x_rot)
        rotation_layout.addLayout(x_rot_layout)
        
        y_rot_layout = QHBoxLayout()
        y_rot_layout.addWidget(QLabel("Y:"))
        self.y_rot = QDoubleSpinBox()
        self.y_rot.setRange(-180.0, 180.0)
        self.y_rot.setValue(0.0)
        y_rot_layout.addWidget(self.y_rot)
        rotation_layout.addLayout(y_rot_layout)
        
        z_rot_layout = QHBoxLayout()
        z_rot_layout.addWidget(QLabel("Z:"))
        self.z_rot = QDoubleSpinBox()
        self.z_rot.setRange(-180.0, 180.0)
        self.z_rot.setValue(0.0)
        z_rot_layout.addWidget(self.z_rot)
        rotation_layout.addLayout(z_rot_layout)
        
        rotation_group.setLayout(rotation_layout)
        layout.addWidget(rotation_group)

        # Кнопка применения трансформации
        self.apply_btn = QPushButton("Применить трансформацию")
        layout.addWidget(self.apply_btn)

        layout.addStretch()
        self.setLayout(layout)

class SliceSettings(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Толщина слоя
        layer_thickness = QHBoxLayout()
        layer_thickness.addWidget(QLabel("Толщина слоя (мм):"))
        self.layer_thickness = QDoubleSpinBox()
        self.layer_thickness.setRange(0.01, 1.0)
        self.layer_thickness.setValue(0.1)
        self.layer_thickness.setSingleStep(0.01)
        layer_thickness.addWidget(self.layer_thickness)
        layout.addLayout(layer_thickness)

        # Толщина стенок
        wall_thickness = QHBoxLayout()
        wall_thickness.addWidget(QLabel("Толщина стенок (мм):"))
        self.wall_thickness = QSpinBox()
        self.wall_thickness.setRange(1, 10)
        self.wall_thickness.setValue(2)
        wall_thickness.addWidget(self.wall_thickness)
        layout.addLayout(wall_thickness)

        # Подложка
        self.raft_enable = QCheckBox("Включить подложку")
        self.raft_enable.setChecked(True)
        layout.addWidget(self.raft_enable)

        raft_layers = QHBoxLayout()
        raft_layers.addWidget(QLabel("Количество слоев подложки:"))
        self.raft_layers = QSpinBox()
        self.raft_layers.setRange(1, 10)
        self.raft_layers.setValue(3)
        raft_layers.addWidget(self.raft_layers)
        layout.addLayout(raft_layers)

        # Разрешение
        resolution_group = QGroupBox("Разрешение изображения")
        resolution_layout = QVBoxLayout()
        
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Ширина:"))
        self.width = QSpinBox()
        self.width.setRange(100, 1000)
        self.width.setValue(320)
        res_layout.addWidget(self.width)
        
        res_layout.addWidget(QLabel("Высота:"))
        self.height = QSpinBox()
        self.height.setRange(100, 1000)
        self.height.setValue(240)
        res_layout.addWidget(self.height)
        
        resolution_layout.addLayout(res_layout)
        resolution_group.setLayout(resolution_layout)
        layout.addWidget(resolution_group)

        # Путь сохранения
        save_path_layout = QHBoxLayout()
        save_path_layout.addWidget(QLabel("Папка для сохранения:"))
        self.save_path = QLineEdit()
        self.save_path.setText(os.getcwd())
        save_path_layout.addWidget(self.save_path)
        
        self.browse_btn = QPushButton("Обзор")
        save_path_layout.addWidget(self.browse_btn)
        layout.addLayout(save_path_layout)

        layout.addStretch()
        self.setLayout(layout)

class FillSettings(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Тип заполнения
        fill_type = QHBoxLayout()
        fill_type.addWidget(QLabel("Тип заполнения:"))
        self.fill_type = QComboBox()
        self.fill_type.addItems(["Полное", "Сетка", "Восьмиугольники", "Квадраты", "Треугольники", "Гироид"])
        fill_type.addWidget(self.fill_type)
        layout.addLayout(fill_type)

        # Плотность заполнения
        fill_density = QHBoxLayout()
        fill_density.addWidget(QLabel("Плотность заполнения (%):"))
        self.fill_density = QSpinBox()
        self.fill_density.setRange(0, 100)
        self.fill_density.setValue(20)
        fill_density.addWidget(self.fill_density)
        layout.addLayout(fill_density)

        # Размер паттерна
        pattern_size = QHBoxLayout()
        pattern_size.addWidget(QLabel("Размер паттерна (мм):"))
        self.pattern_size = QDoubleSpinBox()
        self.pattern_size.setRange(1.0, 10.0)
        self.pattern_size.setValue(5.0)
        self.pattern_size.setSingleStep(0.5)
        pattern_size.addWidget(self.pattern_size)
        layout.addLayout(pattern_size)

        layout.addStretch()
        self.setLayout(layout)

class SupportSettings(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Включение поддержек
        self.support_enable = QCheckBox("Включить поддержки")
        layout.addWidget(self.support_enable)

        # Угол поддержек
        support_angle = QHBoxLayout()
        support_angle.addWidget(QLabel("Угол поддержек (градусы):"))
        self.support_angle = QSpinBox()
        self.support_angle.setRange(0, 90)
        self.support_angle.setValue(45)
        support_angle.addWidget(self.support_angle)
        layout.addLayout(support_angle)

        # Тип поддержек
        support_type = QHBoxLayout()
        support_type.addWidget(QLabel("Тип поддержек:"))
        self.support_type = QComboBox()
        self.support_type.addItems(["Обычные", "Древовидные"])
        support_type.addWidget(self.support_type)
        layout.addLayout(support_type)

        # Плотность поддержек
        support_density = QHBoxLayout()
        support_density.addWidget(QLabel("Плотность поддержек (%):"))
        self.support_density = QSpinBox()
        self.support_density.setRange(5, 50)
        self.support_density.setValue(15)
        support_density.addWidget(self.support_density)
        layout.addLayout(support_density)

        layout.addStretch()
        self.setLayout(layout)

class STLConverter:
    def __init__(self):
        self.mesh = None
        self.transformed_mesh = None
        
    def load_mesh(self, file_path):
        self.mesh = mesh.Mesh.from_file(file_path)
        self.transformed_mesh = self.mesh
        return self.mesh
        
    def apply_transformations(self, scale, rotation, translation):
        if self.mesh is None:
            return None
            
        # Создаем копию меша для трансформаций
        self.transformed_mesh = mesh.Mesh(self.mesh.data.copy())
        
        # Масштабирование
        self.transformed_mesh.vectors *= scale
        
        # Вращение
        rx, ry, rz = rotation
        rx = math.radians(rx)
        ry = math.radians(ry)
        rz = math.radians(rz)
        
        # Матрицы вращения
        rot_x = np.array([[1, 0, 0],
                          [0, math.cos(rx), -math.sin(rx)],
                          [0, math.sin(rx), math.cos(rx)]])
        
        rot_y = np.array([[math.cos(ry), 0, math.sin(ry)],
                          [0, 1, 0],
                          [-math.sin(ry), 0, math.cos(ry)]])
        
        rot_z = np.array([[math.cos(rz), -math.sin(rz), 0],
                          [math.sin(rz), math.cos(rz), 0],
                          [0, 0, 1]])
        
        # Применяем вращение к каждой точке
        for i in range(len(self.transformed_mesh.vectors)):
            for j in range(3):
                vec = self.transformed_mesh.vectors[i, j]
                vec = np.dot(rot_z, np.dot(rot_y, np.dot(rot_x, vec)))
                self.transformed_mesh.vectors[i, j] = vec
        
        # Перенос
        self.transformed_mesh.vectors += np.array(translation)
        
        return self.transformed_mesh
        
    def slice(self, layer_height, width, height, fill_type, fill_density, pattern_size, 
              support_enable=False, support_angle=45, support_type="Обычные", support_density=15):
        if self.transformed_mesh is None:
            return []
            
        # Находим границы модели
        min_z = np.min(self.transformed_mesh.vectors[:, :, 2])
        max_z = np.max(self.transformed_mesh.vectors[:, :, 2])
        
        # Генерируем слои
        layers = []
        z = min_z
        
        while z <= max_z:
            # Получаем контуры для текущего слоя
            contours = self._get_contours(z)
            
            # Создаем изображение слоя
            layer_img = Image.new('1', (width, height), 0)
            draw = ImageDraw.Draw(layer_img)
            
            # Рисуем контуры
            for contour in contours:
                # Масштабируем и переносим контур в центр изображения
                scaled_contour = []
                for point in contour:
                    x = int((point[0] + width/2) * 0.9)
                    y = int((point[1] + height/2) * 0.9)
                    scaled_contour.append((x, y))
                
                if len(scaled_contour) >= 3:
                    draw.polygon(scaled_contour, fill=1, outline=1)
            
            # Применяем заполнение
            if fill_density > 0:
                self._apply_fill_pattern(layer_img, fill_type, fill_density, pattern_size)
            
            layers.append(layer_img)
            z += layer_height
            
        return layers
        
    def _get_contours(self, z):
        # Упрощенная реализация получения контуров
        # В реальном приложении нужно использовать более сложный алгоритм
        contours = []
        
        for triangle in self.transformed_mesh.vectors:
            points_below = []
            points_above = []
            
            for point in triangle:
                if point[2] <= z:
                    points_below.append(point)
                else:
                    points_above.append(point)
            
            # Если треугольник пересекает плоскость Z
            if len(points_below) == 2 and len(points_above) == 1:
                # Находим точки пересечения
                p1, p2 = points_below
                p3 = points_above[0]
                
                # Интерполируем точки пересечения
                t1 = (z - p1[2]) / (p3[2] - p1[2])
                x1 = p1[0] + t1 * (p3[0] - p1[0])
                y1 = p1[1] + t1 * (p3[1] - p1[1])
                
                t2 = (z - p2[2]) / (p3[2] - p2[2])
                x2 = p2[0] + t2 * (p3[0] - p2[0])
                y2 = p2[1] + t2 * (p3[1] - p2[1])
                
                contours.append([(x1, y1), (x2, y2)])
            
            elif len(points_below) == 1 and len(points_above) == 2:
                # Аналогично предыдущему случаю, но наоборот
                p1 = points_below[0]
                p2, p3 = points_above
                
                t1 = (z - p1[2]) / (p2[2] - p1[2])
                x1 = p1[0] + t1 * (p2[0] - p1[0])
                y1 = p1[1] + t1 * (p2[1] - p1[1])
                
                t2 = (z - p1[2]) / (p3[2] - p1[2])
                x2 = p1[0] + t2 * (p3[0] - p1[0])
                y2 = p1[1] + t2 * (p3[1] - p1[1])
                
                contours.append([(x1, y1), (x2, y2)])
        
        # Объединяем сегменты в замкнутые контуры
        # (упрощенная реализация - в реальном приложении нужен более сложный алгоритм)
        closed_contours = []
        while contours:
            current = contours.pop(0)
            found_match = True
            
            while found_match and len(current) < 20:  # ограничение для избежания бесконечного цикла
                found_match = False
                for i, contour in enumerate(contours):
                    if abs(current[-1][0] - contour[0][0]) < 0.1 and abs(current[-1][1] - contour[0][1]) < 0.1:
                        current.extend(contour[1:])
                        contours.pop(i)
                        found_match = True
                        break
                    elif abs(current[-1][0] - contour[-1][0]) < 0.1 and abs(current[-1][1] - contour[-1][1]) < 0.1:
                        current.extend(reversed(contour[:-1]))
                        contours.pop(i)
                        found_match = True
                        break
            
            if len(current) > 2 and abs(current[0][0] - current[-1][0]) < 0.1 and abs(current[0][1] - current[-1][1]) < 0.1:
                closed_contours.append(current)
        
        return closed_contours
        
    def _apply_fill_pattern(self, image, pattern_type, density, pattern_size):
        # Преобразуем изображение в массив numpy
        img_array = np.array(image)
        
        if pattern_type == "Полное":
            # Уже заполнено полностью
            pass
        elif pattern_type == "Сетка":
            # Создаем сетку линий
            h, w = img_array.shape
            step = int(pattern_size * (100 / density))
            
            for i in range(0, h, step):
                cv2.line(img_array, (0, i), (w, i), 1, 1)
            
            for i in range(0, w, step):
                cv2.line(img_array, (i, 0), (i, h), 1, 1)
        elif pattern_type == "Восьмиугольники":
            # Реализация восьмиугольников
            h, w = img_array.shape
            step = int(pattern_size * (100 / density))
            
            for y in range(0, h, step):
                for x in range(0, w, step):
                    if x + step < w and y + step < h:
                        pts = np.array([
                            [x + step//4, y],
                            [x + 3*step//4, y],
                            [x + step, y + step//4],
                            [x + step, y + 3*step//4],
                            [x + 3*step//4, y + step],
                            [x + step//4, y + step],
                            [x, y + 3*step//4],
                            [x, y + step//4]
                        ], np.int32)
                        cv2.fillPoly(img_array, [pts], 1)
        # Другие паттерны можно добавить аналогично
        
        # Обновляем изображение
        image.paste(Image.fromarray(img_array))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stl_mesh = None
        self.converter = STLConverter()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("STL to BMP Slicer")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()

        # Левая панель с настройками
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout()

        # Кнопка загрузки STL
        self.load_btn = QPushButton("Загрузить STL файл")
        self.load_btn.clicked.connect(self.load_stl)
        left_layout.addWidget(self.load_btn)

        # Настройки слайсинга
        self.tabs = QTabWidget()
        self.transform_settings = TransformSettings()
        self.slice_settings = SliceSettings()
        self.fill_settings = FillSettings()
        self.support_settings = SupportSettings()

        self.tabs.addTab(self.transform_settings, "Трансформация")
        self.tabs.addTab(self.slice_settings, "Слайсинг")
        self.tabs.addTab(self.fill_settings, "Заполнение")
        self.tabs.addTab(self.support_settings, "Поддержки")
        left_layout.addWidget(self.tabs)

        # Кнопка генерации
        self.generate_btn = QPushButton("Генерировать BMP")
        self.generate_btn.clicked.connect(self.generate_bmp)
        left_layout.addWidget(self.generate_btn)

        # Кнопка применения трансформации
        self.transform_settings.apply_btn.clicked.connect(self.apply_transformation)

        # Кнопка выбора папки сохранения
        self.slice_settings.browse_btn.clicked.connect(self.browse_save_path)

        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel)

        # Правая панель с визуализацией
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Визуализация 3D модели
        self.model_viewer = ModelViewer(self)
        right_layout.addWidget(self.model_viewer)
        
        # Превью слоя
        preview_label = QLabel("Превью слоя:")
        right_layout.addWidget(preview_label)
        
        self.preview = QLabel()
        self.preview.setMinimumSize(320, 240)
        self.preview.setStyleSheet("border: 1px solid black; background-color: white;")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setText("Превью будет здесь")
        right_layout.addWidget(self.preview)
        
        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel)

        central_widget.setLayout(main_layout)

    def load_stl(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите STL файл", "", "STL Files (*.stl)")
        if file_path:
            try:
                self.stl_mesh = self.converter.load_mesh(file_path)
                self.model_viewer.plot_mesh(self.stl_mesh)
                QMessageBox.information(self, "Успех", "STL файл успешно загружен")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить STL файл: {str(e)}")

    def apply_transformation(self):
        if self.stl_mesh is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите STL файл")
            return

        try:
            scale = self.transform_settings.scale.value()
            rotation = (
                self.transform_settings.x_rot.value(),
                self.transform_settings.y_rot.value(),
                self.transform_settings.z_rot.value()
            )
            translation = (
                self.transform_settings.x_pos.value(),
                self.transform_settings.y_pos.value(),
                self.transform_settings.z_pos.value()
            )
            
            transformed_mesh = self.converter.apply_transformations(scale, rotation, translation)
            self.model_viewer.plot_mesh(transformed_mesh)
            QMessageBox.information(self, "Успех", "Трансформация применена")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при трансформации: {str(e)}")

    def browse_save_path(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")
        if folder_path:
            self.slice_settings.save_path.setText(folder_path)

    def generate_bmp(self):
        if self.stl_mesh is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите STL файл")
            return

        # Получение параметров
        layer_height = self.slice_settings.layer_thickness.value()
        width = self.slice_settings.width.value()
        height = self.slice_settings.height.value()
        fill_type = self.fill_settings.fill_type.currentText()
        fill_density = self.fill_settings.fill_density.value()
        pattern_size = self.fill_settings.pattern_size.value()
        
        support_enable = self.support_settings.support_enable.isChecked()
        support_angle = self.support_settings.support_angle.value()
        support_type = self.support_settings.support_type.currentText()
        support_density = self.support_settings.support_density.value()
        
        save_path = self.slice_settings.save_path.text()

        # Здесь должна быть реализована логика слайсинга и генерации изображений
        try:
            layers = self.converter.slice(
                layer_height, width, height, fill_type, fill_density, pattern_size,
                support_enable, support_angle, support_type, support_density
            )
            
            # Сохранение изображений
            for i, layer_img in enumerate(layers):
                layer_img.save(os.path.join(save_path, f"layer_{i:04d}.bmp"))
            
            # Показ превью первого слоя
            if layers:
                preview_img = layers[0].resize((320, 240), Image.NEAREST)
                preview_img = preview_img.convert("RGB")
                data = preview_img.tobytes("raw", "RGB")
                q_img = QImage(data, preview_img.width, preview_img.height, QImage.Format_RGB888)
                self.preview.setPixmap(QPixmap.fromImage(q_img))
            
            QMessageBox.information(self, "Успех", f"BMP файлы успешно сгенерированы и сохранены в {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при генерации: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
