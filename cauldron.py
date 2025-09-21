import sys
import os
import numpy as np
from stl import mesh
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
                             QCheckBox, QFileDialog, QGroupBox, QTabWidget, QMessageBox, QLineEdit,
                             QSplitter, QSizePolicy, QTextEdit, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent
import cv2
from scipy.spatial import ConvexHull
from skimage import morphology
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import tempfile
import math
import time
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.Qt import QSurfaceFormat
from collections import defaultdict

# Устанавливаем формат поверхности для OpenGL
format = QSurfaceFormat()
format.setDepthBufferSize(24)
format.setStencilBufferSize(8)
format.setVersion(3, 2)
format.setProfile(QSurfaceFormat.CoreProfile)
QSurfaceFormat.setDefaultFormat(format)

# Класс для многопоточной обработки слайсинга
class SlicingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, converter, layer_height, width, height, fill_type, 
                 fill_density, pattern_size, support_enable, support_angle, 
                 support_type, support_density, wall_thickness, solid_layers,
                 table_width_mm, table_depth_mm):
        super().__init__()
        self.converter = converter
        self.layer_height = layer_height
        self.width = width
        self.height = height
        self.fill_type = fill_type
        self.fill_density = fill_density
        self.pattern_size = pattern_size
        self.support_enable = support_enable
        self.support_angle = support_angle
        self.support_type = support_type
        self.support_density = support_density
        self.wall_thickness = wall_thickness
        self.solid_layers = solid_layers
        self.table_width_mm = table_width_mm
        self.table_depth_mm = table_depth_mm

    def run(self):
        try:
            layers = self.converter.slice(
                self.layer_height, self.width, self.height, self.fill_type, 
                self.fill_density, self.pattern_size, self.support_enable, 
                self.support_angle, self.support_type, self.support_density,
                self.wall_thickness, self.solid_layers, self.table_width_mm, self.table_depth_mm
            )
            self.finished.emit(layers)
        except Exception as e:
            self.error.emit(str(e))

# OpenGL виджет для быстрой визуализации 3D моделей
class OpenGLWidget(QGLWidget):
    def __init__(self, parent=None):
        super(OpenGLWidget, self).__init__(parent)
        self.mesh_vertices = None
        self.mesh_normals = None
        self.xRot = 0
        self.yRot = 0
        self.zRot = 0
        self.zoom = 1.0
        self.lastPos = None
        self.display_list = None

    def initializeGL(self):
        try:
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHT0)
            glEnable(GL_LIGHTING)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            
            # Настройка освещения
            light_position = [0.0, 10.0, 10.0, 1.0]
            light_ambient = [0.2, 0.2, 0.2, 1.0]
            light_diffuse = [0.8, 0.8, 0.8, 1.0]
            light_specular = [1.0, 1.0, 1.0, 1.0]
            
            glLightfv(GL_LIGHT0, GL_POSITION, light_position)
            glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
            glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
            glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
            
            glShadeModel(GL_SMOOTH)
        except Exception as e:
            print(f"OpenGL initialization error: {e}")

    def resizeGL(self, w, h):
        try:
            glViewport(0, 0, w, h)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, w/h, 0.1, 100.0)
            glMatrixMode(GL_MODELVIEW)
        except Exception as e:
            print(f"OpenGL resize error: {e}")

    def paintGL(self):
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            glTranslatef(0.0, 0.0, -5.0 * self.zoom)
            glRotatef(self.xRot, 1.0, 0.0, 0.0)
            glRotatef(self.yRot, 0.0, 1.0, 0.0)
            glRotatef(self.zRot, 0.0, 0.0, 1.0)

            if self.mesh_vertices is not None and self.mesh_normals is not None:
                glEnableClientState(GL_VERTEX_ARRAY)
                glEnableClientState(GL_NORMAL_ARRAY)
                
                glVertexPointer(3, GL_FLOAT, 0, self.mesh_vertices)
                glNormalPointer(GL_FLOAT, 0, self.mesh_normals)
                
                glDrawArrays(GL_TRIANGLES, 0, len(self.mesh_vertices) // 3)
                
                glDisableClientState(GL_VERTEX_ARRAY)
                glDisableClientState(GL_NORMAL_ARRAY)
        except Exception as e:
            print(f"OpenGL render error: {e}")

    def setMeshData(self, vertices, normals):
        try:
            # Конвертируем данные в формат, понятный OpenGL
            if vertices is not None and normals is not None:
                self.mesh_vertices = vertices.astype(np.float32)
                self.mesh_normals = normals.astype(np.float32)
            self.update()
        except Exception as e:
            print(f"OpenGL set mesh data error: {e}")

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.xRot += dy * 0.5
            self.yRot += dx * 0.5
            self.update()

        self.lastPos = event.pos()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.zoom *= 1.0 + delta * 0.1
        self.zoom = max(0.1, min(self.zoom, 5.0))
        self.update()

# Класс настроек трансформации модели
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

# Класс настроек слайсинга
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

        # Реальные размеры стола
        table_size_group = QGroupBox("Реальные размеры стола (мм)")
        table_size_layout = QVBoxLayout()
        
        table_width_layout = QHBoxLayout()
        table_width_layout.addWidget(QLabel("Ширина стола (мм):"))
        self.table_width = QDoubleSpinBox()
        self.table_width.setRange(50, 500)
        self.table_width.setValue(200)
        self.table_width.setSingleStep(1)
        table_width_layout.addWidget(self.table_width)
        table_size_layout.addLayout(table_width_layout)
        
        table_depth_layout = QHBoxLayout()
        table_depth_layout.addWidget(QLabel("Глубина стола (мм):"))
        self.table_depth = QDoubleSpinBox()
        self.table_depth.setRange(50, 500)
        self.table_depth.setValue(200)
        self.table_depth.setSingleStep(1)
        table_depth_layout.addWidget(self.table_depth)
        table_size_layout.addLayout(table_depth_layout)
        
        table_size_group.setLayout(table_size_layout)
        layout.addWidget(table_size_group)

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

# Класс настроек заполнения
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

# Класс настроек поддержек
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

# Класс для работы с STL и преобразований
class STLConverter:
    def __init__(self):
        self.mesh = None
        self.transformed_mesh = None
        
    def load_mesh(self, file_path):
        try:
            self.mesh = mesh.Mesh.from_file(file_path)
            self.transformed_mesh = self.mesh
            return self.mesh
        except Exception as e:
            print(f"Error loading mesh: {e}")
            return None
        
    def apply_transformations(self, scale, rotation, translation):
        if self.mesh is None:
            return None
            
        try:
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
        except Exception as e:
            print(f"Error applying transformations: {e}")
            return None
        
    def get_opengl_data(self):
        if self.transformed_mesh is None:
            return None, None
            
        try:
            vertices = []
            normals = []
            
            for triangle in self.transformed_mesh.vectors:
                # Вычисляем нормаль для треугольника
                v1 = triangle[1] - triangle[0]
                v2 = triangle[2] - triangle[0]
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                
                if norm > 0:
                    normal = normal / norm
                else:
                    normal = np.array([0, 0, 1])
                
                # Добавляем вершины и нормали
                for vertex in triangle:
                    vertices.append(vertex)
                for _ in range(3):
                    normals.append(normal)
                    
            return np.array(vertices).flatten(), np.array(normals).flatten()
        except Exception as e:
            print(f"Error preparing OpenGL data: {e}")
            return None, None
        
    def slice(self, layer_height, width, height, fill_type, fill_density, pattern_size, 
              support_enable=False, support_angle=45, support_type="Обычные", support_density=15,
              wall_thickness=1, solid_layers=5, table_width_mm=200, table_depth_mm=200):
        if self.transformed_mesh is None:
            return []
            
        try:
            # Используем улучшенный алгоритм слайсинга
            return self.advanced_slicing(
                layer_height, width, height, fill_type, fill_density,
                wall_thickness, solid_layers, table_width_mm, table_depth_mm
            )
        except Exception as e:
            print(f"Error during slicing: {e}")
            return []
        
    def advanced_slicing(self, layer_height, width, height, fill_type, fill_density,
                         wall_thickness=1, solid_layers=5, table_width_mm=200, table_depth_mm=200):
        """Улучшенный алгоритм слайсинга с учетом реальных размеров"""
        if self.transformed_mesh is None:
            return []
            
        # Определение границ модели по всем осям
        min_coords = np.min(self.transformed_mesh.vectors, axis=(0, 1))
        max_coords = np.max(self.transformed_mesh.vectors, axis=(0, 1))
        
        min_x, min_y, min_z = min_coords
        max_x, max_y, max_z = max_coords
        
        # Вычисляем высоту модели по оси Z
        z_height = max_z - min_z
        
        # Правильный расчет количества слоев: n = h / k
        num_slices = max(1, int(math.ceil(z_height / layer_height)))
        
        # Вычисляем фактическую толщину слоя для равномерного распределения
        actual_layer_height = z_height / num_slices
        
        # Вычисляем размер модели в мм
        model_width_mm = max_x - min_x
        model_depth_mm = max_y - min_y
        
        # Определяем масштаб для сохранения пропорций с учетом реальных размеров стола
        scale_x = (table_width_mm * 0.9) / model_width_mm if model_width_mm > 0 else 1
        scale_y = (table_depth_mm * 0.9) / model_depth_mm if model_depth_mm > 0 else 1
        scale = min(scale_x, scale_y)
        
        # Вычисляем центр модели
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Смещение для центрирования
        offset_x = width / 2 - center_x * scale
        offset_y = height / 2 - center_y * scale

        # Функция для преобразования координат в пиксели
        def to_pixel(x, y):
            px = int(offset_x + x * scale)
            py = int(offset_y - y * scale)  # Инвертируем Y для правильной ориентации
            return (px, py)

        print(f"Высота модели: {z_height:.2f} мм")
        print(f"Толщина слоя: {layer_height:.2f} мм")
        print(f"Количество слоев: {num_slices}")
        print(f"Фактическая толщина слоя: {actual_layer_height:.4f} мм")
        print(f"Слои с полной заливкой: {solid_layers}")
        print(f"Масштаб модели: {scale:.4f}")
        
        # Создание изображений для каждого среза
        layers = []
        for slice_index in range(num_slices):
            current_z = min_z + (slice_index * actual_layer_height)
            
            # Создаем изображение с черным фоном
            img = Image.new('1', (width, height), 0)
            draw = ImageDraw.Draw(img)
            
            # Создаем маску для заполнения
            mask_img = Image.new('1', (width, height), 0)
            mask_draw = ImageDraw.Draw(mask_img)
            
            # Определяем, нужно ли использовать полную заливку для этого слоя
            use_full_infill = False
            
            # Используем полную заливку для первых и последних solid_layers слоев
            if slice_index < solid_layers or slice_index >= num_slices - solid_layers:
                use_full_infill = True
            
            # Обрабатываем каждый треугольник
            for triangle in self.transformed_mesh.vectors:
                # Проверяем, пересекает ли треугольник текущую плоскость Z
                z_values = triangle[:, 2]
                
                # Особый случай: горизонтальная плоскость точно на уровне среза или рядом
                if np.any(np.isclose(z_values, current_z, atol=actual_layer_height/2)):
                    # Это горизонтальная плоскость на уровне среза или рядом
                    # Проецируем весь треугольник на плоскость
                    vertices_2d = triangle[:, :2]
                    
                    # Преобразуем вершины в пиксели
                    pixels = [to_pixel(v[0], v[1]) for v in vertices_2d]
                    
                    # Рисуем заполненный треугольник
                    if len(pixels) >= 3:
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
            if fill_density > 0:
                # Находим точку внутри контура для заливки
                # Используем центр ограничивающего прямоугольника маски
                bbox = mask_img.getbbox()
                if bbox:
                    center_x_mask = (bbox[0] + bbox[2]) // 2
                    center_y_mask = (bbox[1] + bbox[3]) // 2
                    
                    # Заполняем маску
                    ImageDraw.floodfill(mask_img, (center_x_mask, center_y_mask), 1)
                    
                    # Определяем тип заполнения
                    current_infill_type = fill_type
                    current_infill_percent = fill_density
                    
                    # Используем полную заливку, если это необходимо
                    if use_full_infill:
                        current_infill_type = "Полное"
                        current_infill_percent = 100
                        print(f"Слой {slice_index+1}/{num_slices} (полная заливка)")
                    else:
                        print(f"Слой {slice_index+1}/{num_slices} ({fill_type} заполнение)")
                    
                    # Добавляем заполнение с использованием улучшенного алгоритма
                    self.advanced_infill(draw, mask_img, current_infill_type, current_infill_percent, (width, height))
            
            layers.append(img)
            
        return layers
        
    def advanced_infill(self, draw, mask, infill_type, infill_percent, image_size):
        """
        Улучшенный алгоритм заполнения с использованием масок и паттернов
        """
        width, height = image_size
        
        # Создаем изображение для паттерна
        pattern_img = Image.new('1', (width, height), 0)
        pattern_draw = ImageDraw.Draw(pattern_img)
        
        # Определяем шаг заполнения на основе процента
        # Чем выше процент, тем меньше шаг (больше заполнения)
        max_density_step = 20
        min_density_step = 2
        step = max(min_density_step, int(max_density_step * (100 - infill_percent) / 100))
        
        if infill_type == "Сетка":
            # Сетка (два направления линий)
            for i in range(-height, width + height, step):
                pattern_draw.line([(i, 0), (i + height, height)], fill=1, width=1)
            
            for i in range(0, width + height, step):
                pattern_draw.line([(i, 0), (i - height, height)], fill=1, width=1)
        
        elif infill_type == "Восьмиугольники":
            # Восьмиугольный паттерн
            hex_size = step
            for y in range(0, height + hex_size, hex_size):
                for x in range(0, width + hex_size, hex_size):
                    # Смещение для каждого второго ряда
                    x_offset = hex_size // 2 if (y // hex_size) % 2 == 1 else 0
                    
                    # Рисуем шестиугольник
                    points = []
                    for i in range(6):
                        angle = math.pi / 3 * i
                        px = x + x_offset + hex_size * math.cos(angle)
                        py = y + hex_size * math.sin(angle)
                        points.append((px, py))
                    
                    if len(points) >= 3:
                        pattern_draw.polygon(points, outline=1, fill=1)
        
        elif infill_type == "Квадраты":
            # Квадратный паттерн
            for y in range(0, height, step):
                for x in range(0, width, step):
                    pattern_draw.rectangle([x, y, x+step-1, y+step-1], outline=1, fill=1)
        
        elif infill_type == "Треугольники":
            # Треугольный паттерн
            for y in range(0, height + step, step):
                for x in range(0, width + step, step):
                    # Рисуем треугольники, образующие шестиугольник
                    if (x // step + y // step) % 2 == 0:
                        pattern_draw.polygon([(x, y), (x+step, y), (x+step//2, y+step)], outline=1, fill=1)
                    else:
                        pattern_draw.polygon([(x, y+step), (x+step, y+step), (x+step//2, y)], outline=1, fill=1)
        
        elif infill_type == "Гироид":
            # Гироидный паттерн (упрощенная версия)
            for y in range(0, height, step):
                for x in range(0, width, step):
                    # Упрощенный гироид - волнистые линии
                    if (x + y) % (2 * step) < step:
                        pattern_draw.point((x, y), fill=1)
        
        elif infill_type == "Полное":
            # Полное заполнение
            pattern_draw.rectangle([0, 0, width, height], fill=1)
        
        # Накладываем паттерн на маску
        for y in range(height):
            for x in range(width):
                if mask.getpixel((x, y)) == 1 and pattern_img.getpixel((x, y)) == 1:
                    draw.point((x, y), fill=1)

# Главное окно приложения
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stl_mesh = None
        self.converter = STLConverter()
        self.slicing_thread = None
        self.layers = []
        self.current_layer = 0
        self.initUI()
        self.setAcceptDrops(True)

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

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

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
        
        # Визуализация 3D модели с помощью OpenGL
        self.model_viewer = OpenGLWidget(self)
        right_layout.addWidget(self.model_viewer)
        
        # Превью слоя
        preview_label = QLabel("Превью слоя:")
        right_layout.addWidget(preview_label)
        
        self.preview = QLabel()
        self.preview.setMinimumSize(320, 240)
        self.preview.setStyleSheet("border: 1px solid black; background-color: white;")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setText("Перетащите STL файл сюда или используйте кнопку загрузки")
        right_layout.addWidget(self.preview)
        
        # Элементы управления просмотром слоев
        layer_controls = QHBoxLayout()
        layer_controls.addWidget(QLabel("Слой:"))
        
        self.layer_slider = QSlider(Qt.Horizontal)
        self.layer_slider.setMinimum(0)
        self.layer_slider.setMaximum(0)
        self.layer_slider.valueChanged.connect(self.show_layer)
        layer_controls.addWidget(self.layer_slider)
        
        self.layer_spin = QSpinBox()
        self.layer_spin.setMinimum(0)
        self.layer_spin.setMaximum(0)
        self.layer_spin.valueChanged.connect(self.show_layer)
        layer_controls.addWidget(self.layer_spin)
        
        right_layout.addLayout(layer_controls)
        
        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel)

        central_widget.setLayout(main_layout)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().lower().endswith('.stl') for url in urls):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        for url in urls:
            file_path = url.toLocalFile()
            if file_path.lower().endswith('.stl'):
                self.load_stl_file(file_path)
                break

    def load_stl(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите STL файл", "", "STL Files (*.stl)")
        if file_path:
            self.load_stl_file(file_path)

    def load_stl_file(self, file_path):
        try:
            self.stl_mesh = self.converter.load_mesh(file_path)
            
            if self.stl_mesh is None:
                QMessageBox.critical(self, "Ошибка", "Не удалось загрузить STL файл")
                return
            
            # Обновляем OpenGL виджет
            vertices, normals = self.converter.get_opengl_data()
            if vertices is not None and normals is not None:
                self.model_viewer.setMeshData(vertices, normals)
            
            self.preview.setText("Модель загружена. Нажмите 'Генерировать BMP' для создания слоев.")
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
            
            if transformed_mesh is None:
                QMessageBox.critical(self, "Ошибка", "Не удалось применить трансформации")
                return
            
            # Обновляем OpenGL виджет
            vertices, normals = self.converter.get_opengl_data()
            if vertices is not None and normals is not None:
                self.model_viewer.setMeshData(vertices, normals)
                
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
        wall_thickness = self.slice_settings.wall_thickness.value()
        solid_layers = self.slice_settings.raft_layers.value()
        table_width_mm = self.slice_settings.table_width.value()
        table_depth_mm = self.slice_settings.table_depth.value()
        
        save_path = self.slice_settings.save_path.text()

        # Показываем прогресс бар
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Запускаем слайсинг в отдельном потоке
        self.slicing_thread = SlicingThread(
            self.converter, layer_height, width, height, fill_type, fill_density, 
            pattern_size, support_enable, support_angle, support_type, support_density,
            wall_thickness, solid_layers, table_width_mm, table_depth_mm
        )
        
        self.slicing_thread.finished.connect(self.on_slicing_finished)
        self.slicing_thread.error.connect(self.on_slicing_error)
        self.slicing_thread.start()

        # Таймер для обновления прогресса
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(100)

    def update_progress(self):
        # В реальном приложении здесь нужно получать реальный прогресс из потока
        current = self.progress_bar.value()
        if current < 100:
            self.progress_bar.setValue(current + 1)

    def on_slicing_finished(self, layers):
        # Останавливаем таймер прогресса
        self.progress_timer.stop()
        self.progress_bar.setVisible(False)
        
        self.layers = layers
        
        # Сохранение изображений
        save_path = self.slice_settings.save_path.text()
        
        try:
            for i, layer_img in enumerate(layers):
                layer_img.save(os.path.join(save_path, f"layer_{i:04d}.bmp"))
            
            # Настройка элементов управления слоями
            if layers:
                self.layer_slider.setMaximum(len(layers)-1)
                self.layer_spin.setMaximum(len(layers)-1)
                self.show_layer(0)
            
            QMessageBox.information(self, "Успех", f"BMP файлы успешно сгенерированы и сохранены в {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении: {str(e)}")

    def show_layer(self, index):
        if index < len(self.layers):
            self.current_layer = index
            layer_img = self.layers[index]
            preview_img = layer_img.resize((320, 240), Image.NEAREST)
            preview_img = preview_img.convert("RGB")
            data = preview_img.tobytes("raw", "RGB")
            q_img = QImage(data, preview_img.width, preview_img.height, QImage.Format_RGB888)
            self.preview.setPixmap(QPixmap.fromImage(q_img))
            
            # Синхронизация слайдера и спинбокса
            self.layer_slider.setValue(index)
            self.layer_spin.setValue(index)

    def on_slicing_error(self, error_msg):
        # Останавливаем таймер прогресса
        self.progress_timer.stop()
        self.progress_bar.setVisible(False)
        
        QMessageBox.critical(self, "Ошибка", f"Ошибка при генерации: {error_msg}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
