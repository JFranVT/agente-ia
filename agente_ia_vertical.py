"""
Agente IA optimizado para MODO VERTICAL
Diseñado para carrito con cámara vertical (DroidCam)
"""

import cv2
import json
import numpy as np
from datetime import datetime
import sys
import time
import threading
from queue import Queue

from analyzer_mediapipe import PersonAnalyzerMediaPipe
from utils_Version import DataManager, Logger
from ui_improved import ImprovedUI

class AgenteIAVertical:
    """Agente IA optimizado para captura VERTICAL con DroidCam"""
    
    def __init__(self, config_file="config.json"):
        self.config = self._load_config(config_file)
        self.analyzer = PersonAnalyzerMediaPipe()
        self.logger = Logger()
        self.data_manager = DataManager(self.config.get('output_folder', 'resultados'))
        
        self.start_time = time.time()
        self.frame_count = 0
        self.person_count = 0
        self.all_analyses = []
        
        # Configuración para modo vertical
        self.vertical_mode = self.config.get('vertical_mode', True)
        self.rotation_angle = self.config.get('rotation_angle', 90)  # 90 o -90
        
        # Dimensiones en modo vertical
        if self.vertical_mode:
            # Intercambiamos ancho y alto para modo vertical
            self.frame_width = self.config.get('frame_height', 480)
            self.frame_height = self.config.get('frame_width', 640)
        else:
            self.frame_width = self.config.get('frame_width', 640)
            self.frame_height = self.config.get('frame_height', 480)
        
        # Threading
        self.detection_queue = Queue(maxsize=3)
        self.analysis_queue = Queue(maxsize=3)
        self.display_queue = Queue(maxsize=1)
        
        self.running = True
        self.last_analysis = []
        self.last_faces = []
        self.current_frame_display = None
        
        self.logger.log(f"Agente IA Vertical inicializado")
        self.logger.log(f"Modo: {'VERTICAL' if self.vertical_mode else 'HORIZONTAL'}")
        self.logger.log(f"Dimensiones: {self.frame_width}x{self.frame_height}")
    
    def _load_config(self, config_file):
        """Carga configuración con opciones verticales"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Asegurar que existan las opciones verticales
            if 'vertical_mode' not in config:
                config['vertical_mode'] = True
            if 'rotation_angle' not in config:
                config['rotation_angle'] = 90
            
            return config
        except Exception as e:
            print(f"Error cargando config: {e}")
            return {}
    
    def connect_to_camera(self):
        """Conecta a DroidCam"""
        rtsp_url = self.config.get('rtsp_url', 1)
        
        try:
            # Intentar conectar a DroidCam
            # DroidCam usa: http://IP:4747/video
            if isinstance(rtsp_url, str) and rtsp_url.startswith('http'):
                cap = cv2.VideoCapture(rtsp_url)
            else:
                # Cámara USB o índice
                cap = cv2.VideoCapture(rtsp_url)
            
            time.sleep(2)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.logger.log(f"✓ Cámara conectada: {rtsp_url}")
                    self.logger.log(f"  Resolución nativa: {frame.shape[1]}x{frame.shape[0]}")
                    return cap
            
            self.logger.log(f"✗ No se pudo conectar a: {rtsp_url}", "ERROR")
            return None
        except Exception as e:
            self.logger.log(f"Error de conexión: {e}", "ERROR")
            return None
    
    def process_frame_vertical(self, frame):
        """Procesa frame en modo vertical"""
        if frame is None:
            return None, [], []
        
        # Rotar frame según configuración
        if self.vertical_mode:
            if self.rotation_angle == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation_angle == -90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif self.rotation_angle == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        # Redimensionar al tamaño configurado
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # Detectar caras
        faces = self.analyzer.detect_faces(frame)
        
        analyses = []
        for face in faces:
            try:
                bbox_data = face.get('bbox')
                if not bbox_data or len(bbox_data) != 4:
                    continue
                
                x, y, w, h = bbox_data
                face_roi = frame[y:y+h, x:x+w]
                confidence = face.get('confidence', 0.5)
                
                if confidence < 0.5 or face_roi.size == 0:
                    continue
                
                is_back = face.get('type') == 'back'
                analysis = self.analyzer.analyze_person(
                    frame, face_roi, 
                    full_frame=frame, 
                    is_from_back=is_back, 
                    pose_data=face.get('pose')
                )
                analyses.append(analysis)
                self.all_analyses.append(analysis)
                
            except Exception as e:
                self.logger.log(f"Error procesando cara: {e}", "WARNING")
                continue
        
        # Dibujar resultados
        frame = self.analyzer.draw_results(frame, faces, analyses)
        
        return frame, faces, analyses
    
    def save_results(self):
        """Guarda resultados"""
        if self.all_analyses:
            if self.config.get('save_json', True):
                self.data_manager.save_json(self.all_analyses)
            if self.config.get('save_csv', True):
                self.data_manager.save_csv(self.all_analyses)
            self.logger.log("Resultados guardados")
    
    def print_statistics(self):
        """Imprime estadísticas"""
        elapsed = time.time() - self.start_time
        fps_avg = self.frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*60)
        print("ESTADÍSTICAS FINALES")
        print("="*60)
        print(f"Personas detectadas: {self.person_count}")
        print(f"Frames procesados: {self.frame_count}")
        print(f"FPS promedio: {fps_avg:.2f}")
        print(f"Tiempo total: {int(elapsed)} segundos")
        print(f"Modo: {'VERTICAL' if self.vertical_mode else 'HORIZONTAL'}")
        print("="*60 + "\n")
    
    def run(self):
        """Ejecuta el agente IA en modo vertical"""
        print("\n" + "="*60)
        print("   AGENTE IA - MODO VERTICAL (CARRITO)")
        print("="*60)
        print(f"📍 Conectando a: {self.config.get('rtsp_url', 'DroidCam')}")
        print(f"📐 Modo: VERTICAL ({self.frame_width}x{self.frame_height})")
        print("Presiona 'q' para salir, 's' para guardar")
        print("      'r' para rotar 90°")
        print("="*60 + "\n")
        
        cap = self.connect_to_camera()
        
        if cap is None:
            print("\n❌ No se pudo conectar a la cámara")
            print("\n💡 Tips para DroidCam:")
            print("   1. Conecta el celular por WiFi")
            print("   2. Usa: http://[IP_DEL_CELULAR]:4747/video")
            print("   3. Actualiza config.json con la URL")
            return False
        
        ui = ImprovedUI(
            frame_width=self.frame_width,
            frame_height=self.frame_height
        )
        
        try:
            while self.running:
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    self.logger.log("Reconectando...", "WARNING")
                    cap.release()
                    time.sleep(2)
                    cap = self.connect_to_camera()
                    if cap is None:
                        break
                    continue
                
                self.frame_count += 1
                
                # PROCESAR FRAME EN MODO VERTICAL
                frame_display, faces, analyses = self.process_frame_vertical(frame)
                
                if frame_display is None:
                    continue
                
                # Actualizar contador de personas
                if faces:
                    self.person_count += len(faces)
                
                # Calcular FPS
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                # Agregar estadísticas
                frame_display = ui.add_stats(
                    frame_display, fps, self.person_count,
                    self.frame_count, elapsed
                )
                
                # Agregar panel de análisis
                if analyses:
                    try:
                        panel = ui.create_info_panel(analyses[0])
                        frame_display = ui.combine_frames(frame_display, panel)
                    except:
                        pass
                
                # Mostrar
                if self.config.get('display_video', True):
                    cv2.imshow('Agente IA - Modo VERTICAL', frame_display)
                
                # Teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    self.logger.log("Saliendo...")
                    break
                elif key == ord('s'):
                    self.save_results()
                elif key == ord('r'):
                    # Cambiar rotación
                    self.rotation_angle = -90 if self.rotation_angle == 90 else 90
                    self.logger.log(f"Rotación cambiada a: {self.rotation_angle}°")
                
        except KeyboardInterrupt:
            self.logger.log("Interrupción del usuario")
        except Exception as e:
            import traceback
            self.logger.log(f"Error: {e}", "ERROR")
            self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            self.save_results()
            self.print_statistics()
        
        return True

def main():
    """Función principal"""
    agente = AgenteIAVertical(config_file="config.json")
    agente.run()

if __name__ == "__main__":
    main()
