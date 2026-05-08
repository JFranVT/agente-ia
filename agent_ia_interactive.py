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

class AgenteIAInteractivo:
    """Agente IA Interactivo para detección bajo demanda con interfaz gráfica"""
    
    # Estados del sistema
    STATE_SEARCHING = "searching"      # Buscando personas
    STATE_PERSON_FOUND = "person_found"  # Persona detectada, esperando acción
    STATE_ANALYZING = "analyzing"      # Analizando los 20 aspectos
    STATE_SHOWING_RESULTS = "showing_results"  # Mostrando resultados
    
    def __init__(self, config_file="config.json"):
        self.config = self._load_config(config_file)
        self.analyzer = PersonAnalyzerMediaPipe()
        self.logger = Logger()
        self.data_manager = DataManager(self.config['output_folder'])
        
        self.start_time = time.time()
        self.frame_count = 0
        self.person_count = 0
        self.all_analyses = []
        
        # Estado del sistema
        self.state = self.STATE_SEARCHING
        self.current_analysis = None
        self.current_frame_roi = None
        
        # Threading
        self.running = True
        self.analysis_queue = Queue(maxsize=1)
        self.detection_queue = Queue(maxsize=1)
        
        # Datos de persona actual
        self.current_person_bbox = None
        self.detected_frame = None
        self.current_pose_data = None
        
        # GUI - Mouse
        self.mouse_x = 0
        self.mouse_y = 0
        self.analyze_button_clicked = False
        self.new_detection_clicked = False
        
        # Dimensiones
        self.window_width = 1400
        self.window_height = 900
        self.video_width = 1280
        self.video_height = 720
        
        # Contador para detección fallida
        self.failed_detection_frames = 0
        self.max_failed_frames = 30
        
        self.logger.log(f"Agente IA Interactivo inicializado correctamente")
    
    def _load_config(self, config_file):
        """Carga configuración"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.log(f"Error cargando config: {e}", "ERROR")
            return {}
    
    def connect_to_camera(self):
        """Conecta a la cámara"""
        rtsp_url = self.config.get('rtsp_url', 0)
        
        try:
            cap = cv2.VideoCapture(rtsp_url)
            time.sleep(2)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.logger.log(f"✓ Conexión exitosa a la cámara")
                    return cap
            
            self.logger.log(f"✗ No se pudo conectar a: {rtsp_url}", "ERROR")
            return None
        except Exception as e:
            self.logger.log(f"Error de conexión: {e}", "ERROR")
            return None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos del mouse"""
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state == self.STATE_PERSON_FOUND:
                if self._is_in_analyze_button(x, y):
                    self.analyze_button_clicked = True
            
            elif self.state == self.STATE_SHOWING_RESULTS:
                if self._is_in_new_detection_button(x, y):
                    self.new_detection_clicked = True
    
    def _is_in_analyze_button(self, x, y):
        """Verifica si (x,y) está dentro del botón ANALIZAR"""
        btn_width = 200
        btn_height = 50
        btn_y = self.video_height - btn_height - 15
        btn_x = (self.video_width - btn_width) // 2
        
        return (btn_x <= x <= btn_x + btn_width and 
                btn_y <= y <= btn_y + btn_height)
    
    def _is_in_new_detection_button(self, x, y):
        """Verifica si (x,y) está dentro del botón NUEVA"""
        btn_width = 200
        btn_height = 50
        btn_y = self.video_height - btn_height - 15
        btn_x = (self.video_width - btn_width) // 2
        
        return (btn_x <= x <= btn_x + btn_width and 
                btn_y <= y <= btn_y + btn_height)
    
    def _analysis_worker(self):
        """Hilo para análisis detallado de 20 aspectos"""
        while self.running:
            try:
                frame_data = self.analysis_queue.get(timeout=0.5)
                if frame_data is None:
                    break
                
                frame, roi = frame_data
                try:
                    analysis = self.analyzer.analyze_person(
                        frame, roi, full_frame=frame, is_from_back=False, pose_data=self.current_pose_data
                    )
                    self.current_analysis = analysis
                    self.person_count += 1
                    self.all_analyses.append(analysis)
                    
                    self.state = self.STATE_SHOWING_RESULTS
                    self.logger.log(f"✓ Análisis completo de 20 aspectos finalizado")
                    
                except Exception as e:
                    self.logger.log(f"Error en análisis: {e}", "WARNING")
                    self.state = self.STATE_SEARCHING
            except:
                continue
    
    def _detection_worker(self):
        """Hilo para detección en background sin bloquear video"""
        while self.running:
            try:
                frame_data = self.detection_queue.get(timeout=0.1)
                if frame_data is None:
                    break
                
                frame = frame_data
                try:
                    faces = self.analyzer.detect_faces(frame)
                    self.current_pose_data = self.analyzer.last_pose_data
                    
                    if faces:
                        face = faces[0]
                        bbox_data = face['bbox']
                        x, y, w, h = bbox_data
                        self.current_pose_data = face.get('pose', self.current_pose_data)
                        
                        if self.state == self.STATE_PERSON_FOUND:
                            if self.validate_person_detection(frame, face, (x, y, w, h)):
                                self.current_person_bbox = (x, y, w, h)
                                self.current_frame_roi = frame[y:y+h, x:x+w].copy()
                                self.detected_frame = frame.copy()
                            else:
                                self.state = self.STATE_SEARCHING
                        
                        elif self.state == self.STATE_SEARCHING:
                            if self.validate_person_detection(frame, face, (x, y, w, h)):
                                self.current_person_bbox = (x, y, w, h)
                                self.current_frame_roi = frame[y:y+h, x:x+w].copy()
                                self.detected_frame = frame.copy()
                                self.state = self.STATE_PERSON_FOUND
                                self.failed_detection_frames = 0
                                self.logger.log("✓ Persona REAL detectada")
                            else:
                                self.failed_detection_frames += 1
                                if self.failed_detection_frames > self.max_failed_frames:
                                    self.failed_detection_frames = 10
                    else:
                        if self.state == self.STATE_PERSON_FOUND:
                            self.state = self.STATE_SEARCHING
                        elif self.state == self.STATE_SEARCHING:
                            self.failed_detection_frames += 1
                            if self.failed_detection_frames > self.max_failed_frames:
                                self.failed_detection_frames = 10
                
                except Exception as e:
                    self.logger.log(f"Error en detección: {e}", "WARNING")
            except:
                continue
    
    def validate_person_detection(self, frame, face, bbox):
        """Valida si es realmente una persona (no falsa detección)"""
        try:
            x, y, w, h = bbox
            roi = frame[y:y+h, x:x+w]
            
            confidence = face.get('confidence', 0.5)
            if confidence < self.config.get('confidence_threshold', 0.5):
                return False
            
            if w < 30 or h < 30:
                return False
            if w > frame.shape[1] * 0.9 or h > frame.shape[0] * 0.9:
                return False
            
            aspect_ratio = w / h if h != 0 else 0
            if aspect_ratio < 0.3 or aspect_ratio > 2:
                return False
            
            return True
        except:
            return False
    
    def _resize_frame_keep_aspect(self, frame, target_width, target_height):
        """Redimensiona el frame manteniendo aspecto sin distorsionar"""
        h, w = frame.shape[:2]
        
        scale = min(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas, x_offset, y_offset, scale
    
    def draw_buttons(self, frame):
        """Dibuja botones interactivos según el estado actual"""
        btn_width = 200
        btn_height = 50
        btn_y = frame.shape[0] - btn_height - 15
        btn_x = (frame.shape[1] - btn_width) // 2
        
        if self.state == self.STATE_PERSON_FOUND:
            is_hover = (btn_x <= self.mouse_x <= btn_x + btn_width and 
                       btn_y <= self.mouse_y <= btn_y + btn_height)
            
            color = (0, 200, 100) if is_hover else (0, 255, 0)
            cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_width, btn_y + btn_height), 
                         color, -1)
            cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_width, btn_y + btn_height), 
                         (0, 0, 0), 2)
            
            text = "ANALIZAR"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = btn_x + (btn_width - text_size[0]) // 2
            text_y = btn_y + (btn_height + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        elif self.state == self.STATE_SHOWING_RESULTS:
            is_hover = (btn_x <= self.mouse_x <= btn_x + btn_width and 
                       btn_y <= self.mouse_y <= btn_y + btn_height)
            
            color = (200, 100, 0) if is_hover else (255, 0, 0)
            cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_width, btn_y + btn_height), 
                         color, -1)
            cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_width, btn_y + btn_height), 
                         (0, 0, 0), 2)
            
            text = "NUEVA"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = btn_x + (btn_width - text_size[0]) // 2
            text_y = btn_y + (btn_height + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        return frame
    
    def draw_status(self, frame):
        """Dibuja estado actual en la interfaz"""
        status_text = ""
        status_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        if self.state == self.STATE_SEARCHING:
            status_text = "BUSCANDO PERSONAS..."
            status_color = (0, 165, 255)
        elif self.state == self.STATE_PERSON_FOUND:
            status_text = "PERSONA DETECTADA - Haz clic en ANALIZAR"
            status_color = (0, 255, 0)
        elif self.state == self.STATE_ANALYZING:
            status_text = "ANALIZANDO 20 ASPECTOS..."
            status_color = (0, 255, 255)
        elif self.state == self.STATE_SHOWING_RESULTS:
            status_text = "ANALISIS COMPLETADO - Ver panel derecho"
            status_color = (0, 255, 0)
        
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), bg_color, -1)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), status_color, 3)
        
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, status_text, (text_x, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        
        return frame
    
    def draw_person_not_found(self, frame):
        """Dibuja mensaje de persona no detectada"""
        if self.state == self.STATE_SEARCHING and self.failed_detection_frames > 0:
            text = "PERSONA NO DETECTADA - Buscando..."
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = frame.shape[0] // 2 + 80
            
            cv2.rectangle(frame, (text_x - 10, text_y - 30), 
                         (text_x + text_size[0] + 10, text_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        return frame
    
    def run(self):
        """Ejecuta el agente IA interactivo"""
        print("\n" + "="*60)
        print("   AGENTE IA INTERACTIVO - DETECCIÓN BAJO DEMANDA")
        print("="*60)
        print(f"📍 Conectando a: {self.config['rtsp_url']}")
        print("\nCONTROLES:")
        print("  CLICK:   Analizar/Nueva detección (botones en pantalla)")
        print("  ESPACIO: Atajo rápido Analizar/Nueva")
        print("  'q':     Salir")
        print("  's':     Guardar resultados")
        print("="*60 + "\n")
        
        cap = self.connect_to_camera()
        
        if cap is None:
            print("\n❌ No se pudo conectar a la cámara")
            return False
        
        ui = ImprovedUI(
            frame_width=self.video_width,
            frame_height=self.video_height
        )
        
        window_name = 'Agente IA Interactivo'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        detection_thread.start()
        analysis_thread.start()
        
        detection_counter = 0
        person_found_counter = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    self.logger.log("Reconectando a la cámara...", "WARNING")
                    cap.release()
                    time.sleep(2)
                    cap = self.connect_to_camera()
                    if cap is None:
                        break
                    continue
                
                self.frame_count += 1
                detection_counter += 1
                person_found_counter += 1
                
                frame_display, x_offset, y_offset, scale = self._resize_frame_keep_aspect(
                    frame, self.video_width, self.video_height
                )
                frame_display = self.analyzer.pose_analyzer.draw_pose(frame_display, self.current_pose_data)
                
                detection_freq = 3 if self.state == self.STATE_PERSON_FOUND else 4
                
                if detection_counter >= detection_freq:
                    try:
                        self.detection_queue.put_nowait(frame_display.copy())
                    except:
                        pass
                    detection_counter = 0
                
                if self.state in [self.STATE_PERSON_FOUND, self.STATE_SHOWING_RESULTS]:
                    if self.current_person_bbox:
                        x, y, w, h = self.current_person_bbox
                        cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                frame_display = ui.add_stats(
                    frame_display, fps, self.person_count,
                    self.frame_count, elapsed
                )
                
                frame_display = self.draw_status(frame_display)
                frame_display = self.draw_person_not_found(frame_display)
                frame_display = self.draw_buttons(frame_display)
                
                cv2.imshow(window_name, frame_display)
                
                # Mostrar ventana de análisis en ventana separada
                if self.current_analysis and self.state == self.STATE_SHOWING_RESULTS:
                    analysis_window_name = 'Análisis de 20 Aspectos'
                    analysis_image = ui.create_analysis_window(self.current_analysis)
                    cv2.imshow(analysis_window_name, analysis_image)
                
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        self.logger.log("Ventana cerrada - terminando...")
                        break
                except:
                    pass
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    self.logger.log("Usuario presionó 'q' - terminando...")
                    break
                elif key == ord(' '):
                    if self.state == self.STATE_PERSON_FOUND:
                        self.analyze_button_clicked = True
                    elif self.state == self.STATE_SHOWING_RESULTS:
                        self.new_detection_clicked = True
                elif key == ord('s'):
                    self.logger.log("Usuario presionó 's' - guardando resultados...")
                    self.save_results()
                
                if self.analyze_button_clicked and self.state == self.STATE_PERSON_FOUND:
                    self.state = self.STATE_ANALYZING
                    self.logger.log("Iniciando análisis de 20 aspectos...")
                    try:
                        self.analysis_queue.put_nowait(
                            (self.detected_frame, self.current_frame_roi)
                        )
                    except:
                        self.logger.log("Error al enviar para análisis", "WARNING")
                    self.analyze_button_clicked = False
                
                if self.new_detection_clicked and self.state == self.STATE_SHOWING_RESULTS:
                    self.state = self.STATE_SEARCHING
                    self.current_analysis = None
                    self.current_person_bbox = None
                    self.detected_frame = None
                    self.current_pose_data = None
                    self.failed_detection_frames = 0
                    
                    # Cerrar ventana de análisis
                    try:
                        cv2.destroyWindow('Análisis de 20 Aspectos')
                    except:
                        pass
                    
                    self.logger.log("Buscando siguiente persona...")
                    self.new_detection_clicked = False
        
        except KeyboardInterrupt:
            self.logger.log("Interrupción del usuario", "WARNING")
        except Exception as e:
            import traceback
            self.logger.log(f"Error: {e}", "ERROR")
            self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
        finally:
            self.running = False
            detection_thread.join(timeout=2)
            analysis_thread.join(timeout=2)
            cap.release()
            cv2.destroyAllWindows()
            
            self.save_results()
            self.print_statistics()
    
    def save_results(self):
        """Guarda resultados"""
        if self.all_analyses:
            if self.config.get('save_json'):
                self.data_manager.save_json(self.all_analyses)
            if self.config.get('save_csv'):
                self.data_manager.save_csv(self.all_analyses)
            self.logger.log(f"✓ Resultados guardados")
    
    def print_statistics(self):
        """Imprime estadísticas finales"""
        elapsed = time.time() - self.start_time
        fps_avg = self.frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*60)
        print("ESTADÍSTICAS FINALES")
        print("="*60)
        print(f"Personas analizadas: {self.person_count}")
        print(f"Frames procesados: {self.frame_count}")
        print(f"FPS promedio: {fps_avg:.2f}")
        print(f"Tiempo total: {int(elapsed)} segundos")
        print("="*60 + "\n")

def main():
    """Función principal"""
    agente = AgenteIAInteractivo(config_file="config.json")
    agente.run()

if __name__ == "__main__":
    main()
