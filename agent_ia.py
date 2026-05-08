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

class AgenteIA:
    """Agente IA para análisis de 20 aspectos de personas"""
    
    def __init__(self, config_file="config.json"):
        self.config = self._load_config(config_file)
        self.analyzer = PersonAnalyzerMediaPipe()
        self.logger = Logger()
        self.data_manager = DataManager(self.config['output_folder'])
        
        self.start_time = time.time()
        self.frame_count = 0
        self.person_count = 0
        self.all_analyses = []
        
        # Sistema de tracking: mapea personas detectadas para evitar contarlas múltiples veces
        self.tracked_people = {}  # {person_id: {'bbox': (x,y,w,h), 'frames': N, 'analysis': {...}}}
        self.next_person_id = 0
        self.max_distance_threshold = 50  # Píxeles máximos de movimiento para ser misma persona
        
        # Threading para procesamiento paralelo
        self.detection_queue = Queue(maxsize=3)
        self.analysis_queue = Queue(maxsize=3)
        self.display_queue = Queue(maxsize=1)
        
        self.running = True
        self.last_analysis = []
        self.last_faces = []
        self.current_frame_display = None
        
        self.logger.log(f"Agente IA inicializado correctamente")
    
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
    
    def process_frame(self, frame):
        """Procesa un frame del video"""
        h, w = frame.shape[:2]
        new_w = self.config['frame_width']
        new_h = self.config['frame_height']
        frame = cv2.resize(frame, (new_w, new_h))
        
        faces = self.analyzer.detect_faces(frame)
        
        analyses = []
        new_people_in_frame = []
        
        for face in faces:
            try:
                # Validación extra de la estructura de face
                if 'bbox' not in face or 'type' not in face:
                    continue
                
                bbox_data = face['bbox']
                if len(bbox_data) != 4:
                    continue
                
                x, y, width, height = bbox_data
                face_roi = frame[y:y+height, x:x+width]
                confidence = face.get('confidence', 0.5)
                
                # Validar confianza mínima para evitar falsas detecciones
                if confidence < 0.5:
                    continue
                
                if face_roi.size > 0:
                    is_back = face.get('type') == 'back'
                    analysis = self.analyzer.analyze_person(
                        frame, face_roi, full_frame=frame, is_from_back=is_back, pose_data=face.get('pose')
                    )
                    analyses.append(analysis)
                    
                    # Tracking: buscar si es persona conocida o nueva
                    person_id = self._match_person_to_tracked((x, y, width, height))
                    
                    if person_id is None:
                        # Es una PERSONA NUEVA
                        person_id = self.next_person_id
                        self.next_person_id += 1
                        self.person_count += 1  # Contar solo personas NUEVAS
                        new_people_in_frame.append(person_id)
                    
                    # Actualizar tracking
                    self.tracked_people[person_id] = {
                        'bbox': (x, y, width, height),
                        'frames': self.tracked_people.get(person_id, {}).get('frames', 0) + 1,
                        'analysis': analysis
                    }
                    
                    self.all_analyses.append(analysis)
            except Exception as e:
                self.logger.log(f"Error procesando face: {e}", "WARNING")
                continue
        
        # Limpiar tracking de personas que no aparecen
        self._cleanup_tracked_people()
        
        frame = self.analyzer.draw_results(frame, faces, analyses)
        
        return frame, faces, analyses
    
    def _match_person_to_tracked(self, bbox):
        """Busca si el bbox coincide con alguna persona ya rastreada"""
        x, y, w, h = bbox
        bbox_center = (x + w // 2, y + h // 2)
        
        for person_id, data in self.tracked_people.items():
            tx, ty, tw, th = data['bbox']
            tracked_center = (tx + tw // 2, ty + th // 2)
            
            # Calcular distancia entre centros
            distance = self._calculate_distance(bbox_center, tracked_center)
            
            # Si está muy cerca = misma persona
            if distance < self.max_distance_threshold:
                return person_id
        
        return None  # No es persona conocida
    
    def _calculate_distance(self, point1, point2):
        """Calcula distancia euclidiana entre dos puntos"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _cleanup_tracked_people(self):
        """Elimina personas que no se detectan por varios frames"""
        frames_timeout = 30  # Si no se detecta por 30 frames, olvídale
        people_to_remove = [
            pid for pid, data in self.tracked_people.items()
            if (self.frame_count - data.get('last_seen', self.frame_count)) > frames_timeout
        ]
        
        for pid in people_to_remove:
            del self.tracked_people[pid]
    
    def _process_frames_worker(self):
        """Hilo de procesamiento de análisis sin bloquear el video"""
        while self.running:
            try:
                # Esperar máximo 0.1s por un frame
                frame_data = self.analysis_queue.get(timeout=0.1)
                if frame_data is None:
                    break
                
                frame, person_id = frame_data
                try:
                    pose_data = self.analyzer.pose_analyzer.detect_pose(frame)
                    analysis = self.analyzer.analyze_person(
                        frame, frame, full_frame=frame, is_from_back=False, pose_data=pose_data
                    )
                    self.last_analysis = [analysis]
                    self.all_analyses.append(analysis)
                    
                    # Actualizar tracking
                    if person_id is not None:
                        if person_id not in self.tracked_people:
                            self.person_count += 1
                        self.tracked_people[person_id] = {
                            'bbox': (0, 0, frame.shape[1], frame.shape[0]),
                            'frames': 1,
                            'analysis': analysis,
                            'last_seen': self.frame_count
                        }
                except Exception as e:
                    self.logger.log(f"Error en hilo de procesamiento: {e}", "WARNING")
            except:
                continue
    
    def _detection_worker(self):
        """Hilo para detección de caras sin bloquear"""
        while self.running:
            try:
                frame_data = self.detection_queue.get(timeout=0.1)
                if frame_data is None:
                    break
                
                frame = frame_data
                try:
                    faces = self.analyzer.detect_faces(frame)
                    self.last_faces = faces
                    
                    # Enviar caras detectadas para análisis
                    for idx, face in enumerate(faces):
                        try:
                            self.analysis_queue.put_nowait((frame, idx))
                        except:
                            pass
                except Exception as e:
                    self.logger.log(f"Error en detección: {e}", "WARNING")
            except:
                continue
    
    def save_results(self):
        """Guarda resultados"""
        if self.all_analyses:
            if self.config.get('save_json'):
                self.data_manager.save_json(self.all_analyses)
            if self.config.get('save_csv'):
                self.data_manager.save_csv(self.all_analyses)
            self.logger.log(f"Resultados guardados")
    
    def print_statistics(self):
        """Imprime estadísticas finales"""
        elapsed = time.time() - self.start_time
        fps_avg = self.frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*60)
        print("ESTADÍSTICAS FINALES")
        print("="*60)
        print(f"Personas detectadas: {self.person_count}")
        print(f"Frames procesados: {self.frame_count}")
        print(f"FPS promedio: {fps_avg:.2f}")
        print(f"Tiempo total: {int(elapsed)} segundos")
        print("="*60 + "\n")
    
# Reemplazar el método run() en agent_ia.py con esta versión corregida

    def run(self):
        """Ejecuta el agente IA en tiempo real con manejo de errores mejorado"""
        print("\n" + "="*60)
        print("   AGENTE IA - ANÁLISIS DE 20 ASPECTOS DE PERSONAS")
        print("="*60)
        print(f"📍 Conectando a: {self.config.get('rtsp_url', 'Cámara por defecto')}")
        print("Presiona 'q' para salir, 's' para guardar resultados")
        print("="*60 + "\n")
        
        # Validar configuración
        if not self.config:
            self.logger.log("Configuración vacía o no encontrada", "ERROR")
            return False
        
        cap = self.connect_to_camera()
        
        if cap is None:
            print("\n❌ No se pudo conectar a la cámara")
            return False
        
        # Inicializar UI con dimensiones de la configuración
        frame_width = self.config.get('frame_width', 640)
        frame_height = self.config.get('frame_height', 480)
        
        try:
            ui = ImprovedUI(
                frame_width=frame_width,
                frame_height=frame_height
            )
        except Exception as e:
            self.logger.log(f"Error inicializando UI: {e}", "ERROR")
            cap.release()
            return False
        
        # Iniciar hilos de procesamiento
        detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        analysis_thread = threading.Thread(target=self._process_frames_worker, daemon=True)
        detection_thread.start()
        analysis_thread.start()
        
        detection_counter = 0
        frame_display = None  # Inicializar para evitar errores
        
        try:
            while self.running:
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    self.logger.log("Frame no válido, reconectando...", "WARNING")
                    cap.release()
                    time.sleep(2)
                    cap = self.connect_to_camera()
                    if cap is None:
                        self.logger.log("No se pudo reconectar", "ERROR")
                        break
                    continue
                
                # Validar frame
                if frame.size == 0:
                    self.logger.log("Frame vacío, saltando...", "WARNING")
                    continue
                
                # ✓ Incrementar SOLO cuando el frame es válido
                self.frame_count += 1
                detection_counter += 1
                
                # Redimensionar RÁPIDO
                h, w = frame.shape[:2]
                new_w = self.config.get('frame_width', 640)
                new_h = self.config.get('frame_height', 480)
                
                try:
                    frame_display = cv2.resize(frame, (new_w, new_h))
                except Exception as e:
                    self.logger.log(f"Error redimensionando frame: {e}", "WARNING")
                    continue
                
                # Enviar a detección cada 6 frames (sin bloquear)
                if detection_counter >= 6:
                    try:
                        # Limpiar queue si está llena
                        if self.detection_queue.full():
                            try:
                                self.detection_queue.get_nowait()
                            except:
                                pass
                        self.detection_queue.put_nowait(frame_display.copy())
                    except Exception as e:
                        self.logger.log(f"Error enviando a queue: {e}", "WARNING")
                    detection_counter = 0
                
                # Calcular FPS
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                # Dibujar caras detectadas (con validación)
                if self.last_faces and frame_display is not None:
                    for face in self.last_faces:
                        try:
                            bbox_data = face.get('bbox')
                            if bbox_data and len(bbox_data) == 4:
                                x, y, w, h = bbox_data
                                # Validar que bbox esté dentro del frame
                                if (0 <= x < frame_display.shape[1] and 
                                    0 <= y < frame_display.shape[0] and 
                                    x + w <= frame_display.shape[1] and 
                                    y + h <= frame_display.shape[0]):
                                    cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    cv2.putText(frame_display, "Detectado", (x, y-5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        except Exception as e:
                            self.logger.log(f"Error dibujando cara: {e}", "WARNING")
                
                # Agregar estadísticas
                try:
                    frame_display = ui.add_stats(
                        frame_display, fps, self.person_count,
                        self.frame_count, elapsed
                    )
                except Exception as e:
                    self.logger.log(f"Error agregando stats: {e}", "WARNING")
                
                # Agregar panel de análisis (sin esperar)
                if self.last_analysis:
                    try:
                        panel = ui.create_info_panel(self.last_analysis[0])
                        frame_display = ui.combine_frames(frame_display, panel)
                    except Exception as e:
                        self.logger.log(f"Error creando panel: {e}", "WARNING")
                
                # Mostrar video
                if self.config.get('display_video', True) and frame_display is not None:
                    try:
                        cv2.imshow('Agente IA - Análisis de 20 Aspectos', frame_display)
                    except Exception as e:
                        self.logger.log(f"Error mostrando video: {e}", "ERROR")
                        break
                
                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.log("Usuario presionó 'q' - terminando...")
                    break
                elif key == ord('s'):
                    self.logger.log("Usuario presionó 's' - guardando resultados...")
                    self.save_results()
                elif key == ord('r'):
                    self.logger.log("Reset de estadísticas")
                    self.person_count = 0
                    self.frame_count = 0
                    self.start_time = time.time()
        
        except KeyboardInterrupt:
            self.logger.log("Interrupción del usuario", "WARNING")
        except Exception as e:
            import traceback
            self.logger.log(f"Error crítico en loop principal: {e}", "ERROR")
            self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
        finally:
            # Limpiar correctamente
            self.logger.log("Cerrando sistema...", "INFO")
            self.running = False
            
            # Detener hilos
            try:
                self.detection_queue.put(None)
                self.analysis_queue.put(None)
            except:
                pass
            
            detection_thread.join(timeout=2)
            analysis_thread.join(timeout=2)
            
            # Liberar recursos
            try:
                cap.release()
                cv2.destroyAllWindows()
            except:
                pass
            
            # Guardar resultados finales
            self.save_results()
            self.print_statistics()
            
        return True

def main():
    """Función principal"""
    agente = AgenteIA(config_file="config.json")
    agente.run()

if __name__ == "__main__":
    main()
