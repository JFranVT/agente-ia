"""
Agente IA - MODO VERTICAL FLUIDO CON PANEL LATERAL
Optimizado con hilos paralelos + Panel de 25 aspectos
"""

import cv2
import json
import numpy as np
from datetime import datetime
import time
import threading
from queue import Queue, Empty

from analyzer_mediapipe import PersonAnalyzerMediaPipe
from utils_Version import DataManager, Logger
from ui_improved import ImprovedUI

class AgenteIAVertical:
    """Agente IA vertical con procesamiento paralelo y panel lateral"""
    
    def __init__(self, config_file="config.json"):
        self.config = self._load_config(config_file)
        self.logger = Logger()
        self.data_manager = DataManager(self.config.get('output_folder', 'resultados'))
        
        self.start_time = time.time()
        self.frame_count = 0
        self.person_count = 0
        self.all_analyses = []
        
        # Modo vertical
        self.vertical_mode = self.config.get('vertical_mode', True)
        self.rotation_angle = self.config.get('rotation_angle', 90)
        
        # Resolución
        if self.vertical_mode:
            self.frame_width = self.config.get('frame_height', 480)
            self.frame_height = self.config.get('frame_width', 640)
        else:
            self.frame_width = self.config.get('frame_width', 640)
            self.frame_height = self.config.get('frame_height', 480)
        
        # Colas para paralelismo
        self.raw_queue = Queue(maxsize=2)
        self.display_queue = Queue(maxsize=2)
        self.detection_queue = Queue(maxsize=1)
        
        self.running = True
        self.last_faces = []
        self.last_analyses = []
        self.analyzer = None
        
        self.frame_time = 1.0 / 30
        
        self.logger.log(f"✓ Agente IA Vertical + Panel: {self.frame_width}x{self.frame_height}")
    
    def _load_config(self, config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            config.setdefault('vertical_mode', True)
            config.setdefault('rotation_angle', 90)
            return config
        except:
            return {}
    
    def connect_to_camera(self):
        rtsp_url = self.config.get('rtsp_url', 0)
        try:
            if isinstance(rtsp_url, str) and rtsp_url.startswith('http'):
                cap = cv2.VideoCapture(rtsp_url)
            else:
                cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(1)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.logger.log(f"✓ Cámara conectada: {frame.shape[1]}x{frame.shape[0]}")
                    return cap
            return None
        except Exception as e:
            self.logger.log(f"Error: {e}", "ERROR")
            return None
    
    def capture_worker(self, cap):
        """Hilo 1: Captura"""
        while self.running and cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                
                if self.vertical_mode:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
                if self.raw_queue.full():
                    try:
                        self.raw_queue.get_nowait()
                    except:
                        pass
                self.raw_queue.put(frame)
            except:
                continue
    
    def process_worker(self):
        """Hilo 2: Procesamiento"""
        while self.running:
            try:
                frame = self.raw_queue.get(timeout=0.01)
                self.frame_count += 1
                
                frame_resized = cv2.resize(frame, (self.frame_width, self.frame_height))
                
                if self.display_queue.full():
                    try:
                        self.display_queue.get_nowait()
                    except:
                        pass
                self.display_queue.put(frame_resized)
                
                if self.frame_count % 3 == 0 and self.detection_queue.empty():
                    self.detection_queue.put_nowait(frame_resized.copy())
                
            except Empty:
                continue
    
    def detection_worker(self):
        """Hilo 3: Solo detecta cuando hay un rostro real"""
        self.analyzer = PersonAnalyzerMediaPipe()
        
        # Estado: buscando o siguiendo
        self.modo = 'buscando'  # 'buscando' o 'siguiendo'
        self.tracked_person = None
        self.frames_perdidos = 0
        self.max_perdidos = 20  # Frames sin ver antes de volver a buscar
        
        # Para evitar falsos positivos
        self.detecciones_seguidas = 0
        self.min_detecciones = 3  # Debe detectar 3 veces seguidas para confirmar
        
        while self.running:
            try:
                frame = self.detection_queue.get(timeout=0.1)
                
                faces = self.analyzer.detect_faces(frame)
                
                if self.modo == 'buscando':
                    # === MODO BÚSQUEDA: Esperar un rostro confirmado ===
                    if faces:
                        self.detecciones_seguidas += 1
                        
                        # Necesita detectar varias veces seguidas para confirmar
                        if self.detecciones_seguidas >= self.min_detecciones:
                            # ¡Persona confirmada!
                            face = faces[0]  # Primera cara
                            bbox = face.get('bbox')
                            
                            if bbox and len(bbox) == 4:
                                x, y, w, h = bbox
                                face_roi = frame[y:y+h, x:x+w]
                                
                                if face_roi.size > 0:
                                    analysis = self.analyzer.analyze_person(
                                        frame, face_roi, full_frame=frame,
                                        is_from_back=(face.get('type') == 'back'),
                                        pose_data=face.get('pose')
                                    )
                                    
                                    self.last_faces = [face]
                                    self.last_analyses = [analysis]
                                    self.all_analyses.append(analysis)
                                    self.tracked_person = (x, y, w, h)
                                    self.frames_perdidos = 0
                                    self.person_count += 1
                                    self.modo = 'siguiendo'
                                    
                                    self.logger.log("✓ Persona detectada y confirmada")
                    else:
                        # Resetear contador si no hay detección
                        self.detecciones_seguidas = 0
                        self.last_faces = []
                        self.last_analyses = []
                
                else:
                    # === MODO SIGUIENDO: Ya hay una persona ===
                    if faces:
                        # Buscar la cara más cercana a la última posición
                        best_face = None
                        best_dist = float('inf')
                        
                        for face in faces:
                            bbox = face.get('bbox')
                            if not bbox or len(bbox) != 4:
                                continue
                            
                            fx, fy, fw, fh = bbox
                            fc = (fx + fw//2, fy + fh//2)
                            
                            if self.tracked_person:
                                tx, ty, tw, th = self.tracked_person
                                tc = (tx + tw//2, ty + th//2)
                                dist = ((fc[0]-tc[0])**2 + (fc[1]-tc[1])**2)**0.5
                                
                                if dist < best_dist:
                                    best_dist = dist
                                    best_face = face
                        
                        # Si encontró la misma persona
                        if best_face is not None and best_dist < 100:
                            bbox = best_face.get('bbox')
                            x, y, w, h = bbox
                            face_roi = frame[y:y+h, x:x+w]
                            
                            if face_roi.size > 0:
                                analysis = self.analyzer.analyze_person(
                                    frame, face_roi, full_frame=frame,
                                    is_from_back=(best_face.get('type') == 'back'),
                                    pose_data=best_face.get('pose')
                                )
                                
                                self.last_faces = [best_face]
                                self.last_analyses = [analysis]
                                self.all_analyses.append(analysis)
                                self.tracked_person = (x, y, w, h)
                                self.frames_perdidos = 0
                        
                        self.frames_perdidos = 0
                    else:
                        # No detecta nada
                        self.frames_perdidos += 1
                        
                        if self.frames_perdidos > self.max_perdidos:
                            # Se perdió la persona, volver a buscar
                            self.modo = 'buscando'
                            self.detecciones_seguidas = 0
                            self.tracked_person = None
                            self.last_faces = []
                            self.last_analyses = []
                            self.logger.log("Persona perdida - Buscando de nuevo...")
                    
            except Empty:
                continue
    
    def run(self):
        """Ejecuta el sistema con PANEL LATERAL VISIBLE"""
        print("\n" + "="*60)
        print("   AGENTE IA - VERTICAL FLUIDO + PANEL")
        print("="*60)
        print(f"📐 Video: {self.frame_width}x{self.frame_height}")
        print("Controles: q=salir | s=guardar")
        print("="*60 + "\n")
        
        cap = self.connect_to_camera()
        if cap is None:
            return False
        
        # Hilos
        threading.Thread(target=self.capture_worker, args=(cap,), daemon=True).start()
        threading.Thread(target=self.process_worker, daemon=True).start()
        threading.Thread(target=self.detection_worker, daemon=True).start()
        
        # UI
        ui = ImprovedUI(frame_width=self.frame_width, frame_height=self.frame_height)
        
        fps_display = 0
        fps_counter = 0
        fps_timer = time.time()
        
        try:
            while self.running:
                try:
                    frame_display = self.display_queue.get(timeout=0.01)
                    
                    # FPS
                    fps_counter += 1
                    if time.time() - fps_timer >= 1.0:
                        fps_display = fps_counter
                        fps_counter = 0
                        fps_timer = time.time()
                    
                    # 🟢 DIBUJAR RESULTADOS COMPLETOS (bbox, P1, género, edad, pose)
                    if self.last_faces and self.last_analyses:
                        frame_display = self.analyzer.draw_results(
                            frame_display, 
                            self.last_faces, 
                            self.last_analyses
                        )
                    
                    # Estadísticas
                    elapsed = time.time() - self.start_time
                    frame_display = ui.add_stats(frame_display, fps_display, self.person_count,
                                                self.frame_count, elapsed)
                    
                    # 🔵 PANEL LATERAL
                    if self.last_analyses:
                        panel = ui.create_info_panel(self.last_analyses[0])
                        frame_display = ui.combine_frames(frame_display, panel)
                    else:
                        panel_vacio = np.ones((self.frame_height, ui.panel_width, 3), dtype=np.uint8) * 40
                        cv2.putText(panel_vacio, "Esperando", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                        cv2.putText(panel_vacio, "deteccion...", (30, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                        frame_display = ui.combine_frames(frame_display, panel_vacio)
                    
                    # Mostrar
                    cv2.imshow('Agente IA - Panel Completo', frame_display)
                    
                except Empty:
                    pass
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_results()
        
        finally:
            self.running = False
            time.sleep(0.3)
            cap.release()
            cv2.destroyAllWindows()
            self.save_results()
            self.print_statistics()

    def save_results(self):
        if self.all_analyses:
            self.data_manager.save_json(self.all_analyses)
    
    def print_statistics(self):
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        print(f"\n📊 FPS: {fps:.1f} | Personas: {self.person_count}")

def main():
    agente = AgenteIAVertical(config_file="config.json")
    agente.run()

if __name__ == "__main__":
    main()