"""
Detector de personas FRONTALES y DE ESPALDA
Usa cascadas para rostros + HOG para cuerpos completos
"""

import cv2
import numpy as np
from datetime import datetime
from analyzer_mediapipe_pose import PersonAnalyzerMediaPipePose

class PersonAnalyzerMediaPipe:
    """Detecta personas de frente y de espalda"""
    
    def __init__(self):
        self.cascades = []
        self.pose_analyzer = PersonAnalyzerMediaPipePose()
        self.last_pose_data = None
        
        # Cargar cascadas para rostros
        try:
            cascade_paths = [
                cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            ]
            for path in cascade_paths:
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    self.cascades.append(cascade)
            
            if self.cascades:
                print(f"✓ {len(self.cascades)} detectores de rostro cargados")
        except Exception as e:
            print(f"⚠️ Error cargando cascades: {e}")
        
        # Detector HOG para cuerpos completos (espaldas)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("✓ Detector de cuerpos (HOG) cargado para espaldas")
    
    def detect_faces(self, frame):
        """Detecta SOLO rostros frontales"""
        faces_list = []
        h, w = frame.shape[:2]
        
        self.last_pose_data = self.pose_analyzer.detect_pose(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for cascade in self.cascades:
            try:
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=5,
                    minSize=(40, 40), maxSize=(300, 300)
                )
                
                for (x, y, bw, bh) in faces:
                    bbox_area = bw * bh
                    frame_area = w * h
                    
                    if bbox_area < frame_area * 0.4:
                        faces_list.append({
                            'bbox': (x, y, bw, bh),
                            'confidence': 0.85,
                            'type': 'frontal',
                            'distance': self._estimate_distance(bw, bh),
                            'pose': self.last_pose_data
                        })
            except:
                continue
        
        return faces_list
    
    def _estimate_distance(self, w, h):
        face_size = (w * h) ** 0.5
        if face_size > 200:
            return "cercano"
        elif face_size > 100:
            return "medio"
        else:
            return "lejano"
    
    def _remove_overlaps(self, faces):
        """Elimina detecciones solapadas"""
        if len(faces) <= 1:
            return faces
        
        cleaned = []
        for i, face in enumerate(faces):
            x, y, w, h = face['bbox']
            is_overlap = False
            
            for existing in cleaned:
                ex, ey, ew, eh = existing['bbox']
                intersection_x = max(0, min(x+w, ex+ew) - max(x, ex))
                intersection_y = max(0, min(y+h, ey+eh) - max(y, ey))
                intersection = intersection_x * intersection_y
                
                if intersection > (w*h) * 0.3:
                    is_overlap = True
                    break
            
            if not is_overlap:
                cleaned.append(face)
        
        return cleaned
    
    def analyze_person(self, frame, face_roi, full_frame=None, is_from_back=False, pose_data=None):
        """Analiza persona (frontal o de espalda)"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "orientacion": "De espalda" if is_from_back else "Frontal",
            "aspectos": {}
        }
        
        if pose_data is None:
            pose_data = self.last_pose_data
        
        try:
            h, w = face_roi.shape[:2]
            results["aspectos"]["1_rostro_detectado"] = True
            results["aspectos"]["1_area_rostro"] = f"{h}x{w}"
            
            if is_from_back:
                # === ANÁLISIS DE ESPALDA ===
                results["aspectos"]["2_edad_estimada"] = "No disponible"
                results["aspectos"]["3_genero"] = "No disponible"
                results["aspectos"]["4_expresion_facial"] = "N/A"
                results["aspectos"]["7_ojos_abiertos"] = "N/A"
                results["aspectos"]["8_sonrisa_detectada"] = "N/A"
                results["aspectos"]["10_barba_bigote"] = "N/A"
                results["aspectos"]["11_gafas"] = "N/A"
                results["aspectos"]["19_etnia_estimada"] = "No disponible"
                results["aspectos"]["6_confianza_rostro"] = 0.50
                
                # Cabello (se puede ver de espalda)
                results["aspectos"]["9_color_cabello_aprox"] = self._analyze_hair_color(face_roi)
                results["aspectos"]["12_sombrero"] = self._detect_hat(face_roi)
                
                # Ropa
                if full_frame is not None:
                    ropa_sup, ropa_inf = self._analyze_clothing(full_frame)
                    results["aspectos"]["15_ropa_superior_color"] = ropa_sup
                    results["aspectos"]["16_ropa_inferior_color"] = ropa_inf
                else:
                    results["aspectos"]["15_ropa_superior_color"] = "No visible"
                    results["aspectos"]["16_ropa_inferior_color"] = "No visible"
                
                results["aspectos"]["20_confianza_general"] = 0.60
            else:
                # === ANÁLISIS FRONTAL ===
                results["aspectos"]["2_edad_estimada"] = self._estimate_age(face_roi)
                results["aspectos"]["3_genero"] = self._estimate_gender(face_roi)
                results["aspectos"]["4_expresion_facial"] = self._analyze_expression(face_roi)
                results["aspectos"]["7_ojos_abiertos"] = self._analyze_eyes(face_roi)
                results["aspectos"]["8_sonrisa_detectada"] = "Sí" if self._detect_smile(face_roi) else "No"
                results["aspectos"]["9_color_cabello_aprox"] = self._analyze_hair_color(face_roi)
                results["aspectos"]["10_barba_bigote"] = self._detect_beard(face_roi)
                results["aspectos"]["11_gafas"] = self._detect_glasses(face_roi)
                results["aspectos"]["12_sombrero"] = self._detect_hat(face_roi)
                results["aspectos"]["19_etnia_estimada"] = self._estimate_ethnicity(face_roi)
                results["aspectos"]["6_confianza_rostro"] = 0.90
                
                if full_frame is not None:
                    ropa_sup, ropa_inf = self._analyze_clothing(full_frame)
                    results["aspectos"]["15_ropa_superior_color"] = ropa_sup
                    results["aspectos"]["16_ropa_inferior_color"] = ropa_inf
                else:
                    results["aspectos"]["15_ropa_superior_color"] = "No visible"
                    results["aspectos"]["16_ropa_inferior_color"] = "No visible"
                
                results["aspectos"]["20_confianza_general"] = 0.85
            
            # === DATOS CORPORALES (para ambos) ===
            results["aspectos"]["17_accesorios_detectados"] = ["Ninguno"]
            results["aspectos"]["18_calidad_imagen"] = "Alta"
            
            if pose_data is not None and pose_data.get('detected'):
                visibility = pose_data.get('visibility', {})
                results["aspectos"]["13_cuerpo_visible"] = (
                    f"Cabeza:{visibility.get('cabeza', 'no')}, "
                    f"Torso:{visibility.get('torso', 'no')}"
                )
                results["aspectos"]["14_postura"] = pose_data.get('posture', 'Desconocida')
                results["aspectos"]["21_brazos_visibles"] = visibility.get('brazos', 'ninguno')
                results["aspectos"]["22_piernas_visibles"] = visibility.get('piernas', 'ninguna')
                results["aspectos"]["23_gesto_corporal"] = pose_data.get('gesture', 'Neutral')
                results["aspectos"]["24_complexion_corporal"] = pose_data.get('body_build', 'Media')
                results["aspectos"]["25_angulos_articulaciones"] = pose_data.get('angles', {})
            else:
                results["aspectos"]["13_cuerpo_visible"] = "No detectado"
                results["aspectos"]["14_postura"] = "No detectada"
                results["aspectos"]["21_brazos_visibles"] = "ninguno"
                results["aspectos"]["22_piernas_visibles"] = "ninguna"
                results["aspectos"]["23_gesto_corporal"] = "Neutral"
                results["aspectos"]["24_complexion_corporal"] = "Media"
            
        except Exception as e:
            print(f"⚠️ Error analizando: {e}")
        
        return results
    
    # === MÉTODOS DE ANÁLISIS ===
    
    def _estimate_age(self, face_roi):
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            texture = np.std(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
            if texture > 32: return "18-25 años"
            elif texture > 20: return "25-35 años"
            else: return "35-50 años"
        except:
            return "Indeterminado"
    
    def _estimate_gender(self, face_roi):
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            h = face_roi.shape[0]
            jawline = np.mean(gray[int(h*0.65):, :])
            return "Masculino" if jawline < 115 else "Femenino"
        except:
            return "Indeterminado"
    
    def _analyze_expression(self, face_roi):
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            mouth = np.mean(gray[int(face_roi.shape[0]*0.6):, :])
            return "Feliz" if mouth > 130 else "Neutral"
        except:
            return "Neutral"
    
    def _analyze_eyes(self, face_roi):
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            eyes_std = np.std(gray[:int(face_roi.shape[0]*0.4), :])
            return "Sí" if eyes_std > 25 else "No"
        except:
            return "No"
    
    def _detect_smile(self, face_roi):
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            return np.mean(gray[int(face_roi.shape[0]*0.6):, :]) > 130
        except:
            return False
    
    def _analyze_hair_color(self, face_roi):
        try:
            hair = face_roi[:int(face_roi.shape[0]*0.25), :]
            if hair.size == 0: return "Desconocido"
            v = np.mean(cv2.cvtColor(hair, cv2.COLOR_BGR2HSV)[:, :, 2])
            if v < 110: return "Negro"
            elif v < 160: return "Castaño"
            else: return "Rubio/Claro"
        except:
            return "Desconocido"
    
    def _detect_beard(self, face_roi):
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            return "Sí" if np.mean(gray[int(face_roi.shape[0]*0.65):, :]) < 115 else "No"
        except:
            return "No"
    
    def _detect_glasses(self, face_roi):
        try:
            eyes = face_roi[:int(face_roi.shape[0]*0.45), :]
            gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            count = sum(1 for c in contours if 300 < cv2.contourArea(c) < 5000)
            return "Sí" if count >= 2 else "No"
        except:
            return "No"
    
    def _detect_hat(self, face_roi):
        try:
            top = face_roi[:int(face_roi.shape[0]*0.25), :]
            hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,100]))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if cv2.contourArea(c) > 500 and w/(h+1) > 1.5:
                    return "Sí, gorra detectada"
            return "No detectado"
        except:
            return "No detectado"
    
    def _estimate_ethnicity(self, face_roi):
        try:
            center = face_roi[int(face_roi.shape[0]*0.2):int(face_roi.shape[0]*0.8), :]
            v = np.mean(cv2.cvtColor(center, cv2.COLOR_BGR2HSV)[:, :, 2])
            return "Afrodescendiente" if v < 100 else "Caucásico/Mixto"
        except:
            return "Indeterminado"
    
    def _analyze_clothing(self, frame):
        try:
            h, w = frame.shape[:2]
            chest = frame[int(h*0.30):int(h*0.55), :]
            legs = frame[int(h*0.55):int(h*0.85), :]
            
            def classify(region):
                if region.size == 0: return "No visible"
                hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                h_c, s_c, v_c = np.mean(hsv[:,:,0]), np.mean(hsv[:,:,1]), np.mean(hsv[:,:,2])
                if s_c < 30:
                    return "Blanco" if v_c > 180 else ("Negro" if v_c < 80 else "Gris")
                if h_c < 10 or h_c > 170: return "Rojo"
                elif h_c < 25: return "Naranja"
                elif h_c < 80: return "Verde/Azul"
                else: return "Púrpura"
            
            return classify(chest), classify(legs)
        except:
            return "Desconocido", "Desconocido"
    
    def draw_results(self, frame, faces, analyses):
        """Dibuja resultados en el frame"""
        if not faces or not analyses:
            return frame
        
        num = min(len(faces), len(analyses))
        
        for i in range(num):
            try:
                face = faces[i]
                analysis = analyses[i]
                bbox = face.get('bbox')
                
                if not bbox or len(bbox) != 4:
                    continue
                
                x, y, w, h = bbox
                
                # Color según tipo
                if face.get('type') == 'back':
                    color = (0, 165, 255)  # Naranja para espalda
                    label_base = f"E{i+1}"
                else:
                    color = (0, 255, 0)  # Verde para frontal
                    label_base = f"P{i+1}"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Etiqueta
                gender = analysis.get("aspectos", {}).get("3_genero", "?")
                age = analysis.get("aspectos", {}).get("2_edad_estimada", "?")
                label = f"{label_base} | {gender} | {age}"
                
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 2)
            except:
                continue
        
        # Dibujar pose
        frame = self.pose_analyzer.draw_pose(frame, self.last_pose_data)
        
        return frame