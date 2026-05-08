import cv2
import numpy as np
from datetime import datetime
from analyzer_mediapipe_pose import PersonAnalyzerMediaPipePose

class PersonAnalyzer:
    """Detecta personas frontales Y de espalda (por cabeza/cuerpo)"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        self.pose_analyzer = PersonAnalyzerMediaPipePose()
    
    def detect_head_back(self, frame):
        """Detecta cabeza de espalda usando cascadas Haar + filtros HSV mejorados"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_frame, w_frame = frame.shape[:2]
        
        # Mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        heads = []
        
        try:
            # Método 1: Cascadas Haar de perfil con dos niveles de sensibilidad
            # NIVEL 1: Más restrictivo (cabezas cercanas/medianas)
            profile_faces_strict = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40)
            )
            
            border_margin = 30
            for face_data in profile_faces_strict:
                try:
                    if len(face_data) != 4:
                        continue
                    x, y, w, h = face_data
                    aspect_ratio = w / h if h != 0 else 0
                    if (x > border_margin and 
                        (x + w) < (w_frame - border_margin) and 
                        y > border_margin and
                        0.7 < aspect_ratio < 1.4):
                        heads.append((x, y, w, h))
                except Exception:
                    continue
            
            # NIVEL 2: Más sensible (cabezas lejanas)
            profile_faces_loose = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
            )
            
            for face_data in profile_faces_loose:
                try:
                    if len(face_data) != 4:
                        continue
                    x, y, w, h = face_data
                    aspect_ratio = w / h if h != 0 else 0
                    # Evitar duplicados de NIVEL 1
                    is_duplicate = False
                    for (hx, hy, hw, hh) in heads:
                        if abs(x - hx) < 30 and abs(y - hy) < 30:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate and (x > border_margin and 
                        (x + w) < (w_frame - border_margin) and 
                        y > border_margin and
                        0.65 < aspect_ratio < 1.5 and
                        (w * h) > 300):  # Área mínima para evitar ruido
                        heads.append((x, y, w, h))
                except Exception:
                    continue
        except Exception:
            pass
        
        # Método 2: Detección HSV mejorada (solo para cabello realista)
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Rango HSV más restrictivo para cabello oscuro
            lower_hair_dark = np.array([0, 20, 15])
            upper_hair_dark = np.array([25, 255, 120])
            mask_dark = cv2.inRange(hsv, lower_hair_dark, upper_hair_dark)
            
            # Rango HSV para cabello castaño
            lower_hair_brown = np.array([10, 30, 25])
            upper_hair_brown = np.array([35, 220, 130])
            mask_brown = cv2.inRange(hsv, lower_hair_brown, upper_hair_brown)
            
            # Combinar máscaras
            mask = cv2.bitwise_or(mask_dark, mask_brown)
            
            # Aplicar erosión y dilatación para eliminar ruido
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                try:
                    bbox = cv2.boundingRect(contour)
                    if len(bbox) != 4:
                        continue
                    x, y, w, h = bbox
                    area = cv2.contourArea(contour)
                    
                    # Rangos más amplios para detectar a diferentes distancias
                    if (1000 < area < 20000 and 
                        0.6 < (w/h if h != 0 else 0) < 1.5 and
                        x > border_margin and 
                        (x + w) < (w_frame - border_margin) and 
                        y > border_margin):
                        
                        # Calcular solidity
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 0
                        
                        # Solidity más flexible
                        if solidity > 0.5:
                            # Evitar duplicados
                            is_duplicate = False
                            for (hx, hy, hw, hh) in heads:
                                if abs(x - hx) < 40 and abs(y - hy) < 40:
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                heads.append((x, y, w, h))
                except Exception:
                    continue
        except Exception:
            pass
        
        return heads
    
    def analyze_person(self, frame, face_roi, full_frame=None, is_from_back=False, pose_data=None):
        """Analiza persona (frontal o de espalda)"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "orientacion": "De espalda" if is_from_back else "Frontal",
            "aspectos": {}
        }
        if pose_data is None:
            pose_data = self.pose_analyzer.detect_pose(full_frame if full_frame is not None else frame)
        
        try:
            h, w = face_roi.shape[:2]
            results["aspectos"]["1_rostro_detectado"] = True
            results["aspectos"]["1_area_rostro"] = f"{h}x{w}"
            
            if is_from_back:
                results["aspectos"]["2_edad_estimada"] = "No disponible"
                results["aspectos"]["3_genero"] = "No disponible"
                results["aspectos"]["4_expresion_facial"] = "N/A"
                results["aspectos"]["7_ojos_abiertos"] = "N/A"
                results["aspectos"]["8_sonrisa_detectada"] = "N/A"
                results["aspectos"]["11_gafas"] = "N/A"
                color_cabello = self._analyze_hair_color_advanced(face_roi, h, w)
                results["aspectos"]["9_color_cabello_aprox"] = color_cabello
                
                if full_frame is not None:
                    ropa_superior, ropa_inferior = self._analyze_clothing(full_frame)
                    results["aspectos"]["15_ropa_superior_color"] = ropa_superior
                    results["aspectos"]["16_ropa_inferior_color"] = ropa_inferior
                else:
                    results["aspectos"]["15_ropa_superior_color"] = "No disponible"
                    results["aspectos"]["16_ropa_inferior_color"] = "No disponible"
                
                results["aspectos"]["13_cuerpo_visible"] = "De espalda"
                results["aspectos"]["14_postura"] = "De espalda"
                results["aspectos"]["20_confianza_general"] = 0.70
            else:
                results["aspectos"]["2_edad_estimada"] = self._estimate_age_advanced(face_roi, h, w)
                results["aspectos"]["3_genero"] = self._estimate_gender_advanced(face_roi, h, w)
                results["aspectos"]["4_expresion_facial"] = self._analyze_facial_expression(face_roi, h, w)
                
                # Analizar si los ojos estan abiertos por contraste, no por cascadas
                eye_region_h = int(h*0.4)
                if eye_region_h > 0:
                    eye_region = face_roi[:eye_region_h, :]
                    eyes_status = self._analyze_eyes_open_advanced(eye_region)
                else:
                    eyes_status = "No"
                results["aspectos"]["7_ojos_abiertos"] = eyes_status
                
                # Procesar sonrisa solo si hay región suficiente
                smile_region_start = int(h*0.4)
                if smile_region_start < h:
                    smile_region = face_roi[smile_region_start:, :]
                    if smile_region.size > 100:  # Validar región mínima
                        try:
                            smiles = self.smile_cascade.detectMultiScale(smile_region, 1.8, 20)
                            results["aspectos"]["8_sonrisa_detectada"] = len(smiles) > 0
                        except Exception:
                            results["aspectos"]["8_sonrisa_detectada"] = False
                    else:
                        results["aspectos"]["8_sonrisa_detectada"] = False
                else:
                    results["aspectos"]["8_sonrisa_detectada"] = False
                
                results["aspectos"]["9_color_cabello_aprox"] = self._analyze_hair_color_advanced(face_roi, h, w)
                results["aspectos"]["10_barba_bigote"] = self._detect_beard_advanced(face_roi, h, w)
                results["aspectos"]["11_gafas"] = self._detect_glasses_advanced(face_roi, h, w)
                results["aspectos"]["13_cuerpo_visible"] = "Parcialmente"
                results["aspectos"]["14_postura"] = self._analyze_posture(face_roi, h, w)
                results["aspectos"]["20_confianza_general"] = 0.85
            
            results["aspectos"]["6_confianza_rostro"] = 0.90
            results["aspectos"]["12_sombrero"] = self._detect_hat_visor(face_roi, h, w) if not is_from_back else "No visible"
            results["aspectos"]["17_accesorios_detectados"] = ["Ninguno"]
            results["aspectos"]["18_calidad_imagen"] = "Alta"
            results["aspectos"]["19_etnia_estimada"] = self._estimate_ethnicity(face_roi, h, w) if not is_from_back else "No disponible"
            
            if pose_data is not None:
                visibility = self.analyze_visibility(pose_data)
                results["aspectos"]["13_cuerpo_visible"] = (
                    f"Cabeza:{visibility.get('cabeza', 'no')}, "
                    f"Torso:{visibility.get('torso', 'no')}, "
                    f"Extremidades:{visibility.get('total_percent', 0)}%"
                )
                results["aspectos"]["14_postura"] = self.analyze_body_posture(pose_data)
                results["aspectos"]["21_brazos_visibles"] = visibility.get("brazos", "ninguno")
                results["aspectos"]["22_piernas_visibles"] = visibility.get("piernas", "ninguna")
                results["aspectos"]["23_gesto_corporal"] = self.analyze_body_gesture(pose_data)
                results["aspectos"]["24_complexion_corporal"] = pose_data.get("body_build", "Indeterminada")
                results["aspectos"]["25_angulos_articulaciones"] = pose_data.get("angles", {})
            
        except Exception as e:
            print(f"Error: {e}")
        
        return results
    
    def analyze_body_posture(self, pose_data):
        """Determina postura corporal usando keypoints de pose."""
        return pose_data.get("posture", "Desconocida") if pose_data else "Desconocida"
    
    def analyze_visibility(self, pose_data):
        """Visibilidad de cabeza, torso, brazos y piernas."""
        if not pose_data:
            return {"cabeza": "no", "torso": "no", "brazos": "ninguno", "piernas": "ninguna", "total_percent": 0.0}
        return pose_data.get("visibility", {"cabeza": "no", "torso": "no", "brazos": "ninguno", "piernas": "ninguna", "total_percent": 0.0})
    
    def analyze_body_gesture(self, pose_data):
        """Detecta gesto corporal principal."""
        return pose_data.get("gesture", "Neutral") if pose_data else "Neutral"
    
    def _estimate_age_advanced(self, face_roi, h, w):
        """Estima edad usando análisis robusto de múltiples características"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Feature 1: Textura/Arrugas (detecta edad avanzada)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            texture = np.mean(np.sqrt(sobelx**2 + sobely**2))
            
            # Feature 2: Brillo de piel (jóvenes = más claro)
            brightness = np.mean(gray)
            
            # Feature 3: Contraste (jóvenes = más contraste)
            contrast = np.std(gray)
            
            # Feature 4: Oscuridad bajo los ojos (indica edad avanzada)
            eye_region = gray[:int(h*0.4), :]
            eye_darkness = np.mean(eye_region)
            
            # SCORING equilibrado para jóvenes (25 años)
            age_features = []
            
            # Texture: alto = joven, bajo = viejo
            if texture > 32:
                age_features.append("joven")  # Mucho detalle = piel elástica
            elif texture < 20:
                age_features.append("viejo")   # Poco detalle = piel flácida
            else:
                age_features.append("medio")
            
            # Brightness: alto = joven
            if brightness > 120:
                age_features.append("joven")   # Piel clara = joven
            elif brightness < 100:
                age_features.append("viejo")   # Piel oscura/manchada = viejo
            else:
                age_features.append("medio")
            
            # Contrast: alto = joven
            if contrast > 35:
                age_features.append("joven")   # Alto contraste = joven
            elif contrast < 25:
                age_features.append("viejo")
            else:
                age_features.append("medio")
            
            # Eye darkness: bajo = joven (ojos sin bolsas)
            if eye_darkness > 130:
                age_features.append("joven")
            elif eye_darkness < 110:
                age_features.append("viejo")
            else:
                age_features.append("medio")
            
            # Contar votos
            joven_votes = age_features.count("joven")
            viejo_votes = age_features.count("viejo")
            
            # Decisión
            if joven_votes >= 3:
                return "18-25 años"
            elif joven_votes >= 2 and viejo_votes <= 1:
                return "25-35 años"
            elif viejo_votes >= 2 and joven_votes <= 1:
                return "50-65 años"
            elif viejo_votes >= 3:
                return "65+ años"
            else:
                return "35-50 años"
        except:
            return "Indeterminado"
    
    def _estimate_gender_advanced(self, face_roi, h, w):
        """Estima género usando múltiples características"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Feature 1: Intensidad de mandíbula (hombres más oscuro)
        jawline_region = gray[int(h*0.65):, :]
        jawline_intensity = np.mean(jawline_region)
        
        # Feature 2: Textura de piel (hombres más textura/poros)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        skin_texture = np.mean(np.sqrt(sobelx**2 + sobely**2))
        
        # Feature 3: Contraste general (hombres más contraste)
        contrast = np.std(gray)
        
        # Feature 4: Forehead region (frente - hombres suelen tener más luz/menos cabello)
        forehead = gray[:int(h*0.25), :]
        forehead_brightness = np.mean(forehead)
        
        # Scoring - PESOS EQUILIBRADOS
        masculinity_score = 0
        
        # Jawline: hombres < 110 (más oscuro)
        if jawline_intensity < 115:
            masculinity_score += 2
        elif jawline_intensity < 130:
            masculinity_score += 1
        else:
            masculinity_score -= 1
            
        # Texture: hombres > 30 (más poros)
        if skin_texture > 32:
            masculinity_score += 2
        elif skin_texture > 25:
            masculinity_score += 1
        else:
            masculinity_score -= 1
            
        # Contrast: hombres > 28
        if contrast > 30:
            masculinity_score += 1
        
        # Forehead brightness: hombres > 130 (frente más clara, menos cabello)
        if forehead_brightness > 135:
            masculinity_score += 1
        elif forehead_brightness < 110:
            masculinity_score -= 1
        
        return "Masculino" if masculinity_score >= 2 else "Femenino"
    
    def _analyze_facial_expression(self, face_roi, h, w):
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        mouth_region = gray[int(h*0.6):, :]
        mouth_intensity = np.mean(mouth_region)
        return "Feliz" if mouth_intensity > 130 else "Neutral"
    
    def _analyze_hair_color_advanced(self, face_roi, h, w):
        """Analiza color de cabello - INTELIGENTE para ROI de cualquier tamaño"""
        try:
            # Detectar si el ROI es frame completo o anormalmente grande
            # ROI normal de rostro: h 80-300, w 60-250
            is_huge_roi = (h > 400 or w > 500)
            
            if is_huge_roi:
                # Probablemente frame completo
                # Analizar región superior-central donde típicamente está el cabello
                # Zona de cabello: 0-20% del alto, centro horizontal
                start_y = 0
                end_y = int(h * 0.20)
                start_x = int(w * 0.2)
                end_x = int(w * 0.8)
                
                if end_y <= start_y or end_x <= start_x:
                    return "Desconocido"
                
                hair_region = face_roi[start_y:end_y, start_x:end_x]
            else:
                # ROI normal - analizar top 25%
                hair_region = face_roi[:int(h*0.25), :]
            
            if hair_region.size == 0:
                return "Desconocido"
            
            # Convertir a HSV
            hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
            h_channel = hsv[:, :, 0]
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]
            
            avg_h = np.mean(h_channel)
            avg_s = np.mean(s_channel)
            avg_v = np.mean(v_channel)
            
            # UMBRALES FINALES - Optimizados para usuario con cabello oscuro
            
            # Negro: V < 120 y S < 75
            negro_mask = (v_channel < 120) & (s_channel < 75)
            negro_count = np.sum(negro_mask)
            
            # Castaño: V 65-170 y H < 60 y S < 110
            castano_mask = (v_channel >= 65) & (v_channel < 170) & (h_channel < 60) & (s_channel < 110)
            castano_count = np.sum(castano_mask)
            
            # Rubio: V > 170 y S < 85
            rubio_mask = (v_channel > 170) & (s_channel < 85)
            rubio_count = np.sum(rubio_mask)
            
            total_pixels = hair_region.size // 3
            
            # Calcular ratios
            negro_ratio = negro_count / total_pixels if total_pixels > 0 else 0
            castano_ratio = castano_count / total_pixels if total_pixels > 0 else 0
            rubio_ratio = rubio_count / total_pixels if total_pixels > 0 else 0
            
            # DECISIÓN FINAL - Favoreciendo Negro y Castaño fuertemente
            if negro_ratio > 0.15:
                return "Negro"
            elif castano_ratio > 0.12:
                return "Castaño"
            elif rubio_ratio > 0.35:  # Rubio requiere mucho más
                return "Rubio/Claro"
            else:
                # Fallback final: usar promedios
                if avg_v < 120:
                    return "Negro"
                elif avg_v < 170:
                    return "Castaño"
                else:
                    return "Rubio/Claro"
                    
        except Exception as e:
            return "Desconocido"
    
    def _detect_beard_advanced(self, face_roi, h, w):
        chin_region = face_roi[int(h*0.65):, :]
        gray = cv2.cvtColor(chin_region, cv2.COLOR_BGR2GRAY)
        intensity = np.mean(gray)
        return "Sí" if intensity < 110 else "No"
    
    def _detect_glasses_advanced(self, face_roi, h, w):
        """Detecta gafas buscando marcos rectangulares en region de ojos"""
        try:
            eye_region = face_roi[:int(h*0.45), :]
            
            # Validar que eye_region tenga contenido
            if eye_region.size == 0:
                return "No"
            
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Buscar lineas horizontales y verticales (marcos de gafas)
            # Usar deteccion de contornos mas selectiva
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            glasses_count = 0
            for contour in contours:
                bbox = cv2.boundingRect(contour)
                if len(bbox) != 4:  # Validación extra
                    continue
                x, y, cw, ch = bbox
                area = cv2.contourArea(contour)
                # Marcos de gafas: rectangulares, tamano moderado, proporcion casi cuadrada
                if 300 < area < 5000 and 0.6 < (cw / (ch + 1)) < 1.4:
                    glasses_count += 1
            
            # Si detecta 2 marcos (ambas gafas) es mas confiable
            return "Si" if glasses_count >= 2 else "No"
        except Exception as e:
            return "No"
    
    def _analyze_clothing(self, full_frame):
        """Analiza ropa superior e inferior con mejor precisión"""
        h, w = full_frame.shape[:2]
        
        def classify_color(hsv_region):
            """Clasifica color de una región HSV"""
            h_channel = hsv_region[:, :, 0]
            s_channel = hsv_region[:, :, 1]
            v_channel = hsv_region[:, :, 2]
            
            avg_h = np.mean(h_channel)
            avg_s = np.mean(s_channel)
            avg_v = np.mean(v_channel)
            
            # Primero: Blanco/Negro/Gris
            if avg_s < 30:  # Baja saturación = acromático
                if avg_v > 180:
                    return "Blanco"
                elif avg_v < 80:
                    return "Negro"
                else:
                    return "Gris"
            
            # Segundo: Colores saturados
            if avg_v < 60:
                return "Negro/Muy Oscuro"
            elif avg_v > 200 and avg_s < 40:
                return "Blanco"
            
            # Por Hue
            if avg_h < 10 or avg_h > 170:  # Rojo
                return "Rojo"
            elif 10 <= avg_h < 25:  # Naranja
                return "Naranja"
            elif 25 <= avg_h < 35:  # Amarillo
                return "Amarillo"
            elif 35 <= avg_h < 80:  # Verde
                return "Verde"
            elif 80 <= avg_h < 130:  # Azul/Cián
                return "Azul"
            elif 130 <= avg_h < 170:  # Magenta/Púrpura
                return "Púrpura"
            else:
                return "Otro"
        
        # Analizar ropa superior (pecho)
        chest_region = full_frame[int(h*0.30):int(h*0.55), :]
        if chest_region.size > 0:
            hsv_chest = cv2.cvtColor(chest_region, cv2.COLOR_BGR2HSV)
            ropa_superior = classify_color(hsv_chest)
        else:
            ropa_superior = "No visible"
        
        # Analizar ropa inferior (caderas/piernas)
        legs_region = full_frame[int(h*0.55):int(h*0.85), :]
        if legs_region.size > 0:
            hsv_legs = cv2.cvtColor(legs_region, cv2.COLOR_BGR2HSV)
            ropa_inferior = classify_color(hsv_legs)
        else:
            ropa_inferior = "No visible"
        
        return ropa_superior, ropa_inferior
    
    def _detect_hat_visor(self, face_roi, h, w):
        """Detecta gorra con visera analizando la region superior de la cabeza"""
        try:
            # Region superior donde podria haber gorra/visera
            top_region = face_roi[:int(h*0.25), :]
            
            # Validar que top_region tenga contenido
            if top_region.size == 0:
                return "No detectado"
            
            # Buscar pixeles oscuros (gorra tipicamente oscura)
            hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
            # Detectar colores oscuros que podrian ser gorra
            lower_dark = np.array([0, 0, 0])
            upper_dark = np.array([180, 255, 100])
            mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
            
            # Buscar contornos en la region superior
            contours, _ = cv2.findContours(mask_dark, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analizar forma de visera (rectangular, mas ancha que alta)
            for contour in contours:
                bbox = cv2.boundingRect(contour)
                if len(bbox) != 4:  # Validación extra
                    continue
                x, y, cw, ch = bbox
                area = cv2.contourArea(contour)
                
                # Visera tipicamente: area significante, mas ancha que alta (ratio > 1.5)
                if area > 500 and (cw / (ch + 1)) > 1.5:
                    return "Si, gorra con visera detectada"
            
            return "No detectado"
        except Exception as e:
            return "No detectado"
    
    def _estimate_ethnicity(self, face_roi, h, w):
        """Estima etnia basada en tono de piel"""
        # Analizar el tono de piel en la region central del rostro
        center_region = face_roi[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        
        # Convertir a HSV para analizar tono de piel
        hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]  # Hue
        s_channel = hsv[:, :, 1]  # Saturation
        v_channel = hsv[:, :, 2]  # Value
        
        avg_hue = np.mean(h_channel)
        avg_sat = np.mean(s_channel)
        avg_val = np.mean(v_channel)
        
        # Clasificacion basada en caracteristicas HSV del tono de piel
        # Hue ~10-25 (rojo-naranja) es tipico de piel
        # Saturation baja a media (piel tiene menos saturacion)
        # Value alta (piel es relativamente clara)
        
        if avg_val < 100:  # Muy oscuro
            return "Afrodescendiente"
        elif avg_hue < 15 and avg_sat < 80:  # Rojo-naranja puro, baja saturacion
            return "Caucasico/Europeo"
        elif 15 < avg_hue < 30 and avg_sat > 50:  # Mas naranja, mas saturado
            return "Indio/Sudamericano"
        elif avg_hue > 30 and avg_val > 150:  # Mas amarillento, muy claro
            return "Asiatico"
        else:
            return "Mixto/Indeterminado"
    
    def _analyze_eyes_open_advanced(self, eye_region):
        """Detecta si los ojos estan abiertos analizando contraste y distribucion de intensidad"""
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Analizar la distribucion de intensidades (ojos abiertos tienen contraste alto)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Si hay alto contraste (std) es probablemente porque estan abiertos
        # Los ojos cerrados tienen intensidad uniforme
        if std_intensity > 25:  # Alto contraste indica ojos abiertos
            return "Si"
        else:
            return "No"
    
    def _analyze_posture(self, face_roi, h, w):
        """Analiza la postura del rostro (inclinacion)"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Analizar simetria vertical del rostro
        left_half = gray[:, :int(w/2)]
        right_half = gray[:, int(w/2):]
        
        # Si los dos lados tienen diferente intensidad promedio, esta inclinado
        left_intensity = np.mean(left_half)
        right_intensity = np.mean(right_half)
        diff = abs(left_intensity - right_intensity)
        
        if diff > 20:
            if left_intensity > right_intensity:
                return "Inclinado a la derecha"
            else:
                return "Inclinado a la izquierda"
        else:
            return "Frontal"
    
    def detect_faces(self, frame):
        """Detecta rostros con máxima sensibilidad pero rechaza falsos positivos"""
        h_frame, w_frame = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # MEJORAR IMAGEN: Histograma + Contraste
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Bilateral filter para suavizar pero mantener bordes
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        face_list = []
        
        # INTENTO 1: Cascada frontal SENSIBLE (máxima detección)
        try:
            faces_loose = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,      
                minNeighbors=3,        
                minSize=(20, 20),      
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces_loose:
                if w > 0 and h > 0:
                    aspect_ratio = w / h if h != 0 else 0
                    # Rechazar si es demasiado grande (>40% del frame)
                    if 0.5 < aspect_ratio < 2.0 and (w*h) < (h_frame * w_frame * 0.4):
                        face_list.append({
                            'bbox': (x, y, w, h),
                            'confidence': 0.85,
                            'type': 'frontal',
                            'distance': 'unknown'
                        })
        except Exception as e:
            pass
        
        # INTENTO 2: Cascada de perfil (rostros de lado/3/4)
        try:
            faces_profile = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces_profile:
                if w > 0 and h > 0 and (w*h) < (h_frame * w_frame * 0.4):
                    # Evitar duplicados
                    is_dup = False
                    for existing in face_list:
                        ex, ey, ew, eh = existing['bbox']
                        if abs(x - ex) < 30 and abs(y - ey) < 30:
                            is_dup = True
                            break
                    
                    if not is_dup:
                        aspect_ratio = w / h if h != 0 else 0
                        if 0.5 < aspect_ratio < 2.0:
                            face_list.append({
                                'bbox': (x, y, w, h),
                                'confidence': 0.80,
                                'type': 'profile',
                                'distance': 'unknown'
                            })
        except Exception:
            pass
        
        # Si NO detecta rostros, intenta DIVISIÓN EN CUADRANTES
        if len(face_list) == 0:
            try:
                # Dividir imagen en 4 cuadrantes
                h_half = h_frame // 2
                w_half = w_frame // 2
                
                quadrants = [
                    (gray[0:h_half, 0:w_half], 0, 0),           
                    (gray[0:h_half, w_half:w_frame], w_half, 0),  
                    (gray[h_half:h_frame, 0:w_half], 0, h_half),  
                    (gray[h_half:h_frame, w_half:w_frame], w_half, h_half)  
                ]
                
                for quad_gray, quad_x_offset, quad_y_offset in quadrants:
                    try:
                        faces_quad = self.face_cascade.detectMultiScale(
                            quad_gray,
                            scaleFactor=1.05,
                            minNeighbors=3,
                            minSize=(15, 15),
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        
                        for (x, y, w, h) in faces_quad:
                            # Ajustar coordenadas al frame completo
                            abs_x = x + quad_x_offset
                            abs_y = y + quad_y_offset
                            
                            if w > 0 and h > 0:
                                face_list.append({
                                    'bbox': (abs_x, abs_y, w, h),
                                    'confidence': 0.75,
                                    'type': 'frontal',
                                    'distance': 'unknown'
                                })
                    except:
                        pass
            except:
                pass
        
        # Remover duplicados antes de retornar
        face_list = self._remove_overlapping_faces(face_list)
        
        # Si AÚN no detecta nada, usar detección de piel (fallback)
        # PERO SOLO si no hay mucha piel (evitar frame completo)
        if len(face_list) == 0:
            faces_skin = self._detect_by_skin_color(frame)
            face_list.extend(faces_skin)
        
        return face_list
    
    def _remove_overlapping_faces(self, faces):
        """Remove duplicate/overlapping face detections"""
        if len(faces) <= 1:
            return faces
        
        cleaned = []
        for face in faces:
            x, y, w, h = face['bbox']
            is_overlap = False
            
            for existing in cleaned:
                ex, ey, ew, eh = existing['bbox']
                # Calculate intersection over union
                xi1, yi1 = max(x, ex), max(y, ey)
                xi2, yi2 = min(x+w, ex+ew), min(y+h, ey+eh)
                
                if xi2 > xi1 and yi2 > yi1:
                    intersection = (xi2 - xi1) * (yi2 - yi1)
                    if intersection > (w*h)*0.3:  # 30% overlap = duplicate
                        is_overlap = True
                        break
            
            if not is_overlap:
                cleaned.append(face)
        
        return cleaned
    
    def _detect_by_skin_color(self, frame):
        """Fallback: detecta rostros por color de piel (HSV) - EXTREMADAMENTE RESTRICTIVO"""
        faces = []
        try:
            h_frame, w_frame = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Rangos de piel en HSV - MUY RESTRICTIVOS
            lower_skin = np.array([0, 20, 60], dtype=np.uint8)
            upper_skin = np.array([50, 150, 255], dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Morphology para limpiar
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                try:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)
                    frame_area = h_frame * w_frame
                    
                    # FILTROS EXTREMADAMENTE RESTRICTIVOS
                    # 1. Área: rostro típico 2000-25000 píxeles
                    if not (2000 < area < 25000):
                        continue
                    
                    # 2. No puede ser más del 15% del frame
                    if area > (frame_area * 0.15):
                        continue
                    
                    # 3. Aspect ratio: rostro cuadrado (0.6-1.6)
                    aspect = w / h if h > 0 else 0
                    if not (0.6 < aspect < 1.6):
                        continue
                    
                    # 4. Margen desde bordes: mínimo 20 píxeles
                    margin = 20
                    if not (x > margin and (x+w) < (w_frame-margin) and 
                            y > margin and (y+h) < (h_frame-margin)):
                        continue
                    
                    # Si pasó todos los filtros, agregar
                    faces.append({
                        'bbox': (x, y, w, h),
                        'confidence': 0.60,
                        'type': 'frontal',
                        'distance': 'unknown'
                    })
                except:
                    pass
        except:
            pass
        
        return faces
    
    def draw_results(self, frame, faces, analyses):
        for i, (face, analysis) in enumerate(zip(faces, analyses)):
            try:
                bbox_data = face.get('bbox')
                if bbox_data is None or len(bbox_data) != 4:
                    continue
                    
                x, y, w, h = bbox_data
                color = (0, 165, 255) if face.get('type') == 'back' else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"Espalda {i+1}" if face.get('type') == 'back' else f"Persona {i+1}"
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception:
                continue
        return frame
