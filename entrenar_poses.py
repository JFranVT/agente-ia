"""
Sistema de entrenamiento de poses corporales
Aprende tus poses específicas para mejorar la detección
"""

import cv2
import numpy as np
import json
from datetime import datetime

class EntrenadorPoses:
    """Entrena al sistema para reconocer tus poses"""
    
    def __init__(self):
        self.poses_entrenadas = {}
        self.cargar_poses()
        
        # Conectar cámara
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ No se pudo conectar a la cámara")
            exit(1)
        
        # Detector de personas
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Lista de poses a entrenar
        self.lista_poses = [
            "parado",
            "brazo_derecho_arriba",
            "brazo_izquierdo_arriba",
            "ambos_brazos_arriba",
            "brazos_cruzados",
            "brazos_extendidos",
            "pierna_derecha_levantada",
            "pierna_izquierda_levantada",
            "sentado",
            "agachado",
            "arrodillado",
        ]
        
        # Instrucciones para cada pose
        self.instrucciones = {
            "parado": "Parate derecho, brazos abajo",
            "brazo_derecho_arriba": "Levanta SOLO el brazo DERECHO",
            "brazo_izquierdo_arriba": "Levanta SOLO el brazo IZQUIERDO",
            "ambos_brazos_arriba": "Levanta AMBOS brazos arriba",
            "brazos_cruzados": "Cruza los brazos sobre el pecho",
            "brazos_extendidos": "Extiende brazos en forma de T",
            "pierna_derecha_levantada": "Levanta la pierna DERECHA",
            "pierna_izquierda_levantada": "Levanta la pierna IZQUIERDA",
            "sentado": "Sientate en una silla",
            "agachado": "Agachate con rodillas flexionadas",
            "arrodillado": "Arrodillate en el suelo",
        }
        
        print(f"✅ Listo! {len(self.lista_poses)} poses para entrenar")
    
    def cargar_poses(self):
        try:
            with open('poses_entrenadas.json', 'r') as f:
                self.poses_entrenadas = json.load(f)
            print(f"✓ {len(self.poses_entrenadas)} poses cargadas")
        except:
            self.poses_entrenadas = {}
    
    def guardar_poses(self):
        with open('poses_entrenadas.json', 'w') as f:
            json.dump(self.poses_entrenadas, f, indent=2)
    
    def extraer_caracteristicas(self, frame):
        """Extrae características del cuerpo"""
        h, w = frame.shape[:2]
        boxes, _ = self.hog.detectMultiScale(frame, winStride=(4,4), padding=(16,16), scale=1.05)
        
        if len(boxes) == 0:
            return None
        
        best = max(boxes, key=lambda b: b[2]*b[3])
        x, y, bw, bh = best
        roi = frame[y:y+bh, x:x+bw]
        
        if roi.size == 0:
            return None
        
        ph, pw = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        features = {}
        
        # Energía en laterales (brazos)
        features['izq'] = float(np.sum(gray[int(ph*0.2):int(ph*0.5), :int(pw*0.2)]))
        features['der'] = float(np.sum(gray[int(ph*0.2):int(ph*0.5), int(pw*0.8):]))
        features['centro'] = float(np.sum(gray[int(ph*0.2):int(ph*0.5), int(pw*0.3):int(pw*0.7)]))
        
        # Ratio brazos
        features['ratio_brazos'] = features['izq'] / (features['der'] + 1)
        
        # Piernas
        features['pierna_izq'] = float(np.sum(gray[int(ph*0.5):, :int(pw*0.45)]))
        features['pierna_der'] = float(np.sum(gray[int(ph*0.5):, int(pw*0.55):]))
        
        # Proporción
        features['aspect'] = float(ph / pw)
        
        # Centro de masa
        m = cv2.moments(gray)
        if m['m00'] > 0:
            features['cm_x'] = float(m['m10'] / m['m00'] / pw)
            features['cm_y'] = float(m['m01'] / m['m00'] / ph)
        else:
            features['cm_x'] = 0.5
            features['cm_y'] = 0.5
        
        return {'features': features, 'bbox': (x, y, bw, bh)}
    
    def entrenar_pose(self, nombre_pose, num_muestras=10):
        """Entrena UNA pose"""
        instruccion = self.instrucciones.get(nombre_pose, nombre_pose)
        
        print(f"\n{'='*50}")
        print(f"🎯 {nombre_pose}")
        print(f"📋 {instruccion}")
        print(f"📸 {num_muestras} muestras | ESPACIO = capturar | q = cancelar")
        
        muestras = []
        contador = 0
        
        while contador < num_muestras:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (480, 640))
            
            # Overlay de información
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (480, 120), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            cv2.putText(frame, instruccion, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Muestra {contador+1}/{num_muestras}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Barra de progreso
            bar_x, bar_y, bar_w, bar_h = 90, 95, 300, 8
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (100,100,100), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+int(bar_w*contador/num_muestras), bar_y+bar_h), (0,255,0), -1)
            
            # Detectar persona
            datos = self.extraer_caracteristicas(frame)
            if datos:
                x, y, bw, bh = datos['bbox']
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                cv2.putText(frame, "DETECTADO - Presiona ESPACIO", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "NO DETECTADO - Ajusta posicion", (100, 350), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Entrenamiento', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and datos is not None:
                muestras.append(datos['features'])
                contador += 1
                print(f"  ✓ {contador}/{num_muestras}")
                cv2.waitKey(300)
            
            elif key == ord('q'):
                cv2.destroyWindow('Entrenamiento')
                cv2.waitKey(100)
                return False
        
        # 🟢 CERRAR VENTANA AL TERMINAR (ESTO FALTABA)
        cv2.destroyWindow('Entrenamiento')
        cv2.waitKey(100)
        
        # Guardar
        self.poses_entrenadas[nombre_pose] = {
            'muestras': muestras,
            'fecha': datetime.now().isoformat(),
            'num_muestras': num_muestras
        }
        self.guardar_poses()
        print(f"  ✅ '{nombre_pose}' entrenada!")
        return True
    
    def probar(self):
        """Prueba prediccion en tiempo real"""
        print("\n🔍 PROBANDO - Presiona 'q' para salir")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (480, 640))
            
            datos = self.extraer_caracteristicas(frame)
            
            if datos and self.poses_entrenadas:
                # Buscar pose más cercana
                f1 = datos['features']
                mejor_pose = None
                mejor_dist = float('inf')
                
                for nombre, info in self.poses_entrenadas.items():
                    for muestra in info['muestras']:
                        dist = 0
                        for k in ['izq', 'der', 'centro', 'pierna_izq', 'pierna_der']:
                            if k in f1 and k in muestra:
                                dist += abs(f1[k] - muestra[k]) / (max(abs(f1[k]), abs(muestra[k]), 1))
                        dist += abs(f1.get('aspect', 1) - muestra.get('aspect', 1)) * 5
                        dist += abs(f1.get('cm_y', 0.5) - muestra.get('cm_y', 0.5)) * 10
                        
                        if dist < mejor_dist:
                            mejor_dist = dist
                            mejor_pose = nombre
                
                # Dibujar
                x, y, bw, bh = datos['bbox']
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                
                if mejor_pose:
                    confianza = max(0, int(100 - mejor_dist * 10))
                    cv2.putText(frame, f"Pose: {mejor_pose}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confianza: {confianza}%", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            cv2.imshow('Probando Prediccion', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow('Probando Prediccion')
    
    def run(self):
        """Ejecuta el entrenador"""
        print("\n" + "="*50)
        print("   ENTRENADOR DE POSES CORPORALES")
        print("="*50)
        print("\nOpciones:")
        print("  1 = Entrenar TODAS las poses")
        print("  2 = Entrenar UNA pose")
        print("  3 = Probar prediccion")
        print("  4 = Salir")
        
        while True:
            opcion = input("\nElige: ").strip()
            
            if opcion == "1":
                for i, pose in enumerate(self.lista_poses):
                    print(f"\n{'='*40}")
                    print(f"POSE {i+1} de {len(self.lista_poses)}")
                    print(f"{'='*40}")
                    input(f"Preparate para: {pose} (ENTER para empezar)")
                    
                    resultado = self.entrenar_pose(pose, 10)
                    
                    if not resultado:
                        print("⚠️ Entrenamiento cancelado")
                        break
                    
                    # Pequeña pausa entre poses
                    cv2.waitKey(500)
                
                print("\n✅ ENTRENAMIENTO COMPLETADO!")
            
            elif opcion == "2":
                print("\nPoses:")
                for i, p in enumerate(self.lista_poses, 1):
                    print(f"  {i}. {p}")
                try:
                    idx = int(input("Numero: ")) - 1
                    if 0 <= idx < len(self.lista_poses):
                        self.entrenar_pose(self.lista_poses[idx], 10)
                except:
                    print("Invalido")
            
            elif opcion == "3":
                self.probar()
            
            elif opcion == "4":
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Entrenador cerrado")

if __name__ == "__main__":
    e = EntrenadorPoses()
    e.run()