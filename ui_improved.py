import cv2
import numpy as np

class ImprovedUI:
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.panel_width = 380  # Panel más ancho
        
    def create_analysis_window(self, analysis):
        """Crea una ventana optimizada para mostrar los 25 aspectos"""
        window_width = 520
        window_height = 850  # Más alto para que quepan todos
        window = np.ones((window_height, window_width, 3), dtype=np.uint8) * 35  # Fondo oscuro
        
        aspectos = analysis.get('aspectos', {})
        
        # Colores mejorados para mejor legibilidad
        color_titulo = (0, 255, 200)  # Verde azulado brillante
        color_seccion = (255, 200, 0)  # Naranja/dorado para secciones
        color_valor = (230, 230, 230)  # Blanco suave
        color_linea = (80, 80, 80)  # Gris para líneas
        
        y_offset = 15
        line_height = 28  # Altura de línea optimizada
        
        # Título principal
        cv2.putText(window, "ANALISIS DE 25 ASPECTOS", (15, y_offset), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, color_titulo, 2)
        y_offset += 30
        cv2.line(window, (15, y_offset), (window_width - 15, y_offset), color_seccion, 2)
        y_offset += 20
        
        # === SECCIÓN 1: DATOS BÁSICOS ===
        cv2.putText(window, "── DATOS BASICOS ──", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_seccion, 1)
        y_offset += 25
        
        basicos = [
            ("Rostro detectado:", str(aspectos.get('1_rostro_detectado', 'N/A'))),
            ("Area rostro:", str(aspectos.get('1_area_rostro', 'N/A'))),
            ("Edad estimada:", str(aspectos.get('2_edad_estimada', 'N/A'))),
            ("Genero:", str(aspectos.get('3_genero', 'N/A'))),
            ("Etnia:", str(aspectos.get('19_etnia_estimada', 'N/A'))),
        ]
        
        for label, valor in basicos:
            self._draw_row(window, label, valor, y_offset, color_valor)
            y_offset += line_height
        
        y_offset += 5
        cv2.line(window, (15, y_offset), (window_width - 15, y_offset), color_linea, 1)
        y_offset += 15
        
        # === SECCIÓN 2: ROSTRO ===
        cv2.putText(window, "── ROSTRO ──", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_seccion, 1)
        y_offset += 25
        
        rostro = [
            ("Expresion:", str(aspectos.get('4_expresion_facial', 'N/A'))),
            ("Confianza:", f"{aspectos.get('6_confianza_rostro', 0):.0%}"),
            ("Ojos:", "Abiertos" if aspectos.get('7_ojos_abiertos') == 'Sí' else "Cerrados"),
            ("Sonrisa:", str(aspectos.get('8_sonrisa_detectada', 'No'))),
            ("Cabello:", str(aspectos.get('9_color_cabello_aprox', 'N/A'))),
            ("Barba/Bigote:", str(aspectos.get('10_barba_bigote', 'No'))),
            ("Gafas:", str(aspectos.get('11_gafas', 'No'))),
            ("Sombrero/Gorra:", str(aspectos.get('12_sombrero', 'No'))),
        ]
        
        for label, valor in rostro:
            self._draw_row(window, label, valor, y_offset, color_valor)
            y_offset += line_height
        
        y_offset += 5
        cv2.line(window, (15, y_offset), (window_width - 15, y_offset), color_linea, 1)
        y_offset += 15
        
        # === SECCIÓN 3: CUERPO ===
        cv2.putText(window, "── CUERPO ──", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_seccion, 1)
        y_offset += 25
        
        cuerpo = [
            ("Visibilidad:", str(aspectos.get('13_cuerpo_visible', 'N/A'))),
            ("Postura:", str(aspectos.get('14_postura', 'N/A'))),
            ("Complexion:", str(aspectos.get('24_complexion_corporal', 'N/A'))),
        ]
        
        for label, valor in cuerpo:
            self._draw_row(window, label, valor, y_offset, color_valor)
            y_offset += line_height
        
        y_offset += 5
        
        # === SECCIÓN 4: EXTREMIDADES ===
        cv2.putText(window, "── EXTREMIDADES ──", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_seccion, 1)
        y_offset += 25
        
        extremidades = [
            ("Brazos visibles:", str(aspectos.get('21_brazos_visibles', 'N/A'))),
            ("Piernas visibles:", str(aspectos.get('22_piernas_visibles', 'N/A'))),
            ("Gesto corporal:", str(aspectos.get('23_gesto_corporal', 'N/A'))),
        ]
        
        for label, valor in extremidades:
            self._draw_row(window, label, valor, y_offset, color_valor)
            y_offset += line_height
        
        y_offset += 5
        cv2.line(window, (15, y_offset), (window_width - 15, y_offset), color_linea, 1)
        y_offset += 15
        
        # === SECCIÓN 5: VESTIMENTA ===
        cv2.putText(window, "── VESTIMENTA ──", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_seccion, 1)
        y_offset += 25
        
        vestimenta = [
            ("Ropa superior:", str(aspectos.get('15_ropa_superior_color', 'N/A'))),
            ("Ropa inferior:", str(aspectos.get('16_ropa_inferior_color', 'N/A'))),
            ("Accesorios:", str(aspectos.get('17_accesorios_detectados', 'Ninguno'))),
        ]
        
        for label, valor in vestimenta:
            self._draw_row(window, label, valor, y_offset, color_valor)
            y_offset += line_height
        
        y_offset += 5
        cv2.line(window, (15, y_offset), (window_width - 15, y_offset), color_linea, 1)
        y_offset += 15
        
        # === SECCIÓN 6: CALIDAD Y CONFIANZA ===
        cv2.putText(window, "── CALIDAD ──", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_seccion, 1)
        y_offset += 25
        
        calidad = [
            ("Calidad imagen:", str(aspectos.get('18_calidad_imagen', 'N/A'))),
            ("Confianza gral:", f"{aspectos.get('20_confianza_general', 0):.0%}"),
        ]
        
        for label, valor in calidad:
            self._draw_row(window, label, valor, y_offset, color_valor)
            y_offset += line_height
        
        # Si hay ángulos, mostrarlos
        angulos = aspectos.get('25_angulos_articulaciones', {})
        if angulos:
            y_offset += 10
            cv2.line(window, (15, y_offset), (window_width - 15, y_offset), color_linea, 1)
            y_offset += 15
            cv2.putText(window, "── ANGULOS ARTICULACIONES ──", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_seccion, 1)
            y_offset += 22
            
            for articulacion, angulo in angulos.items():
                self._draw_row(window, f"{articulacion}:", f"{angulo}°", y_offset, color_valor)
                y_offset += 22
        
        return window
    
    def _draw_row(self, window, label, valor, y, color_valor):
        """Dibuja una fila con etiqueta y valor"""
        # Etiqueta en la izquierda
        cv2.putText(window, label, (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_valor, 1)
        
        # Valor alineado a la derecha
        valor_str = str(valor)[:30]  # Limitar longitud
        text_size = cv2.getTextSize(valor_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x_valor = window.shape[1] - text_size[0] - 20
        cv2.putText(window, valor_str, (x_valor, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 1)  # Verde para valores
        
    def create_info_panel(self, analysis):
        """Versión compacta para panel lateral"""
        panel_height = self.frame_height
        panel = np.ones((panel_height, self.panel_width, 3), dtype=np.uint8) * 40
        
        aspectos = analysis.get('aspectos', {})
        
        color_titulo = (0, 255, 200)
        color_seccion = (255, 200, 0)
        color_valor = (220, 220, 220)
        color_valor_destacado = (0, 255, 150)
        
        y_offset = 10
        line_height = 22  # Más compacto
        
        # Título
        cv2.putText(panel, "ANALISIS DE PERSONA", (10, y_offset), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.55, color_titulo, 2)
        y_offset += 28
        
        # Datos en formato compacto
        datos = [
            ("SECCION", "VALOR", True),  # Encabezado
            ("Edad", str(aspectos.get('2_edad_estimada', '?'))),
            ("Genero", str(aspectos.get('3_genero', '?'))),
            ("Expresion", str(aspectos.get('4_expresion_facial', '?'))),
            ("Ojos", "Abiertos" if aspectos.get('7_ojos_abiertos') == 'Sí' else "Cerr."),
            ("Sonrisa", "Si" if aspectos.get('8_sonrisa_detectada') == 'Sí' else "No"),
            ("Cabello", str(aspectos.get('9_color_cabello_aprox', '?'))[:10]),
            ("Barba", str(aspectos.get('10_barba_bigote', '?'))),
            ("Gafas", str(aspectos.get('11_gafas', '?'))),
            ("Sombrero", str(aspectos.get('12_sombrero', '?'))[:8]),
            ("Postura", str(aspectos.get('14_postura', '?'))),
            ("Complexion", str(aspectos.get('24_complexion', '?'))),
            ("Brazos", str(aspectos.get('21_brazos_visibles', '?'))),
            ("Piernas", str(aspectos.get('22_piernas_visibles', '?'))),
            ("Gesto", str(aspectos.get('23_gesto_corporal', '?'))),
            ("Ropa Sup.", str(aspectos.get('15_ropa_superior_color', '?'))[:10]),
            ("Ropa Inf.", str(aspectos.get('16_ropa_inferior_color', '?'))[:10]),
            ("Confianza", f"{aspectos.get('20_confianza_general', 0):.0%}"),
        ]
        
        for i, dato in enumerate(datos):
            if len(dato) == 3:  # Es encabezado
                cv2.putText(panel, f"{dato[0]:<15} {dato[1]:>15}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_seccion, 1)
                y_offset += 5
                cv2.line(panel, (10, y_offset), (self.panel_width - 10, y_offset), (100, 100, 100), 1)
            else:
                label, valor = dato
                # Formato: etiqueta a la izquierda, valor a la derecha
                cv2.putText(panel, f"{label}:", (15, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_valor, 1)
                
                text_size = cv2.getTextSize(str(valor), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
                x_valor = self.panel_width - text_size[0] - 15
                cv2.putText(panel, str(valor), (x_valor, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_valor_destacado, 1)
            
            y_offset += line_height
        
        return panel
    
    def combine_frames(self, frame, panel):
        """Combina frame de video con panel de análisis"""
        h, w = frame.shape[:2]
        if h != self.frame_height:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        if panel.shape[0] != self.frame_height:
            panel = cv2.resize(panel, (self.panel_width, self.frame_height))
        
        combined = np.hstack([frame, panel])
        return combined
    
    def add_stats(self, frame, fps, personas_total, frames_procesados, tiempo_elapsed):
        """Agrega estadísticas en la parte superior del frame"""
        stats = [
            f"FPS: {fps:.1f}",
            f"Personas: {personas_total}",
            f"Frames: {frames_procesados}",
            f"Tiempo: {int(tiempo_elapsed)}s"
        ]
        
        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (380, 85), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Estadísticas en formato horizontal para ahorrar espacio
        y_offset = 22
        for i, stat in enumerate(stats):
            x_offset = 10 + (i * 95)
            cv2.putText(frame, stat, (x_offset, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame