import json
import csv
import os
from datetime import datetime

class Logger:
    """Gestor de logs mejorado"""
    
    def __init__(self, log_file=None):
        self.log_file = log_file
        
    def log(self, message, level="INFO"):
        """Registra un mensaje con timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"
        print(formatted_message)
        
        # Opcional: guardar en archivo
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(formatted_message + '\n')
            except Exception:
                pass

class DataManager:
    """Gestor de datos y almacenamiento robusto"""
    
    def __init__(self, output_folder="resultados"):
        self.output_folder = output_folder
        self._create_folder()
    
    def _create_folder(self):
        """Crea la carpeta de salida si no existe"""
        try:
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
                print(f"✓ Carpeta creada: {self.output_folder}")
        except Exception as e:
            print(f"⚠️ Error creando carpeta: {e}")
    
    def save_json(self, analyses):
        """Guarda análisis en JSON con manejo de errores"""
        if not analyses:
            print("⚠️ No hay datos para guardar en JSON")
            return False
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_folder, f"analisis_{timestamp}.json")
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(analyses, f, indent=2, ensure_ascii=False)
            
            print(f"✓ JSON guardado exitosamente: {filename}")
            return True
        except Exception as e:
            print(f"✗ Error guardando JSON: {e}")
            return False
    
    def save_csv(self, analyses):
        """Guarda análisis en CSV con manejo de errores"""
        if not analyses:
            print("⚠️ No hay datos para guardar en CSV")
            return False
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_folder, f"analisis_{timestamp}.csv")
            
            # Obtener todas las claves de aspectos de manera segura
            fieldnames = ['timestamp', 'orientacion']
            if analyses and analyses[0].get('aspectos'):
                fieldnames.extend(analyses[0]['aspectos'].keys())
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for analysis in analyses:
                    row = {
                        'timestamp': analysis.get('timestamp', ''),
                        'orientacion': analysis.get('orientacion', 'N/A')
                    }
                    if analysis.get('aspectos'):
                        row.update(analysis['aspectos'])
                    writer.writerow(row)
            
            print(f"✓ CSV guardado exitosamente: {filename}")
            return True
        except Exception as e:
            print(f"✗ Error guardando CSV: {e}")
            return False