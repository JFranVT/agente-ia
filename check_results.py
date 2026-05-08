import json
import os

# Obtener último archivo
files = sorted([f for f in os.listdir('resultados') if f.endswith('.json')])
if files:
    latest = files[-1]
    with open(f'resultados/{latest}') as f:
        data = json.load(f)
    
    print(f'\n=== RESULTADOS: {latest} ===\n')
    print(f'Total de análisis: {len(data)}\n')
    
    for i, analysis in enumerate(data):
        aspects = analysis.get('aspectos', {})
        print(f'--- Persona {i+1} ---')
        print(f'  Edad: {aspects.get("2_edad_estimada", "?")}')
        print(f'  Género: {aspects.get("3_genero", "?")}')
        print(f'  Color cabello: {aspects.get("9_color_cabello_aprox", "?")}')
        print(f'  Área rostro: {aspects.get("1_area_rostro", "?")}')
        print(f'  Barba: {aspects.get("10_barba_bigote", "?")}')
        print()
