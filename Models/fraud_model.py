import shutil
import os

# Ruta de descargas (ajusta si es necesario)
descargas = os.path.expanduser('~/Downloads/')

# Ruta de tu carpeta pkl (ajusta según tu proyecto)
carpeta_pkl = 'pkl/'  # o la ruta completa como 'C:/Users/TuUsuario/proyecto/pkl/'

# Archivos a mover
archivos = [ 'fraud_model.pkl']

for archivo in archivos:
    origen = os.path.join(descargas, archivo)
    destino = os.path.join(carpeta_pkl, archivo)
    
    if os.path.exists(origen):
        shutil.move(origen, destino)
        print(f'✓ {archivo} movido a {carpeta_pkl}')
    else:
        print(f'✗ {archivo} no encontrado en Descargas')