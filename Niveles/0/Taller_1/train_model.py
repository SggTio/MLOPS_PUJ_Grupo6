

#!/usr/bin/env python3
"""
Script maestro para entrenar y persistir el modelo de clasificación de pingüinos.

Este script orquesta todo el pipeline de ML, desde el procesamiento de datos
hasta la persistencia del modelo entrenado. Actúa como el punto de entrada
único para reproducir el entrenamiento completo del modelo.
"""

import logging
import sys
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Función principal que ejecuta el pipeline completo de entrenamiento.
    """
    try:
        logger.info("=== Iniciando pipeline completo de entrenamiento ===")
        
        # Importar módulos después de configurar logging
        from src.data_processing import process_penguins_data
        from src.model_training import train_penguin_classifier  
        from src.model_manager import save_trained_model
        
        # Paso 1: Procesar datos
        logger.info("Paso 1: Procesando datos...")
        X, y, processing_info = process_penguins_data()
        logger.info(f"Datos procesados: {len(X)} muestras, {len(X.columns)} features")
        
        # Paso 2: Entrenar modelo
        logger.info("Paso 2: Entrenando modelo...")
        model, scaler, metadata = train_penguin_classifier(X, y, processing_info)
        logger.info(f"Modelo entrenado con accuracy: {metadata['performance_metrics']['accuracy']:.4f}")
        
        # Paso 3: Guardar artefactos
        logger.info("Paso 3: Guardando artefactos...")
        saved_paths = save_trained_model(model, scaler, metadata)
        
        logger.info("=== Pipeline completado exitosamente ===")
        logger.info("Archivos guardados:")
        for artifact_type, path in saved_paths.items():
            logger.info(f"  {artifact_type}: {path}")
            
    except Exception as e:
        logger.error(f"Error en pipeline de entrenamiento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
