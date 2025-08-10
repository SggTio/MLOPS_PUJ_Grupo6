"""
Módulo responsable del procesamiento y transformación de datos del dataset Palmer Penguins.

Este módulo encapsula toda la lógica relacionada con la carga, limpieza, y transformación
de datos. La separación de estas responsabilidades facilita el testing, mantenimiento,
y reutilización del código en diferentes contextos.
"""

import pandas as pd
from palmerpenguins import load_penguins
from typing import Tuple, Dict, Any
import logging

# Configurar logging para este módulo
logger = logging.getLogger(__name__)


class PenguinDataProcessor:
    """
    Clase responsable del procesamiento completo de datos de pingüinos Palmer.
    
    Esta clase implementa el patrón de responsabilidad única, encapsulando toda
    la lógica de transformación de datos en un componente reutilizable y testeable.
    """
    
    def __init__(self):
        """
        Inicializar el procesador con configuración predeterminada.
        """
        self.species_mapping = {}
        self.original_columns = []
        self.processed_columns = []
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Cargar los datos originales del dataset Palmer Penguins.
        
        Returns:
            pd.DataFrame: Dataset original sin procesamiento
            
        Raises:
            Exception: Si hay problemas cargando los datos
        """
        try:
            logger.info("Iniciando carga de datos Palmer Penguins")
            df = load_penguins()
            logger.info(f"Datos cargados exitosamente: {len(df)} filas, {len(df.columns)} columnas")
            
            # Guardar información sobre las columnas originales
            self.original_columns = df.columns.tolist()
            
            return df
        except Exception as e:
            logger.error(f"Error cargando datos: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpiar el dataset eliminando valores nulos y validando datos.
        
        Args:
            df (pd.DataFrame): Dataset original
            
        Returns:
            pd.DataFrame: Dataset limpio sin valores nulos
        """
        logger.info("Iniciando limpieza de datos")
        
        # Registrar estadísticas antes de la limpieza
        initial_rows = len(df)
        null_counts = df.isnull().sum()
        
        logger.info(f"Valores nulos por columna antes de limpieza:\n{null_counts}")
        
        # Eliminar filas con valores nulos
        df_clean = df.dropna()
        
        # Registrar estadísticas después de la limpieza
        final_rows = len(df_clean)
        removed_rows = initial_rows - final_rows
        
        logger.info(f"Limpieza completada: {removed_rows} filas eliminadas ({final_rows} restantes)")
        
        return df_clean
    
    def create_feature_target_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separar características (features) de la variable objetivo (target).
        
        Args:
            df (pd.DataFrame): Dataset limpio
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features y target separados
        """
        logger.info("Separando features y target")
        
        # Separar features (todas las columnas excepto 'species')
        features = df.drop(['species'], axis=1)
        
        # Extraer target y crear mapeo de especies
        target = df['species'].copy()
        unique_species = target.unique()
        
        # Crear mapeo de especies a números (para compatibilidad con sklearn)
        self.species_mapping = {species: i+1 for i, species in enumerate(unique_species)}
        target_encoded = target.map(self.species_mapping)
        
        logger.info(f"Mapeo de especies creado: {self.species_mapping}")
        logger.info(f"Features extraídas: {features.columns.tolist()}")
        
        return features, target_encoded
    
    def apply_one_hot_encoding(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Aplicar codificación one-hot a variables categóricas.
        
        Args:
            features (pd.DataFrame): Features sin codificar
            
        Returns:
            pd.DataFrame: Features con codificación one-hot aplicada
        """
        logger.info("Aplicando codificación one-hot")
        
        # Identificar columnas categóricas
        categorical_columns = features.select_dtypes(include=['object']).columns.tolist()
        logger.info(f"Columnas categóricas identificadas: {categorical_columns}")
        
        # Aplicar one-hot encoding
        features_encoded = pd.get_dummies(features, columns=categorical_columns).astype(float)
        
        # Guardar información sobre las columnas procesadas
        self.processed_columns = features_encoded.columns.tolist()
        
        logger.info(f"Codificación completada: {len(features_encoded.columns)} features finales")
        logger.info(f"Nuevas columnas: {features_encoded.columns.tolist()}")
        
        return features_encoded
    
    def get_processing_info(self) -> Dict[str, Any]:
        """
        Obtener información detallada sobre el procesamiento realizado.
        
        Returns:
            Dict[str, Any]: Información sobre transformaciones aplicadas
        """
        return {
            "original_columns": self.original_columns,
            "processed_columns": self.processed_columns,
            "species_mapping": self.species_mapping,
            "num_original_features": len(self.original_columns) - 1,  # -1 por 'species'
            "num_processed_features": len(self.processed_columns),
            "encoding_applied": "one_hot"
        }
    
    def process_full_pipeline(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Ejecutar el pipeline completo de procesamiento de datos.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]: 
                Features procesadas, target, información del procesamiento
        """
        logger.info("=== Iniciando pipeline completo de procesamiento ===")
        
        # Paso 1: Cargar datos
        raw_data = self.load_raw_data()
        
        # Paso 2: Limpiar datos
        clean_data = self.clean_data(raw_data)
        
        # Paso 3: Separar features y target
        features, target = self.create_feature_target_split(clean_data)
        
        # Paso 4: Aplicar one-hot encoding
        features_processed = self.apply_one_hot_encoding(features)
        
        # Paso 5: Generar información del procesamiento
        processing_info = self.get_processing_info()
        
        logger.info("=== Pipeline de procesamiento completado exitosamente ===")
        
        return features_processed, target, processing_info


def process_penguins_data() -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Función de conveniencia para procesar datos Palmer Penguins.
    
    Esta función proporciona una interfaz simple para ejecutar todo el pipeline
    de procesamiento sin necesidad de instanciar directamente la clase.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]: 
            Features procesadas, target, información del procesamiento
    """
    processor = PenguinDataProcessor()
    return processor.process_full_pipeline()
