"""
API principal para el servicio de clasificación de pingüinos Palmer - SISTEMA MULTI-MODELO.

Este módulo orquesta todos los componentes del sistema ML (procesamiento, modelos, predicción)
en una interfaz REST coherente con capacidad de múltiples algoritmos. Implementa patrones de 
diseño robustos para manejo de errores, logging, selección dinámica de modelos y observabilidad 
que son esenciales en sistemas de ML en producción.
"""

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
from threading import Lock

# Agregar el directorio padre al path para permitir imports relativos
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import logging
import time
from datetime import datetime
from typing import Optional
import uuid

# Imports de nuestros módulos personalizados
from src.model_manager import MultiModelManager
from src.multi_model_trainer import train_multiple_algorithms
from src.data_processing import process_penguins_data
from api.schemas import (
    PenguinFeaturesSimple, 
    PenguinFeaturesComplete, 
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelsListResponse,
    ModelActivationResponse,
    ModelComparisonResponse,
    ModelDeletionResponse,
    MultiModelTrainingRequest,
    ErrorResponse
)

# Configurar logging global para el API
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# Variables globales para gestión de estado del modelo
model_manager: Optional[MultiModelManager] = None
cached_model = None
cached_scaler = None
cached_metadata = None
model_load_time = None
_reload_lock = Lock()  # Evita condiciones de carrera al recargar el modelo


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestor de ciclo de vida de la aplicación usando el nuevo patrón lifespan.
    
    Este patrón reemplaza los decoradores @app.on_event("startup") y @app.on_event("shutdown")
    con un enfoque más elegante y robusto que garantiza la limpieza adecuada de recursos.
    """
    # Código de startup
    global model_manager, cached_model, cached_scaler, cached_metadata, model_load_time
    
    logger.info("=== Iniciando carga de artefactos del modelo ===")
    
    try:
        # Inicializar el gestor de modelos multi-modelo
        models_dir = os.getenv("MODELS_DIR", "models")
        model_manager = MultiModelManager(models_dir=models_dir)
                
        # Intentar cargar todos los artefactos del modelo activo
        start_time = time.time()
        cached_model, cached_scaler, cached_metadata = model_manager.load_active_model()
        load_duration = time.time() - start_time
        model_load_time = datetime.now().isoformat()
        
        logger.info(f"Artefactos cargados exitosamente en {load_duration:.2f} segundos")
        logger.info(f"Modelo activo: {model_manager.get_active_model_name()}")
        logger.info(f"Accuracy: {cached_metadata.get('accuracy', {}).get('test', 'unknown')}")
        
    except FileNotFoundError as e:
        logger.error(f"Artefactos del modelo no encontrados: {e}")
        logger.warning("API iniciará sin modelo cargado - solo endpoints de información disponibles")
    except Exception as e:
        logger.error(f"Error cargando artefactos del modelo: {e}")
        logger.warning("API iniciará sin modelo cargado")
    
    # Yield controla cuando el servidor está listo para recibir requests
    yield
    
    # Código de shutdown (si fuera necesario)
    logger.info("=== Cerrando aplicación ===")


# Inicializar la aplicación FastAPI con el nuevo lifespan manager
app = FastAPI(
    title="Palmer Penguins Multi-Model Classifier API",
    description="""
    API RESTful para clasificación de especies de pingüinos usando múltiples algoritmos de ML.
    
    🎯 **Características principales:**
    - **Múltiples algoritmos**: Logistic Regression, Random Forest, Gradient Boosting, SVM, Neural Network
    - **Selección dinámica**: Cambio de modelo activo sin reiniciar el servicio
    - **Comparación automática**: Métricas y rankings de todos los modelos entrenados
    - **Entrenamiento on-demand**: Re-entrenamiento de algoritmos via API
    - **A/B Testing**: Soporte nativo para experimentación con modelos
    
    🐧 **Dataset**: Palmer Penguins - Clasificación entre especies Adelie, Chinstrap, y Gentoo
    
    🚀 **MLOps Ready**: Versionado, logging, monitoring y gestión completa del ciclo de vida
    """,
    version="2.0.0",
    contact={
        "name": "MLOps Team",
        "email": "mlops@university.edu"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan  # Usar el nuevo sistema de lifespan
)

# Configurar CORS para permitir requests desde diferentes orígenes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependencia para request tracking y logging
async def log_request_info(request: Request):
    """
    Dependency que registra información de cada request para observabilidad.
    
    En sistemas de producción, este tipo de logging es crucial para
    debugging, monitoring, y análisis de patrones de uso.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"[{request_id}] {request.method} {request.url.path} - Iniciando request")
    
    # Agregar información al contexto del request para uso posterior
    request.state.request_id = request_id
    request.state.start_time = start_time
    
    return request_id


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handler personalizado para excepciones HTTP que proporciona logging estructurado.
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.warning(f"[{request_id}] HTTP {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP{exc.status_code}",
            message=exc.detail,
            timestamp=datetime.now().isoformat(),
            request_id=request_id
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handler para excepciones no anticipadas que proporciona logging detallado.
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(f"[{request_id}] Error no manejado: {type(exc).__name__}: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="Error interno del servidor",
            timestamp=datetime.now().isoformat(),
            request_id=request_id
        ).model_dump()
    )

def convert_simple_to_complete_features(simple_features: PenguinFeaturesSimple) -> PenguinFeaturesComplete:
    """
    Converter características simples a formato completo con one-hot encoding.
    
    ACTUALIZADO para incluir la feature "year" y usar nombres exactos que espera el modelo:
    - sex_female, sex_male (minúsculas, como en el modelo entrenado)
    - Incluir year como feature numérica
    - Mantener el orden exacto de features del modelo
    """
    # Inicializar todas las variables categóricas en 0
    island_encoding = {"island_Biscoe": 0, "island_Dream": 0, "island_Torgersen": 0}
    sex_encoding = {"sex_female": 0, "sex_male": 0}  # Nombres corregidos
    
    # Aplicar one-hot encoding para isla
    island_key = f"island_{simple_features.island.value}"
    if island_key in island_encoding:
        island_encoding[island_key] = 1
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Isla no válida: {simple_features.island}. Valores permitidos: Biscoe, Dream, Torgersen"
        )
    
    # Aplicar one-hot encoding para sexo (usando nombres corregidos)
    if simple_features.sex.value == "Female":
        sex_encoding["sex_female"] = 1
    elif simple_features.sex.value == "Male":
        sex_encoding["sex_male"] = 1
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Sexo no válido: {simple_features.sex}. Valores permitidos: Male, Female"
        )
    
    # Construir objeto completo con TODAS las features que espera el modelo
    return PenguinFeaturesComplete(
        bill_length_mm=simple_features.bill_length_mm,
        bill_depth_mm=simple_features.bill_depth_mm,
        flipper_length_mm=simple_features.flipper_length_mm,
        body_mass_g=simple_features.body_mass_g,
        year=simple_features.year,  # FEATURE AGREGADA
        **island_encoding,
        **sex_encoding  # Usando nombres corregidos
    )


def get_species_name(species_code: int) -> str:
    """
    Convertir código numérico de especie a nombre legible.
    
    Esta función proporciona la traducción inversa del mapeo creado
    durante entrenamiento, permitiendo respuestas user-friendly.
    """
    species_mapping = {1: "Adelie", 2: "Chinstrap", 3: "Gentoo"}
    return species_mapping.get(species_code, "Unknown")

def fix_target_classes_for_api(species_mapping: dict[str, int]) -> dict[str, str]:
    """
    Convertir el mapeo de especies del formato de entrenamiento al formato del API.

    Durante el entrenamiento: {"Adelie": 1, "Gentoo": 2, "Chinstrap": 3}
    Para el API necesitamos: {"1": "Adelie", "2": "Gentoo", "3": "Chinstrap"}
    """
    if not species_mapping:
        return {}

    # Invertir el mapeo: de {nombre: código} a {código_string: nombre}
    inverted_mapping = {str(code): name for name, code in species_mapping.items()}
    return inverted_mapping


def validate_model_availability():
    """
    Dependency function que valida que el modelo esté disponible.
    
    Este patrón de validación como dependency permite reutilización
    across múltiples endpoints que requieren modelo cargado.
    """
    if cached_model is None or cached_scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Modelo no disponible. El servicio se está iniciando o hay un problema con los artefactos del modelo."
        )


@app.get("/", response_model=dict)
async def root(request_id: str = Depends(log_request_info)):
    """
    Endpoint raíz que proporciona información general sobre el API.
    
    Este endpoint actúa como "health check" básico y punto de entrada
    para desarrolladores explorando el API.
    """
    # Obtener información del modelo activo
    active_model_name = model_manager.get_active_model_name() if model_manager else None
    
    return {
        "service": "Palmer Penguins Multi-Model Classifier",
        "version": "2.0.0",
        "status": "active",
        "model_loaded": cached_model is not None,
        "active_model": active_model_name,
        "model_load_time": model_load_time,
        "features": {
            "multi_model_support": True,
            "dynamic_model_switching": True,
            "auto_comparison": True,
            "on_demand_training": True
        },
        "endpoints": {
            # Endpoints básicos
            "/predict/simple": "Predicción con entrada user-friendly",
            "/predict/complete": "Predicción con one-hot encoding explícito",
            "/health": "Estado detallado del servicio",
            "/model/info": "Información sobre el modelo activo",
            "/model/reload": "Recargar modelo activo",
            # Endpoints multi-modelo
            "/models/list": "Listar todos los modelos disponibles",
            "/models/activate/{model_name}": "Activar un modelo específico",
            "/models/comparison": "Comparación detallada entre modelos",
            "/models/train": "Entrenar múltiples modelos",
            "/models/{model_name}": "Eliminar un modelo específico",
            # Documentación
            "/docs": "Documentación interactiva (Swagger UI)",
            "/redoc": "Documentación alternativa (ReDoc)"
        },
        "request_id": request_id
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(request_id: str = Depends(log_request_info)):
    """
    Endpoint de salud comprehensivo para monitoring y orchestration.
    
    Este endpoint es fundamental para sistemas de producción donde
    tools como Kubernetes necesitan verificar la salud del servicio.
    """
    # Determinar estado general basado en disponibilidad de componentes
    model_available = cached_model is not None
    scaler_available = cached_scaler is not None
    overall_healthy = model_available and scaler_available
    
    status = "healthy" if overall_healthy else "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=model_available,
        scaler_loaded=scaler_available,
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(
    request_id: str = Depends(log_request_info),
    _: None = Depends(validate_model_availability)
):
    """
    Obtener información detallada sobre el modelo actualmente activo.
    
    Este endpoint es crucial para debugging, auditing, y verificación
    de que el modelo correcto está siendo usado en producción.
    """
    try:
        # Extraer información del metadata cacheado (formato multi-modelo)
        model_config = cached_metadata.get("config", {})
        accuracy_info = cached_metadata.get("accuracy", {})
        
        # Obtener información del modelo activo
        active_model_name = model_manager.get_active_model_name()
        
        return ModelInfoResponse(
            model_type=model_config.get("algorithm", "unknown"),
            version=cached_metadata.get("model_info", {}).get("saved_timestamp", "unknown"),
            training_date=cached_metadata.get("model_info", {}).get("saved_timestamp", "unknown"),
            accuracy=accuracy_info.get("test", 0.0),
            feature_count=len(cached_metadata.get("feature_columns", [])),
            target_classes={"1": "Adelie", "2": "Chinstrap", "3": "Gentoo"},  # Hardcoded por compatibilidad
            features=cached_metadata.get("feature_columns", [])
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Error obteniendo información del modelo: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo información del modelo")


@app.post("/model/reload", response_model=ModelInfoResponse)
async def reload_model(
    request_id: str = Depends(log_request_info)
):
    """
    Recargar artefactos del modelo activo desde MODELS_DIR sin reiniciar el servicio.
    
    Útil cuando un nuevo modelo ha sido guardado (por ejemplo, desde Jupyter)
    en el volumen compartido. Este endpoint vuelve a cargar el modelo activo,
    scaler y metadata, y retorna la información del modelo recién cargado.
    """
    global cached_model, cached_scaler, cached_metadata, model_load_time, model_manager

    try:
        with _reload_lock:
            # Asegurar que usamos el mismo directorio de modelos que el resto del servicio
            models_dir = os.getenv("MODELS_DIR", "models")
            if model_manager is None:
                model_manager = MultiModelManager(models_dir=models_dir)

            # Volver a cargar artefactos del modelo activo
            new_model, new_scaler, new_metadata = model_manager.load_active_model()

            # Actualizar caches atómicamente
            cached_model = new_model
            cached_scaler = new_scaler
            cached_metadata = new_metadata
            model_load_time = datetime.now().isoformat()

        # Preparar respuesta consistente con /model/info
        model_info = new_metadata.get("config", {})
        performance = new_metadata.get("accuracy", {})
        
        active_model_name = model_manager.get_active_model_name()

        logger.info(f"[{request_id}] Modelo {active_model_name} recargado exitosamente desde {models_dir}")

        return ModelInfoResponse(
            model_type=model_info.get("algorithm", "unknown"),
            version=new_metadata.get("model_info", {}).get("saved_timestamp", "unknown"),
            training_date=new_metadata.get("model_info", {}).get("saved_timestamp", "unknown"),
            accuracy=performance.get("test", 0.0),
            feature_count=len(new_metadata.get("feature_columns", [])),
            target_classes={"1": "Adelie", "2": "Chinstrap", "3": "Gentoo"},  # Hardcoded por compatibilidad
            features=new_metadata.get("feature_columns", [])
        )

    except FileNotFoundError:
        logger.warning(f"[{request_id}] No se encontraron artefactos en MODELS_DIR")
        raise HTTPException(status_code=404, detail="Artefactos del modelo no encontrados en MODELS_DIR")
    except Exception as e:
        logger.error(f"[{request_id}] Error recargando modelo: {e}")
        raise HTTPException(status_code=500, detail="Error recargando artefactos del modelo")


# ========== ENDPOINTS MULTI-MODELO ==========

@app.get("/models/list", response_model=ModelsListResponse)
async def list_available_models(request_id: str = Depends(log_request_info)):
    """
    Listar todos los modelos disponibles con sus métricas y estado.
    
    Este endpoint proporciona una vista completa de todos los modelos entrenados,
    sus métricas de performance, estado de activación, y metadatos relevantes.
    """
    try:
        models_list = model_manager.list_available_models()
        
        if "error" in models_list:
            raise HTTPException(status_code=500, detail=f"Error listando modelos: {models_list['error']}")
        
        logger.info(f"[{request_id}] Lista de modelos obtenida: {models_list['total_models']} modelos")
        
        return ModelsListResponse(**models_list)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error listando modelos: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo lista de modelos")


@app.post("/models/activate/{model_name}", response_model=ModelActivationResponse)
async def activate_model(
    model_name: str,
    request_id: str = Depends(log_request_info)
):
    """
    Activar un modelo específico para uso en predicciones.
    
    Este endpoint permite cambiar dinámicamente qué modelo se usa para predicciones
    sin reiniciar el servicio. Útil para A/B testing y experimentación en producción.
    """
    global cached_model, cached_scaler, cached_metadata
    
    try:
        # Obtener modelo activo actual
        previous_active = model_manager.get_active_model_name()
        
        # Establecer nuevo modelo activo
        success = model_manager.set_active_model(model_name)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' no encontrado")
        
        # Cargar el nuevo modelo en cache
        with _reload_lock:
            try:
                new_model, new_scaler, new_metadata = model_manager.load_active_model()
                cached_model = new_model
                cached_scaler = new_scaler  
                cached_metadata = new_metadata
                
                logger.info(f"[{request_id}] Modelo activado y cargado: {model_name}")
                
            except Exception as e:
                # Si falló la carga, revertir el cambio
                if previous_active:
                    model_manager.set_active_model(previous_active)
                raise HTTPException(status_code=500, detail=f"Error cargando modelo {model_name}: {str(e)}")
        
        # Preparar información del modelo activado
        model_info = {
            "algorithm": new_metadata.get("config", {}).get("algorithm", "unknown"),
            "test_accuracy": new_metadata.get("accuracy", {}).get("test", 0.0),
            "training_date": new_metadata.get("model_info", {}).get("saved_timestamp", "unknown")
        }
        
        return ModelActivationResponse(
            success=True,
            message=f"Modelo '{model_name}' activado exitosamente",
            previous_active=previous_active,
            new_active=model_name,
            model_info=model_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error activando modelo {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error activando modelo: {str(e)}")


@app.get("/models/comparison", response_model=ModelComparisonResponse)
async def get_models_comparison(request_id: str = Depends(log_request_info)):
    """
    Obtener comparación detallada entre todos los modelos entrenados.
    
    Este endpoint proporciona métricas comparativas, rankings, y análisis detallado
    de performance de todos los modelos disponibles. Útil para análisis de modelos
    y toma de decisiones sobre qué modelo usar en producción.
    """
    try:
        comparison = model_manager.get_models_comparison()
        
        if "error" in comparison:
            raise HTTPException(status_code=500, detail=f"Error obteniendo comparación: {comparison['error']}")
        
        logger.info(f"[{request_id}] Comparación de modelos obtenida")
        
        return ModelComparisonResponse(**comparison)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error obteniendo comparación: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo comparación de modelos")


@app.delete("/models/{model_name}", response_model=ModelDeletionResponse)
async def delete_model(
    model_name: str,
    request_id: str = Depends(log_request_info)
):
    """
    Eliminar un modelo específico del sistema.
    
    Este endpoint permite limpiar modelos obsoletos o experimentales.
    Si se elimina el modelo activo, automáticamente se activará el mejor modelo disponible.
    """
    global cached_model, cached_scaler, cached_metadata
    
    try:
        # Verificar que el modelo existe
        models_list = model_manager.list_available_models()
        if model_name not in [m["name"] for m in models_list.get("models", [])]:
            raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' no encontrado")
        
        # Verificar que no sea el único modelo
        if models_list.get("total_models", 0) <= 1:
            raise HTTPException(status_code=400, detail="No se puede eliminar el único modelo disponible")
        
        # Verificar si es el modelo activo
        active_model = model_manager.get_active_model_name()
        was_active = (model_name == active_model)
        
        # Eliminar modelo
        success = model_manager.delete_model(model_name)
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Error eliminando modelo '{model_name}'")
        
        # Si era el modelo activo, activar el mejor disponible
        new_active = None
        if was_active:
            try:
                # Obtener lista actualizada
                updated_list = model_manager.list_available_models()
                if updated_list.get("models"):
                    # Activar el mejor modelo disponible
                    best_available = max(updated_list["models"], key=lambda x: x["test_accuracy"])
                    model_manager.set_active_model(best_available["name"])
                    new_active = best_available["name"]
                    
                    # Recargar en cache
                    with _reload_lock:
                        cached_model, cached_scaler, cached_metadata = model_manager.load_active_model()
                    
                    logger.info(f"[{request_id}] Nuevo modelo activo: {new_active}")
                    
            except Exception as e:
                logger.error(f"[{request_id}] Error activando nuevo modelo después de eliminación: {e}")
        
        # Obtener recuento actualizado
        updated_list = model_manager.list_available_models()
        remaining_count = updated_list.get("total_models", 0)
        
        logger.info(f"[{request_id}] Modelo {model_name} eliminado. Restantes: {remaining_count}")
        
        return ModelDeletionResponse(
            success=True,
            message=f"Modelo '{model_name}' eliminado exitosamente",
            deleted_model=model_name,
            remaining_models=remaining_count,
            new_active=new_active
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error eliminando modelo {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error eliminando modelo: {str(e)}")


@app.post("/models/train", response_model=ModelComparisonResponse)
async def train_multiple_models(
    training_request: MultiModelTrainingRequest,
    request_id: str = Depends(log_request_info)
):
    """
    Entrenar múltiples algoritmos simultáneamente y compararlos.
    
    Este endpoint ejecuta el pipeline completo de entrenamiento multi-modelo:
    1. Procesa los datos Palmer Penguins
    2. Entrena los algoritmos especificados (o todos si no se especifica)
    3. Evalúa y compara todos los modelos
    4. Guarda todos los artefactos
    5. Opcionalmente activa el mejor modelo
    
    Útil para experimentación con nuevos algoritmos y re-entrenamiento periódico.
    """
    global cached_model, cached_scaler, cached_metadata
    
    try:
        logger.info(f"[{request_id}] Iniciando entrenamiento multi-modelo...")
        
        # Procesar datos
        logger.info(f"[{request_id}] Procesando datos...")
        X, y, processing_info = process_penguins_data()
        
        # Configurar entrenador
        from src.multi_model_trainer import MultiModelTrainer
        trainer = MultiModelTrainer(
            test_size=training_request.test_size,
            cv_folds=training_request.cv_folds
        )
        
        # Si se especificaron algoritmos específicos, filtrar
        if training_request.algorithms:
            available_algorithms = list(trainer.model_configs.keys())
            invalid_algorithms = set(training_request.algorithms) - set(available_algorithms)
            
            if invalid_algorithms:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Algoritmos no válidos: {list(invalid_algorithms)}. Disponibles: {available_algorithms}"
                )
            
            # Filtrar configuraciones
            trainer.model_configs = {name: config for name, config in trainer.model_configs.items() 
                                   if name in training_request.algorithms}
        
        # Entrenar modelos
        logger.info(f"[{request_id}] Entrenando {len(trainer.model_configs)} algoritmos...")
        comparison_results = trainer.train_all_models(X, y)
        
        # Guardar todos los modelos
        logger.info(f"[{request_id}] Guardando artefactos...")
        models_dir = os.getenv("MODELS_DIR", "models")
        model_manager_temp = MultiModelManager(models_dir)
        
        save_result = model_manager_temp.save_multiple_models(
            trainer.trained_models, 
            trainer.scaler, 
            comparison_results
        )
        
        # Activar el mejor modelo si se solicita
        if training_request.auto_activate_best:
            best_model = comparison_results.get("experiment_info", {}).get("best_model")
            if best_model:
                logger.info(f"[{request_id}] Activando mejor modelo: {best_model}")
                model_manager_temp.set_active_model(best_model)
                
                # Cargar en cache
                with _reload_lock:
                    cached_model, cached_scaler, cached_metadata = model_manager_temp.load_active_model()
        
        logger.info(f"[{request_id}] Entrenamiento multi-modelo completado exitosamente")
        logger.info(f"[{request_id}] Mejor modelo: {comparison_results.get('experiment_info', {}).get('best_model')}")
        
        return ModelComparisonResponse(**comparison_results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error en entrenamiento multi-modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Error en entrenamiento: {str(e)}")


# ========== ENDPOINTS DE PREDICCIÓN ==========

@app.post("/predict/simple", response_model=PredictionResponse)
async def predict_species_simple(
    features: PenguinFeaturesSimple,
    request_id: str = Depends(log_request_info),
    _: None = Depends(validate_model_availability)
):
    """
    Predecir especie usando entrada simplificada con conversión automática.
    
    Este endpoint proporciona la experiencia más user-friendly al manejar
    automáticamente la conversión de valores categóricos legibles a la
    representación interna que requiere el modelo.
    """
    try:
        start_inference = time.time()
        
        # Convertir a formato completo
        complete_features = convert_simple_to_complete_features(features)
        
        # Ejecutar predicción usando el endpoint interno
        result = await execute_prediction(complete_features, request_id, start_inference)
        
        logger.info(f"[{request_id}] Predicción simple exitosa: {result.species}")
        return result
        
    except HTTPException:
        # Re-lanzar excepciones HTTP específicas
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error en predicción simple: {e}")
        raise HTTPException(status_code=500, detail="Error ejecutando predicción")


@app.post("/predict/complete", response_model=PredictionResponse)
async def predict_species_complete(
    features: PenguinFeaturesComplete,
    request_id: str = Depends(log_request_info),
    _: None = Depends(validate_model_availability)
):
    """
    Predecir especie usando características con one-hot encoding explícito.
    
    Este endpoint está diseñado para integraciones avanzadas donde el cliente
    prefiere manejar la codificación categórica directamente, proporcionando
    mayor control sobre la entrada al modelo.
    """
    try:
        start_inference = time.time()
        result = await execute_prediction(features, request_id, start_inference)
        
        logger.info(f"[{request_id}] Predicción completa exitosa: {result.species}")
        return result
        
    except Exception as e:
        logger.error(f"[{request_id}] Error en predicción completa: {e}")
        raise HTTPException(status_code=500, detail="Error ejecutando predicción")


async def execute_prediction(features: PenguinFeaturesComplete, request_id: str, start_time: float) -> PredictionResponse:
    """
    Función helper que ejecuta la predicción actual del modelo.

    ACTUALIZADA para usar el orden exacto de features que espera el modelo
    y metadatos del sistema multi-modelo.
    """
    try:
        # Convertir características a array numpy en el ORDEN EXACTO del modelo
        feature_array = np.array([[
            features.bill_length_mm,
            features.bill_depth_mm,
            features.flipper_length_mm,
            features.body_mass_g,
            features.year,           # AGREGADO en posición correcta
            features.island_Biscoe,
            features.island_Dream,
            features.island_Torgersen,
            features.sex_female,     # Nombre corregido
            features.sex_male        # Nombre corregido
        ]])

        # Aplicar escalado usando el mismo scaler del entrenamiento
        feature_array_scaled = cached_scaler.transform(feature_array)
        
        # Ejecutar predicción
        prediction = cached_model.predict(feature_array_scaled)[0]
        probabilities = cached_model.predict_proba(feature_array_scaled)[0]
        
        # Calcular confianza y crear diccionario de probabilidades
        confidence = float(max(probabilities))
        prob_dict = {
            "Adelie": float(probabilities[0]),
            "Chinstrap": float(probabilities[1]),
            "Gentoo": float(probabilities[2])
        }
        
        # Obtener nombre de especie
        species_name = get_species_name(int(prediction))
        
        # Calcular tiempo de procesamiento
        processing_time = (time.time() - start_time) * 1000  # Convertir a milliseconds
        
        # Crear metadata de predicción (ACTUALIZADA para multi-modelo)
        prediction_metadata = {
            "processing_time_ms": round(processing_time, 2),
            "model_version": cached_metadata.get("model_info", {}).get("saved_timestamp", 
                                                cached_metadata.get("registry_info", {}).get("last_updated", "unknown")),
            "request_timestamp": datetime.now().isoformat(),
            "active_model": model_manager.get_active_model_name() if model_manager else "unknown"
        }
        
        return PredictionResponse(
            species=species_name,
            species_code=int(prediction),
            confidence=confidence,
            probabilities=prob_dict,
            prediction_metadata=prediction_metadata
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Error en execute_prediction: {e}")
        raise


if __name__ == "__main__":
    """
    Entry point para ejecutar el servidor directamente.
    
    Esta configuración inteligente permite tanto desarrollo con recarga automática
    como ejecución directa para producción, dependiendo de cómo se invoque el script.
    """
    import sys
    
    # Detectar si estamos ejecutando desde python -m api.main
    if len(sys.argv) > 0 and 'api.main' in sys.argv[0]:
        # Ejecución a través de python -m: usar string de importación para reload
        uvicorn.run(
            "api.main:app",  # String de importación para permitir reload
            host="0.0.0.0",
            port=8989,
            reload=True,  # Reload funcionará correctamente ahora
            log_level="info"
        )
    else:
        # Ejecución directa: usar objeto app directamente
        uvicorn.run(
            app,  # Objeto directo para ejecución simple
            host="0.0.0.0",
            port=8989,
            reload=False,  # Sin reload para ejecución directa
            log_level="info"
        )
