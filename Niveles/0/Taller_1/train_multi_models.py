#!/usr/bin/env python3
"""
Script maestro para entrenar múltiples algoritmos de ML y compararlos.

Este script orquesta el pipeline completo de experimentación multi-modelo,
desde el procesamiento de datos hasta la comparación y persistencia de todos
los algoritmos entrenados. Reemplaza el train_model.py con capacidades avanzadas.
"""

import logging
import sys
import argparse
from pathlib import Path
import time

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_argument_parser():
    """Crear parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenar y comparar múltiples algoritmos de ML para clasificación de pingüinos"
    )
    
    parser.add_argument(
        '--algorithms', 
        nargs='*',
        choices=['logistic_regression', 'random_forest', 'gradient_boosting', 'svm_rbf', 'neural_network'],
        help='Algoritmos específicos a entrenar. Si no se especifica, entrena todos.'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directorio donde guardar los modelos (default: models)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporción de datos para testing (default: 0.2)'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Número de folds para validación cruzada (default: 5)'
    )
    
    parser.add_argument(
        '--no-auto-activate',
        action='store_true',
        help='No activar automáticamente el mejor modelo'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mostrar información detallada de entrenamiento'
    )
    
    return parser


def print_results_summary(comparison_results):
    """Imprimir resumen de resultados de manera legible."""
    print("\n" + "="*80)
    print("🏆 RESULTADOS DEL ENTRENAMIENTO MULTI-MODELO")
    print("="*80)
    
    # Información del experimento
    exp_info = comparison_results.get("experiment_info", {})
    print(f"📊 Total de modelos entrenados: {exp_info.get('total_models_trained', 0)}")
    print(f"🏅 Mejor modelo: {exp_info.get('best_model', 'N/A')}")
    print(f"⏱️  Timestamp: {exp_info.get('timestamp', 'N/A')}")
    print(f"🔄 CV folds: {exp_info.get('cv_folds', 'N/A')}")
    print(f"📝 Test size: {exp_info.get('test_size', 'N/A')}")
    
    # Ranking de modelos
    print(f"\n🥇 RANKING POR ACCURACY:")
    ranking = comparison_results.get("ranking", [])
    for rank_info in ranking:
        rank = rank_info.get("rank", 0)
        model = rank_info.get("model", "unknown")
        accuracy = rank_info.get("accuracy", 0.0)
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        print(f"   {medal} {model}: {accuracy:.4f}")
    
    # Detalles por modelo
    print(f"\n📈 MÉTRICAS DETALLADAS:")
    models_summary = comparison_results.get("models_summary", {})
    
    for model_name, summary in models_summary.items():
        is_best = "👑" if summary.get("is_best", False) else "  "
        algorithm = summary.get("algorithm", "unknown")
        test_acc = summary.get("test_accuracy", 0.0)
        cv_acc = summary.get("cv_accuracy", 0.0)
        cv_std = summary.get("cv_std", 0.0)
        precision = summary.get("precision", 0.0)
        recall = summary.get("recall", 0.0)
        f1 = summary.get("f1_score", 0.0)
        time_s = summary.get("training_time", 0.0)
        
        print(f"\n{is_best} {model_name.upper()} ({algorithm})")
        print(f"   📊 Test Accuracy:    {test_acc:.4f}")
        print(f"   🔄 CV Accuracy:      {cv_acc:.4f} ± {cv_std:.4f}")
        print(f"   🎯 Precision:        {precision:.4f}")
        print(f"   🔍 Recall:           {recall:.4f}")
        print(f"   ⚖️  F1-Score:         {f1:.4f}")
        print(f"   ⏱️  Training Time:    {time_s:.2f}s")
    
    print("\n" + "="*80)


def main():
    """Función principal que ejecuta el pipeline multi-modelo."""
    # Parsear argumentos
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configurar logging verbose si se solicita
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    start_time = time.time()
    
    try:
        logger.info("=== INICIANDO PIPELINE MULTI-MODELO ===")
        logger.info(f"Directorio de modelos: {args.models_dir}")
        logger.info(f"Test size: {args.test_size}")
        logger.info(f"CV folds: {args.cv_folds}")
        logger.info(f"Algoritmos: {args.algorithms if args.algorithms else 'todos'}")
        
        # Importar módulos después de configurar logging
        from src.data_processing import process_penguins_data
        from src.multi_model_trainer import MultiModelTrainer
        from src.model_manager import MultiModelManager
        
        # Paso 1: Procesar datos
        logger.info("📊 Paso 1: Procesando datos Palmer Penguins...")
        X, y, processing_info = process_penguins_data()
        logger.info(f"Datos procesados: {len(X)} muestras, {len(X.columns)} features")
        
        # Paso 2: Configurar entrenador multi-modelo
        logger.info("🤖 Paso 2: Configurando entrenador multi-modelo...")
        trainer = MultiModelTrainer(
            test_size=args.test_size,
            cv_folds=args.cv_folds
        )
        
        # Filtrar algoritmos si se especificaron
        if args.algorithms:
            available_algorithms = list(trainer.model_configs.keys())
            invalid_algorithms = set(args.algorithms) - set(available_algorithms)
            
            if invalid_algorithms:
                logger.error(f"Algoritmos no válidos: {list(invalid_algorithms)}")
                logger.error(f"Algoritmos disponibles: {available_algorithms}")
                sys.exit(1)
            
            # Filtrar configuraciones
            original_count = len(trainer.model_configs)
            trainer.model_configs = {name: config for name, config in trainer.model_configs.items() 
                                   if name in args.algorithms}
            
            logger.info(f"Filtrado: {len(trainer.model_configs)}/{original_count} algoritmos seleccionados")
        
        # Paso 3: Entrenar todos los modelos
        logger.info(f"🚀 Paso 3: Entrenando {len(trainer.model_configs)} algoritmos...")
        comparison_results = trainer.train_all_models(X, y)
        
        # Paso 4: Guardar artefactos
        logger.info("💾 Paso 4: Guardando artefactos...")
        model_manager = MultiModelManager(models_dir=args.models_dir)
        
        save_result = model_manager.save_multiple_models(
            trainer.trained_models,
            trainer.scaler,
            comparison_results
        )
        
        logger.info("Artefactos guardados:")
        for artifact_type, path in save_result["files_saved"].items():
            logger.info(f"  📄 {artifact_type}: {path}")
        
        # Paso 5: Activar el mejor modelo (si se solicita)
        best_model = comparison_results.get("experiment_info", {}).get("best_model")
        
        if not args.no_auto_activate and best_model:
            logger.info(f"👑 Paso 5: Activando mejor modelo: {best_model}")
            model_manager.set_active_model(best_model)
        else:
            logger.info("⏭️  Paso 5: Saltado - no se activó automáticamente el mejor modelo")
        
        # Mostrar resultados
        if not args.verbose:
            print_results_summary(comparison_results)
        
        # Estadísticas finales
        total_time = time.time() - start_time
        
        logger.info("=== PIPELINE MULTI-MODELO COMPLETADO ===")
        logger.info(f"🏆 Mejor modelo: {best_model}")
        logger.info(f"⏱️  Tiempo total: {total_time:.2f} segundos")
        logger.info(f"💾 Modelos guardados en: {args.models_dir}")
        
        if not args.no_auto_activate and best_model:
            logger.info(f"✅ Modelo {best_model} está activo y listo para predicciones")
        
    except Exception as e:
        logger.error(f"❌ Error en pipeline multi-modelo: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
