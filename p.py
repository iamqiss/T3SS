import os
import shutil

# Define the project directory name
PROJECT_DIR = "T3SS"

# Mapping from file extension to comment style
COMMENT_STYLES = {
    '.py': '#',
    '.go': '//',
    '.rs': '//',
    '.js': '//',
    '.l': '//',
    '.y': '//',
    '.ml': '(* *)', # OCaml/F# style
    '.html': '',
    '.yaml': '#',
    '.sh': '#',
    '.tf': '#',
    '.Dockerfile': '#',
    '.md': '', # No comments needed for markdown content typically
    '.dot': '//'
}

def generate_header(file_relative_path, comment_style):
    """
    Generates a copyright header with file path for each file.
    """
    if not comment_style:
        return "" # Return empty string for file types without comments

    lines = [
        "T3SS Project",
        f"File: {file_relative_path}",
        "(c) 2025 Qiss Labs. All Rights Reserved.",
        "Unauthorized copying or distribution of this file is strictly prohibited.",
        "For internal use only."
    ]

    if comment_style == '':  # Special case for HTML
        header = ""
    elif comment_style == '(* *)': # Special case for ML family
        header = "(*\n"
        for line in lines:
            header += f" * {line}\n"
        header += " *)"
    else:
        header = "\n".join(f"{comment_style} {line}" for line in lines)
        
    return header + "\n\n"

# Define the directory and file structure using a dictionary
# The keys are directory paths, and the values are lists of files to create in that directory.
# Nested dictionaries represent subdirectories.
structure = {
    "core": {
        "indexing": {
            "crawler": {
                "parser": ["html_parser.rs", "content_extractor.js", "pdf_parser.py", "xml_json_handler.go"],
                "url_frontier": ["bloom_filter.rs"],
                "async_crawl_jobs": ["worker_orchestrator.py"],
                "_files": ["scheduler.py", "fetcher.go", "robots_handler.py", "politeness_enforcer.go"]
            },
            "indexer": {
                "document_store": ["compression_utils.go", "sharding_manager.py", "ttl_expirer.rs"],
                "ranking_signals": ["pagerank_calculator.rs", "freshness_analyzer.py", "authority_scorer.go", "clickstream_aggregator.py", "social_signal_integrator.go"],
                "entity_extractor": ["ner_model.py", "knowledge_graph_integrator.rs", "relation_miner.go"],
                "index_merger": ["conflict_resolver.py"],
                "_files": ["inverted_index_builder.rs"]
            },
            "_files": ["deduplication_service.go", "spam_detector/ml_classifier.py", "spam_detector/heuristic_rules.rs", "spam_detector/blacklist_manager.go", "quality_assessor/content_quality_model.py", "quality_assessor/domain_reputation_tracker.rs"]
        },
        "querying": {
            "query_parser": ["lexer.l", "parser.y", "intent_classifier.ml", "query_expander.py", "spell_corrector.go"],
            "searcher": ["query_executor.rs", "relevance_scorer.go", "result_aggregator.py", "diversity_enforcer.rs", "federated_search_integrator.go"],
            "ranking": {
                "machine_learning_ranker": ["model_trainer.py", "inference_engine.rs", "a_b_tester.go", "feature_extractor.py"],
                "contextual_ranker": ["session_analyzer.py", "geo_location_handler.go"],
                "_files": ["fairness_auditor.rs"]
            },
            "query_logging": ["anomaly_flagger.py"],
            "_files": [] # No files directly in querying
        },
        "storage": {
            "distributed_fs": ["chunk_server.go", "master_server.rs", "backup_restore_pipeline.py"],
            "database": ["table_manager.rs", "query_engine.go", "transaction_coordinator.py"],
            "caching": ["memcache_wrapper.py", "cache_invalidator.rs"],
            "archival": ["glacier_analog.go"],
            "_files": []
        },
        "nlp_core": {
            "tokenizer": ["multilingual_tokenizer.rs"],
            "semantic_search": ["embedding_generator.go", "vector_index.rs", "reranker.py"],
            "translation_service": ["mt_model.rs"],
            "_files": ["sentiment_analyzer.py"]
        },
        "graph_core": ["link_graph_builder.go", "community_detector.rs"],
        "_files": []
    },
    "frontend": {
        "web_ui": {
            "html_templates": ["search_page.html", "results_renderer.js", "snippet_highlighter.py"],
            "api_endpoints": ["search_api.go", "autocomplete_service.py", "voice_search_handler.rs", "related_queries_generator.go"],
            "zero_click_features": ["knowledge_panel_extractor.py"],
            "_files": []
        },
        "mobile_ui": {
            "app_integration": ["push_notification_service.py"],
            "_files": ["amp_handler.go"]
        },
        "accessibility": ["screen_reader_optimizer.js"],
        "verticals": {
            "images_ui": ["thumbnail_generator.rs"],
            "news_ui": ["headline_clusterer.py"],
            "videos_ui": ["transcript_indexer.go"],
            "shopping_ui": ["price_tracker.rs"]
        },
        "_files": []
    },
    "backend_services": {
        "auth": ["oauth_handler.go", "session_manager.py", "privacy_controls.rs", "mfa_enforcer.go"],
        "logging": ["event_logger.rs", "metrics_aggregator.go", "anomaly_detector.py", "trace_distributor.rs"],
        "ml_services": {
            "training_pipeline": ["data_preprocessor.py", "trainer_job.go", "hyperparam_tuner.rs", "model_validator.py"],
            "serving": ["inference_server.rs", "model_updater.go"],
            "computer_vision": ["image_classifier.py", "object_detector.go", "ocr_engine.rs"],
            "nlp_services": ["summarizer.py", "qa_model.go"],
            "_files": []
        },
        "ads": ["auction_system.go", "targeting_engine.py", "fraud_prevention.rs", "creative_optimizer.go"],
        "data_pipelines": {
            "data_warehouse": ["sql_analyzer.rs"],
            "real_time_analytics": ["stream_enricher.py"],
            "_files": ["ingestion_service.go", "batch_processor.py"]
        },
        "security": ["encryption_service.rs", "vulnerability_scanner.py", "intrusion_detection.go", "access_control_manager.rs"],
        "billing": ["usage_tracker.py"],
        "notification": ["email_sms_dispatcher.go"],
        "experimentation": ["flag_manager.py", "metrics_evaluator.rs"],
        "_files": []
    },
    "infrastructure": {
        "config": ["prod_config.yaml", "staging_config.yaml"],
        "deployment": {
            "docker": ["base_image.Dockerfile"],
            "k8s": ["deployment.yaml", "autoscaler_config.rs", "istio_mesh.yaml"],
            "terraform": ["gcp_aws_modules.tf"],
            "_files": []
        },
        "ci_cd": {
            "build_scripts": ["build.sh"],
            "tests": ["e2e_tests.py", "chaos_tests.py", "unit_tests", "integration_tests"],
            "_files": ["release_manager.go"]
        },
        "monitoring": ["alerting_system.go", "dashboard_generator.py", "log_analyzer.rs"],
        "disaster_recovery": ["backup_orchestrator.py"],
        "_files": []
    },
    "shared_libs": {
        "utils": ["string_utils.rs", "date_time.py", "math_helpers.go"],
        "networking": ["grpc_wrappers.go", "http_client.rs"],
        "crypto": ["tls_handler.rs", "hash_utils.go"],
        "i18n": ["locale_manager.py"],
        "testing": ["stub_generator.rs"],
        "_files": []
    },
    "experimental": {
        "quantum_search": ["qubit_indexer.rs"],
        "ai_agents": ["multi_agent_system.py", "planning_engine.go"],
        "metaverse_integration": ["spatial_query_handler.go"],
        "blockchain_index": ["nft_metadata_extractor.rs"],
        "neuromorphic": ["spiking_nn.py"],
        "_files": []
    },
    "verticals": {
        "images": {
            "ingestion": ["uploader_service.go", "crawler_integration.py", "feed_processor.rs", "batch_ingester.py", "duplicate_checker.go"],
            "processing": ["resizer_micro.rs", "thumbnail_generator.py", "watermark_remover.go", "format_converter.rs", "exif_extractor.py", "compression_pipeline.go"],
            "analysis": ["object_detector.py", "face_recognition.rs", "ocr_engine.go", "scene_classifier.py", "color_analyzer.rs", "texture_feature_extractor.go", "landmark_identifier.py", "aesthetic_scorer.rs"],
            "embedding": ["clip_embedder.py", "vit_encoder.go", "vector_db_manager.rs", "dimensionality_reducer.py", "hybrid_search_integrator.go"],
            "moderation": ["nsfw_detector.py", "violence_gore_classifier.rs", "hate_symbol_recognizer.go", "deepfake_spotter.py", "copyright_scanner.rs", "human_review_queue.go"],
            "enrichment": ["caption_generator.py", "tag_suggester.rs", "entity_linker.go", "sentiment_tagger.py", "style_classifier.rs"],
            "search_engine": ["reverse_image_search.go", "text_to_image_query_parser.py", "filter_applier.rs", "result_ranker.go", "diversity_sampler.py", "pagination_handler.rs"],
            "storage": ["hot_storage.go", "cold_archive.py", "cdn_integrator.rs", "metadata_db.go", "replication_manager.py"],
            "caching": ["thumbnail_cache.rs", "embedding_cache.py", "query_result_cache.go", "invalidator_service.rs"],
            "monitoring": ["latency_tracker.py", "error_aggregator.go", "usage_analytics.rs", "alert_dispatcher.py"],
            "experimentation": ["variant_tester.go", "ui_ab_test.py", "metric_evaluator.rs"],
            "integrations": ["stock_photo_api.go", "social_image_puller.py", "creative_commons_verifier.rs"],
            "security": ["stegano_detector.py", "malware_scanner.go", "access_control.rs"],
            "creative_tools": ["style_transfer.py", "inpainting_service.go", "upscaler.rs", "collage_generator.py"],
            "accessibility": ["alt_text_generator.go", "color_blind_simulator.py"],
            "performance": ["gpu_scheduler.rs", "load_balancer.py", "bottleneck_analyzer.go"],
            "data_pipelines": ["feature_extraction_job.py", "cleanup_cron.rs", "export_service.go"],
            "ui_components": ["grid_renderer.js", "zoom_viewer.py", "filter_ui.go"],
            "ml_training": ["dataset_curator.rs", "trainer_orchestrator.py", "evaluator.go"],
            "deployment": {
                "helm_charts": ["images_vertical.yaml"],
                "_files": ["autoscaler_config.py"]
            },
            "docs": {
                "model_catalog": ["clip_variants.md"],
                "troubleshooting": ["ocr_failures.md"],
                "_files": ["architecture.md"]
            }
        },
        "news": ["event_detector.py", "fact_checker.go", "timeline_builder.rs"],
        "maps": ["geocoding_service.py", "routing_engine.go"],
        "scholarly": ["citation_graph.rs", "plagiarism_checker.py"],
        "finance": ["market_predictor.ml"],
        "_files": []
    },
    "integrations": {
        "api_partners": ["weather_api_wrapper.go"],
        "social_connectors": ["post_embedder.py"],
        "enterprise": ["custom_index_builder.rs"],
        "_files": []
    },
    "docs": {
        "api_docs": [],
        "onboarding": ["quickstart.md"],
        "wild_ideas": ["infinite_index.md"],
        "dependency_maps": ["full_madness.dot"],
        "_files": ["architecture.md"]
    },
    "_files": []
}

def create_structure(base_path, content_dict, parent_path=""):
    """Recursively creates directories and files with copyright headers including file path."""
    # Helper function to create a single file with a header
    def create_file_with_header(full_path, relative_path):
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        # Determine the file extension, with a special case for 'Dockerfile'
        base_name = os.path.basename(full_path)
        ext = os.path.splitext(base_name)[1]
        if 'dockerfile' in base_name.lower():
            ext = '.Dockerfile'

        comment_style = COMMENT_STYLES.get(ext, '//') # Default to '//' for unknown types
        header = generate_header(relative_path.replace("\\", "/"), comment_style)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(header)

    # Main loop for directories and files
    for key, value in content_dict.items():
        if key == "_files":
            # Handle files in the current directory
            for file_name in value:
                full_file_path = os.path.join(base_path, file_name)
                relative_file_path = os.path.join(parent_path, file_name)
                create_file_with_header(full_file_path, relative_file_path)
        else:
            # Handle subdirectories
            new_dir_path = os.path.join(base_path, key)
            os.makedirs(new_dir_path, exist_ok=True)
            new_parent_path = os.path.join(parent_path, key)
            if isinstance(value, dict):
                # Recurse for nested dictionary (more subdirectories)
                create_structure(new_dir_path, value, new_parent_path)
            elif isinstance(value, list):
                # Handle list of files in the new subdirectory
                for file_name in value:
                    full_file_path = os.path.join(new_dir_path, file_name)
                    relative_file_path = os.path.join(new_parent_path, file_name)
                    create_file_with_header(full_file_path, relative_file_path)

if __name__ == "__main__":
    if os.path.exists(PROJECT_DIR) and os.path.isdir(PROJECT_DIR):
        print(f"Existing '{PROJECT_DIR}' found. Removing it to regenerate the full structure...")
        shutil.rmtree(PROJECT_DIR)

    print("Creating the T3SS project structure with detailed file headers...")
    
    # Create the base directory
    os.makedirs(PROJECT_DIR)

    # Begin the recursive creation process
    create_structure(PROJECT_DIR, structure)

    print(f"\n{PROJECT_DIR} project scaffolding is complete! 🎉")
    print(f"Navigate into the project directory: cd {PROJECT_DIR}")

