"""
Analysis Worker - Consumes claims from Redis queue and performs fraud detection
"""
import redis
import json
import logging
import time
import signal
import sys
import os
from typing import Dict, Any

from model_loader import model_loader
from preprocessor import ClaimPreprocessor
from fraud_detector import FraudDetector
from db import Database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True


class AnalysisWorker:
    """Main worker process for analyzing claims"""
    
    def __init__(self):
        self.redis_client = None
        self.database = None
        self.fraud_detector = None
        self.stats = {
            'processed': 0,
            'anomalies_detected': 0,
            'errors': 0,
            'started_at': time.time()
        }
    
    def initialize(self):
        """Initialize all components"""
        logger.info("=== Initializing Analysis Worker ===")
        
        # Connect to Redis
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        logger.info(f"Connecting to Redis: {redis_url}")
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.redis_client.ping()
        logger.info("✓ Redis connection established")
        
        # Initialize database
        logger.info("Initializing database connection...")
        self.database = Database()
        logger.info("✓ Database connection established")
        
        # Load ML models
        logger.info("Loading ML models...")
        scaler = model_loader.scaler
        clinical_norms = model_loader.clinical_norms
        autoencoder = model_loader.autoencoder
        iforest = model_loader.iforest
        logger.info("✓ All models loaded")
        
        # Initialize preprocessor and detector
        preprocessor = ClaimPreprocessor(scaler, clinical_norms)
        self.fraud_detector = FraudDetector(autoencoder, iforest, preprocessor)
        logger.info("✓ Fraud detector initialized")
        
        logger.info("=== Worker Initialization Complete ===\n")
    
    def process_claim(self, claim_data: Dict[str, Any]):
        """
        Process a single claim through the fraud detection pipeline
        
        Args:
            claim_data: Dictionary containing claim information
        """
        claim_id = claim_data.get('claim_id', 'UNKNOWN')
        
        try:
            logger.info(f"[{claim_id}] Processing claim...")
            
            # Step 1: Store claim in database
            if not self.database.insert_claim(claim_data):
                logger.warning(f"[{claim_id}] Failed to insert claim, but continuing with analysis")
            
            # Step 2: Run fraud detection
            analysis_result = self.fraud_detector.detect(claim_data)
            
            # Step 3: Store analysis results
            if self.database.insert_analysis_result(analysis_result):
                logger.info(
                    f"[{claim_id}] ✓ COMPLETE - Risk: {analysis_result['risk_level']}, "
                    f"Score: {analysis_result['anomaly_score']:.4f}"
                )
            else:
                logger.error(f"[{claim_id}] Failed to store analysis results")
                self.stats['errors'] += 1
                return
            
            # Update stats
            self.stats['processed'] += 1
            if analysis_result['is_anomaly']:
                self.stats['anomalies_detected'] += 1
            
            # Log progress every 10 claims
            if self.stats['processed'] % 10 == 0:
                self.print_stats()
                
        except Exception as e:
            logger.error(f"[{claim_id}] Error processing claim: {e}", exc_info=True)
            self.stats['errors'] += 1
    
    def run(self):
        """Main worker loop - poll Redis queue and process claims"""
        logger.info("🚀 Worker started. Listening for claims on 'claims_queue'...")
        logger.info("Press Ctrl+C to stop gracefully\n")
        
        while not shutdown_requested:
            try:
                # BRPOP: Block until item available, timeout after 5 seconds
                result = self.redis_client.brpop('claims_queue', timeout=5)
                
                if result is None:
                    # No data available, continue polling
                    continue
                
                # result is tuple: (queue_name, value)
                _, claim_json = result
                
                # Deserialize claim data
                claim_data = json.loads(claim_json)
                
                # Process the claim
                self.process_claim(claim_data)
                
            except redis.ConnectionError as e:
                logger.error(f"Redis connection error: {e}")
                logger.info("Attempting to reconnect in 5 seconds...")
                time.sleep(5)
                try:
                    self.redis_client.ping()
                    logger.info("Reconnected to Redis")
                except:
                    logger.error("Reconnection failed")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in queue: {e}")
                self.stats['errors'] += 1
                
            except KeyboardInterrupt:
                logger.info("\nKeyboard interrupt detected")
                break
                
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)
                time.sleep(1)  # Prevent tight error loop
        
        self.shutdown()
    
    def print_stats(self):
        """Print worker statistics"""
        runtime = time.time() - self.stats['started_at']
        rate = self.stats['processed'] / runtime if runtime > 0 else 0
        fraud_rate = (self.stats['anomalies_detected'] / self.stats['processed'] * 100 
                      if self.stats['processed'] > 0 else 0)
        
        logger.info(
            f"📊 STATS: Processed={self.stats['processed']}, "
            f"Anomalies={self.stats['anomalies_detected']} ({fraud_rate:.1f}%), "
            f"Errors={self.stats['errors']}, "
            f"Rate={rate:.2f} claims/sec"
        )
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("\n=== Shutting Down Worker ===")
        self.print_stats()
        
        if self.database:
            self.database.close()
            logger.info("✓ Database connections closed")
        
        if self.redis_client:
            self.redis_client.close()
            logger.info("✓ Redis connection closed")
        
        logger.info("=== Shutdown Complete ===")


def main():
    """Entry point"""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    worker = AnalysisWorker()
    
    try:
        worker.initialize()
        worker.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
