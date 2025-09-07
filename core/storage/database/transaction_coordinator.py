# T3SS Project
# File: core/storage/database/transaction_coordinator.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis
from contextlib import asynccontextmanager
import weakref
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionState(Enum):
    """Transaction states"""
    PENDING = "pending"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"
    TIMEOUT = "timeout"

class IsolationLevel(Enum):
    """Transaction isolation levels"""
    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"

@dataclass
class TransactionConfig:
    """Configuration for transaction coordinator"""
    # Timeout settings
    default_timeout: int = 30  # seconds
    max_timeout: int = 300  # seconds
    prepare_timeout: int = 10  # seconds
    commit_timeout: int = 10  # seconds
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    exponential_backoff: bool = True
    
    # Performance settings
    max_concurrent_transactions: int = 1000
    cleanup_interval: int = 60  # seconds
    max_transaction_age: int = 3600  # seconds
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_key_prefix: str = "txn:"
    
    # Logging settings
    enable_audit_log: bool = True
    log_level: str = "INFO"

@dataclass
class Transaction:
    """Transaction representation"""
    transaction_id: str
    state: TransactionState
    created_at: datetime
    updated_at: datetime
    timeout: int
    isolation_level: IsolationLevel
    participants: Set[str] = field(default_factory=set)
    operations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    error_message: Optional[str] = None

@dataclass
class TransactionResult:
    """Result of transaction operation"""
    success: bool
    transaction_id: str
    state: TransactionState
    error_message: Optional[str] = None
    participants_committed: List[str] = field(default_factory=list)
    participants_aborted: List[str] = field(default_factory=list)
    execution_time: float = 0.0

class ParticipantManager:
    """Manages transaction participants"""
    
    def __init__(self):
        self.participants: Dict[str, Any] = {}
        self.participant_health: Dict[str, bool] = {}
        self.lock = threading.RLock()
    
    def register_participant(self, participant_id: str, participant: Any):
        """Register a transaction participant"""
        with self.lock:
            self.participants[participant_id] = participant
            self.participant_health[participant_id] = True
            logger.info(f"Registered participant: {participant_id}")
    
    def unregister_participant(self, participant_id: str):
        """Unregister a transaction participant"""
        with self.lock:
            if participant_id in self.participants:
                del self.participants[participant_id]
                del self.participant_health[participant_id]
                logger.info(f"Unregistered participant: {participant_id}")
    
    def get_participant(self, participant_id: str) -> Optional[Any]:
        """Get a participant by ID"""
        with self.lock:
            return self.participants.get(participant_id)
    
    def get_healthy_participants(self) -> Dict[str, Any]:
        """Get all healthy participants"""
        with self.lock:
            return {
                pid: participant for pid, participant in self.participants.items()
                if self.participant_health.get(pid, False)
            }
    
    def mark_participant_unhealthy(self, participant_id: str):
        """Mark a participant as unhealthy"""
        with self.lock:
            self.participant_health[participant_id] = False
            logger.warning(f"Marked participant as unhealthy: {participant_id}")
    
    def mark_participant_healthy(self, participant_id: str):
        """Mark a participant as healthy"""
        with self.lock:
            self.participant_health[participant_id] = True
            logger.info(f"Marked participant as healthy: {participant_id}")

class TwoPhaseCommitProtocol:
    """Two-Phase Commit protocol implementation"""
    
    def __init__(self, participant_manager: ParticipantManager):
        self.participant_manager = participant_manager
    
    async def prepare_phase(self, transaction: Transaction) -> bool:
        """Execute prepare phase of 2PC"""
        logger.info(f"Starting prepare phase for transaction {transaction.transaction_id}")
        
        participants = list(transaction.participants)
        if not participants:
            logger.warning(f"No participants for transaction {transaction.transaction_id}")
            return True
        
        # Send prepare requests to all participants
        prepare_tasks = []
        for participant_id in participants:
            task = self._prepare_participant(participant_id, transaction)
            prepare_tasks.append(task)
        
        # Wait for all prepare responses
        results = await asyncio.gather(*prepare_tasks, return_exceptions=True)
        
        # Check if all participants voted to commit
        all_prepared = True
        for i, result in enumerate(results):
            participant_id = participants[i]
            if isinstance(result, Exception):
                logger.error(f"Prepare failed for participant {participant_id}: {result}")
                self.participant_manager.mark_participant_unhealthy(participant_id)
                all_prepared = False
            elif not result:
                logger.warning(f"Participant {participant_id} voted to abort")
                all_prepared = False
        
        if all_prepared:
            logger.info(f"All participants prepared for transaction {transaction.transaction_id}")
        else:
            logger.warning(f"Some participants failed to prepare for transaction {transaction.transaction_id}")
        
        return all_prepared
    
    async def commit_phase(self, transaction: Transaction) -> bool:
        """Execute commit phase of 2PC"""
        logger.info(f"Starting commit phase for transaction {transaction.transaction_id}")
        
        participants = list(transaction.participants)
        if not participants:
            return True
        
        # Send commit requests to all participants
        commit_tasks = []
        for participant_id in participants:
            task = self._commit_participant(participant_id, transaction)
            commit_tasks.append(task)
        
        # Wait for all commit responses
        results = await asyncio.gather(*commit_tasks, return_exceptions=True)
        
        # Check if all participants committed successfully
        all_committed = True
        for i, result in enumerate(results):
            participant_id = participants[i]
            if isinstance(result, Exception):
                logger.error(f"Commit failed for participant {participant_id}: {result}")
                self.participant_manager.mark_participant_unhealthy(participant_id)
                all_committed = False
            elif not result:
                logger.warning(f"Participant {participant_id} failed to commit")
                all_committed = False
        
        if all_committed:
            logger.info(f"All participants committed for transaction {transaction.transaction_id}")
        else:
            logger.warning(f"Some participants failed to commit for transaction {transaction.transaction_id}")
        
        return all_committed
    
    async def abort_phase(self, transaction: Transaction) -> bool:
        """Execute abort phase of 2PC"""
        logger.info(f"Starting abort phase for transaction {transaction.transaction_id}")
        
        participants = list(transaction.participants)
        if not participants:
            return True
        
        # Send abort requests to all participants
        abort_tasks = []
        for participant_id in participants:
            task = self._abort_participant(participant_id, transaction)
            abort_tasks.append(task)
        
        # Wait for all abort responses
        results = await asyncio.gather(*abort_tasks, return_exceptions=True)
        
        # Log results (abort is best-effort)
        for i, result in enumerate(results):
            participant_id = participants[i]
            if isinstance(result, Exception):
                logger.error(f"Abort failed for participant {participant_id}: {result}")
            elif not result:
                logger.warning(f"Participant {participant_id} failed to abort")
        
        logger.info(f"Abort phase completed for transaction {transaction.transaction_id}")
        return True
    
    async def _prepare_participant(self, participant_id: str, transaction: Transaction) -> bool:
        """Send prepare request to a participant"""
        participant = self.participant_manager.get_participant(participant_id)
        if not participant:
            logger.error(f"Participant not found: {participant_id}")
            return False
        
        try:
            # Check if participant has prepare method
            if hasattr(participant, 'prepare'):
                result = await participant.prepare(transaction.transaction_id, transaction.operations)
                return result
            else:
                logger.warning(f"Participant {participant_id} does not support prepare")
                return False
        except Exception as e:
            logger.error(f"Prepare request failed for participant {participant_id}: {e}")
            return False
    
    async def _commit_participant(self, participant_id: str, transaction: Transaction) -> bool:
        """Send commit request to a participant"""
        participant = self.participant_manager.get_participant(participant_id)
        if not participant:
            logger.error(f"Participant not found: {participant_id}")
            return False
        
        try:
            # Check if participant has commit method
            if hasattr(participant, 'commit'):
                result = await participant.commit(transaction.transaction_id)
                return result
            else:
                logger.warning(f"Participant {participant_id} does not support commit")
                return False
        except Exception as e:
            logger.error(f"Commit request failed for participant {participant_id}: {e}")
            return False
    
    async def _abort_participant(self, participant_id: str, transaction: Transaction) -> bool:
        """Send abort request to a participant"""
        participant = self.participant_manager.get_participant(participant_id)
        if not participant:
            logger.error(f"Participant not found: {participant_id}")
            return False
        
        try:
            # Check if participant has abort method
            if hasattr(participant, 'abort'):
                result = await participant.abort(transaction.transaction_id)
                return result
            else:
                logger.warning(f"Participant {participant_id} does not support abort")
                return False
        except Exception as e:
            logger.error(f"Abort request failed for participant {participant_id}: {e}")
            return False

class TransactionCoordinator:
    """Advanced transaction coordinator with 2PC support"""
    
    def __init__(self, config: TransactionConfig = None):
        self.config = config or TransactionConfig()
        self.participant_manager = ParticipantManager()
        self.two_phase_commit = TwoPhaseCommitProtocol(self.participant_manager)
        
        # Transaction storage
        self.active_transactions: Dict[str, Transaction] = {}
        self.completed_transactions: Dict[str, Transaction] = {}
        self.lock = threading.RLock()
        
        # Redis connection for persistence
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=True
            )
            self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage only.")
            self.redis_client = None
        
        # Background tasks
        self.cleanup_task = None
        self.health_check_task = None
        
        # Statistics
        self.stats = {
            'total_transactions': 0,
            'committed_transactions': 0,
            'aborted_transactions': 0,
            'timeout_transactions': 0,
            'average_commit_time': 0.0,
            'average_abort_time': 0.0
        }
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self.cleanup_task = asyncio.create_task(self._cleanup_old_transactions())
        self.health_check_task = asyncio.create_task(self._health_check_participants())
    
    async def _cleanup_old_transactions(self):
        """Clean up old completed transactions"""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(seconds=self.config.max_transaction_age)
                
                with self.lock:
                    # Clean up completed transactions
                    to_remove = []
                    for txn_id, txn in self.completed_transactions.items():
                        if txn.updated_at < cutoff_time:
                            to_remove.append(txn_id)
                    
                    for txn_id in to_remove:
                        del self.completed_transactions[txn_id]
                    
                    if to_remove:
                        logger.info(f"Cleaned up {len(to_remove)} old transactions")
                
                # Clean up Redis if available
                if self.redis_client:
                    await self._cleanup_redis_transactions(cutoff_time)
                
            except Exception as e:
                logger.error(f"Cleanup task failed: {e}")
    
    async def _health_check_participants(self):
        """Check health of participants"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                healthy_participants = self.participant_manager.get_healthy_participants()
                for participant_id, participant in healthy_participants.items():
                    try:
                        # Check if participant has health check method
                        if hasattr(participant, 'health_check'):
                            is_healthy = await participant.health_check()
                            if not is_healthy:
                                self.participant_manager.mark_participant_unhealthy(participant_id)
                        else:
                            # Assume healthy if no health check method
                            pass
                    except Exception as e:
                        logger.warning(f"Health check failed for participant {participant_id}: {e}")
                        self.participant_manager.mark_participant_unhealthy(participant_id)
                
            except Exception as e:
                logger.error(f"Health check task failed: {e}")
    
    async def _cleanup_redis_transactions(self, cutoff_time: datetime):
        """Clean up old transactions from Redis"""
        try:
            # Get all transaction keys
            pattern = f"{self.config.redis_key_prefix}*"
            keys = self.redis_client.keys(pattern)
            
            for key in keys:
                txn_data = self.redis_client.get(key)
                if txn_data:
                    txn_dict = json.loads(txn_data)
                    updated_at = datetime.fromisoformat(txn_dict['updated_at'])
                    if updated_at < cutoff_time:
                        self.redis_client.delete(key)
        except Exception as e:
            logger.warning(f"Redis cleanup failed: {e}")
    
    def register_participant(self, participant_id: str, participant: Any):
        """Register a transaction participant"""
        self.participant_manager.register_participant(participant_id, participant)
    
    def unregister_participant(self, participant_id: str):
        """Unregister a transaction participant"""
        self.participant_manager.unregister_participant(participant_id)
    
    async def begin_transaction(self, 
                              timeout: Optional[int] = None,
                              isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """Begin a new transaction"""
        transaction_id = str(uuid.uuid4())
        timeout = timeout or self.config.default_timeout
        
        transaction = Transaction(
            transaction_id=transaction_id,
            state=TransactionState.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            timeout=timeout,
            isolation_level=isolation_level,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.active_transactions[transaction_id] = transaction
        
        # Persist to Redis if available
        if self.redis_client:
            await self._persist_transaction(transaction)
        
        logger.info(f"Started transaction: {transaction_id}")
        return transaction_id
    
    async def add_participant(self, transaction_id: str, participant_id: str) -> bool:
        """Add a participant to a transaction"""
        with self.lock:
            transaction = self.active_transactions.get(transaction_id)
            if not transaction:
                logger.error(f"Transaction not found: {transaction_id}")
                return False
            
            if transaction.state != TransactionState.PENDING:
                logger.error(f"Cannot add participant to transaction in state: {transaction.state}")
                return False
            
            transaction.participants.add(participant_id)
            transaction.updated_at = datetime.now()
        
        # Persist to Redis if available
        if self.redis_client:
            await self._persist_transaction(transaction)
        
        logger.info(f"Added participant {participant_id} to transaction {transaction_id}")
        return True
    
    async def add_operation(self, transaction_id: str, operation: Dict[str, Any]) -> bool:
        """Add an operation to a transaction"""
        with self.lock:
            transaction = self.active_transactions.get(transaction_id)
            if not transaction:
                logger.error(f"Transaction not found: {transaction_id}")
                return False
            
            if transaction.state != TransactionState.PENDING:
                logger.error(f"Cannot add operation to transaction in state: {transaction.state}")
                return False
            
            transaction.operations.append(operation)
            transaction.updated_at = datetime.now()
        
        # Persist to Redis if available
        if self.redis_client:
            await self._persist_transaction(transaction)
        
        logger.info(f"Added operation to transaction {transaction_id}")
        return True
    
    async def commit_transaction(self, transaction_id: str) -> TransactionResult:
        """Commit a transaction using 2PC"""
        start_time = time.time()
        
        with self.lock:
            transaction = self.active_transactions.get(transaction_id)
            if not transaction:
                return TransactionResult(
                    success=False,
                    transaction_id=transaction_id,
                    state=TransactionState.ABORTED,
                    error_message="Transaction not found"
                )
            
            if transaction.state != TransactionState.PENDING:
                return TransactionResult(
                    success=False,
                    transaction_id=transaction_id,
                    state=transaction.state,
                    error_message=f"Transaction in invalid state: {transaction.state}"
                )
            
            # Update state to preparing
            transaction.state = TransactionState.PREPARING
            transaction.updated_at = datetime.now()
        
        try:
            # Persist state change
            if self.redis_client:
                await self._persist_transaction(transaction)
            
            # Phase 1: Prepare
            logger.info(f"Starting prepare phase for transaction {transaction_id}")
            prepare_success = await self.two_phase_commit.prepare_phase(transaction)
            
            if not prepare_success:
                # Abort transaction
                logger.warning(f"Prepare phase failed for transaction {transaction_id}, aborting")
                await self.abort_transaction(transaction_id)
                return TransactionResult(
                    success=False,
                    transaction_id=transaction_id,
                    state=TransactionState.ABORTED,
                    error_message="Prepare phase failed"
                )
            
            # Update state to prepared
            with self.lock:
                transaction.state = TransactionState.PREPARED
                transaction.updated_at = datetime.now()
            
            if self.redis_client:
                await self._persist_transaction(transaction)
            
            # Phase 2: Commit
            logger.info(f"Starting commit phase for transaction {transaction_id}")
            commit_success = await self.two_phase_commit.commit_phase(transaction)
            
            if not commit_success:
                # This is a critical error - some participants may have committed
                logger.error(f"Commit phase failed for transaction {transaction_id}")
                # Still mark as committed since some participants may have committed
                with self.lock:
                    transaction.state = TransactionState.COMMITTED
                    transaction.updated_at = datetime.now()
                    self.completed_transactions[transaction_id] = transaction
                    del self.active_transactions[transaction_id]
                
                if self.redis_client:
                    await self._persist_transaction(transaction)
                
                return TransactionResult(
                    success=False,
                    transaction_id=transaction_id,
                    state=TransactionState.COMMITTED,
                    error_message="Commit phase failed - some participants may not have committed"
                )
            
            # Success
            with self.lock:
                transaction.state = TransactionState.COMMITTED
                transaction.updated_at = datetime.now()
                self.completed_transactions[transaction_id] = transaction
                del self.active_transactions[transaction_id]
            
            if self.redis_client:
                await self._persist_transaction(transaction)
            
            # Update statistics
            execution_time = time.time() - start_time
            self.stats['committed_transactions'] += 1
            self.stats['average_commit_time'] = (
                (self.stats['average_commit_time'] * (self.stats['committed_transactions'] - 1) + 
                 execution_time) / self.stats['committed_transactions']
            )
            
            logger.info(f"Successfully committed transaction {transaction_id}")
            return TransactionResult(
                success=True,
                transaction_id=transaction_id,
                state=TransactionState.COMMITTED,
                participants_committed=list(transaction.participants),
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Commit failed for transaction {transaction_id}: {e}")
            
            # Abort transaction
            await self.abort_transaction(transaction_id)
            
            return TransactionResult(
                success=False,
                transaction_id=transaction_id,
                state=TransactionState.ABORTED,
                error_message=str(e)
            )
    
    async def abort_transaction(self, transaction_id: str) -> TransactionResult:
        """Abort a transaction"""
        start_time = time.time()
        
        with self.lock:
            transaction = self.active_transactions.get(transaction_id)
            if not transaction:
                return TransactionResult(
                    success=False,
                    transaction_id=transaction_id,
                    state=TransactionState.ABORTED,
                    error_message="Transaction not found"
                )
            
            # Update state to aborting
            transaction.state = TransactionState.ABORTING
            transaction.updated_at = datetime.now()
        
        try:
            # Persist state change
            if self.redis_client:
                await self._persist_transaction(transaction)
            
            # Send abort requests to all participants
            await self.two_phase_commit.abort_phase(transaction)
            
            # Update state to aborted
            with self.lock:
                transaction.state = TransactionState.ABORTED
                transaction.updated_at = datetime.now()
                self.completed_transactions[transaction_id] = transaction
                del self.active_transactions[transaction_id]
            
            if self.redis_client:
                await self._persist_transaction(transaction)
            
            # Update statistics
            execution_time = time.time() - start_time
            self.stats['aborted_transactions'] += 1
            self.stats['average_abort_time'] = (
                (self.stats['average_abort_time'] * (self.stats['aborted_transactions'] - 1) + 
                 execution_time) / self.stats['aborted_transactions']
            )
            
            logger.info(f"Successfully aborted transaction {transaction_id}")
            return TransactionResult(
                success=True,
                transaction_id=transaction_id,
                state=TransactionState.ABORTED,
                participants_aborted=list(transaction.participants),
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Abort failed for transaction {transaction_id}: {e}")
            
            # Still mark as aborted
            with self.lock:
                transaction.state = TransactionState.ABORTED
                transaction.updated_at = datetime.now()
                self.completed_transactions[transaction_id] = transaction
                if transaction_id in self.active_transactions:
                    del self.active_transactions[transaction_id]
            
            return TransactionResult(
                success=False,
                transaction_id=transaction_id,
                state=TransactionState.ABORTED,
                error_message=str(e)
            )
    
    async def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get a transaction by ID"""
        with self.lock:
            # Check active transactions first
            if transaction_id in self.active_transactions:
                return self.active_transactions[transaction_id]
            
            # Check completed transactions
            if transaction_id in self.completed_transactions:
                return self.completed_transactions[transaction_id]
            
            # Check Redis if available
            if self.redis_client:
                return await self._load_transaction_from_redis(transaction_id)
            
            return None
    
    async def _persist_transaction(self, transaction: Transaction):
        """Persist transaction to Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"{self.config.redis_key_prefix}{transaction.transaction_id}"
            txn_dict = {
                'transaction_id': transaction.transaction_id,
                'state': transaction.state.value,
                'created_at': transaction.created_at.isoformat(),
                'updated_at': transaction.updated_at.isoformat(),
                'timeout': transaction.timeout,
                'isolation_level': transaction.isolation_level.value,
                'participants': list(transaction.participants),
                'operations': transaction.operations,
                'metadata': transaction.metadata,
                'retry_count': transaction.retry_count,
                'error_message': transaction.error_message
            }
            
            self.redis_client.setex(
                key, 
                self.config.max_transaction_age,
                json.dumps(txn_dict, default=str)
            )
        except Exception as e:
            logger.warning(f"Failed to persist transaction {transaction.transaction_id}: {e}")
    
    async def _load_transaction_from_redis(self, transaction_id: str) -> Optional[Transaction]:
        """Load transaction from Redis"""
        if not self.redis_client:
            return None
        
        try:
            key = f"{self.config.redis_key_prefix}{transaction_id}"
            txn_data = self.redis_client.get(key)
            
            if not txn_data:
                return None
            
            txn_dict = json.loads(txn_data)
            
            transaction = Transaction(
                transaction_id=txn_dict['transaction_id'],
                state=TransactionState(txn_dict['state']),
                created_at=datetime.fromisoformat(txn_dict['created_at']),
                updated_at=datetime.fromisoformat(txn_dict['updated_at']),
                timeout=txn_dict['timeout'],
                isolation_level=IsolationLevel(txn_dict['isolation_level']),
                participants=set(txn_dict['participants']),
                operations=txn_dict['operations'],
                metadata=txn_dict['metadata'],
                retry_count=txn_dict.get('retry_count', 0),
                error_message=txn_dict.get('error_message')
            )
            
            return transaction
        except Exception as e:
            logger.warning(f"Failed to load transaction {transaction_id} from Redis: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        with self.lock:
            return {
                **self.stats,
                'active_transactions': len(self.active_transactions),
                'completed_transactions': len(self.completed_transactions),
                'registered_participants': len(self.participant_manager.participants)
            }
    
    def get_active_transactions(self) -> List[Transaction]:
        """Get all active transactions"""
        with self.lock:
            return list(self.active_transactions.values())
    
    def get_completed_transactions(self) -> List[Transaction]:
        """Get all completed transactions"""
        with self.lock:
            return list(self.completed_transactions.values())
    
    async def shutdown(self):
        """Shutdown the coordinator"""
        logger.info("Shutting down transaction coordinator")
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.health_check_task:
            self.health_check_task.cancel()
        
        # Abort all active transactions
        active_txns = list(self.active_transactions.keys())
        for txn_id in active_txns:
            await self.abort_transaction(txn_id)
        
        logger.info("Transaction coordinator shutdown complete")

# Example usage
async def main():
    """Example usage of the transaction coordinator"""
    config = TransactionConfig(
        default_timeout=30,
        max_concurrent_transactions=100
    )
    
    coordinator = TransactionCoordinator(config)
    
    # Register some mock participants
    class MockParticipant:
        def __init__(self, participant_id: str):
            self.participant_id = participant_id
        
        async def prepare(self, transaction_id: str, operations: List[Dict[str, Any]]) -> bool:
            print(f"Participant {self.participant_id} prepared for transaction {transaction_id}")
            return True
        
        async def commit(self, transaction_id: str) -> bool:
            print(f"Participant {self.participant_id} committed transaction {transaction_id}")
            return True
        
        async def abort(self, transaction_id: str) -> bool:
            print(f"Participant {self.participant_id} aborted transaction {transaction_id}")
            return True
        
        async def health_check(self) -> bool:
            return True
    
    # Register participants
    participant1 = MockParticipant("db1")
    participant2 = MockParticipant("db2")
    participant3 = MockParticipant("cache1")
    
    coordinator.register_participant("db1", participant1)
    coordinator.register_participant("db2", participant2)
    coordinator.register_participant("cache1", participant3)
    
    # Begin a transaction
    txn_id = await coordinator.begin_transaction(
        timeout=30,
        isolation_level=IsolationLevel.READ_COMMITTED,
        metadata={"user_id": "123", "operation": "update_profile"}
    )
    
    # Add participants
    await coordinator.add_participant(txn_id, "db1")
    await coordinator.add_participant(txn_id, "db2")
    await coordinator.add_participant(txn_id, "cache1")
    
    # Add operations
    await coordinator.add_operation(txn_id, {
        "type": "update",
        "table": "users",
        "data": {"name": "John Doe", "email": "john@example.com"}
    })
    
    await coordinator.add_operation(txn_id, {
        "type": "invalidate",
        "cache_key": "user:123"
    })
    
    # Commit the transaction
    result = await coordinator.commit_transaction(txn_id)
    
    print(f"Transaction result: {result}")
    
    # Get statistics
    stats = coordinator.get_stats()
    print(f"Coordinator stats: {stats}")
    
    # Shutdown
    await coordinator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
