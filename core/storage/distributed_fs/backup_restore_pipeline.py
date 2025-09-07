# T3SS Project
# File: core/storage/distributed_fs/backup_restore_pipeline.py
# (c) 2025 Qiss Labs. All Rights Reserved.
# Unauthorized copying or distribution of this file is strictly prohibited.
# For internal use only.

import asyncio
import logging
import hashlib
import shutil
import tarfile
import gzip
import bz2
import lzma
import zlib
import time
import os
import json
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis
import psutil
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import tempfile
import subprocess
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompressionType(Enum):
    """Compression types for backup files"""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"
    ZLIB = "zlib"

class EncryptionType(Enum):
    """Encryption types for backup files"""
    NONE = "none"
    AES256 = "aes256"
    FERNET = "fernet"

class BackupStatus(Enum):
    """Backup operation status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    VERIFYING = "verifying"

class RestoreStatus(Enum):
    """Restore operation status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    VERIFYING = "verifying"

@dataclass
class BackupConfig:
    """Configuration for backup operations"""
    # Source settings
    source_paths: List[str]
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)
    
    # Destination settings
    backup_root: str = "/backups"
    backup_name: str = "backup"
    timestamp_format: str = "%Y%m%d_%H%M%S"
    
    # Compression settings
    compression_type: CompressionType = CompressionType.GZIP
    compression_level: int = 6
    
    # Encryption settings
    encryption_type: EncryptionType = EncryptionType.FERNET
    encryption_key: Optional[str] = None
    
    # Performance settings
    max_workers: int = 4
    chunk_size: int = 1024 * 1024  # 1MB
    buffer_size: int = 64 * 1024  # 64KB
    
    # Verification settings
    verify_checksums: bool = True
    verify_integrity: bool = True
    
    # Retention settings
    max_backups: int = 30
    retention_days: int = 90
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

@dataclass
class RestoreConfig:
    """Configuration for restore operations"""
    # Source settings
    backup_path: str
    restore_path: str
    
    # Filter settings
    include_files: List[str] = field(default_factory=list)
    exclude_files: List[str] = field(default_factory=list)
    
    # Performance settings
    max_workers: int = 4
    chunk_size: int = 1024 * 1024  # 1MB
    
    # Verification settings
    verify_checksums: bool = True
    verify_integrity: bool = True

@dataclass
class BackupMetadata:
    """Metadata for backup operations"""
    backup_id: str
    backup_name: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    source_paths: List[str] = field(default_factory=list)
    backup_path: str = ""
    compression_type: CompressionType = CompressionType.NONE
    encryption_type: EncryptionType = EncryptionType.NONE
    file_count: int = 0
    total_size: int = 0
    compressed_size: int = 0
    checksum: str = ""
    status: BackupStatus = BackupStatus.PENDING
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RestoreMetadata:
    """Metadata for restore operations"""
    restore_id: str
    backup_id: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    backup_path: str = ""
    restore_path: str = ""
    file_count: int = 0
    total_size: int = 0
    status: RestoreStatus = RestoreStatus.PENDING
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class CompressionManager:
    """Manages compression operations"""
    
    @staticmethod
    def compress_file(input_path: str, output_path: str, 
                     compression_type: CompressionType, 
                     compression_level: int = 6) -> bool:
        """Compress a file"""
        try:
            if compression_type == CompressionType.NONE:
                shutil.copy2(input_path, output_path)
                return True
            elif compression_type == CompressionType.GZIP:
                with open(input_path, 'rb') as f_in:
                    with gzip.open(output_path, 'wb', compresslevel=compression_level) as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return True
            elif compression_type == CompressionType.BZIP2:
                with open(input_path, 'rb') as f_in:
                    with bz2.open(output_path, 'wb', compresslevel=compression_level) as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return True
            elif compression_type == CompressionType.LZMA:
                with open(input_path, 'rb') as f_in:
                    with lzma.open(output_path, 'wb', preset=compression_level) as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return True
            elif compression_type == CompressionType.ZLIB:
                with open(input_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        f_out.write(zlib.compress(f_in.read(), compression_level))
                return True
            else:
                logger.error(f"Unsupported compression type: {compression_type}")
                return False
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return False
    
    @staticmethod
    def decompress_file(input_path: str, output_path: str, 
                       compression_type: CompressionType) -> bool:
        """Decompress a file"""
        try:
            if compression_type == CompressionType.NONE:
                shutil.copy2(input_path, output_path)
                return True
            elif compression_type == CompressionType.GZIP:
                with gzip.open(input_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return True
            elif compression_type == CompressionType.BZIP2:
                with bz2.open(input_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return True
            elif compression_type == CompressionType.LZMA:
                with lzma.open(input_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return True
            elif compression_type == CompressionType.ZLIB:
                with open(input_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        f_out.write(zlib.decompress(f_in.read()))
                return True
            else:
                logger.error(f"Unsupported compression type: {compression_type}")
                return False
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return False

class EncryptionManager:
    """Manages encryption operations"""
    
    def __init__(self, encryption_type: EncryptionType, key: Optional[str] = None):
        self.encryption_type = encryption_type
        self.key = key or self._generate_key()
        self.cipher = self._create_cipher()
    
    def _generate_key(self) -> str:
        """Generate a new encryption key"""
        if self.encryption_type == EncryptionType.FERNET:
            return Fernet.generate_key().decode()
        elif self.encryption_type == EncryptionType.AES256:
            return base64.urlsafe_b64encode(os.urandom(32)).decode()
        else:
            return ""
    
    def _create_cipher(self):
        """Create encryption cipher"""
        if self.encryption_type == EncryptionType.FERNET:
            return Fernet(self.key.encode())
        elif self.encryption_type == EncryptionType.AES256:
            # Use PBKDF2 to derive key from password
            password = self.key.encode()
            salt = b'salt'  # In production, use random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            return Fernet(key)
        else:
            return None
    
    def encrypt_file(self, input_path: str, output_path: str) -> bool:
        """Encrypt a file"""
        try:
            if self.encryption_type == EncryptionType.NONE:
                shutil.copy2(input_path, output_path)
                return True
            
            if not self.cipher:
                logger.error("No cipher available for encryption")
                return False
            
            with open(input_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    data = f_in.read()
                    encrypted_data = self.cipher.encrypt(data)
                    f_out.write(encrypted_data)
            return True
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return False
    
    def decrypt_file(self, input_path: str, output_path: str) -> bool:
        """Decrypt a file"""
        try:
            if self.encryption_type == EncryptionType.NONE:
                shutil.copy2(input_path, output_path)
                return True
            
            if not self.cipher:
                logger.error("No cipher available for decryption")
                return False
            
            with open(input_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    encrypted_data = f_in.read()
                    decrypted_data = self.cipher.decrypt(encrypted_data)
                    f_out.write(decrypted_data)
            return True
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return False

class FileScanner:
    """Scans files for backup operations"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
    
    def scan_files(self) -> List[Tuple[str, int, str]]:
        """Scan files to backup"""
        files = []
        
        for source_path in self.config.source_paths:
            if os.path.isfile(source_path):
                files.append((source_path, os.path.getsize(source_path), 
                            self._calculate_checksum(source_path)))
            elif os.path.isdir(source_path):
                files.extend(self._scan_directory(source_path))
        
        return files
    
    def _scan_directory(self, directory: str) -> List[Tuple[str, int, str]]:
        """Scan directory recursively"""
        files = []
        
        for root, dirs, filenames in os.walk(directory):
            # Apply exclude patterns
            dirs[:] = [d for d in dirs if not self._matches_patterns(d, self.config.exclude_patterns)]
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                
                # Apply include/exclude patterns
                if self._should_include_file(file_path):
                    try:
                        size = os.path.getsize(file_path)
                        checksum = self._calculate_checksum(file_path)
                        files.append((file_path, size, checksum))
                    except (OSError, IOError) as e:
                        logger.warning(f"Cannot access file {file_path}: {e}")
        
        return files
    
    def _should_include_file(self, file_path: str) -> bool:
        """Check if file should be included in backup"""
        # Check exclude patterns
        if self._matches_patterns(file_path, self.config.exclude_patterns):
            return False
        
        # Check include patterns
        if self.config.include_patterns:
            return self._matches_patterns(file_path, self.config.include_patterns)
        
        return True
    
    def _matches_patterns(self, file_path: str, patterns: List[str]) -> bool:
        """Check if file matches any pattern"""
        import fnmatch
        
        for pattern in patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True
        return False
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Cannot calculate checksum for {file_path}: {e}")
            return ""

class BackupPipeline:
    """Advanced backup and restore pipeline"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.compression_manager = CompressionManager()
        self.encryption_manager = EncryptionManager(
            config.encryption_type, 
            config.encryption_key
        )
        self.file_scanner = FileScanner(config)
        
        # Redis connection for metadata storage
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
                decode_responses=True
            )
            self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage only.")
            self.redis_client = None
        
        # Statistics
        self.stats = {
            'total_backups': 0,
            'successful_backups': 0,
            'failed_backups': 0,
            'total_restores': 0,
            'successful_restores': 0,
            'failed_restores': 0,
            'total_data_backed_up': 0,
            'total_data_restored': 0
        }
    
    async def create_backup(self) -> str:
        """Create a new backup"""
        backup_id = f"backup_{int(time.time())}"
        timestamp = datetime.now().strftime(self.config.timestamp_format)
        backup_name = f"{self.config.backup_name}_{timestamp}"
        backup_path = os.path.join(self.config.backup_root, backup_name)
        
        # Create backup metadata
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_name=backup_name,
            created_at=datetime.now(),
            source_paths=self.config.source_paths.copy(),
            backup_path=backup_path,
            compression_type=self.config.compression_type,
            encryption_type=self.config.encryption_type,
            status=BackupStatus.PENDING
        )
        
        # Store metadata
        await self._store_backup_metadata(metadata)
        
        try:
            # Create backup directory
            os.makedirs(backup_path, exist_ok=True)
            
            # Update status
            metadata.status = BackupStatus.RUNNING
            await self._store_backup_metadata(metadata)
            
            # Scan files
            logger.info("Scanning files for backup...")
            files = self.file_scanner.scan_files()
            metadata.file_count = len(files)
            metadata.total_size = sum(size for _, size, _ in files)
            
            # Create file list
            file_list_path = os.path.join(backup_path, "file_list.json")
            file_list = []
            
            # Process files
            processed_size = 0
            for file_path, size, checksum in files:
                try:
                    # Calculate relative path
                    rel_path = self._get_relative_path(file_path)
                    backup_file_path = os.path.join(backup_path, rel_path)
                    
                    # Create directory if needed
                    os.makedirs(os.path.dirname(backup_file_path), exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(file_path, backup_file_path)
                    
                    # Compress if needed
                    if self.config.compression_type != CompressionType.NONE:
                        compressed_path = backup_file_path + ".compressed"
                        if self.compression_manager.compress_file(
                            backup_file_path, compressed_path, 
                            self.config.compression_type, 
                            self.config.compression_level
                        ):
                            os.remove(backup_file_path)
                            backup_file_path = compressed_path
                    
                    # Encrypt if needed
                    if self.config.encryption_type != EncryptionType.NONE:
                        encrypted_path = backup_file_path + ".encrypted"
                        if self.encryption_manager.encrypt_file(
                            backup_file_path, encrypted_path
                        ):
                            os.remove(backup_file_path)
                            backup_file_path = encrypted_path
                    
                    # Add to file list
                    file_list.append({
                        'original_path': file_path,
                        'backup_path': rel_path,
                        'size': size,
                        'checksum': checksum,
                        'compressed': self.config.compression_type != CompressionType.NONE,
                        'encrypted': self.config.encryption_type != EncryptionType.NONE
                    })
                    
                    processed_size += size
                    
                    # Update progress
                    if processed_size % (100 * 1024 * 1024) == 0:  # Every 100MB
                        logger.info(f"Processed {processed_size / (1024 * 1024):.1f} MB")
                
                except Exception as e:
                    logger.error(f"Failed to backup file {file_path}: {e}")
                    continue
            
            # Save file list
            with open(file_list_path, 'w') as f:
                json.dump(file_list, f, indent=2)
            
            # Calculate final checksum
            if self.config.verify_checksums:
                metadata.checksum = self._calculate_directory_checksum(backup_path)
            
            # Update metadata
            metadata.file_count = len(file_list)
            metadata.compressed_size = self._calculate_directory_size(backup_path)
            metadata.status = BackupStatus.VERIFYING
            await self._store_backup_metadata(metadata)
            
            # Verify backup
            if self.config.verify_integrity:
                if await self._verify_backup(metadata):
                    metadata.status = BackupStatus.COMPLETED
                else:
                    metadata.status = BackupStatus.FAILED
                    metadata.error_message = "Backup verification failed"
            else:
                metadata.status = BackupStatus.COMPLETED
            
            metadata.completed_at = datetime.now()
            await self._store_backup_metadata(metadata)
            
            # Update statistics
            self.stats['total_backups'] += 1
            if metadata.status == BackupStatus.COMPLETED:
                self.stats['successful_backups'] += 1
                self.stats['total_data_backed_up'] += metadata.total_size
            else:
                self.stats['failed_backups'] += 1
            
            logger.info(f"Backup {backup_id} completed with status {metadata.status.value}")
            return backup_id
            
        except Exception as e:
            logger.error(f"Backup {backup_id} failed: {e}")
            metadata.status = BackupStatus.FAILED
            metadata.error_message = str(e)
            metadata.completed_at = datetime.now()
            await self._store_backup_metadata(metadata)
            
            self.stats['total_backups'] += 1
            self.stats['failed_backups'] += 1
            
            return backup_id
    
    async def restore_backup(self, backup_id: str, restore_path: str) -> str:
        """Restore a backup"""
        restore_id = f"restore_{int(time.time())}"
        
        # Get backup metadata
        metadata = await self._get_backup_metadata(backup_id)
        if not metadata:
            raise ValueError(f"Backup {backup_id} not found")
        
        # Create restore metadata
        restore_metadata = RestoreMetadata(
            restore_id=restore_id,
            backup_id=backup_id,
            created_at=datetime.now(),
            backup_path=metadata.backup_path,
            restore_path=restore_path,
            status=RestoreStatus.PENDING
        )
        
        # Store restore metadata
        await self._store_restore_metadata(restore_metadata)
        
        try:
            # Create restore directory
            os.makedirs(restore_path, exist_ok=True)
            
            # Update status
            restore_metadata.status = RestoreStatus.RUNNING
            await self._store_restore_metadata(restore_metadata)
            
            # Load file list
            file_list_path = os.path.join(metadata.backup_path, "file_list.json")
            if not os.path.exists(file_list_path):
                raise ValueError("File list not found in backup")
            
            with open(file_list_path, 'r') as f:
                file_list = json.load(f)
            
            # Restore files
            restored_size = 0
            for file_info in file_list:
                try:
                    backup_file_path = os.path.join(metadata.backup_path, file_info['backup_path'])
                    restore_file_path = os.path.join(restore_path, file_info['original_path'])
                    
                    # Create directory if needed
                    os.makedirs(os.path.dirname(restore_file_path), exist_ok=True)
                    
                    # Decrypt if needed
                    if file_info.get('encrypted', False):
                        decrypted_path = backup_file_path + ".decrypted"
                        if self.encryption_manager.decrypt_file(backup_file_path, decrypted_path):
                            os.remove(backup_file_path)
                            backup_file_path = decrypted_path
                    
                    # Decompress if needed
                    if file_info.get('compressed', False):
                        decompressed_path = backup_file_path + ".decompressed"
                        if self.compression_manager.decompress_file(
                            backup_file_path, decompressed_path, 
                            metadata.compression_type
                        ):
                            os.remove(backup_file_path)
                            backup_file_path = decompressed_path
                    
                    # Copy file
                    shutil.copy2(backup_file_path, restore_file_path)
                    
                    # Verify checksum if needed
                    if self.config.verify_checksums and file_info.get('checksum'):
                        if self._calculate_checksum(restore_file_path) != file_info['checksum']:
                            logger.warning(f"Checksum mismatch for {restore_file_path}")
                    
                    restored_size += file_info['size']
                    
                    # Update progress
                    if restored_size % (100 * 1024 * 1024) == 0:  # Every 100MB
                        logger.info(f"Restored {restored_size / (1024 * 1024):.1f} MB")
                
                except Exception as e:
                    logger.error(f"Failed to restore file {file_info['original_path']}: {e}")
                    continue
            
            # Update metadata
            restore_metadata.file_count = len(file_list)
            restore_metadata.total_size = restored_size
            restore_metadata.status = RestoreStatus.VERIFYING
            await self._store_restore_metadata(restore_metadata)
            
            # Verify restore
            if self.config.verify_integrity:
                if await self._verify_restore(restore_metadata):
                    restore_metadata.status = RestoreStatus.COMPLETED
                else:
                    restore_metadata.status = RestoreStatus.FAILED
                    restore_metadata.error_message = "Restore verification failed"
            else:
                restore_metadata.status = RestoreStatus.COMPLETED
            
            restore_metadata.completed_at = datetime.now()
            await self._store_restore_metadata(restore_metadata)
            
            # Update statistics
            self.stats['total_restores'] += 1
            if restore_metadata.status == RestoreStatus.COMPLETED:
                self.stats['successful_restores'] += 1
                self.stats['total_data_restored'] += restored_size
            else:
                self.stats['failed_restores'] += 1
            
            logger.info(f"Restore {restore_id} completed with status {restore_metadata.status.value}")
            return restore_id
            
        except Exception as e:
            logger.error(f"Restore {restore_id} failed: {e}")
            restore_metadata.status = RestoreStatus.FAILED
            restore_metadata.error_message = str(e)
            restore_metadata.completed_at = datetime.now()
            await self._store_restore_metadata(restore_metadata)
            
            self.stats['total_restores'] += 1
            self.stats['failed_restores'] += 1
            
            return restore_id
    
    def _get_relative_path(self, file_path: str) -> str:
        """Get relative path from source paths"""
        for source_path in self.config.source_paths:
            if file_path.startswith(source_path):
                return os.path.relpath(file_path, source_path)
        return os.path.basename(file_path)
    
    def _calculate_directory_checksum(self, directory: str) -> str:
        """Calculate checksum for entire directory"""
        hash_sha256 = hashlib.sha256()
        
        for root, dirs, files in os.walk(directory):
            for filename in sorted(files):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_sha256.update(chunk)
                except (OSError, IOError):
                    continue
        
        return hash_sha256.hexdigest()
    
    def _calculate_directory_size(self, directory: str) -> int:
        """Calculate total size of directory"""
        total_size = 0
        for root, dirs, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    total_size += os.path.getsize(file_path)
                except (OSError, IOError):
                    continue
        return total_size
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Cannot calculate checksum for {file_path}: {e}")
            return ""
    
    async def _verify_backup(self, metadata: BackupMetadata) -> bool:
        """Verify backup integrity"""
        try:
            # Check if backup directory exists
            if not os.path.exists(metadata.backup_path):
                return False
            
            # Check file list
            file_list_path = os.path.join(metadata.backup_path, "file_list.json")
            if not os.path.exists(file_list_path):
                return False
            
            # Verify checksum if available
            if metadata.checksum:
                calculated_checksum = self._calculate_directory_checksum(metadata.backup_path)
                if calculated_checksum != metadata.checksum:
                    logger.warning("Backup checksum verification failed")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
    
    async def _verify_restore(self, metadata: RestoreMetadata) -> bool:
        """Verify restore integrity"""
        try:
            # Check if restore directory exists
            if not os.path.exists(metadata.restore_path):
                return False
            
            # Check file count
            restored_files = 0
            for root, dirs, files in os.walk(metadata.restore_path):
                restored_files += len(files)
            
            if restored_files != metadata.file_count:
                logger.warning(f"File count mismatch: expected {metadata.file_count}, got {restored_files}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Restore verification failed: {e}")
            return False
    
    async def _store_backup_metadata(self, metadata: BackupMetadata):
        """Store backup metadata"""
        if self.redis_client:
            try:
                key = f"backup:{metadata.backup_id}"
                data = {
                    'backup_id': metadata.backup_id,
                    'backup_name': metadata.backup_name,
                    'created_at': metadata.created_at.isoformat(),
                    'completed_at': metadata.completed_at.isoformat() if metadata.completed_at else None,
                    'source_paths': metadata.source_paths,
                    'backup_path': metadata.backup_path,
                    'compression_type': metadata.compression_type.value,
                    'encryption_type': metadata.encryption_type.value,
                    'file_count': metadata.file_count,
                    'total_size': metadata.total_size,
                    'compressed_size': metadata.compressed_size,
                    'checksum': metadata.checksum,
                    'status': metadata.status.value,
                    'error_message': metadata.error_message,
                    'metadata': metadata.metadata
                }
                self.redis_client.setex(key, 86400 * 30, json.dumps(data, default=str))  # 30 days
            except Exception as e:
                logger.warning(f"Failed to store backup metadata: {e}")
    
    async def _get_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get backup metadata"""
        if self.redis_client:
            try:
                key = f"backup:{backup_id}"
                data = self.redis_client.get(key)
                if data:
                    data_dict = json.loads(data)
                    return BackupMetadata(
                        backup_id=data_dict['backup_id'],
                        backup_name=data_dict['backup_name'],
                        created_at=datetime.fromisoformat(data_dict['created_at']),
                        completed_at=datetime.fromisoformat(data_dict['completed_at']) if data_dict['completed_at'] else None,
                        source_paths=data_dict['source_paths'],
                        backup_path=data_dict['backup_path'],
                        compression_type=CompressionType(data_dict['compression_type']),
                        encryption_type=EncryptionType(data_dict['encryption_type']),
                        file_count=data_dict['file_count'],
                        total_size=data_dict['total_size'],
                        compressed_size=data_dict['compressed_size'],
                        checksum=data_dict['checksum'],
                        status=BackupStatus(data_dict['status']),
                        error_message=data_dict.get('error_message'),
                        metadata=data_dict.get('metadata', {})
                    )
            except Exception as e:
                logger.warning(f"Failed to get backup metadata: {e}")
        
        return None
    
    async def _store_restore_metadata(self, metadata: RestoreMetadata):
        """Store restore metadata"""
        if self.redis_client:
            try:
                key = f"restore:{metadata.restore_id}"
                data = {
                    'restore_id': metadata.restore_id,
                    'backup_id': metadata.backup_id,
                    'created_at': metadata.created_at.isoformat(),
                    'completed_at': metadata.completed_at.isoformat() if metadata.completed_at else None,
                    'backup_path': metadata.backup_path,
                    'restore_path': metadata.restore_path,
                    'file_count': metadata.file_count,
                    'total_size': metadata.total_size,
                    'status': metadata.status.value,
                    'error_message': metadata.error_message,
                    'metadata': metadata.metadata
                }
                self.redis_client.setex(key, 86400 * 7, json.dumps(data, default=str))  # 7 days
            except Exception as e:
                logger.warning(f"Failed to store restore metadata: {e}")
    
    async def _get_restore_metadata(self, restore_id: str) -> Optional[RestoreMetadata]:
        """Get restore metadata"""
        if self.redis_client:
            try:
                key = f"restore:{restore_id}"
                data = self.redis_client.get(key)
                if data:
                    data_dict = json.loads(data)
                    return RestoreMetadata(
                        restore_id=data_dict['restore_id'],
                        backup_id=data_dict['backup_id'],
                        created_at=datetime.fromisoformat(data_dict['created_at']),
                        completed_at=datetime.fromisoformat(data_dict['completed_at']) if data_dict['completed_at'] else None,
                        backup_path=data_dict['backup_path'],
                        restore_path=data_dict['restore_path'],
                        file_count=data_dict['file_count'],
                        total_size=data_dict['total_size'],
                        status=RestoreStatus(data_dict['status']),
                        error_message=data_dict.get('error_message'),
                        metadata=data_dict.get('metadata', {})
                    )
            except Exception as e:
                logger.warning(f"Failed to get restore metadata: {e}")
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.stats.copy()
    
    async def cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        try:
            # Get all backup metadata
            if not self.redis_client:
                return
            
            keys = self.redis_client.keys("backup:*")
            backups = []
            
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    data_dict = json.loads(data)
                    backups.append(data_dict)
            
            # Sort by creation time
            backups.sort(key=lambda x: x['created_at'])
            
            # Remove old backups
            removed_count = 0
            for backup in backups[:-self.config.max_backups]:
                # Remove backup directory
                backup_path = backup['backup_path']
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path)
                
                # Remove metadata
                self.redis_client.delete(f"backup:{backup['backup_id']}")
                removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} old backups")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Example usage
async def main():
    """Example usage of the backup pipeline"""
    config = BackupConfig(
        source_paths=["/home/user/documents", "/home/user/photos"],
        exclude_patterns=["*.tmp", "*.log"],
        backup_root="/backups",
        backup_name="user_backup",
        compression_type=CompressionType.GZIP,
        encryption_type=EncryptionType.FERNET,
        verify_checksums=True,
        verify_integrity=True
    )
    
    pipeline = BackupPipeline(config)
    
    # Create backup
    print("Creating backup...")
    backup_id = await pipeline.create_backup()
    print(f"Backup created: {backup_id}")
    
    # Restore backup
    print("Restoring backup...")
    restore_id = await pipeline.restore_backup(backup_id, "/restore")
    print(f"Restore completed: {restore_id}")
    
    # Get statistics
    stats = pipeline.get_stats()
    print(f"Pipeline statistics: {stats}")
    
    # Cleanup old backups
    await pipeline.cleanup_old_backups()

if __name__ == "__main__":
    asyncio.run(main())
