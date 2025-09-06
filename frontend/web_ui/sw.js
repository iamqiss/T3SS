/**
 * T3SS Project - Service Worker
 * (c) 2025 Qiss Labs. All Rights Reserved.
 */

const CACHE_NAME = 't3ss-v1.0.0';
const STATIC_CACHE = 't3ss-static-v1.0.0';
const DYNAMIC_CACHE = 't3ss-dynamic-v1.0.0';

// Files to cache on install
const STATIC_FILES = [
    '/',
    '/index.html',
    '/css/main.css',
    '/js/app.js',
    '/js/components.js',
    '/js/utils.js',
    '/manifest.json',
    '/favicon.ico',
    '/images/logo.png',
    '/images/logo-dark.png',
    '/images/icon-192.png',
    '/images/icon-512.png'
];

// API endpoints to cache
const API_CACHE_PATTERNS = [
    /^\/api\/v1\/search\/suggest/,
    /^\/api\/v1\/search\/history/,
    /^\/api\/v1\/search\/trending/
];

// Install event - cache static files
self.addEventListener('install', (event) => {
    console.log('Service Worker: Installing...');
    
    event.waitUntil(
        caches.open(STATIC_CACHE)
            .then((cache) => {
                console.log('Service Worker: Caching static files');
                return cache.addAll(STATIC_FILES);
            })
            .then(() => {
                console.log('Service Worker: Static files cached');
                return self.skipWaiting();
            })
            .catch((error) => {
                console.error('Service Worker: Failed to cache static files', error);
            })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    console.log('Service Worker: Activating...');
    
    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames.map((cacheName) => {
                        if (cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE) {
                            console.log('Service Worker: Deleting old cache', cacheName);
                            return caches.delete(cacheName);
                        }
                    })
                );
            })
            .then(() => {
                console.log('Service Worker: Activated');
                return self.clients.claim();
            })
    );
});

// Fetch event - serve from cache or network
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);
    
    // Skip non-GET requests
    if (request.method !== 'GET') {
        return;
    }
    
    // Skip chrome-extension and other non-http requests
    if (!url.protocol.startsWith('http')) {
        return;
    }
    
    event.respondWith(
        handleRequest(request)
    );
});

async function handleRequest(request) {
    const url = new URL(request.url);
    
    try {
        // Handle static files
        if (isStaticFile(url.pathname)) {
            return await handleStaticFile(request);
        }
        
        // Handle API requests
        if (url.pathname.startsWith('/api/')) {
            return await handleApiRequest(request);
        }
        
        // Handle navigation requests
        if (request.mode === 'navigate') {
            return await handleNavigationRequest(request);
        }
        
        // Default: try network first, then cache
        return await networkFirst(request);
        
    } catch (error) {
        console.error('Service Worker: Request failed', error);
        return new Response('Service Worker Error', { status: 500 });
    }
}

async function handleStaticFile(request) {
    // Try cache first for static files
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
        return cachedResponse;
    }
    
    // If not in cache, fetch from network and cache
    try {
        const networkResponse = await fetch(request);
        if (networkResponse.ok) {
            const cache = await caches.open(STATIC_CACHE);
            cache.put(request, networkResponse.clone());
        }
        return networkResponse;
    } catch (error) {
        // Return offline page for navigation requests
        if (request.mode === 'navigate') {
            return await caches.match('/offline.html') || new Response('Offline', { status: 503 });
        }
        throw error;
    }
}

async function handleApiRequest(request) {
    const url = new URL(request.url);
    
    // Check if this API endpoint should be cached
    const shouldCache = API_CACHE_PATTERNS.some(pattern => pattern.test(url.pathname));
    
    if (shouldCache) {
        return await cacheFirst(request);
    } else {
        return await networkFirst(request);
    }
}

async function handleNavigationRequest(request) {
    // For navigation requests, try network first
    try {
        const networkResponse = await fetch(request);
        return networkResponse;
    } catch (error) {
        // If network fails, try cache
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }
        
        // If no cache, return offline page
        return await caches.match('/offline.html') || new Response('Offline', { status: 503 });
    }
}

async function cacheFirst(request) {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
        return cachedResponse;
    }
    
    try {
        const networkResponse = await fetch(request);
        if (networkResponse.ok) {
            const cache = await caches.open(DYNAMIC_CACHE);
            cache.put(request, networkResponse.clone());
        }
        return networkResponse;
    } catch (error) {
        throw error;
    }
}

async function networkFirst(request) {
    try {
        const networkResponse = await fetch(request);
        if (networkResponse.ok) {
            const cache = await caches.open(DYNAMIC_CACHE);
            cache.put(request, networkResponse.clone());
        }
        return networkResponse;
    } catch (error) {
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }
        throw error;
    }
}

function isStaticFile(pathname) {
    const staticExtensions = ['.html', '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.woff', '.woff2', '.ttf', '.eot'];
    return staticExtensions.some(ext => pathname.endsWith(ext));
}

// Background sync for offline actions
self.addEventListener('sync', (event) => {
    console.log('Service Worker: Background sync', event.tag);
    
    if (event.tag === 'search-sync') {
        event.waitUntil(syncSearchHistory());
    }
});

async function syncSearchHistory() {
    try {
        // Get offline search history from IndexedDB
        const offlineSearches = await getOfflineSearches();
        
        if (offlineSearches.length > 0) {
            // Sync with server
            const response = await fetch('/api/v1/search/history/sync', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ searches: offlineSearches })
            });
            
            if (response.ok) {
                // Clear offline searches
                await clearOfflineSearches();
                console.log('Service Worker: Search history synced');
            }
        }
    } catch (error) {
        console.error('Service Worker: Failed to sync search history', error);
    }
}

// Push notifications
self.addEventListener('push', (event) => {
    console.log('Service Worker: Push received', event);
    
    const options = {
        body: event.data ? event.data.text() : 'New update available',
        icon: '/images/icon-192.png',
        badge: '/images/badge-72.png',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
        },
        actions: [
            {
                action: 'explore',
                title: 'View',
                icon: '/images/checkmark.png'
            },
            {
                action: 'close',
                title: 'Close',
                icon: '/images/xmark.png'
            }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification('T3SS Search', options)
    );
});

// Notification click
self.addEventListener('notificationclick', (event) => {
    console.log('Service Worker: Notification clicked', event);
    
    event.notification.close();
    
    if (event.action === 'explore') {
        event.waitUntil(
            clients.openWindow('/')
        );
    }
});

// Message handling
self.addEventListener('message', (event) => {
    console.log('Service Worker: Message received', event.data);
    
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
    
    if (event.data && event.data.type === 'GET_VERSION') {
        event.ports[0].postMessage({ version: CACHE_NAME });
    }
});

// IndexedDB helpers for offline storage
async function getOfflineSearches() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('t3ss_offline', 1);
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
            const db = request.result;
            const transaction = db.transaction(['searches'], 'readonly');
            const store = transaction.objectStore('searches');
            const getAllRequest = store.getAll();
            
            getAllRequest.onsuccess = () => resolve(getAllRequest.result);
            getAllRequest.onerror = () => reject(getAllRequest.error);
        };
        
        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains('searches')) {
                db.createObjectStore('searches', { keyPath: 'id', autoIncrement: true });
            }
        };
    });
}

async function clearOfflineSearches() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('t3ss_offline', 1);
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
            const db = request.result;
            const transaction = db.transaction(['searches'], 'readwrite');
            const store = transaction.objectStore('searches');
            const clearRequest = store.clear();
            
            clearRequest.onsuccess = () => resolve();
            clearRequest.onerror = () => reject(clearRequest.error);
        };
    });
}

// Cache management
async function cleanOldCaches() {
    const cacheNames = await caches.keys();
    const oldCaches = cacheNames.filter(name => 
        name.startsWith('t3ss-') && name !== STATIC_CACHE && name !== DYNAMIC_CACHE
    );
    
    return Promise.all(
        oldCaches.map(cacheName => caches.delete(cacheName))
    );
}

// Periodic cache cleanup
setInterval(() => {
    cleanOldCaches().then(() => {
        console.log('Service Worker: Old caches cleaned');
    });
}, 24 * 60 * 60 * 1000); // Daily cleanup