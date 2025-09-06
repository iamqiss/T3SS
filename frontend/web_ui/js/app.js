/**
 * T3SS Project - Main Application JavaScript
 * (c) 2025 Qiss Labs. All Rights Reserved.
 */

class T3SSApp {
    constructor() {
        this.config = {
            apiBaseUrl: '/api/v1',
            maxSuggestions: 8,
            debounceDelay: 300,
            searchTimeout: 10000,
            cacheSize: 100,
            enableVoiceSearch: true,
            enableImageSearch: true,
            enableRealTimeUpdates: true
        };
        
        this.state = {
            currentQuery: '',
            currentPage: 1,
            resultsPerPage: 20,
            totalResults: 0,
            searchTime: 0,
            isLoading: false,
            suggestions: [],
            searchHistory: [],
            userPreferences: this.loadUserPreferences()
        };
        
        this.cache = new Map();
        this.debounceTimer = null;
        this.searchAbortController = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.setupKeyboardShortcuts();
        this.setupTheme();
        this.setupServiceWorker();
        this.loadSearchHistory();
        this.setupRealTimeUpdates();
        
        // Focus search input on page load
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.focus();
        }
        
        console.log('T3SS App initialized');
    }
    
    setupEventListeners() {
        // Search form
        const searchForm = document.getElementById('searchForm');
        if (searchForm) {
            searchForm.addEventListener('submit', (e) => this.handleSearch(e));
        }
        
        // Search input
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => this.handleInputChange(e));
            searchInput.addEventListener('keydown', (e) => this.handleInputKeydown(e));
            searchInput.addEventListener('focus', () => this.showSuggestions());
            searchInput.addEventListener('blur', () => this.hideSuggestions());
        }
        
        // Voice search
        const voiceBtn = document.getElementById('voiceBtn');
        if (voiceBtn) {
            voiceBtn.addEventListener('click', () => this.startVoiceSearch());
        }
        
        // Image search
        const cameraBtn = document.getElementById('cameraBtn');
        if (cameraBtn) {
            cameraBtn.addEventListener('click', () => this.startImageSearch());
        }
        
        // Settings
        const settingsBtn = document.getElementById('settingsBtn');
        const settingsModal = document.getElementById('settingsModal');
        const settingsClose = document.getElementById('settingsClose');
        const settingsCancel = document.getElementById('settingsCancel');
        const settingsSave = document.getElementById('settingsSave');
        
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => this.showSettings());
        }
        
        if (settingsClose) {
            settingsClose.addEventListener('click', () => this.hideSettings());
        }
        
        if (settingsCancel) {
            settingsCancel.addEventListener('click', () => this.hideSettings());
        }
        
        if (settingsSave) {
            settingsSave.addEventListener('click', () => this.saveSettings());
        }
        
        // Modal backdrop click
        if (settingsModal) {
            settingsModal.addEventListener('click', (e) => {
                if (e.target === settingsModal) {
                    this.hideSettings();
                }
            });
        }
        
        // Filter changes
        const filters = ['timeFilter', 'languageFilter', 'regionFilter', 'safeSearchFilter'];
        filters.forEach(filterId => {
            const filter = document.getElementById(filterId);
            if (filter) {
                filter.addEventListener('change', () => this.handleFilterChange());
            }
        });
        
        // Pagination
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('pagination-btn')) {
                const page = parseInt(e.target.dataset.page);
                if (page) {
                    this.goToPage(page);
                }
            }
        });
        
        // Result clicks
        document.addEventListener('click', (e) => {
            if (e.target.closest('.result-item')) {
                const resultItem = e.target.closest('.result-item');
                const url = resultItem.dataset.url;
                if (url) {
                    this.trackResultClick(url, resultItem.dataset.docId);
                }
            }
        });
        
        // Window events
        window.addEventListener('resize', () => this.handleResize());
        window.addEventListener('beforeunload', () => this.saveState());
        
        // Online/offline events
        window.addEventListener('online', () => this.handleOnline());
        window.addEventListener('offline', () => this.handleOffline());
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K to focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                const searchInput = document.getElementById('searchInput');
                if (searchInput) {
                    searchInput.focus();
                    searchInput.select();
                }
            }
            
            // Escape to clear search or close modals
            if (e.key === 'Escape') {
                this.hideSuggestions();
                this.hideSettings();
            }
            
            // Arrow keys for suggestions
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                this.handleSuggestionNavigation(e);
            }
            
            // Enter to select suggestion
            if (e.key === 'Enter' && e.target.classList.contains('search-suggestion')) {
                e.preventDefault();
                this.selectSuggestion(e.target);
            }
        });
    }
    
    setupTheme() {
        const theme = this.state.userPreferences.theme || 'light';
        document.documentElement.setAttribute('data-theme', theme);
        
        // Auto theme detection
        if (theme === 'auto') {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            document.documentElement.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
        }
    }
    
    setupServiceWorker() {
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js')
                .then(registration => {
                    console.log('Service Worker registered:', registration);
                })
                .catch(error => {
                    console.log('Service Worker registration failed:', error);
                });
        }
    }
    
    setupRealTimeUpdates() {
        if (this.config.enableRealTimeUpdates && 'EventSource' in window) {
            this.eventSource = new EventSource('/api/v1/events');
            
            this.eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleRealTimeUpdate(data);
            };
            
            this.eventSource.onerror = (error) => {
                console.error('EventSource error:', error);
            };
        }
    }
    
    async handleSearch(event) {
        event.preventDefault();
        
        const searchInput = document.getElementById('searchInput');
        const query = searchInput.value.trim();
        
        if (!query) {
            return;
        }
        
        this.state.currentQuery = query;
        this.state.currentPage = 1;
        
        // Add to search history
        this.addToSearchHistory(query);
        
        // Hide suggestions
        this.hideSuggestions();
        
        // Perform search
        await this.performSearch(query, 1);
    }
    
    async performSearch(query, page = 1) {
        if (this.state.isLoading) {
            return;
        }
        
        this.state.isLoading = true;
        this.showLoading();
        
        try {
            // Cancel previous search if still running
            if (this.searchAbortController) {
                this.searchAbortController.abort();
            }
            
            this.searchAbortController = new AbortController();
            
            // Check cache first
            const cacheKey = this.getCacheKey(query, page);
            if (this.cache.has(cacheKey)) {
                const cachedResult = this.cache.get(cacheKey);
                this.displayResults(cachedResult);
                this.state.isLoading = false;
                this.hideLoading();
                return;
            }
            
            // Build search parameters
            const params = this.buildSearchParams(query, page);
            
            // Make API request
            const startTime = performance.now();
            const response = await fetch(`${this.config.apiBaseUrl}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify(params),
                signal: this.searchAbortController.signal
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            const endTime = performance.now();
            
            // Update state
            this.state.searchTime = endTime - startTime;
            this.state.totalResults = result.total || 0;
            this.state.currentPage = page;
            
            // Cache result
            this.cache.set(cacheKey, result);
            this.manageCacheSize();
            
            // Display results
            this.displayResults(result);
            
            // Track search
            this.trackSearch(query, result);
            
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error('Search error:', error);
                this.showError('Search failed. Please try again.');
            }
        } finally {
            this.state.isLoading = false;
            this.hideLoading();
        }
    }
    
    buildSearchParams(query, page) {
        const filters = this.getActiveFilters();
        
        return {
            query: query,
            page: page,
            page_size: this.state.resultsPerPage,
            filters: filters,
            user_id: this.getUserId(),
            session_id: this.getSessionId(),
            timestamp: Date.now()
        };
    }
    
    getActiveFilters() {
        const filters = {};
        
        const timeFilter = document.getElementById('timeFilter');
        if (timeFilter && timeFilter.value) {
            filters.time_range = timeFilter.value;
        }
        
        const languageFilter = document.getElementById('languageFilter');
        if (languageFilter && languageFilter.value) {
            filters.language = languageFilter.value;
        }
        
        const regionFilter = document.getElementById('regionFilter');
        if (regionFilter && regionFilter.value) {
            filters.region = regionFilter.value;
        }
        
        const safeSearchFilter = document.getElementById('safeSearchFilter');
        if (safeSearchFilter && safeSearchFilter.value) {
            filters.safe_search = safeSearchFilter.value;
        }
        
        return filters;
    }
    
    displayResults(result) {
        const resultsSection = document.getElementById('resultsSection');
        const resultsContainer = document.getElementById('resultsContainer');
        const resultsCount = document.getElementById('resultsCount');
        const resultsTime = document.getElementById('resultsTime');
        const relatedSection = document.getElementById('relatedSection');
        const relatedSearches = document.getElementById('relatedSearches');
        
        if (!resultsSection || !resultsContainer) {
            return;
        }
        
        // Show results section
        resultsSection.style.display = 'block';
        
        // Update results info
        if (resultsCount) {
            resultsCount.textContent = `About ${result.total.toLocaleString()} results`;
        }
        
        if (resultsTime) {
            resultsTime.textContent = `(${(this.state.searchTime / 1000).toFixed(2)} seconds)`;
        }
        
        // Clear previous results
        resultsContainer.innerHTML = '';
        
        // Display results
        if (result.results && result.results.length > 0) {
            result.results.forEach((item, index) => {
                const resultElement = this.createResultElement(item, index);
                resultsContainer.appendChild(resultElement);
            });
        } else {
            resultsContainer.innerHTML = `
                <div class="no-results">
                    <h3>No results found</h3>
                    <p>Try different keywords or check your spelling.</p>
                </div>
            `;
        }
        
        // Display related searches
        if (result.suggestions && result.suggestions.length > 0) {
            if (relatedSection && relatedSearches) {
                relatedSearches.innerHTML = '';
                result.suggestions.forEach(suggestion => {
                    const suggestionElement = document.createElement('a');
                    suggestionElement.href = '#';
                    suggestionElement.className = 'related-search';
                    suggestionElement.textContent = suggestion;
                    suggestionElement.addEventListener('click', (e) => {
                        e.preventDefault();
                        this.searchSuggestion(suggestion);
                    });
                    relatedSearches.appendChild(suggestionElement);
                });
                relatedSection.style.display = 'block';
            }
        }
        
        // Update pagination
        this.updatePagination(result);
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    createResultElement(item, index) {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'result-item';
        resultDiv.dataset.url = item.url;
        resultDiv.dataset.docId = item.id;
        
        const title = document.createElement('h3');
        title.className = 'result-title';
        title.innerHTML = this.highlightText(item.title, this.state.currentQuery);
        
        const url = document.createElement('div');
        url.className = 'result-url';
        url.textContent = this.formatUrl(item.url);
        
        const snippet = document.createElement('div');
        snippet.className = 'result-snippet';
        snippet.innerHTML = this.highlightText(item.snippet, this.state.currentQuery);
        
        const meta = document.createElement('div');
        meta.className = 'result-meta';
        
        if (item.metadata && item.metadata.timestamp) {
            const date = document.createElement('div');
            date.className = 'result-date';
            date.innerHTML = `<i class="fas fa-clock"></i> ${this.formatDate(item.metadata.timestamp)}`;
            meta.appendChild(date);
        }
        
        if (item.metadata && item.metadata.size) {
            const size = document.createElement('div');
            size.className = 'result-size';
            size.innerHTML = `<i class="fas fa-file"></i> ${this.formatSize(item.metadata.size)}`;
            meta.appendChild(size);
        }
        
        resultDiv.appendChild(title);
        resultDiv.appendChild(url);
        resultDiv.appendChild(snippet);
        resultDiv.appendChild(meta);
        
        return resultDiv;
    }
    
    highlightText(text, query) {
        if (!query || !text) {
            return text;
        }
        
        const regex = new RegExp(`(${this.escapeRegex(query)})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }
    
    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
    
    formatUrl(url) {
        try {
            const urlObj = new URL(url);
            return urlObj.hostname + urlObj.pathname;
        } catch {
            return url;
        }
    }
    
    formatDate(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) { // Less than 1 minute
            return 'Just now';
        } else if (diff < 3600000) { // Less than 1 hour
            return `${Math.floor(diff / 60000)} minutes ago`;
        } else if (diff < 86400000) { // Less than 1 day
            return `${Math.floor(diff / 3600000)} hours ago`;
        } else if (diff < 604800000) { // Less than 1 week
            return `${Math.floor(diff / 86400000)} days ago`;
        } else {
            return date.toLocaleDateString();
        }
    }
    
    formatSize(bytes) {
        if (bytes < 1024) {
            return `${bytes} B`;
        } else if (bytes < 1048576) {
            return `${(bytes / 1024).toFixed(1)} KB`;
        } else {
            return `${(bytes / 1048576).toFixed(1)} MB`;
        }
    }
    
    updatePagination(result) {
        const pagination = document.getElementById('pagination');
        if (!pagination) {
            return;
        }
        
        const totalPages = Math.ceil(result.total / this.state.resultsPerPage);
        if (totalPages <= 1) {
            pagination.innerHTML = '';
            return;
        }
        
        let paginationHTML = '';
        const currentPage = this.state.currentPage;
        
        // Previous button
        if (currentPage > 1) {
            paginationHTML += `
                <button class="pagination-btn" data-page="${currentPage - 1}">
                    <i class="fas fa-chevron-left"></i>
                </button>
            `;
        }
        
        // Page numbers
        const startPage = Math.max(1, currentPage - 2);
        const endPage = Math.min(totalPages, currentPage + 2);
        
        for (let i = startPage; i <= endPage; i++) {
            const isActive = i === currentPage ? 'active' : '';
            paginationHTML += `
                <button class="pagination-btn ${isActive}" data-page="${i}">
                    ${i}
                </button>
            `;
        }
        
        // Next button
        if (currentPage < totalPages) {
            paginationHTML += `
                <button class="pagination-btn" data-page="${currentPage + 1}">
                    <i class="fas fa-chevron-right"></i>
                </button>
            `;
        }
        
        pagination.innerHTML = paginationHTML;
    }
    
    async goToPage(page) {
        if (page === this.state.currentPage || this.state.isLoading) {
            return;
        }
        
        await this.performSearch(this.state.currentQuery, page);
    }
    
    handleInputChange(event) {
        const query = event.target.value.trim();
        
        // Clear previous debounce timer
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        
        // Set new debounce timer
        this.debounceTimer = setTimeout(() => {
            if (query.length > 0) {
                this.fetchSuggestions(query);
            } else {
                this.hideSuggestions();
            }
        }, this.config.debounceDelay);
    }
    
    async fetchSuggestions(query) {
        try {
            const response = await fetch(`${this.config.apiBaseUrl}/search/suggest?q=${encodeURIComponent(query)}`);
            if (!response.ok) {
                throw new Error('Failed to fetch suggestions');
            }
            
            const data = await response.json();
            this.state.suggestions = data.suggestions || [];
            this.displaySuggestions();
        } catch (error) {
            console.error('Error fetching suggestions:', error);
            this.hideSuggestions();
        }
    }
    
    displaySuggestions() {
        const suggestionsContainer = document.getElementById('searchSuggestions');
        if (!suggestionsContainer) {
            return;
        }
        
        if (this.state.suggestions.length === 0) {
            this.hideSuggestions();
            return;
        }
        
        suggestionsContainer.innerHTML = '';
        
        this.state.suggestions.slice(0, this.config.maxSuggestions).forEach(suggestion => {
            const suggestionElement = document.createElement('div');
            suggestionElement.className = 'search-suggestion';
            suggestionElement.innerHTML = `
                <i class="fas fa-search search-suggestion-icon"></i>
                <span class="search-suggestion-text">${suggestion}</span>
            `;
            suggestionElement.addEventListener('click', () => {
                this.selectSuggestion(suggestionElement);
            });
            suggestionsContainer.appendChild(suggestionElement);
        });
        
        suggestionsContainer.style.display = 'block';
    }
    
    selectSuggestion(suggestionElement) {
        const suggestionText = suggestionElement.querySelector('.search-suggestion-text').textContent;
        const searchInput = document.getElementById('searchInput');
        
        if (searchInput) {
            searchInput.value = suggestionText;
            this.hideSuggestions();
            this.handleSearch({ preventDefault: () => {} });
        }
    }
    
    showSuggestions() {
        if (this.state.suggestions.length > 0) {
            this.displaySuggestions();
        }
    }
    
    hideSuggestions() {
        const suggestionsContainer = document.getElementById('searchSuggestions');
        if (suggestionsContainer) {
            suggestionsContainer.style.display = 'none';
        }
    }
    
    async startVoiceSearch() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            this.showError('Voice search is not supported in your browser.');
            return;
        }
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        const searchInput = document.getElementById('searchInput');
        const voiceBtn = document.getElementById('voiceBtn');
        
        voiceBtn.innerHTML = '<i class="fas fa-stop"></i>';
        voiceBtn.classList.add('recording');
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            if (searchInput) {
                searchInput.value = transcript;
                this.handleSearch({ preventDefault: () => {} });
            }
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.showError('Voice search failed. Please try again.');
        };
        
        recognition.onend = () => {
            voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
            voiceBtn.classList.remove('recording');
        };
        
        recognition.start();
    }
    
    startImageSearch() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.style.display = 'none';
        
        input.onchange = (event) => {
            const file = event.target.files[0];
            if (file) {
                this.performImageSearch(file);
            }
        };
        
        document.body.appendChild(input);
        input.click();
        document.body.removeChild(input);
    }
    
    async performImageSearch(file) {
        this.showLoading();
        
        try {
            const formData = new FormData();
            formData.append('image', file);
            
            const response = await fetch(`${this.config.apiBaseUrl}/search/image`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Image search failed');
            }
            
            const result = await response.json();
            this.displayResults(result);
        } catch (error) {
            console.error('Image search error:', error);
            this.showError('Image search failed. Please try again.');
        } finally {
            this.hideLoading();
        }
    }
    
    showLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'flex';
        }
    }
    
    hideLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
    }
    
    showError(message) {
        // Create error notification
        const notification = document.createElement('div');
        notification.className = 'notification notification-error';
        notification.innerHTML = `
            <i class="fas fa-exclamation-circle"></i>
            <span>${message}</span>
            <button class="notification-close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
        
        // Close button
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        });
    }
    
    showSettings() {
        const settingsModal = document.getElementById('settingsModal');
        if (settingsModal) {
            settingsModal.style.display = 'flex';
            this.loadSettings();
        }
    }
    
    hideSettings() {
        const settingsModal = document.getElementById('settingsModal');
        if (settingsModal) {
            settingsModal.style.display = 'none';
        }
    }
    
    loadSettings() {
        const prefs = this.state.userPreferences;
        
        const resultsPerPage = document.getElementById('resultsPerPage');
        if (resultsPerPage) {
            resultsPerPage.value = prefs.resultsPerPage || 20;
        }
        
        const defaultLanguage = document.getElementById('defaultLanguage');
        if (defaultLanguage) {
            defaultLanguage.value = prefs.language || 'en';
        }
        
        const defaultSafeSearch = document.getElementById('defaultSafeSearch');
        if (defaultSafeSearch) {
            defaultSafeSearch.value = prefs.safeSearch || 'moderate';
        }
        
        const theme = document.getElementById('theme');
        if (theme) {
            theme.value = prefs.theme || 'light';
        }
        
        const fontSize = document.getElementById('fontSize');
        if (fontSize) {
            fontSize.value = prefs.fontSize || 'medium';
        }
    }
    
    saveSettings() {
        const prefs = {
            resultsPerPage: parseInt(document.getElementById('resultsPerPage').value),
            language: document.getElementById('defaultLanguage').value,
            safeSearch: document.getElementById('defaultSafeSearch').value,
            theme: document.getElementById('theme').value,
            fontSize: document.getElementById('fontSize').value
        };
        
        this.state.userPreferences = prefs;
        this.saveUserPreferences();
        this.setupTheme();
        this.hideSettings();
        
        this.showSuccess('Settings saved successfully!');
    }
    
    loadUserPreferences() {
        try {
            const saved = localStorage.getItem('t3ss_preferences');
            return saved ? JSON.parse(saved) : {};
        } catch {
            return {};
        }
    }
    
    saveUserPreferences() {
        try {
            localStorage.setItem('t3ss_preferences', JSON.stringify(this.state.userPreferences));
        } catch (error) {
            console.error('Failed to save preferences:', error);
        }
    }
    
    addToSearchHistory(query) {
        if (!query || query.length < 2) {
            return;
        }
        
        // Remove if already exists
        this.state.searchHistory = this.state.searchHistory.filter(item => item !== query);
        
        // Add to beginning
        this.state.searchHistory.unshift(query);
        
        // Keep only last 50 searches
        this.state.searchHistory = this.state.searchHistory.slice(0, 50);
        
        this.saveSearchHistory();
    }
    
    loadSearchHistory() {
        try {
            const saved = localStorage.getItem('t3ss_search_history');
            this.state.searchHistory = saved ? JSON.parse(saved) : [];
        } catch {
            this.state.searchHistory = [];
        }
    }
    
    saveSearchHistory() {
        try {
            localStorage.setItem('t3ss_search_history', JSON.stringify(this.state.searchHistory));
        } catch (error) {
            console.error('Failed to save search history:', error);
        }
    }
    
    getCacheKey(query, page) {
        const filters = this.getActiveFilters();
        return `${query}:${page}:${JSON.stringify(filters)}`;
    }
    
    manageCacheSize() {
        if (this.cache.size > this.config.cacheSize) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
    }
    
    getUserId() {
        // In a real app, this would come from authentication
        return localStorage.getItem('t3ss_user_id') || 'anonymous';
    }
    
    getSessionId() {
        let sessionId = sessionStorage.getItem('t3ss_session_id');
        if (!sessionId) {
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            sessionStorage.setItem('t3ss_session_id', sessionId);
        }
        return sessionId;
    }
    
    trackSearch(query, result) {
        // Analytics tracking
        if (typeof gtag !== 'undefined') {
            gtag('event', 'search', {
                search_term: query,
                results_count: result.total,
                search_time: this.state.searchTime
            });
        }
    }
    
    trackResultClick(url, docId) {
        // Analytics tracking
        if (typeof gtag !== 'undefined') {
            gtag('event', 'click', {
                event_category: 'search_result',
                event_label: url,
                value: docId
            });
        }
    }
    
    handleRealTimeUpdate(data) {
        // Handle real-time updates from server
        console.log('Real-time update:', data);
    }
    
    handleResize() {
        // Handle window resize
        this.hideSuggestions();
    }
    
    handleOnline() {
        console.log('Connection restored');
    }
    
    handleOffline() {
        console.log('Connection lost');
    }
    
    saveState() {
        // Save current state
        try {
            sessionStorage.setItem('t3ss_state', JSON.stringify({
                currentQuery: this.state.currentQuery,
                currentPage: this.state.currentPage
            }));
        } catch (error) {
            console.error('Failed to save state:', error);
        }
    }
    
    showSuccess(message) {
        // Create success notification
        const notification = document.createElement('div');
        notification.className = 'notification notification-success';
        notification.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <span>${message}</span>
            <button class="notification-close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.t3ssApp = new T3SSApp();
});