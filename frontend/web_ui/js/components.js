/**
 * T3SS Project - UI Components
 * (c) 2025 Qiss Labs. All Rights Reserved.
 */

class T3SSComponents {
    constructor() {
        this.components = new Map();
        this.init();
    }
    
    init() {
        this.registerComponents();
    }
    
    registerComponents() {
        // Search Input Component
        this.components.set('search-input', {
            template: `
                <div class="search-input-container">
                    <input type="text" id="searchInput" class="search-input" placeholder="Search the web..." autocomplete="off" spellcheck="false">
                    <div class="search-suggestions" id="searchSuggestions"></div>
                </div>
            `,
            styles: `
                .search-input-container {
                    position: relative;
                    width: 100%;
                    max-width: 600px;
                    margin: 0 auto;
                }
                
                .search-input {
                    width: 100%;
                    padding: 12px 16px;
                    font-size: 16px;
                    border: 2px solid #e1e5e9;
                    border-radius: 24px;
                    outline: none;
                    transition: all 0.3s ease;
                    background: #fff;
                }
                
                .search-input:focus {
                    border-color: #4285f4;
                    box-shadow: 0 0 0 3px rgba(66, 133, 244, 0.1);
                }
                
                .search-suggestions {
                    position: absolute;
                    top: 100%;
                    left: 0;
                    right: 0;
                    background: #fff;
                    border: 1px solid #e1e5e9;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                    z-index: 1000;
                    display: none;
                    max-height: 300px;
                    overflow-y: auto;
                }
                
                .search-suggestion {
                    display: flex;
                    align-items: center;
                    padding: 12px 16px;
                    cursor: pointer;
                    transition: background-color 0.2s ease;
                }
                
                .search-suggestion:hover {
                    background-color: #f8f9fa;
                }
                
                .search-suggestion-icon {
                    margin-right: 12px;
                    color: #5f6368;
                    width: 16px;
                }
                
                .search-suggestion-text {
                    flex: 1;
                    color: #202124;
                }
            `
        });
        
        // Search Button Component
        this.components.set('search-button', {
            template: `
                <button type="submit" class="search-button" id="searchBtn">
                    <i class="fas fa-search"></i>
                </button>
            `,
            styles: `
                .search-button {
                    position: absolute;
                    right: 8px;
                    top: 50%;
                    transform: translateY(-50%);
                    width: 40px;
                    height: 40px;
                    border: none;
                    border-radius: 50%;
                    background: #4285f4;
                    color: white;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .search-button:hover {
                    background: #3367d6;
                    transform: translateY(-50%) scale(1.05);
                }
                
                .search-button:active {
                    transform: translateY(-50%) scale(0.95);
                }
            `
        });
        
        // Voice Search Button Component
        this.components.set('voice-button', {
            template: `
                <button type="button" class="voice-button" id="voiceBtn" title="Voice search">
                    <i class="fas fa-microphone"></i>
                </button>
            `,
            styles: `
                .voice-button {
                    position: absolute;
                    right: 60px;
                    top: 50%;
                    transform: translateY(-50%);
                    width: 40px;
                    height: 40px;
                    border: none;
                    border-radius: 50%;
                    background: transparent;
                    color: #5f6368;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .voice-button:hover {
                    background: #f8f9fa;
                    color: #4285f4;
                }
                
                .voice-button.recording {
                    background: #ea4335;
                    color: white;
                    animation: pulse 1.5s infinite;
                }
                
                @keyframes pulse {
                    0% { transform: translateY(-50%) scale(1); }
                    50% { transform: translateY(-50%) scale(1.1); }
                    100% { transform: translateY(-50%) scale(1); }
                }
            `
        });
        
        // Camera Button Component
        this.components.set('camera-button', {
            template: `
                <button type="button" class="camera-button" id="cameraBtn" title="Image search">
                    <i class="fas fa-camera"></i>
                </button>
            `,
            styles: `
                .camera-button {
                    position: absolute;
                    right: 110px;
                    top: 50%;
                    transform: translateY(-50%);
                    width: 40px;
                    height: 40px;
                    border: none;
                    border-radius: 50%;
                    background: transparent;
                    color: #5f6368;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .camera-button:hover {
                    background: #f8f9fa;
                    color: #4285f4;
                }
            `
        });
        
        // Settings Button Component
        this.components.set('settings-button', {
            template: `
                <button type="button" class="settings-button" id="settingsBtn" title="Settings">
                    <i class="fas fa-cog"></i>
                </button>
            `,
            styles: `
                .settings-button {
                    position: absolute;
                    right: 160px;
                    top: 50%;
                    transform: translateY(-50%);
                    width: 40px;
                    height: 40px;
                    border: none;
                    border-radius: 50%;
                    background: transparent;
                    color: #5f6368;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .settings-button:hover {
                    background: #f8f9fa;
                    color: #4285f4;
                }
            `
        });
        
        // Result Item Component
        this.components.set('result-item', {
            template: (data) => `
                <div class="result-item" data-url="${data.url}" data-doc-id="${data.id}">
                    <h3 class="result-title">
                        <a href="${data.url}" target="_blank" rel="noopener noreferrer">
                            ${data.title}
                        </a>
                    </h3>
                    <div class="result-url">${data.displayUrl}</div>
                    <div class="result-snippet">${data.snippet}</div>
                    <div class="result-meta">
                        ${data.meta ? this.renderMeta(data.meta) : ''}
                    </div>
                </div>
            `,
            styles: `
                .result-item {
                    padding: 20px 0;
                    border-bottom: 1px solid #e8eaed;
                }
                
                .result-item:last-child {
                    border-bottom: none;
                }
                
                .result-title {
                    margin: 0 0 8px 0;
                    font-size: 20px;
                    font-weight: 400;
                    line-height: 1.3;
                }
                
                .result-title a {
                    color: #1a0dab;
                    text-decoration: none;
                }
                
                .result-title a:hover {
                    text-decoration: underline;
                }
                
                .result-url {
                    color: #006621;
                    font-size: 14px;
                    margin-bottom: 8px;
                }
                
                .result-snippet {
                    color: #545454;
                    font-size: 14px;
                    line-height: 1.4;
                    margin-bottom: 8px;
                }
                
                .result-meta {
                    display: flex;
                    gap: 16px;
                    font-size: 12px;
                    color: #5f6368;
                }
                
                .result-meta-item {
                    display: flex;
                    align-items: center;
                    gap: 4px;
                }
                
                .result-meta-item i {
                    width: 12px;
                }
            `
        });
        
        // Pagination Component
        this.components.set('pagination', {
            template: (data) => {
                const { currentPage, totalPages, onPageChange } = data;
                let html = '<div class="pagination">';
                
                // Previous button
                if (currentPage > 1) {
                    html += `
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
                    html += `
                        <button class="pagination-btn ${isActive}" data-page="${i}">
                            ${i}
                        </button>
                    `;
                }
                
                // Next button
                if (currentPage < totalPages) {
                    html += `
                        <button class="pagination-btn" data-page="${currentPage + 1}">
                            <i class="fas fa-chevron-right"></i>
                        </button>
                    `;
                }
                
                html += '</div>';
                return html;
            },
            styles: `
                .pagination {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 8px;
                    margin: 40px 0;
                }
                
                .pagination-btn {
                    padding: 8px 12px;
                    border: 1px solid #dadce0;
                    border-radius: 4px;
                    background: #fff;
                    color: #5f6368;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    font-size: 14px;
                    min-width: 40px;
                    height: 40px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .pagination-btn:hover {
                    border-color: #4285f4;
                    color: #4285f4;
                }
                
                .pagination-btn.active {
                    background: #4285f4;
                    border-color: #4285f4;
                    color: white;
                }
                
                .pagination-btn:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }
            `
        });
        
        // Loading Spinner Component
        this.components.set('loading-spinner', {
            template: `
                <div class="loading-spinner">
                    <div class="spinner"></div>
                    <div class="loading-text">Searching...</div>
                </div>
            `,
            styles: `
                .loading-spinner {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 40px;
                }
                
                .spinner {
                    width: 40px;
                    height: 40px;
                    border: 4px solid #f3f3f3;
                    border-top: 4px solid #4285f4;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                .loading-text {
                    margin-top: 16px;
                    color: #5f6368;
                    font-size: 14px;
                }
            `
        });
        
        // Notification Component
        this.components.set('notification', {
            template: (data) => `
                <div class="notification notification-${data.type}">
                    <i class="fas fa-${this.getNotificationIcon(data.type)}"></i>
                    <span>${data.message}</span>
                    <button class="notification-close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `,
            styles: `
                .notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #fff;
                    border: 1px solid #e1e5e9;
                    border-radius: 8px;
                    padding: 16px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                    z-index: 10000;
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    min-width: 300px;
                    max-width: 500px;
                    animation: slideIn 0.3s ease;
                }
                
                .notification-success {
                    border-left: 4px solid #34a853;
                }
                
                .notification-error {
                    border-left: 4px solid #ea4335;
                }
                
                .notification-warning {
                    border-left: 4px solid #fbbc04;
                }
                
                .notification-info {
                    border-left: 4px solid #4285f4;
                }
                
                .notification i {
                    font-size: 18px;
                }
                
                .notification-success i {
                    color: #34a853;
                }
                
                .notification-error i {
                    color: #ea4335;
                }
                
                .notification-warning i {
                    color: #fbbc04;
                }
                
                .notification-info i {
                    color: #4285f4;
                }
                
                .notification span {
                    flex: 1;
                    color: #202124;
                }
                
                .notification-close {
                    background: none;
                    border: none;
                    color: #5f6368;
                    cursor: pointer;
                    padding: 4px;
                    border-radius: 4px;
                    transition: background-color 0.2s ease;
                }
                
                .notification-close:hover {
                    background: #f8f9fa;
                }
                
                @keyframes slideIn {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
            `
        });
        
        // Modal Component
        this.components.set('modal', {
            template: (data) => `
                <div class="modal-overlay" id="${data.id}">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h2>${data.title}</h2>
                            <button class="modal-close">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                        <div class="modal-body">
                            ${data.content}
                        </div>
                        <div class="modal-footer">
                            ${data.footer || ''}
                        </div>
                    </div>
                </div>
            `,
            styles: `
                .modal-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0, 0, 0, 0.5);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 10000;
                    opacity: 0;
                    visibility: hidden;
                    transition: all 0.3s ease;
                }
                
                .modal-overlay.show {
                    opacity: 1;
                    visibility: visible;
                }
                
                .modal-content {
                    background: #fff;
                    border-radius: 8px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                    max-width: 500px;
                    width: 90%;
                    max-height: 80vh;
                    overflow-y: auto;
                    transform: scale(0.9);
                    transition: transform 0.3s ease;
                }
                
                .modal-overlay.show .modal-content {
                    transform: scale(1);
                }
                
                .modal-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 20px;
                    border-bottom: 1px solid #e1e5e9;
                }
                
                .modal-header h2 {
                    margin: 0;
                    font-size: 20px;
                    font-weight: 500;
                    color: #202124;
                }
                
                .modal-close {
                    background: none;
                    border: none;
                    color: #5f6368;
                    cursor: pointer;
                    padding: 8px;
                    border-radius: 4px;
                    transition: background-color 0.2s ease;
                }
                
                .modal-close:hover {
                    background: #f8f9fa;
                }
                
                .modal-body {
                    padding: 20px;
                }
                
                .modal-footer {
                    padding: 20px;
                    border-top: 1px solid #e1e5e9;
                    display: flex;
                    justify-content: flex-end;
                    gap: 12px;
                }
            `
        });
        
        // Filter Component
        this.components.set('filter', {
            template: (data) => `
                <div class="filter-container">
                    <label for="${data.id}" class="filter-label">${data.label}</label>
                    <select id="${data.id}" class="filter-select">
                        ${data.options.map(option => `
                            <option value="${option.value}" ${option.selected ? 'selected' : ''}>
                                ${option.label}
                            </option>
                        `).join('')}
                    </select>
                </div>
            `,
            styles: `
                .filter-container {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }
                
                .filter-label {
                    font-size: 14px;
                    font-weight: 500;
                    color: #202124;
                }
                
                .filter-select {
                    padding: 8px 12px;
                    border: 1px solid #dadce0;
                    border-radius: 4px;
                    background: #fff;
                    color: #202124;
                    font-size: 14px;
                    cursor: pointer;
                    transition: border-color 0.2s ease;
                }
                
                .filter-select:focus {
                    outline: none;
                    border-color: #4285f4;
                }
            `
        });
    }
    
    renderMeta(meta) {
        let html = '';
        
        if (meta.timestamp) {
            html += `
                <div class="result-meta-item">
                    <i class="fas fa-clock"></i>
                    <span>${this.formatDate(meta.timestamp)}</span>
                </div>
            `;
        }
        
        if (meta.size) {
            html += `
                <div class="result-meta-item">
                    <i class="fas fa-file"></i>
                    <span>${this.formatSize(meta.size)}</span>
                </div>
            `;
        }
        
        if (meta.language) {
            html += `
                <div class="result-meta-item">
                    <i class="fas fa-language"></i>
                    <span>${meta.language}</span>
                </div>
            `;
        }
        
        return html;
    }
    
    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
    
    formatDate(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) {
            return 'Just now';
        } else if (diff < 3600000) {
            return `${Math.floor(diff / 60000)} minutes ago`;
        } else if (diff < 86400000) {
            return `${Math.floor(diff / 3600000)} hours ago`;
        } else if (diff < 604800000) {
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
    
    // Component rendering methods
    render(componentName, data = {}) {
        const component = this.components.get(componentName);
        if (!component) {
            console.error(`Component '${componentName}' not found`);
            return '';
        }
        
        if (typeof component.template === 'function') {
            return component.template(data);
        }
        return component.template;
    }
    
    renderToElement(componentName, element, data = {}) {
        const html = this.render(componentName, data);
        if (element && html) {
            element.innerHTML = html;
        }
        return html;
    }
    
    showNotification(type, message, duration = 5000) {
        const notification = document.createElement('div');
        notification.innerHTML = this.render('notification', { type, message });
        document.body.appendChild(notification);
        
        // Auto remove
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, duration);
        
        // Close button
        const closeBtn = notification.querySelector('.notification-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            });
        }
    }
    
    showModal(id, title, content, footer = '') {
        const modal = document.createElement('div');
        modal.innerHTML = this.render('modal', { id, title, content, footer });
        document.body.appendChild(modal);
        
        // Show modal
        setTimeout(() => {
            modal.classList.add('show');
        }, 10);
        
        // Close button
        const closeBtn = modal.querySelector('.modal-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                this.hideModal(modal);
            });
        }
        
        // Backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.hideModal(modal);
            }
        });
        
        return modal;
    }
    
    hideModal(modal) {
        modal.classList.remove('show');
        setTimeout(() => {
            if (modal.parentNode) {
                modal.parentNode.removeChild(modal);
            }
        }, 300);
    }
    
    // Utility methods
    createElement(tag, className, content) {
        const element = document.createElement(tag);
        if (className) {
            element.className = className;
        }
        if (content) {
            element.innerHTML = content;
        }
        return element;
    }
    
    addEventListeners(element, events) {
        Object.entries(events).forEach(([event, handler]) => {
            element.addEventListener(event, handler);
        });
    }
    
    removeEventListeners(element, events) {
        Object.entries(events).forEach(([event, handler]) => {
            element.removeEventListener(event, handler);
        });
    }
}

// Initialize components
window.t3ssComponents = new T3SSComponents();