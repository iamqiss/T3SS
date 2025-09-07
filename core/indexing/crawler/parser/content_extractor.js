// T3SS Project
// File: core/indexing/crawler/parser/content_extractor.js
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

const cheerio = require('cheerio');
const jsdom = require('jsdom');
const { JSDOM } = jsdom;
const natural = require('natural');
const { SentimentAnalyzer, PorterStemmer } = natural;
const stopwords = require('stopwords');
const crypto = require('crypto');
const url = require('url');
const punycode = require('punycode');
const he = require('he');

/**
 * Advanced content extraction engine for web pages
 * Supports multiple extraction strategies and content analysis
 */
class ContentExtractor {
    constructor(options = {}) {
        this.options = {
            // Extraction settings
            extractText: true,
            extractLinks: true,
            extractImages: true,
            extractMetadata: true,
            extractTables: true,
            extractForms: true,
            extractScripts: true,
            extractStyles: true,
            
            // Content filtering
            minTextLength: 10,
            maxTextLength: 1000000,
            removeScripts: true,
            removeStyles: true,
            removeComments: true,
            removeEmptyElements: true,
            
            // Language detection
            detectLanguage: true,
            supportedLanguages: ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko'],
            
            // Content analysis
            analyzeSentiment: true,
            extractKeywords: true,
            extractEntities: true,
            calculateReadability: true,
            
            // Performance
            maxContentSize: 10 * 1024 * 1024, // 10MB
            timeout: 30000, // 30 seconds
            enableCaching: true,
            
            // Quality thresholds
            minContentQuality: 0.3,
            minReadabilityScore: 30,
            
            ...options
        };
        
        // Initialize analyzers
        this.sentimentAnalyzer = new SentimentAnalyzer('English', PorterStemmer, stopwords.english);
        this.stemmer = PorterStemmer;
        
        // Content cache
        this.cache = new Map();
        
        // Statistics
        this.stats = {
            totalExtractions: 0,
            successfulExtractions: 0,
            failedExtractions: 0,
            averageProcessingTime: 0,
            cacheHits: 0,
            cacheMisses: 0
        };
    }

    /**
     * Extract content from HTML string
     * @param {string} html - HTML content
     * @param {string} baseUrl - Base URL for resolving relative links
     * @param {Object} options - Extraction options
     * @returns {Promise<Object>} Extracted content
     */
    async extractContent(html, baseUrl = '', options = {}) {
        const startTime = Date.now();
        const extractionOptions = { ...this.options, ...options };
        
        try {
            // Validate input
            if (!html || typeof html !== 'string') {
                throw new Error('Invalid HTML content provided');
            }
            
            if (html.length > extractionOptions.maxContentSize) {
                throw new Error(`Content too large: ${html.length} bytes`);
            }
            
            // Check cache
            const cacheKey = this._generateCacheKey(html, baseUrl, extractionOptions);
            if (extractionOptions.enableCaching && this.cache.has(cacheKey)) {
                this.stats.cacheHits++;
                return this.cache.get(cacheKey);
            }
            
            this.stats.cacheMisses++;
            
            // Parse HTML
            const dom = new JSDOM(html, {
                url: baseUrl,
                includeNodeLocations: true,
                runScripts: 'dangerously'
            });
            
            const document = dom.window.document;
            const $ = cheerio.load(html);
            
            // Initialize result object
            const result = {
                url: baseUrl,
                extractedAt: new Date().toISOString(),
                processingTime: 0,
                content: {},
                metadata: {},
                analysis: {},
                quality: {},
                errors: [],
                warnings: []
            };
            
            // Extract different content types
            if (extractionOptions.extractText) {
                result.content.text = await this._extractText(document, $, extractionOptions);
            }
            
            if (extractionOptions.extractLinks) {
                result.content.links = await this._extractLinks(document, $, baseUrl, extractionOptions);
            }
            
            if (extractionOptions.extractImages) {
                result.content.images = await this._extractImages(document, $, baseUrl, extractionOptions);
            }
            
            if (extractionOptions.extractMetadata) {
                result.metadata = await this._extractMetadata(document, $, extractionOptions);
            }
            
            if (extractionOptions.extractTables) {
                result.content.tables = await this._extractTables(document, $, extractionOptions);
            }
            
            if (extractionOptions.extractForms) {
                result.content.forms = await this._extractForms(document, $, extractionOptions);
            }
            
            if (extractionOptions.extractScripts) {
                result.content.scripts = await this._extractScripts(document, $, extractionOptions);
            }
            
            if (extractionOptions.extractStyles) {
                result.content.styles = await this._extractStyles(document, $, extractionOptions);
            }
            
            // Perform content analysis
            if (extractionOptions.analyzeSentiment || extractionOptions.extractKeywords || 
                extractionOptions.extractEntities || extractionOptions.calculateReadability) {
                result.analysis = await this._analyzeContent(result.content.text, extractionOptions);
            }
            
            // Calculate content quality
            result.quality = await this._calculateContentQuality(result, extractionOptions);
            
            // Set processing time
            result.processingTime = Date.now() - startTime;
            
            // Update statistics
            this._updateStats(result.processingTime, true);
            
            // Cache result
            if (extractionOptions.enableCaching) {
                this.cache.set(cacheKey, result);
            }
            
            return result;
            
        } catch (error) {
            this._updateStats(Date.now() - startTime, false);
            throw new Error(`Content extraction failed: ${error.message}`);
        }
    }

    /**
     * Extract text content from document
     * @private
     */
    async _extractText(document, $, options) {
        const textContent = {
            raw: '',
            clean: '',
            structured: [],
            headings: [],
            paragraphs: [],
            lists: [],
            quotes: [],
            code: []
        };
        
        // Remove unwanted elements
        const $clean = $.clone();
        if (options.removeScripts) {
            $clean('script').remove();
        }
        if (options.removeStyles) {
            $clean('style').remove();
        }
        if (options.removeComments) {
            $clean('*').contents().filter(function() {
                return this.nodeType === 8; // Comment node
            }).remove();
        }
        
        // Extract raw text
        textContent.raw = $clean.text();
        
        // Clean text
        textContent.clean = this._cleanText(textContent.raw);
        
        // Extract structured content
        textContent.headings = this._extractHeadings($clean);
        textContent.paragraphs = this._extractParagraphs($clean);
        textContent.lists = this._extractLists($clean);
        textContent.quotes = this._extractQuotes($clean);
        textContent.code = this._extractCode($clean);
        
        // Filter by length
        if (textContent.clean.length < options.minTextLength) {
            throw new Error(`Text too short: ${textContent.clean.length} characters`);
        }
        
        if (textContent.clean.length > options.maxTextLength) {
            textContent.clean = textContent.clean.substring(0, options.maxTextLength);
        }
        
        return textContent;
    }

    /**
     * Extract links from document
     * @private
     */
    async _extractLinks(document, $, baseUrl, options) {
        const links = [];
        
        $('a[href]').each((index, element) => {
            const $el = $(element);
            const href = $el.attr('href');
            const text = $el.text().trim();
            const title = $el.attr('title') || '';
            const rel = $el.attr('rel') || '';
            
            if (!href) return;
            
            try {
                const absoluteUrl = url.resolve(baseUrl, href);
                const parsedUrl = new url.URL(absoluteUrl);
                
                links.push({
                    url: absoluteUrl,
                    text: text,
                    title: title,
                    rel: rel,
                    domain: parsedUrl.hostname,
                    path: parsedUrl.pathname,
                    query: parsedUrl.search,
                    fragment: parsedUrl.hash,
                    isInternal: this._isInternalLink(absoluteUrl, baseUrl),
                    isExternal: !this._isInternalLink(absoluteUrl, baseUrl),
                    isNofollow: rel.includes('nofollow'),
                    isSponsored: rel.includes('sponsored'),
                    isUgc: rel.includes('ugc'),
                    quality: this._calculateLinkQuality(text, title, href)
                });
            } catch (error) {
                // Skip invalid URLs
            }
        });
        
        return links;
    }

    /**
     * Extract images from document
     * @private
     */
    async _extractImages(document, $, baseUrl, options) {
        const images = [];
        
        $('img[src]').each((index, element) => {
            const $el = $(element);
            const src = $el.attr('src');
            const alt = $el.attr('alt') || '';
            const title = $el.attr('title') || '';
            const width = parseInt($el.attr('width')) || null;
            const height = parseInt($el.attr('height')) || null;
            const loading = $el.attr('loading') || 'eager';
            
            if (!src) return;
            
            try {
                const absoluteUrl = url.resolve(baseUrl, src);
                const parsedUrl = new url.URL(absoluteUrl);
                
                images.push({
                    url: absoluteUrl,
                    alt: alt,
                    title: title,
                    width: width,
                    height: height,
                    loading: loading,
                    domain: parsedUrl.hostname,
                    path: parsedUrl.pathname,
                    extension: this._getFileExtension(parsedUrl.pathname),
                    quality: this._calculateImageQuality(alt, title, width, height)
                });
            } catch (error) {
                // Skip invalid URLs
            }
        });
        
        return images;
    }

    /**
     * Extract metadata from document
     * @private
     */
    async _extractMetadata(document, $, options) {
        const metadata = {
            title: '',
            description: '',
            keywords: [],
            author: '',
            language: '',
            charset: '',
            viewport: '',
            robots: '',
            canonical: '',
            og: {},
            twitter: {},
            custom: {}
        };
        
        // Basic meta tags
        metadata.title = $('title').text().trim();
        metadata.description = $('meta[name="description"]').attr('content') || '';
        metadata.keywords = ($('meta[name="keywords"]').attr('content') || '')
            .split(',')
            .map(k => k.trim())
            .filter(k => k.length > 0);
        metadata.author = $('meta[name="author"]').attr('content') || '';
        metadata.language = $('html').attr('lang') || $('meta[http-equiv="content-language"]').attr('content') || '';
        metadata.charset = $('meta[charset]').attr('charset') || $('meta[http-equiv="content-type"]').attr('content') || '';
        metadata.viewport = $('meta[name="viewport"]').attr('content') || '';
        metadata.robots = $('meta[name="robots"]').attr('content') || '';
        metadata.canonical = $('link[rel="canonical"]').attr('href') || '';
        
        // Open Graph tags
        $('meta[property^="og:"]').each((index, element) => {
            const $el = $(element);
            const property = $el.attr('property');
            const content = $el.attr('content');
            if (property && content) {
                metadata.og[property] = content;
            }
        });
        
        // Twitter Card tags
        $('meta[name^="twitter:"]').each((index, element) => {
            const $el = $(element);
            const name = $el.attr('name');
            const content = $el.attr('content');
            if (name && content) {
                metadata.twitter[name] = content;
            }
        });
        
        // Custom meta tags
        $('meta[name]:not([name="description"]):not([name="keywords"]):not([name="author"]):not([name="viewport"]):not([name="robots"])').each((index, element) => {
            const $el = $(element);
            const name = $el.attr('name');
            const content = $el.attr('content');
            if (name && content) {
                metadata.custom[name] = content;
            }
        });
        
        return metadata;
    }

    /**
     * Extract tables from document
     * @private
     */
    async _extractTables(document, $, options) {
        const tables = [];
        
        $('table').each((index, element) => {
            const $table = $(element);
            const tableData = {
                index: index,
                caption: $table.find('caption').text().trim(),
                headers: [],
                rows: [],
                summary: $table.attr('summary') || '',
                className: $table.attr('class') || '',
                id: $table.attr('id') || ''
            };
            
            // Extract headers
            $table.find('thead th, tr:first-child th, tr:first-child td').each((i, header) => {
                tableData.headers.push($(header).text().trim());
            });
            
            // Extract rows
            $table.find('tbody tr, tr').each((rowIndex, row) => {
                const $row = $(row);
                const rowData = [];
                
                $row.find('td, th').each((cellIndex, cell) => {
                    rowData.push($(cell).text().trim());
                });
                
                if (rowData.length > 0) {
                    tableData.rows.push(rowData);
                }
            });
            
            tables.push(tableData);
        });
        
        return tables;
    }

    /**
     * Extract forms from document
     * @private
     */
    async _extractForms(document, $, options) {
        const forms = [];
        
        $('form').each((index, element) => {
            const $form = $(element);
            const formData = {
                index: index,
                action: $form.attr('action') || '',
                method: $form.attr('method') || 'get',
                enctype: $form.attr('enctype') || 'application/x-www-form-urlencoded',
                className: $form.attr('class') || '',
                id: $form.attr('id') || '',
                fields: []
            };
            
            // Extract form fields
            $form.find('input, textarea, select').each((fieldIndex, field) => {
                const $field = $(field);
                const fieldData = {
                    type: $field.attr('type') || $field.prop('tagName').toLowerCase(),
                    name: $field.attr('name') || '',
                    value: $field.attr('value') || $field.text(),
                    placeholder: $field.attr('placeholder') || '',
                    required: $field.attr('required') !== undefined,
                    disabled: $field.attr('disabled') !== undefined,
                    readonly: $field.attr('readonly') !== undefined,
                    className: $field.attr('class') || '',
                    id: $field.attr('id') || ''
                };
                
                formData.fields.push(fieldData);
            });
            
            forms.push(formData);
        });
        
        return forms;
    }

    /**
     * Extract scripts from document
     * @private
     */
    async _extractScripts(document, $, options) {
        const scripts = [];
        
        $('script').each((index, element) => {
            const $script = $(element);
            const scriptData = {
                index: index,
                src: $script.attr('src') || '',
                type: $script.attr('type') || 'text/javascript',
                async: $script.attr('async') !== undefined,
                defer: $script.attr('defer') !== undefined,
                content: $script.html() || '',
                className: $script.attr('class') || '',
                id: $script.attr('id') || ''
            };
            
            scripts.push(scriptData);
        });
        
        return scripts;
    }

    /**
     * Extract styles from document
     * @private
     */
    async _extractStyles(document, $, options) {
        const styles = [];
        
        $('style, link[rel="stylesheet"]').each((index, element) => {
            const $style = $(element);
            const styleData = {
                index: index,
                type: $style.prop('tagName').toLowerCase(),
                href: $style.attr('href') || '',
                media: $style.attr('media') || 'all',
                content: $style.html() || '',
                className: $style.attr('class') || '',
                id: $style.attr('id') || ''
            };
            
            styles.push(styleData);
        });
        
        return styles;
    }

    /**
     * Analyze content for sentiment, keywords, entities, and readability
     * @private
     */
    async _analyzeContent(text, options) {
        const analysis = {
            sentiment: null,
            keywords: [],
            entities: [],
            readability: null,
            language: null,
            wordCount: 0,
            sentenceCount: 0,
            paragraphCount: 0,
            averageWordLength: 0,
            averageSentenceLength: 0
        };
        
        if (!text || text.length === 0) {
            return analysis;
        }
        
        // Basic text statistics
        const words = text.split(/\s+/).filter(word => word.length > 0);
        const sentences = text.split(/[.!?]+/).filter(sentence => sentence.trim().length > 0);
        const paragraphs = text.split(/\n\s*\n/).filter(paragraph => paragraph.trim().length > 0);
        
        analysis.wordCount = words.length;
        analysis.sentenceCount = sentences.length;
        analysis.paragraphCount = paragraphs.length;
        analysis.averageWordLength = words.length > 0 ? words.reduce((sum, word) => sum + word.length, 0) / words.length : 0;
        analysis.averageSentenceLength = sentences.length > 0 ? words.length / sentences.length : 0;
        
        // Sentiment analysis
        if (options.analyzeSentiment) {
            try {
                analysis.sentiment = this.sentimentAnalyzer.getSentiment(words);
            } catch (error) {
                // Sentiment analysis failed
            }
        }
        
        // Keyword extraction
        if (options.extractKeywords) {
            analysis.keywords = this._extractKeywords(words);
        }
        
        // Entity extraction (simplified)
        if (options.extractEntities) {
            analysis.entities = this._extractEntities(text);
        }
        
        // Readability calculation
        if (options.calculateReadability) {
            analysis.readability = this._calculateReadability(words, sentences);
        }
        
        // Language detection (simplified)
        if (options.detectLanguage) {
            analysis.language = this._detectLanguage(text);
        }
        
        return analysis;
    }

    /**
     * Calculate content quality metrics
     * @private
     */
    async _calculateContentQuality(result, options) {
        const quality = {
            overall: 0,
            textQuality: 0,
            structureQuality: 0,
            metadataQuality: 0,
            linkQuality: 0,
            imageQuality: 0,
            factors: []
        };
        
        // Text quality
        if (result.content.text) {
            const textLength = result.content.text.clean.length;
            const wordCount = result.content.text.clean.split(/\s+/).length;
            
            if (textLength > 100) quality.textQuality += 0.2;
            if (wordCount > 50) quality.textQuality += 0.2;
            if (result.content.text.headings.length > 0) quality.textQuality += 0.2;
            if (result.content.text.paragraphs.length > 0) quality.textQuality += 0.2;
            if (result.analysis && result.analysis.readability > 30) quality.textQuality += 0.2;
        }
        
        // Structure quality
        if (result.content.text && result.content.text.headings.length > 0) quality.structureQuality += 0.3;
        if (result.content.tables && result.content.tables.length > 0) quality.structureQuality += 0.2;
        if (result.content.lists && result.content.lists.length > 0) quality.structureQuality += 0.2;
        if (result.metadata.title) quality.structureQuality += 0.3;
        
        // Metadata quality
        if (result.metadata.title) quality.metadataQuality += 0.3;
        if (result.metadata.description) quality.metadataQuality += 0.3;
        if (result.metadata.keywords && result.metadata.keywords.length > 0) quality.metadataQuality += 0.2;
        if (result.metadata.author) quality.metadataQuality += 0.2;
        
        // Link quality
        if (result.content.links) {
            const totalLinks = result.content.links.length;
            const internalLinks = result.content.links.filter(link => link.isInternal).length;
            const highQualityLinks = result.content.links.filter(link => link.quality > 0.7).length;
            
            if (totalLinks > 0) {
                quality.linkQuality += Math.min(0.5, internalLinks / totalLinks);
                quality.linkQuality += Math.min(0.5, highQualityLinks / totalLinks);
            }
        }
        
        // Image quality
        if (result.content.images) {
            const totalImages = result.content.images.length;
            const imagesWithAlt = result.content.images.filter(img => img.alt.length > 0).length;
            const highQualityImages = result.content.images.filter(img => img.quality > 0.7).length;
            
            if (totalImages > 0) {
                quality.imageQuality += Math.min(0.5, imagesWithAlt / totalImages);
                quality.imageQuality += Math.min(0.5, highQualityImages / totalImages);
            }
        }
        
        // Calculate overall quality
        quality.overall = (
            quality.textQuality * 0.4 +
            quality.structureQuality * 0.2 +
            quality.metadataQuality * 0.2 +
            quality.linkQuality * 0.1 +
            quality.imageQuality * 0.1
        );
        
        // Add quality factors
        if (quality.textQuality > 0.7) quality.factors.push('high_text_quality');
        if (quality.structureQuality > 0.7) quality.factors.push('good_structure');
        if (quality.metadataQuality > 0.7) quality.factors.push('complete_metadata');
        if (quality.linkQuality > 0.7) quality.factors.push('quality_links');
        if (quality.imageQuality > 0.7) quality.factors.push('quality_images');
        
        return quality;
    }

    /**
     * Clean text content
     * @private
     */
    _cleanText(text) {
        return text
            .replace(/\s+/g, ' ')  // Normalize whitespace
            .replace(/\n\s*\n/g, '\n')  // Remove empty lines
            .trim();
    }

    /**
     * Extract headings from document
     * @private
     */
    _extractHeadings($) {
        const headings = [];
        $('h1, h2, h3, h4, h5, h6').each((index, element) => {
            const $el = $(element);
            headings.push({
                level: parseInt($el.prop('tagName').substring(1)),
                text: $el.text().trim(),
                id: $el.attr('id') || '',
                className: $el.attr('class') || ''
            });
        });
        return headings;
    }

    /**
     * Extract paragraphs from document
     * @private
     */
    _extractParagraphs($) {
        const paragraphs = [];
        $('p').each((index, element) => {
            const $el = $(element);
            const text = $el.text().trim();
            if (text.length > 0) {
                paragraphs.push({
                    text: text,
                    className: $el.attr('class') || '',
                    id: $el.attr('id') || ''
                });
            }
        });
        return paragraphs;
    }

    /**
     * Extract lists from document
     * @private
     */
    _extractLists($) {
        const lists = [];
        $('ul, ol').each((index, element) => {
            const $el = $(element);
            const items = [];
            $el.find('li').each((itemIndex, item) => {
                items.push($(item).text().trim());
            });
            lists.push({
                type: $el.prop('tagName').toLowerCase(),
                items: items,
                className: $el.attr('class') || '',
                id: $el.attr('id') || ''
            });
        });
        return lists;
    }

    /**
     * Extract quotes from document
     * @private
     */
    _extractQuotes($) {
        const quotes = [];
        $('blockquote, q').each((index, element) => {
            const $el = $(element);
            quotes.push({
                text: $el.text().trim(),
                cite: $el.attr('cite') || '',
                className: $el.attr('class') || '',
                id: $el.attr('id') || ''
            });
        });
        return quotes;
    }

    /**
     * Extract code from document
     * @private
     */
    _extractCode($) {
        const code = [];
        $('code, pre').each((index, element) => {
            const $el = $(element);
            code.push({
                text: $el.text().trim(),
                language: $el.attr('class')?.match(/language-(\w+)/)?.[1] || '',
                className: $el.attr('class') || '',
                id: $el.attr('id') || ''
            });
        });
        return code;
    }

    /**
     * Check if link is internal
     * @private
     */
    _isInternalLink(linkUrl, baseUrl) {
        try {
            const linkDomain = new url.URL(linkUrl).hostname;
            const baseDomain = new url.URL(baseUrl).hostname;
            return linkDomain === baseDomain;
        } catch {
            return false;
        }
    }

    /**
     * Calculate link quality score
     * @private
     */
    _calculateLinkQuality(text, title, href) {
        let quality = 0.5; // Base score
        
        // Text quality
        if (text.length > 5 && text.length < 100) quality += 0.2;
        if (text.length === 0) quality -= 0.3;
        
        // Title attribute
        if (title.length > 0) quality += 0.1;
        
        // URL quality
        if (href.includes('#')) quality -= 0.1; // Fragment links
        if (href.includes('javascript:')) quality -= 0.2; // JavaScript links
        if (href.includes('mailto:')) quality -= 0.1; // Email links
        
        return Math.max(0, Math.min(1, quality));
    }

    /**
     * Calculate image quality score
     * @private
     */
    _calculateImageQuality(alt, title, width, height) {
        let quality = 0.5; // Base score
        
        // Alt text
        if (alt.length > 0) quality += 0.3;
        if (alt.length > 10) quality += 0.1;
        
        // Title attribute
        if (title.length > 0) quality += 0.1;
        
        // Dimensions
        if (width && height) {
            const aspectRatio = width / height;
            if (aspectRatio > 0.5 && aspectRatio < 2) quality += 0.1; // Good aspect ratio
        }
        
        return Math.max(0, Math.min(1, quality));
    }

    /**
     * Extract keywords from text
     * @private
     */
    _extractKeywords(words) {
        const wordFreq = {};
        const stopWords = new Set(stopwords.english);
        
        // Count word frequencies
        words.forEach(word => {
            const cleanWord = word.toLowerCase().replace(/[^\w]/g, '');
            if (cleanWord.length > 2 && !stopWords.has(cleanWord)) {
                wordFreq[cleanWord] = (wordFreq[cleanWord] || 0) + 1;
            }
        });
        
        // Sort by frequency and return top keywords
        return Object.entries(wordFreq)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 20)
            .map(([word, freq]) => ({ word, frequency: freq }));
    }

    /**
     * Extract entities from text (simplified)
     * @private
     */
    _extractEntities(text) {
        const entities = [];
        
        // Simple regex patterns for common entities
        const patterns = {
            email: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g,
            phone: /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/g,
            url: /https?:\/\/[^\s]+/g,
            date: /\b\d{1,2}\/\d{1,2}\/\d{4}\b/g,
            currency: /\$\d+(?:\.\d{2})?/g
        };
        
        Object.entries(patterns).forEach(([type, pattern]) => {
            const matches = text.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    entities.push({ type, value: match });
                });
            }
        });
        
        return entities;
    }

    /**
     * Calculate readability score (Flesch Reading Ease)
     * @private
     */
    _calculateReadability(words, sentences) {
        if (words.length === 0 || sentences.length === 0) return 0;
        
        const avgWordsPerSentence = words.length / sentences.length;
        const avgSyllablesPerWord = this._calculateAverageSyllables(words);
        
        const score = 206.835 - (1.015 * avgWordsPerSentence) - (84.6 * avgSyllablesPerWord);
        return Math.max(0, Math.min(100, score));
    }

    /**
     * Calculate average syllables per word
     * @private
     */
    _calculateAverageSyllables(words) {
        let totalSyllables = 0;
        words.forEach(word => {
            totalSyllables += this._countSyllables(word);
        });
        return totalSyllables / words.length;
    }

    /**
     * Count syllables in a word
     * @private
     */
    _countSyllables(word) {
        word = word.toLowerCase();
        if (word.length <= 3) return 1;
        
        word = word.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, '');
        word = word.replace(/^y/, '');
        const matches = word.match(/[aeiouy]{1,2}/g);
        return matches ? matches.length : 1;
    }

    /**
     * Detect language (simplified)
     * @private
     */
    _detectLanguage(text) {
        // Simple language detection based on common words
        const languagePatterns = {
            en: ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'],
            es: ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo'],
            fr: ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour'],
            de: ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf']
        };
        
        const words = text.toLowerCase().split(/\s+/);
        const scores = {};
        
        Object.entries(languagePatterns).forEach(([lang, patterns]) => {
            scores[lang] = patterns.reduce((score, pattern) => {
                return score + (words.includes(pattern) ? 1 : 0);
            }, 0);
        });
        
        const bestMatch = Object.entries(scores).reduce((a, b) => scores[a[0]] > scores[b[0]] ? a : b);
        return bestMatch[1] > 0 ? bestMatch[0] : 'unknown';
    }

    /**
     * Get file extension from URL path
     * @private
     */
    _getFileExtension(path) {
        const match = path.match(/\.([^.]+)$/);
        return match ? match[1].toLowerCase() : '';
    }

    /**
     * Generate cache key
     * @private
     */
    _generateCacheKey(html, baseUrl, options) {
        const keyData = {
            html: html.substring(0, 1000), // First 1000 chars
            baseUrl,
            options: JSON.stringify(options)
        };
        return crypto.createHash('md5').update(JSON.stringify(keyData)).digest('hex');
    }

    /**
     * Update statistics
     * @private
     */
    _updateStats(processingTime, success) {
        this.stats.totalExtractions++;
        if (success) {
            this.stats.successfulExtractions++;
        } else {
            this.stats.failedExtractions++;
        }
        
        // Update average processing time
        this.stats.averageProcessingTime = (
            (this.stats.averageProcessingTime * (this.stats.totalExtractions - 1) + processingTime) /
            this.stats.totalExtractions
        );
    }

    /**
     * Get extraction statistics
     * @returns {Object} Statistics object
     */
    getStats() {
        return { ...this.stats };
    }

    /**
     * Clear cache
     */
    clearCache() {
        this.cache.clear();
    }

    /**
     * Reset statistics
     */
    resetStats() {
        this.stats = {
            totalExtractions: 0,
            successfulExtractions: 0,
            failedExtractions: 0,
            averageProcessingTime: 0,
            cacheHits: 0,
            cacheMisses: 0
        };
    }
}

module.exports = ContentExtractor;
