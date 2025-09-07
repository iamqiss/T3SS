// T3SS Project
// File: core/indexing/crawler/parser/xml_json_handler.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package parser

import (
	"bytes"
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"log"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/beevik/etree"
	"github.com/antchfx/xmlquery"
	"github.com/antchfx/xpath"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// DocumentType represents the type of document being processed
type DocumentType string

const (
	DocumentTypeXML  DocumentType = "xml"
	DocumentTypeJSON DocumentType = "json"
	DocumentTypeHTML DocumentType = "html"
	DocumentTypeRSS  DocumentType = "rss"
	DocumentTypeAtom DocumentType = "atom"
	DocumentTypeSitemap DocumentType = "sitemap"
)

// ProcessingMode defines how the document should be processed
type ProcessingMode string

const (
	ModeStrict   ProcessingMode = "strict"
	ModeLenient  ProcessingMode = "lenient"
	ModeFast     ProcessingMode = "fast"
	ModeComplete ProcessingMode = "complete"
)

// Config holds configuration for XML/JSON processing
type Config struct {
	// General settings
	DocumentType    DocumentType    `json:"document_type"`
	ProcessingMode  ProcessingMode  `json:"processing_mode"`
	MaxSize         int64          `json:"max_size"`
	Timeout         time.Duration  `json:"timeout"`
	EnableValidation bool          `json:"enable_validation"`
	EnableSchemaValidation bool    `json:"enable_schema_validation"`
	
	// XML specific settings
	XMLNamespacePrefixes map[string]string `json:"xml_namespace_prefixes"`
	XMLSchemaLocation    string            `json:"xml_schema_location"`
	XMLDTDLocation       string            `json:"xml_dtd_location"`
	XMLPreserveWhitespace bool             `json:"xml_preserve_whitespace"`
	XMLTrimWhitespace    bool              `json:"xml_trim_whitespace"`
	
	// JSON specific settings
	JSONUseNumber        bool `json:"json_use_number"`
	JSONDisallowUnknownFields bool `json:"json_disallow_unknown_fields"`
	JSONValidateSchema   bool `json:"json_validate_schema"`
	
	// Processing settings
	EnableXPathQueries   bool `json:"enable_xpath_queries"`
	EnableJSONPathQueries bool `json:"enable_jsonpath_queries"`
	EnableContentExtraction bool `json:"enable_content_extraction"`
	EnableMetadataExtraction bool `json:"enable_metadata_extraction"`
	EnableLinkExtraction bool `json:"enable_link_extraction"`
	
	// Output settings
	PrettyPrint         bool `json:"pretty_print"`
	IncludeComments     bool `json:"include_comments"`
	IncludeProcessingInstructions bool `json:"include_processing_instructions"`
	IncludeNamespaces   bool `json:"include_namespaces"`
}

// DefaultConfig returns a default configuration
func DefaultConfig() *Config {
	return &Config{
		DocumentType:         DocumentTypeXML,
		ProcessingMode:       ModeLenient,
		MaxSize:             10 * 1024 * 1024, // 10MB
		Timeout:             30 * time.Second,
		EnableValidation:    true,
		EnableSchemaValidation: false,
		XMLNamespacePrefixes: make(map[string]string),
		XMLPreserveWhitespace: false,
		XMLTrimWhitespace:   true,
		JSONUseNumber:       false,
		JSONDisallowUnknownFields: false,
		JSONValidateSchema:  false,
		EnableXPathQueries:  true,
		EnableJSONPathQueries: true,
		EnableContentExtraction: true,
		EnableMetadataExtraction: true,
		EnableLinkExtraction: true,
		PrettyPrint:         false,
		IncludeComments:     false,
		IncludeProcessingInstructions: false,
		IncludeNamespaces:   true,
	}
}

// Document represents a parsed document
type Document struct {
	Type        DocumentType            `json:"type"`
	Content     interface{}             `json:"content"`
	Metadata    map[string]interface{}  `json:"metadata"`
	Links       []Link                  `json:"links"`
	Text        string                  `json:"text"`
	Attributes  map[string]interface{}  `json:"attributes"`
	Namespaces  map[string]string       `json:"namespaces"`
	ProcessingTime time.Duration        `json:"processing_time"`
	Size        int64                   `json:"size"`
	Error       string                  `json:"error,omitempty"`
}

// Link represents a link found in the document
type Link struct {
	URL         string            `json:"url"`
	Text        string            `json:"text"`
	Type        string            `json:"type"`
	Attributes  map[string]string `json:"attributes"`
	Context     string            `json:"context"`
}

// XMLNode represents an XML node
type XMLNode struct {
	Name        string                 `json:"name"`
	Value       string                 `json:"value"`
	Attributes  map[string]string      `json:"attributes"`
	Children    []*XMLNode             `json:"children"`
	Namespace   string                 `json:"namespace"`
	Prefix      string                 `json:"prefix"`
	Path        string                 `json:"path"`
	Line        int                    `json:"line"`
	Column      int                    `json:"column"`
}

// JSONNode represents a JSON node
type JSONNode struct {
	Key         string                 `json:"key"`
	Value       interface{}            `json:"value"`
	Type        string                 `json:"type"`
	Path        string                 `json:"path"`
	Parent      *JSONNode              `json:"parent,omitempty"`
	Children    []*JSONNode            `json:"children,omitempty"`
}

// Parser handles XML and JSON document parsing
type Parser struct {
	config     *Config
	logger     *log.Logger
	mu         sync.RWMutex
	stats      *ParserStats
	cache      map[string]*Document
	cacheSize  int
	maxCacheSize int
}

// ParserStats holds parsing statistics
type ParserStats struct {
	TotalDocuments    int64         `json:"total_documents"`
	SuccessfulParses  int64         `json:"successful_parses"`
	FailedParses      int64         `json:"failed_parses"`
	AverageParseTime  time.Duration `json:"average_parse_time"`
	TotalParseTime    time.Duration `json:"total_parse_time"`
	CacheHits         int64         `json:"cache_hits"`
	CacheMisses       int64         `json:"cache_misses"`
	XMLDocuments      int64         `json:"xml_documents"`
	JSONDocuments     int64         `json:"json_documents"`
	RSSDocuments      int64         `json:"rss_documents"`
	AtomDocuments     int64         `json:"atom_documents"`
	SitemapDocuments  int64         `json:"sitemap_documents"`
}

// NewParser creates a new XML/JSON parser
func NewParser(config *Config) *Parser {
	if config == nil {
		config = DefaultConfig()
	}
	
	return &Parser{
		config:       config,
		logger:       log.New(io.Discard, "", 0),
		stats:        &ParserStats{},
		cache:        make(map[string]*Document),
		maxCacheSize: 1000,
	}
}

// SetLogger sets the logger for the parser
func (p *Parser) SetLogger(logger *log.Logger) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.logger = logger
}

// Parse parses a document from the given reader
func (p *Parser) Parse(ctx context.Context, reader io.Reader) (*Document, error) {
	start := time.Now()
	
	// Read the document
	data, err := io.ReadAll(reader)
	if err != nil {
		p.updateStats(start, false)
		return nil, fmt.Errorf("failed to read document: %w", err)
	}
	
	// Check size limit
	if int64(len(data)) > p.config.MaxSize {
		p.updateStats(start, false)
		return nil, fmt.Errorf("document too large: %d bytes (max: %d)", len(data), p.config.MaxSize)
	}
	
	// Check cache
	cacheKey := p.generateCacheKey(data)
	if doc, exists := p.getFromCache(cacheKey); exists {
		p.stats.CacheHits++
		return doc, nil
	}
	
	p.stats.CacheMisses++
	
	// Create document
	doc := &Document{
		Type:     p.config.DocumentType,
		Size:     int64(len(data)),
		Metadata: make(map[string]interface{}),
		Links:    make([]Link, 0),
		Attributes: make(map[string]interface{}),
		Namespaces: make(map[string]string),
	}
	
	// Parse based on document type
	switch p.config.DocumentType {
	case DocumentTypeXML, DocumentTypeRSS, DocumentTypeAtom, DocumentTypeSitemap:
		err = p.parseXML(ctx, data, doc)
	case DocumentTypeJSON:
		err = p.parseJSON(ctx, data, doc)
	default:
		err = p.autoDetectAndParse(ctx, data, doc)
	}
	
	if err != nil {
		doc.Error = err.Error()
		p.updateStats(start, false)
		return doc, err
	}
	
	// Extract content if enabled
	if p.config.EnableContentExtraction {
		p.extractContent(doc)
	}
	
	// Extract metadata if enabled
	if p.config.EnableMetadataExtraction {
		p.extractMetadata(doc)
	}
	
	// Extract links if enabled
	if p.config.EnableLinkExtraction {
		p.extractLinks(doc)
	}
	
	doc.ProcessingTime = time.Since(start)
	p.updateStats(start, true)
	p.addToCache(cacheKey, doc)
	
	return doc, nil
}

// parseXML parses XML documents
func (p *Parser) parseXML(ctx context.Context, data []byte, doc *Document) error {
	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, p.config.Timeout)
	defer cancel()
	
	// Parse XML
	root, err := xmlquery.Parse(bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("XML parsing failed: %w", err)
	}
	
	// Convert to our XMLNode structure
	doc.Content = p.convertXMLNode(root)
	
	// Extract namespaces
	p.extractNamespaces(root, doc)
	
	// Set document type based on root element
	doc.Type = p.detectXMLDocumentType(root)
	
	return nil
}

// parseJSON parses JSON documents
func (p *Parser) parseJSON(ctx context.Context, data []byte, doc *Document) error {
	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, p.config.Timeout)
	defer cancel()
	
	// Parse JSON
	var content interface{}
	decoder := json.NewDecoder(bytes.NewReader(data))
	
	if p.config.JSONUseNumber {
		decoder.UseNumber()
	}
	
	if p.config.JSONDisallowUnknownFields {
		decoder.DisallowUnknownFields()
	}
	
	if err := decoder.Decode(&content); err != nil {
		return fmt.Errorf("JSON parsing failed: %w", err)
	}
	
	// Convert to our JSONNode structure
	doc.Content = p.convertJSONNode("", content, nil)
	doc.Type = DocumentTypeJSON
	
	return nil
}

// autoDetectAndParse automatically detects document type and parses
func (p *Parser) autoDetectAndParse(ctx context.Context, data []byte, doc *Document) error {
	// Try to detect document type
	docType := p.detectDocumentType(data)
	doc.Type = docType
	
	// Parse based on detected type
	switch docType {
	case DocumentTypeXML, DocumentTypeRSS, DocumentTypeAtom, DocumentTypeSitemap:
		return p.parseXML(ctx, data, doc)
	case DocumentTypeJSON:
		return p.parseJSON(ctx, data, doc)
	default:
		return fmt.Errorf("unsupported document type: %s", docType)
	}
}

// detectDocumentType automatically detects the document type
func (p *Parser) detectDocumentType(data []byte) DocumentType {
	// Check for XML declaration or root element
	if bytes.HasPrefix(bytes.TrimSpace(data), []byte("<?xml")) || 
	   bytes.HasPrefix(bytes.TrimSpace(data), []byte("<")) {
		
		// Check for specific XML document types
		content := string(data)
		
		if strings.Contains(content, "<rss") || strings.Contains(content, "<channel") {
			return DocumentTypeRSS
		}
		if strings.Contains(content, "<feed") && strings.Contains(content, "xmlns=\"http://www.w3.org/2005/Atom\"") {
			return DocumentTypeAtom
		}
		if strings.Contains(content, "<urlset") && strings.Contains(content, "xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\"") {
			return DocumentTypeSitemap
		}
		
		return DocumentTypeXML
	}
	
	// Check for JSON
	if bytes.HasPrefix(bytes.TrimSpace(data), []byte("{")) || 
	   bytes.HasPrefix(bytes.TrimSpace(data), []byte("[")) {
		return DocumentTypeJSON
	}
	
	// Default to XML
	return DocumentTypeXML
}

// detectXMLDocumentType detects specific XML document types
func (p *Parser) detectXMLDocumentType(root *xmlquery.Node) DocumentType {
	if root == nil {
		return DocumentTypeXML
	}
	
	name := strings.ToLower(root.Data)
	
	switch name {
	case "rss", "channel":
		return DocumentTypeRSS
	case "feed":
		// Check for Atom namespace
		for _, attr := range root.Attr {
			if attr.Name.Local == "xmlns" && strings.Contains(attr.Value, "atom") {
				return DocumentTypeAtom
			}
		}
		return DocumentTypeAtom
	case "urlset":
		// Check for sitemap namespace
		for _, attr := range root.Attr {
			if attr.Name.Local == "xmlns" && strings.Contains(attr.Value, "sitemap") {
				return DocumentTypeSitemap
			}
		}
		return DocumentTypeSitemap
	default:
		return DocumentTypeXML
	}
}

// convertXMLNode converts xmlquery.Node to XMLNode
func (p *Parser) convertXMLNode(node *xmlquery.Node) *XMLNode {
	if node == nil {
		return nil
	}
	
	xmlNode := &XMLNode{
		Name:       node.Data,
		Value:      node.InnerText(),
		Attributes: make(map[string]string),
		Children:   make([]*XMLNode, 0),
		Namespace:  node.NamespaceURI,
		Prefix:     node.Prefix,
		Path:       p.getNodePath(node),
		Line:       node.Line,
		Column:     node.Column,
	}
	
	// Convert attributes
	for _, attr := range node.Attr {
		xmlNode.Attributes[attr.Name.Local] = attr.Value
	}
	
	// Convert children
	for child := node.FirstChild; child != nil; child = child.NextSibling {
		if child.Type == xmlquery.ElementNode {
			xmlNode.Children = append(xmlNode.Children, p.convertXMLNode(child))
		}
	}
	
	return xmlNode
}

// convertJSONNode converts JSON data to JSONNode
func (p *Parser) convertJSONNode(key string, value interface{}, parent *JSONNode) *JSONNode {
	node := &JSONNode{
		Key:    key,
		Value:  value,
		Type:   reflect.TypeOf(value).String(),
		Parent: parent,
	}
	
	// Set path
	if parent != nil {
		if parent.Key != "" {
			node.Path = parent.Path + "." + key
		} else {
			node.Path = key
		}
	} else {
		node.Path = key
	}
	
	// Convert children based on type
	switch v := value.(type) {
	case map[string]interface{}:
		node.Children = make([]*JSONNode, 0, len(v))
		for k, val := range v {
			node.Children = append(node.Children, p.convertJSONNode(k, val, node))
		}
	case []interface{}:
		node.Children = make([]*JSONNode, 0, len(v))
		for i, val := range v {
			key := fmt.Sprintf("[%d]", i)
			node.Children = append(node.Children, p.convertJSONNode(key, val, node))
		}
	}
	
	return node
}

// extractNamespaces extracts namespace information from XML
func (p *Parser) extractNamespaces(root *xmlquery.Node, doc *Document) {
	if root == nil {
		return
	}
	
	// Extract namespaces from root element
	for _, attr := range root.Attr {
		if attr.Name.Space == "xmlns" || attr.Name.Local == "xmlns" {
			prefix := attr.Name.Local
			if prefix == "xmlns" {
				prefix = ""
			}
			doc.Namespaces[prefix] = attr.Value
		}
	}
	
	// Recursively extract namespaces from all elements
	p.extractNamespacesRecursive(root, doc)
}

// extractNamespacesRecursive recursively extracts namespaces
func (p *Parser) extractNamespacesRecursive(node *xmlquery.Node, doc *Document) {
	if node == nil {
		return
	}
	
	// Extract namespaces from current node
	for _, attr := range node.Attr {
		if attr.Name.Space == "xmlns" || attr.Name.Local == "xmlns" {
			prefix := attr.Name.Local
			if prefix == "xmlns" {
				prefix = ""
			}
			doc.Namespaces[prefix] = attr.Value
		}
	}
	
	// Process children
	for child := node.FirstChild; child != nil; child = child.NextSibling {
		if child.Type == xmlquery.ElementNode {
			p.extractNamespacesRecursive(child, doc)
		}
	}
}

// extractContent extracts text content from the document
func (p *Parser) extractContent(doc *Document) {
	switch doc.Type {
	case DocumentTypeXML, DocumentTypeRSS, DocumentTypeAtom, DocumentTypeSitemap:
		p.extractXMLContent(doc)
	case DocumentTypeJSON:
		p.extractJSONContent(doc)
	}
}

// extractXMLContent extracts text content from XML
func (p *Parser) extractXMLContent(doc *Document) {
	if xmlNode, ok := doc.Content.(*XMLNode); ok {
		doc.Text = p.extractTextFromXMLNode(xmlNode)
	}
}

// extractJSONContent extracts text content from JSON
func (p *Parser) extractJSONContent(doc *Document) {
	if jsonNode, ok := doc.Content.(*JSONNode); ok {
		doc.Text = p.extractTextFromJSONNode(jsonNode)
	}
}

// extractTextFromXMLNode extracts text from XML node
func (p *Parser) extractTextFromXMLNode(node *XMLNode) string {
	if node == nil {
		return ""
	}
	
	var text strings.Builder
	
	// Add node value
	if node.Value != "" {
		text.WriteString(node.Value)
		text.WriteString(" ")
	}
	
	// Add children text
	for _, child := range node.Children {
		text.WriteString(p.extractTextFromXMLNode(child))
	}
	
	return strings.TrimSpace(text.String())
}

// extractTextFromJSONNode extracts text from JSON node
func (p *Parser) extractTextFromJSONNode(node *JSONNode) string {
	if node == nil {
		return ""
	}
	
	var text strings.Builder
	
	// Add string values
	if str, ok := node.Value.(string); ok {
		text.WriteString(str)
		text.WriteString(" ")
	}
	
	// Add children text
	for _, child := range node.Children {
		text.WriteString(p.extractTextFromJSONNode(child))
	}
	
	return strings.TrimSpace(text.String())
}

// extractMetadata extracts metadata from the document
func (p *Parser) extractMetadata(doc *Document) {
	switch doc.Type {
	case DocumentTypeXML, DocumentTypeRSS, DocumentTypeAtom, DocumentTypeSitemap:
		p.extractXMLMetadata(doc)
	case DocumentTypeJSON:
		p.extractJSONMetadata(doc)
	}
}

// extractXMLMetadata extracts metadata from XML
func (p *Parser) extractXMLMetadata(doc *Document) {
	if xmlNode, ok := doc.Content.(*XMLNode); ok {
		p.extractMetadataFromXMLNode(xmlNode, doc)
	}
}

// extractJSONMetadata extracts metadata from JSON
func (p *Parser) extractJSONMetadata(doc *Document) {
	if jsonNode, ok := doc.Content.(*JSONNode); ok {
		p.extractMetadataFromJSONNode(jsonNode, doc)
	}
}

// extractMetadataFromXMLNode extracts metadata from XML node
func (p *Parser) extractMetadataFromXMLNode(node *XMLNode, doc *Document) {
	if node == nil {
		return
	}
	
	// Extract common metadata fields
	switch strings.ToLower(node.Name) {
	case "title":
		doc.Metadata["title"] = node.Value
	case "description":
		doc.Metadata["description"] = node.Value
	case "author":
		doc.Metadata["author"] = node.Value
	case "pubdate", "published":
		doc.Metadata["published"] = node.Value
	case "updated", "lastmod":
		doc.Metadata["updated"] = node.Value
	case "link":
		doc.Metadata["link"] = node.Value
	case "language", "lang":
		doc.Metadata["language"] = node.Value
	}
	
	// Process children
	for _, child := range node.Children {
		p.extractMetadataFromXMLNode(child, doc)
	}
}

// extractMetadataFromJSONNode extracts metadata from JSON node
func (p *Parser) extractMetadataFromJSONNode(node *JSONNode, doc *Document) {
	if node == nil {
		return
	}
	
	// Extract common metadata fields
	switch strings.ToLower(node.Key) {
	case "title":
		if str, ok := node.Value.(string); ok {
			doc.Metadata["title"] = str
		}
	case "description":
		if str, ok := node.Value.(string); ok {
			doc.Metadata["description"] = str
		}
	case "author":
		if str, ok := node.Value.(string); ok {
			doc.Metadata["author"] = str
		}
	case "published", "pubdate":
		if str, ok := node.Value.(string); ok {
			doc.Metadata["published"] = str
		}
	case "updated", "lastmod":
		if str, ok := node.Value.(string); ok {
			doc.Metadata["updated"] = str
		}
	case "url", "link":
		if str, ok := node.Value.(string); ok {
			doc.Metadata["link"] = str
		}
	case "language", "lang":
		if str, ok := node.Value.(string); ok {
			doc.Metadata["language"] = str
		}
	}
	
	// Process children
	for _, child := range node.Children {
		p.extractMetadataFromJSONNode(child, doc)
	}
}

// extractLinks extracts links from the document
func (p *Parser) extractLinks(doc *Document) {
	switch doc.Type {
	case DocumentTypeXML, DocumentTypeRSS, DocumentTypeAtom, DocumentTypeSitemap:
		p.extractXMLLinks(doc)
	case DocumentTypeJSON:
		p.extractJSONLinks(doc)
	}
}

// extractXMLLinks extracts links from XML
func (p *Parser) extractXMLLinks(doc *Document) {
	if xmlNode, ok := doc.Content.(*XMLNode); ok {
		p.extractLinksFromXMLNode(xmlNode, doc)
	}
}

// extractJSONLinks extracts links from JSON
func (p *Parser) extractJSONLinks(doc *Document) {
	if jsonNode, ok := doc.Content.(*JSONNode); ok {
		p.extractLinksFromJSONNode(jsonNode, doc)
	}
}

// extractLinksFromXMLNode extracts links from XML node
func (p *Parser) extractLinksFromXMLNode(node *XMLNode, doc *Document) {
	if node == nil {
		return
	}
	
	// Check if this node contains a link
	if strings.ToLower(node.Name) == "link" || strings.ToLower(node.Name) == "url" {
		link := Link{
			URL:        node.Value,
			Text:       node.Value,
			Type:       node.Name,
			Attributes: make(map[string]string),
			Context:    node.Path,
		}
		
		// Copy attributes
		for k, v := range node.Attributes {
			link.Attributes[k] = v
		}
		
		doc.Links = append(doc.Links, link)
	}
	
	// Check attributes for links
	for attrName, attrValue := range node.Attributes {
		if strings.Contains(strings.ToLower(attrName), "url") || 
		   strings.Contains(strings.ToLower(attrName), "link") ||
		   strings.Contains(strings.ToLower(attrName), "href") {
			link := Link{
				URL:        attrValue,
				Text:       attrValue,
				Type:       attrName,
				Attributes: make(map[string]string),
				Context:    node.Path,
			}
			
			// Copy all attributes
			for k, v := range node.Attributes {
				link.Attributes[k] = v
			}
			
			doc.Links = append(doc.Links, link)
		}
	}
	
	// Process children
	for _, child := range node.Children {
		p.extractLinksFromXMLNode(child, doc)
	}
}

// extractLinksFromJSONNode extracts links from JSON node
func (p *Parser) extractLinksFromJSONNode(node *JSONNode, doc *Document) {
	if node == nil {
		return
	}
	
	// Check if this node contains a link
	if strings.Contains(strings.ToLower(node.Key), "url") || 
	   strings.Contains(strings.ToLower(node.Key), "link") ||
	   strings.Contains(strings.ToLower(node.Key), "href") {
		if str, ok := node.Value.(string); ok {
			link := Link{
				URL:        str,
				Text:       str,
				Type:       node.Key,
				Attributes: make(map[string]string),
				Context:    node.Path,
			}
			
			doc.Links = append(doc.Links, link)
		}
	}
	
	// Process children
	for _, child := range node.Children {
		p.extractLinksFromJSONNode(child, doc)
	}
}

// QueryXPath executes an XPath query on XML documents
func (p *Parser) QueryXPath(doc *Document, query string) ([]*XMLNode, error) {
	if doc.Type != DocumentTypeXML && doc.Type != DocumentTypeRSS && 
	   doc.Type != DocumentTypeAtom && doc.Type != DocumentTypeSitemap {
		return nil, fmt.Errorf("XPath queries only supported for XML documents")
	}
	
	// Convert our XMLNode back to xmlquery.Node for XPath processing
	// This is a simplified implementation
	return nil, fmt.Errorf("XPath queries not yet implemented")
}

// QueryJSONPath executes a JSONPath query on JSON documents
func (p *Parser) QueryJSONPath(doc *Document, query string) ([]interface{}, error) {
	if doc.Type != DocumentTypeJSON {
		return nil, fmt.Errorf("JSONPath queries only supported for JSON documents")
	}
	
	// Use gjson for JSONPath queries
	result := gjson.Get(doc.Text, query)
	
	if result.IsArray() {
		var results []interface{}
		for _, item := range result.Array() {
			results = append(results, item.Value())
		}
		return results, nil
	}
	
	return []interface{}{result.Value()}, nil
}

// getNodePath returns the XPath-like path to a node
func (p *Parser) getNodePath(node *xmlquery.Node) string {
	if node == nil {
		return ""
	}
	
	var path strings.Builder
	p.buildNodePath(node, &path)
	return path.String()
}

// buildNodePath builds the path to a node
func (p *Parser) buildNodePath(node *xmlquery.Node, path *strings.Builder) {
	if node.Parent != nil {
		p.buildNodePath(node.Parent, path)
		if path.Len() > 0 {
			path.WriteString("/")
		}
	}
	
	if node.Type == xmlquery.ElementNode {
		path.WriteString(node.Data)
	}
}

// generateCacheKey generates a cache key for the document
func (p *Parser) generateCacheKey(data []byte) string {
	// Simple hash-based cache key
	hash := 0
	for _, b := range data {
		hash = hash*31 + int(b)
	}
	return fmt.Sprintf("%x", hash)
}

// getFromCache retrieves a document from cache
func (p *Parser) getFromCache(key string) (*Document, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	doc, exists := p.cache[key]
	return doc, exists
}

// addToCache adds a document to cache
func (p *Parser) addToCache(key string, doc *Document) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	// Check cache size
	if len(p.cache) >= p.maxCacheSize {
		// Remove oldest entry (simple FIFO)
		for k := range p.cache {
			delete(p.cache, k)
			break
		}
	}
	
	p.cache[key] = doc
}

// updateStats updates parsing statistics
func (p *Parser) updateStats(start time.Time, success bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.stats.TotalDocuments++
	if success {
		p.stats.SuccessfulParses++
	} else {
		p.stats.FailedParses++
	}
	
	elapsed := time.Since(start)
	p.stats.TotalParseTime += elapsed
	p.stats.AverageParseTime = p.stats.TotalParseTime / time.Duration(p.stats.TotalDocuments)
}

// GetStats returns parsing statistics
func (p *Parser) GetStats() *ParserStats {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	// Return a copy
	stats := *p.stats
	return &stats
}

// ClearCache clears the document cache
func (p *Parser) ClearCache() {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.cache = make(map[string]*Document)
}

// ResetStats resets parsing statistics
func (p *Parser) ResetStats() {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.stats = &ParserStats{}
}

// SetMaxCacheSize sets the maximum cache size
func (p *Parser) SetMaxCacheSize(size int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.maxCacheSize = size
}

// Validate validates a document against its schema
func (p *Parser) Validate(doc *Document) error {
	if !p.config.EnableValidation {
		return nil
	}
	
	switch doc.Type {
	case DocumentTypeXML, DocumentTypeRSS, DocumentTypeAtom, DocumentTypeSitemap:
		return p.validateXML(doc)
	case DocumentTypeJSON:
		return p.validateJSON(doc)
	default:
		return fmt.Errorf("validation not supported for document type: %s", doc.Type)
	}
}

// validateXML validates XML documents
func (p *Parser) validateXML(doc *Document) error {
	// Basic XML validation
	if doc.Content == nil {
		return fmt.Errorf("document has no content")
	}
	
	// Additional validation can be added here
	return nil
}

// validateJSON validates JSON documents
func (p *Parser) validateJSON(doc *Document) error {
	// Basic JSON validation
	if doc.Content == nil {
		return fmt.Errorf("document has no content")
	}
	
	// Additional validation can be added here
	return nil
}

// ToJSON converts a document to JSON
func (p *Parser) ToJSON(doc *Document) ([]byte, error) {
	return json.MarshalIndent(doc, "", "  ")
}

// ToXML converts a document to XML
func (p *Parser) ToXML(doc *Document) ([]byte, error) {
	if doc.Type != DocumentTypeXML && doc.Type != DocumentTypeRSS && 
	   doc.Type != DocumentTypeAtom && doc.Type != DocumentTypeSitemap {
		return nil, fmt.Errorf("can only convert XML documents to XML")
	}
	
	// Convert XMLNode back to XML
	// This is a simplified implementation
	return []byte(doc.Text), nil
}

// NewParserWithDefaults creates a new parser with default configuration
func NewParserWithDefaults() *Parser {
	return NewParser(DefaultConfig())
}

// ParseDocument is a convenience function to parse a document
func ParseDocument(reader io.Reader, docType DocumentType) (*Document, error) {
	config := DefaultConfig()
	config.DocumentType = docType
	
	parser := NewParser(config)
	return parser.Parse(context.Background(), reader)
}

// ParseDocumentWithConfig parses a document with custom configuration
func ParseDocumentWithConfig(reader io.Reader, config *Config) (*Document, error) {
	parser := NewParser(config)
	return parser.Parse(context.Background(), reader)
}
