package partialloader

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/sourcegraph/scip/bindings/go/scip"
	"github.com/uber/scip-lsp/src/scip-lib/mapper"
	"github.com/uber/scip-lsp/src/scip-lib/model"
	scanner "github.com/uber/scip-lsp/src/scip-lib/scanner"
)

// PartialIndex is a partial index of a SCIP index
type PartialIndex interface {
	SetDocumentLoadedCallback(func(*model.Document))
	LoadIndex(indexPath string, indexReader scanner.ScipReader) error
	LoadIndexFile(file string) error
	LoadDocument(relativeDocPath string) (*model.Document, error)
	GetSymbolInformation(symbol string) (*model.SymbolInformation, string, error)
	GetSymbolInformationFromDescriptors(descriptors []model.Descriptor, version string) (*model.SymbolInformation, string, error)
	References(symbol string) (map[string][]*model.Occurrence, error)
	Implementations(symbol string) ([]string, error)
	Tidy() error
}

type docNodes struct {
	nodes    []*SymbolPrefixTreeNode
	revision int64
}

// PartialLoadedIndex is a partial index of a SCIP index
type PartialLoadedIndex struct {
	// PrefixTreeRoot is the root of the symbol prefix tree
	PrefixTreeRoot *SymbolPrefixTree
	prefixTreeMu   sync.RWMutex
	// LoadedDocuments maps a document path to the SCIP Document
	LoadedDocuments map[string]*model.Document
	loadedDocsMu    sync.RWMutex
	// DocTreeNodes maps a document path to each of the symbol prefix tree nodes that the doc refers to
	DocTreeNodes   map[string]*docNodes
	docTreeNodesMu sync.RWMutex
	// Current revision number for tracking updates
	revision atomic.Int64
	// Documents that were updated in the current revision, mapped to their latest revision number
	updatedDocs   map[string]int64
	updatedDocsMu sync.RWMutex
	// Document to index map
	docToIndex   map[string]string
	docToIndexMu sync.RWMutex
	// Mutex to protect index modifications
	modificationMu sync.Mutex

	indexFolder      string
	pool             *scanner.BufferPool
	onDocumentLoaded func(*model.Document)

	// ImplementorsBySymbol maps abstract/interface symbol -> set of implementing symbols
	implementorsMu       sync.RWMutex
	ImplementorsBySymbol map[string]map[string]struct{}

	// If enabled, Implementations() will use a Bloom filter prefilter and scan matching index files
	// on demand, instead of relying only on the in-memory ImplementorsBySymbol map.
	useBloomImplementations bool
	// indexBlooms maps index file path -> Bloom filter containing abstract symbols that have at least one implementor in that index.
	indexBloomMu sync.RWMutex
	indexBlooms  map[string]*BloomFilter
	// loadedIndexPaths tracks the set of loaded index file paths.
	loadedIndexMu    sync.RWMutex
	loadedIndexPaths map[string]struct{}
}

// Options configure PartialLoadedIndex behavior.
type Options struct {
	// UseBloomImplementations enables the bloom-filter-backed implementation lookup path.
	UseBloomImplementations bool
}

// NewPartialLoadedIndex creates a new PartialLoadedIndex
func NewPartialLoadedIndex(indexFolder string) PartialIndex {
	useBloom := strings.EqualFold(os.Getenv("SCIP_LSP_USE_BLOOM_IMPLEMENTATIONS"), "true")
	return NewPartialLoadedIndexWithOptions(indexFolder, Options{UseBloomImplementations: useBloom})
}

// NewPartialLoadedIndexWithOptions creates a new PartialLoadedIndex using explicit options.
// Options take precedence, but we also keep the legacy env var as a fallback to preserve old behavior.
func NewPartialLoadedIndexWithOptions(indexFolder string, opts Options) PartialIndex {
	useBloom := opts.UseBloomImplementations || strings.EqualFold(os.Getenv("SCIP_LSP_USE_BLOOM_IMPLEMENTATIONS"), "true")
	return &PartialLoadedIndex{
		PrefixTreeRoot:       NewSymbolPrefixTree(),
		DocTreeNodes:         make(map[string]*docNodes),
		LoadedDocuments:      make(map[string]*model.Document),
		updatedDocs:          make(map[string]int64),
		docToIndex:           make(map[string]string, 0),
		indexFolder:          indexFolder,
		pool:                 scanner.NewBufferPool(1024, 12),
		onDocumentLoaded:     func(*model.Document) {},
		ImplementorsBySymbol: make(map[string]map[string]struct{}),
		useBloomImplementations: useBloom,
		indexBlooms:             make(map[string]*BloomFilter),
		loadedIndexPaths:        make(map[string]struct{}),
	}
}

// SetDocumentLoadedCallback sets the callback for when a document is loaded
func (p *PartialLoadedIndex) SetDocumentLoadedCallback(callback func(*model.Document)) {
	p.onDocumentLoaded = callback
}

// LoadIndexFile loads a SCIP index file into the PartialLoadedIndex
func (p *PartialLoadedIndex) LoadIndexFile(file string) error {
	indexReader, err := os.Open(file)
	if err != nil {
		return err
	}
	defer indexReader.Close()

	return p.LoadIndex(file, indexReader)
}

// References returns the occurrences of a symbol in the index
func (p *PartialLoadedIndex) References(symbol string) (map[string][]*model.Occurrence, error) {
	if scip.IsLocalSymbol(symbol) {
		return nil, nil
	}

	occMutex := sync.Mutex{}
	occurrences := make(map[string][]*model.Occurrence)
	docScanner := &scanner.IndexScannerImpl{
		Pool: p.pool,
		MatchOccurrence: func(occSymbol []byte) bool {
			return string(occSymbol) == symbol
		},
		VisitOccurrence: func(docPath string, occ *scip.Occurrence) {
			occMutex.Lock()
			defer occMutex.Unlock()
			modelOcc := mapper.ScipOccurrenceToModelOccurrence(occ)
			if occurrences[docPath] == nil {
				occurrences[docPath] = make([]*model.Occurrence, 0)
			}
			occurrences[docPath] = append(occurrences[docPath], modelOcc)
		},
	}

	docScanner.InitBuffers()
	err := docScanner.ScanIndexFolder(p.indexFolder, true)
	if err != nil {
		return nil, err
	}

	return occurrences, nil
}

// LoadIndex loads a SCIP index into the PartialLoadedIndex
func (p *PartialLoadedIndex) LoadIndex(indexPath string, indexReader scanner.ScipReader) error {
	localPrefixTree := &SymbolPrefixTreeNode{
		Children: make(map[model.Descriptor]*SymbolPrefixTreeNode),
	}
	localDocTreeNodes := make(map[string]*docNodes)
	localUpdatedDocs := make(map[string]int64)
	localDocToIndex := make(map[string]string)
	localImplementorsBySymbol := make(map[string]map[string]struct{})
	localAbsSymbols := make(map[string]struct{})

	loadScanner := &scanner.IndexScannerImpl{
		Pool: p.pool,
		MatchSymbol: func(symbol []byte) bool {
			return !scip.IsLocalSymbol(string(symbol))
		},
		MatchDocumentPath: func(indexDocPath string) bool {
			docPath := filepath.Clean(indexDocPath)
			localDocToIndex[docPath] = indexPath
			if localDocTreeNodes[docPath] == nil {
				localDocTreeNodes[docPath] = &docNodes{
					nodes:    make([]*SymbolPrefixTreeNode, 0),
					revision: p.revision.Add(1),
				}
			}

			localUpdatedDocs[docPath] = localDocTreeNodes[docPath].revision
			p.loadedDocsMu.RLock()
			_, docLoaded := p.LoadedDocuments[docPath]
			p.loadedDocsMu.RUnlock()
			return docLoaded // We should only load the document if it had already been loaded
		},
		VisitSymbol: func(indexDocPath string, info *scip.SymbolInformation) {
			docPath := filepath.Clean(indexDocPath)
			modelInfo := mapper.ScipSymbolInformationToModelSymbolInformation(info)
			leafNode, isNew := localPrefixTree.AddSymbol(docPath, modelInfo, p.revision.Load())

			// Populate reverse implementors for quick lookup (impl -> abs)
			for _, rel := range modelInfo.Relationships {
				if rel != nil && rel.IsImplementation {
					abs := rel.Symbol
					localAbsSymbols[abs] = struct{}{}
					if localImplementorsBySymbol[abs] == nil {
						localImplementorsBySymbol[abs] = make(map[string]struct{})
					}
					localImplementorsBySymbol[abs][modelInfo.Symbol] = struct{}{}
				}
			}

			if isNew {
				localDocTreeNodes[docPath].nodes = append(localDocTreeNodes[docPath].nodes, leafNode)
			}
		},
		VisitDocument: func(doc *scip.Document) {
			docPath := filepath.Clean(doc.RelativePath)
			modelDoc := mapper.ScipDocumentToModelDocument(doc)
			p.loadedDocsMu.Lock()
			p.LoadedDocuments[docPath] = modelDoc
			p.loadedDocsMu.Unlock()
			p.onDocumentLoaded(modelDoc)
		},
	}

	loadScanner.InitBuffers()
	defer func() {
		p.modificationMu.Lock()
		defer p.modificationMu.Unlock()
		p.mergePrefixTree(localPrefixTree)
		p.mergeDocTreeNodes(localDocTreeNodes)
		p.mergeUpdatedDocs(localUpdatedDocs)
		p.mergeDocToIndex(localDocToIndex)
		p.mergeImplementors(localImplementorsBySymbol)

		// Update Bloom filter and loaded index list for this index file.
		if indexPath != "" {
			p.loadedIndexMu.Lock()
			p.loadedIndexPaths[indexPath] = struct{}{}
			p.loadedIndexMu.Unlock()

			// Build a bloom filter for abs symbols that appear in implementation relationships in this index.
			bf := NewBloomFilter(len(localAbsSymbols), 0.01)
			for abs := range localAbsSymbols {
				bf.Add(abs)
			}
			p.indexBloomMu.Lock()
			p.indexBlooms[indexPath] = bf
			p.indexBloomMu.Unlock()
		}
	}()
	return loadScanner.ScanIndexReader(indexReader)
}

// GetSymbolInformationFromDescriptors accepts a slice of descriptors and returns the matching symbol information
// If version is specified, it will return the symbol information for the specified version
// If version is not specified, it will return the symbol information for the first version it finds
func (p *PartialLoadedIndex) GetSymbolInformationFromDescriptors(descriptors []model.Descriptor, version string) (*model.SymbolInformation, string, error) {
	p.prefixTreeMu.RLock()
	defer p.prefixTreeMu.RUnlock()

	if len(descriptors) == 0 {
		return nil, "", errors.New("no descriptors provided")
	}

	node := &p.PrefixTreeRoot.SymbolPrefixTreeNode
	for _, part := range descriptors {
		node = node.Children[part]
		if node == nil {
			return nil, "", nil
		}
	}

	if node.SymbolVersions == nil || len(node.SymbolVersions) == 0 || version == "" {
		return nil, "", nil
	}

	// If version is specified, try to get that specific version
	if versionInfo, exists := node.SymbolVersions[version]; exists {
		return versionInfo.Info, versionInfo.DocumentPath, nil
	}

	// If version is not specified or not found, return the first sorted version we find
	versions := make([]string, 0, len(node.SymbolVersions))
	for v := range node.SymbolVersions {
		versions = append(versions, v)
	}
	sort.Strings(versions)
	firstVersionInfo := node.SymbolVersions[versions[0]]

	return firstVersionInfo.Info, firstVersionInfo.DocumentPath, nil
}

// GetSymbolInformation returns the symbol information for a given symbol as well as the document path where it was found
func (p *PartialLoadedIndex) GetSymbolInformation(symbol string) (*model.SymbolInformation, string, error) {
	if scip.IsLocalSymbol(symbol) {
		return nil, "", nil
	}

	// Parse symbol and walk the tree to find the symbol information
	sy, err := model.ParseScipSymbol(symbol)
	if err != nil {
		return nil, "", err
	}

	return p.GetSymbolInformationFromDescriptors(mapper.ScipDescriptorsToModelDescriptors(sy.Descriptors), sy.Package.Version)
}

// mergePrefixTree merges the local prefix tree into the main index
func (p *PartialLoadedIndex) mergePrefixTree(localPrefixTree *SymbolPrefixTreeNode) {
	p.prefixTreeMu.Lock()
	defer p.prefixTreeMu.Unlock()

	p.PrefixTreeRoot.Merge(localPrefixTree)
}

// mergeDocTreeNodes merges the local doc tree nodes into the main index
func (p *PartialLoadedIndex) mergeDocTreeNodes(localDocTreeNodes map[string]*docNodes) {
	p.docTreeNodesMu.Lock()
	defer p.docTreeNodesMu.Unlock()

	for docPath, nodes := range localDocTreeNodes {
		if p.DocTreeNodes[docPath] == nil {
			p.DocTreeNodes[docPath] = nodes
		} else {
			p.DocTreeNodes[docPath].nodes = append(p.DocTreeNodes[docPath].nodes, nodes.nodes...)
			p.DocTreeNodes[docPath].revision = int64(math.Max(float64(p.DocTreeNodes[docPath].revision), float64(nodes.revision)))
		}
	}
}

// mergeUpdatedDocs merges the local updated docs into the main index
func (p *PartialLoadedIndex) mergeUpdatedDocs(localUpdatedDocs map[string]int64) {
	p.updatedDocsMu.Lock()
	defer p.updatedDocsMu.Unlock()

	for docPath, revision := range localUpdatedDocs {
		p.updatedDocs[docPath] = int64(math.Max(float64(p.updatedDocs[docPath]), float64(revision)))
	}
}

// mergeDocToIndex merges the local docToIndex into the main index
func (p *PartialLoadedIndex) mergeDocToIndex(localDocToIndex map[string]string) {
	p.docToIndexMu.Lock()
	defer p.docToIndexMu.Unlock()

	for docPath, indexPath := range localDocToIndex {
		p.docToIndex[docPath] = indexPath
	}
}

// mergeImplementors merges a local reverse implementors map into the main index
func (p *PartialLoadedIndex) mergeImplementors(local map[string]map[string]struct{}) {
	if len(local) == 0 {
		return
	}
	p.implementorsMu.Lock()
	defer p.implementorsMu.Unlock()
	for abs, impls := range local {
		if p.ImplementorsBySymbol[abs] == nil {
			p.ImplementorsBySymbol[abs] = make(map[string]struct{})
		}
		for impl := range impls {
			p.ImplementorsBySymbol[abs][impl] = struct{}{}
		}
	}
}

// LoadDocument loads a document into the PartialLoadedIndex
func (p *PartialLoadedIndex) LoadDocument(relativeDocPath string) (*model.Document, error) {
	p.loadedDocsMu.RLock()
	if p.LoadedDocuments[relativeDocPath] != nil {
		doc := p.LoadedDocuments[relativeDocPath]
		p.loadedDocsMu.RUnlock()
		return doc, nil
	}
	p.loadedDocsMu.RUnlock()

	p.docToIndexMu.RLock()
	index, ok := p.docToIndex[relativeDocPath]
	p.docToIndexMu.RUnlock()
	var (
		doc *model.Document
		err error
	)
	if ok {
		doc, err = p.loadDocumentFromIndex(index, relativeDocPath)
	} else {
		doc, err = p.loadDocumentFromIndexFolder(relativeDocPath)
	}

	if doc != nil {
		p.onDocumentLoaded(doc)
	}

	return doc, err
}

func (p *PartialLoadedIndex) loadDocumentFromIndex(indexPath, docPath string) (*model.Document, error) {
	docScanner := &scanner.IndexScannerImpl{
		Pool: p.pool,
		MatchDocumentPath: func(indexRelativeDocPath string) bool {
			indexRelativeDocPath = filepath.Clean(indexRelativeDocPath)
			return indexRelativeDocPath == docPath
		},
		VisitDocument: func(doc *scip.Document) {
			modelDoc := mapper.ScipDocumentToModelDocument(doc)
			p.loadedDocsMu.Lock()
			p.LoadedDocuments[docPath] = modelDoc
			p.loadedDocsMu.Unlock()
		},
	}
	docScanner.InitBuffers()
	err := docScanner.ScanIndexFile(indexPath)
	if err != nil {
		return nil, err
	}

	p.loadedDocsMu.RLock()
	doc := p.LoadedDocuments[docPath]
	p.loadedDocsMu.RUnlock()
	return doc, nil
}

// loadDocumentFromIndexFolder loads a document from the index folder into the PartialLoadedIndex
func (p *PartialLoadedIndex) loadDocumentFromIndexFolder(relativeDocPath string) (*model.Document, error) {
	docScanner := &scanner.IndexScannerImpl{
		Pool: p.pool,
		MatchDocumentPath: func(s string) bool {
			s = filepath.Clean(s)
			return s == relativeDocPath
		},
		VisitDocument: func(doc *scip.Document) {
			modelDoc := mapper.ScipDocumentToModelDocument(doc)
			p.loadedDocsMu.Lock()
			p.LoadedDocuments[relativeDocPath] = modelDoc
			p.loadedDocsMu.Unlock()
		},
	}

	docScanner.InitBuffers()
	err := docScanner.ScanIndexFolder(p.indexFolder, true)
	if err != nil {
		return nil, err
	}

	p.loadedDocsMu.RLock()
	doc := p.LoadedDocuments[relativeDocPath]
	p.loadedDocsMu.RUnlock()
	return doc, nil
}

// Implementations returns the list of implementing symbols for a given abstract/interface symbol
func (p *PartialLoadedIndex) Implementations(symbol string) ([]string, error) {
	if scip.IsLocalSymbol(symbol) {
		return []string{}, nil
	}

	if p.useBloomImplementations {
		return p.implementationsByBloom(symbol)
	}

	p.implementorsMu.RLock()
	set := p.ImplementorsBySymbol[symbol]
	p.implementorsMu.RUnlock()
	if set == nil {
		return []string{}, nil
	}
	res := make([]string, 0, len(set))
	for s := range set {
		res = append(res, s)
	}
	sort.Strings(res)
	return res, nil
}

// implementationsByBloom uses per-index Bloom filters to select candidate index files and then scans
// only those files to collect implementing symbols for the given abstract/interface symbol.
func (p *PartialLoadedIndex) implementationsByBloom(symbol string) ([]string, error) {
	p.loadedIndexMu.RLock()
	indices := make([]string, 0, len(p.loadedIndexPaths))
	for idx := range p.loadedIndexPaths {
		indices = append(indices, idx)
	}
	p.loadedIndexMu.RUnlock()

	// If we have no known loaded index files (e.g. LoadIndex called without a path), fall back to in-memory map.
	if len(indices) == 0 {
		p.implementorsMu.RLock()
		set := p.ImplementorsBySymbol[symbol]
		p.implementorsMu.RUnlock()
		if set == nil {
			return []string{}, nil
		}
		res := make([]string, 0, len(set))
		for s := range set {
			res = append(res, s)
		}
		sort.Strings(res)
		return res, nil
	}

	results := make(map[string]struct{})
	for _, indexPath := range indices {
		// Bloom prefilter: skip index files that definitely don't contain impl relationships for this symbol.
		p.indexBloomMu.RLock()
		bf := p.indexBlooms[indexPath]
		p.indexBloomMu.RUnlock()
		if bf != nil && !bf.MightContain(symbol) {
			continue
		}

		var mu sync.Mutex
		sc := &scanner.IndexScannerImpl{
			Pool: p.pool,
			MatchSymbol: func(_ []byte) bool {
				// We only care about global symbols for relationships.
				return true
			},
			VisitSymbol: func(_ string, info *scip.SymbolInformation) {
				modelInfo := mapper.ScipSymbolInformationToModelSymbolInformation(info)
				for _, rel := range modelInfo.Relationships {
					if rel != nil && rel.IsImplementation && rel.Symbol == symbol {
						mu.Lock()
						results[modelInfo.Symbol] = struct{}{}
						mu.Unlock()
					}
				}
			},
		}
		sc.InitBuffers()
		if err := sc.ScanIndexFile(indexPath); err != nil {
			return nil, err
		}
	}

	res := make([]string, 0, len(results))
	for s := range results {
		res = append(res, s)
	}
	sort.Strings(res)
	return res, nil
}

// Tidy prunes nodes for documents that were updated in the current revision
func (p *PartialLoadedIndex) Tidy() error {
	// Acquire the modification mutex to prevent new index loads during cleanup
	p.modificationMu.Lock()
	defer p.modificationMu.Unlock()

	p.updatedDocsMu.RLock()
	for docPath, nodes := range p.DocTreeNodes {
		p.updatedDocsMu.RUnlock()
		p.docTreeNodesMu.RLock()
		if nodes != nil {
			// Get the common parent of all nodes for this document
			// and prune from there to avoid affecting other documents
			parent := nodes.nodes[0].Parent
			if parent != nil {
				p.prefixTreeMu.Lock()
				parent.PruneNodes(docPath, nodes.revision)
				p.prefixTreeMu.Unlock()
			}
		}
		p.docTreeNodesMu.RUnlock()
		p.updatedDocsMu.RLock()
	}
	p.updatedDocsMu.RUnlock()

	// Clear the updated docs map after cleanup
	p.updatedDocsMu.Lock()
	p.updatedDocs = make(map[string]int64)
	p.updatedDocsMu.Unlock()
	return nil
}
