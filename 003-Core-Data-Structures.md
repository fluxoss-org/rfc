# 003-Core-Data-Structures.md

# Core Data Structures and Algorithms: Implementation Foundation for Compositional Source Control

## Abstract

The compositional source control paradigm requires fundamentally new data structures and algorithms to support semantic change tracking, event sourcing with deterministic replay, and layered caching for performance optimization. This research presents a comprehensive exploration of the core technical infrastructure needed to implement compositional source control systems, including optimized storage formats for Atomic Change Units (ACUs), event sourcing implementation patterns, layer caching algorithms, and composition engine architecture.

Building upon the semantic change understanding foundation, this work establishes the technical blueprints for storing, retrieving, and manipulating compositional changes at scale. The proposed data structures are designed to handle the exponential complexity of change composition while maintaining logarithmic access times and linear storage requirements. The algorithms presented enable real-time branch composition, conflict detection, and deterministic replay across distributed development teams.

## 1. Introduction & Problem Statement

### 1.1 The Data Structure Challenge

Traditional version control systems like Git store snapshots of file states, which works well for linear development but becomes exponentially complex when handling compositional changes. The fundamental challenges in compositional source control include:

1. **Change Composition Complexity**: Combining N changes can potentially result in 2^N different compositions
2. **Deterministic Replay Requirements**: Same input sequence must always produce identical output
3. **Performance at Scale**: Systems must handle thousands of concurrent ACUs across large codebases
4. **Memory Efficiency**: Avoid storing redundant information while enabling fast access
5. **Distributed Consistency**: Maintain consistency across distributed development teams

### 1.2 Requirements Analysis

#### 1.2.1 Functional Requirements

**ACU Storage and Retrieval**:
- Efficient storage of semantic change information
- Fast retrieval by ACU ID, intent, or content hash
- Support for ACU versioning and evolution
- Cross-referencing capabilities for dependencies and conflicts

**Event Sourcing Infrastructure**:
- Immutable event stream storage
- Deterministic replay from any point in history
- Efficient snapshotting for performance optimization
- Support for temporal queries and analysis

**Layer Caching System**:
- Docker-inspired layered storage architecture
- Copy-on-write semantics for efficient storage
- Layer deduplication and garbage collection
- Fast layer composition and materialization

**Composition Engine**:
- Real-time branch materialization from ACU recipes
- Conflict detection during composition
- Optimization for common composition patterns
- Support for speculative execution and what-if analysis

#### 1.2.2 Non-Functional Requirements

**Performance**:
- Sub-second response times for branch materialization
- Linear storage growth with number of unique ACUs
- Logarithmic access times for ACU retrieval
- Efficient memory utilization for large repositories

**Scalability**:
- Support for repositories with 10M+ lines of code
- Handle 1000+ concurrent developers
- Process 10K+ ACUs per day
- Distribute across multiple data centers

**Reliability**:
- 99.9% system availability
- Data durability guarantees
- Automatic backup and disaster recovery
- Consistent state across distributed replicas

### 1.3 Architecture Overview

The core data structures and algorithms are organized into five main subsystems:

1. **ACU Storage Layer**: Optimized storage and indexing of Atomic Change Units
2. **Event Sourcing Engine**: Immutable event streams with deterministic replay
3. **Layer Cache System**: Docker-inspired layered storage with copy-on-write
4. **Composition Engine**: Real-time branch materialization and conflict detection
5. **Indexing and Query System**: Multi-dimensional indexing for efficient retrieval

## 2. Theoretical Foundations

### 2.1 Change Composition Mathematics

#### 2.1.1 Compositional Complexity Theory

**Definition 2.1** (Change Composition Graph): A directed acyclic graph `G = (V, E)` where:
- `V` represents the set of all ACUs
- `E` represents dependency relationships between ACUs
- Each path through `G` represents a valid composition sequence

**Theorem 2.1** (Composition Complexity Bound): For a repository with `n` ACUs and maximum dependency depth `d`, the number of valid compositions is bounded by `O(n^d)`.

*Proof*: Each ACU can depend on at most `n-1` other ACUs, and the maximum dependency chain length is `d`. The number of valid topological orderings is therefore bounded by `n^d`.

#### 2.1.2 Deterministic Replay Theory

**Definition 2.2** (Replay Function): A replay function `R: E* → S` maps a sequence of events to a system state, where:
- `E*` is the set of all finite event sequences
- `S` is the set of all possible system states
- `R` is deterministic: `∀e ∈ E*: R(e)` produces the same state

**Theorem 2.2** (Replay Determinism): If all ACUs are deterministic transformations and event ordering is consistent, then replay is deterministic.

*Proof*: By induction on event sequence length. Base case: empty sequence produces initial state. Inductive step: if replay is deterministic for sequence of length `k`, and ACU application is deterministic, then replay is deterministic for sequence of length `k+1`.

### 2.2 Layer Caching Theory

#### 2.2.1 Copy-on-Write Semantics

**Definition 2.3** (Layer State): A layer state `L` is a partial function `L: Path → Content` mapping file paths to their contents in that layer.

**Definition 2.4** (Layer Composition): The composition of layers `L₁ ∘ L₂` is defined as:
```
(L₁ ∘ L₂)(p) = {
  L₁(p) if p ∈ domain(L₁)
  L₂(p) if p ∉ domain(L₁) ∧ p ∈ domain(L₂)
  undefined otherwise
}
```

#### 2.2.2 Storage Efficiency Analysis

**Theorem 2.3** (Storage Efficiency): For `n` ACUs with average change size `s` and layer sharing factor `α`, total storage is `O(n × s × (1-α))`.

*Proof*: Each ACU creates a layer of size `s`. Layer sharing reduces storage by factor `α`. Total storage is `n × s × (1-α)`.

### 2.3 Indexing Theory

#### 2.3.1 Multi-Dimensional Access Patterns

ACUs must be efficiently retrievable by multiple dimensions:
- **By ID**: Direct ACU lookup
- **By Content Hash**: Deduplication and equivalence checking
- **By Intent**: Finding ACUs with similar purposes
- **By Dependencies**: Dependency graph traversal
- **By Author**: Developer-specific queries
- **By Timestamp**: Temporal analysis

**Definition 2.5** (Multi-Dimensional Index): An index structure `I` supporting queries across `d` dimensions with time complexity `O(log^d n)` for point queries and `O(log^{d-1} n + k)` for range queries returning `k` results.

## 3. ACU Storage Architecture

### 3.1 ACU Data Model

#### 3.1.1 Core ACU Structure

```typescript
interface AtomicChangeUnit {
  // Identity and metadata
  id: ACUId;                          // Immutable unique identifier
  version: number;                    // ACU version (for schema evolution)
  createdAt: Timestamp;              // Creation timestamp
  author: AuthorId;                  // Original author
  
  // Semantic content
  intent: IntentDescriptor;          // High-level purpose
  semanticSignature: SemanticHash;   // Content-based fingerprint
  changeset: ChangeOperation[];      // Concrete file operations
  
  // Relationships
  dependencies: ACUId[];             // Required predecessor ACUs
  conflicts: ACUId[];                // Known conflicting ACUs
  equivalentTo: ACUId[];             // Semantically equivalent ACUs
  
  // Performance optimizations
  contentHash: ContentHash;          // Fast equality checking
  storageLocation: StoragePointer;   // Physical storage reference
  compressionType: CompressionType;  // Storage optimization
  
  // Validation and integrity
  checksum: CryptographicHash;       // Integrity verification
  signature: DigitalSignature;       // Authenticity verification
}

type ACUId = string;              // UUID or content-based hash
type SemanticHash = string;       // Hash of semantic content
type ContentHash = string;        // Hash of all ACU content
type CryptographicHash = string;  // SHA-256 or similar
```

#### 3.1.2 Change Operation Details

```typescript
interface ChangeOperation {
  type: OperationType;
  path: FilePath;
  metadata: OperationMetadata;
  
  // Operation-specific data
  content?: FileContent;           // For CREATE operations
  diff?: StructuredDiff;          // For MODIFY operations  
  newPath?: FilePath;             // For MOVE operations
  
  // Semantic information
  semanticType: SemanticOperationType;
  affectedElements: SemanticElement[];
  impact: ImpactAnalysis;
}

enum OperationType {
  CREATE = 'create',
  MODIFY = 'modify', 
  DELETE = 'delete',
  MOVE = 'move',
  COPY = 'copy'
}

enum SemanticOperationType {
  STRUCTURAL_CHANGE = 'structural',
  BEHAVIORAL_CHANGE = 'behavioral',
  ARCHITECTURAL_CHANGE = 'architectural',
  COSMETIC_CHANGE = 'cosmetic'
}

interface StructuredDiff {
  additions: CodeBlock[];
  deletions: CodeBlock[];
  modifications: CodeModification[];
  moves: CodeMove[];
}
```

### 3.2 Storage Layer Implementation

#### 3.2.1 Hierarchical Storage Architecture

```typescript
class ACUStorageLayer {
  private primaryIndex: BTreeIndex<ACUId, ACUPointer>;
  private contentIndex: HashIndex<ContentHash, ACUId[]>;
  private semanticIndex: LSHIndex<SemanticHash, ACUId[]>;
  private intentIndex: InvertedIndex<Intent, ACUId[]>;
  private temporalIndex: TimeSeriesIndex<Timestamp, ACUId>;
  private dependencyGraph: GraphStore<ACUId, DependencyType>;
  
  async storeACU(acu: AtomicChangeUnit): Promise<void> {
    // Validate ACU integrity
    await this.validateACU(acu);
    
    // Store primary data
    const pointer = await this.storePrimaryData(acu);
    
    // Update all indices
    await Promise.all([
      this.primaryIndex.insert(acu.id, pointer),
      this.contentIndex.insert(acu.contentHash, [acu.id]),
      this.semanticIndex.insert(acu.semanticSignature, [acu.id]),
      this.intentIndex.insert(acu.intent, [acu.id]),
      this.temporalIndex.insert(acu.createdAt, acu.id),
      this.updateDependencyGraph(acu)
    ]);
  }
  
  async retrieveACU(id: ACUId): Promise<AtomicChangeUnit | null> {
    const pointer = await this.primaryIndex.lookup(id);
    if (!pointer) return null;
    
    return await this.loadACUFromPointer(pointer);
  }
  
  async findSimilarACUs(acu: AtomicChangeUnit): Promise<ACUSimilarityResult[]> {
    // Semantic similarity search using LSH
    const semanticallySimilar = await this.semanticIndex.findSimilar(
      acu.semanticSignature, 
      SEMANTIC_SIMILARITY_THRESHOLD
    );
    
    // Intent-based similarity
    const intentSimilar = await this.intentIndex.lookup(acu.intent);
    
    // Combine and rank results
    return this.rankSimilarityResults(semanticallySimilar, intentSimilar);
  }
}
```

#### 3.2.2 Compression and Deduplication

```typescript
class ACUCompressionEngine {
  async compressACU(acu: AtomicChangeUnit): Promise<CompressedACU> {
    // Analyze change patterns for optimal compression
    const analysisResult = await this.analyzeCompressionOpportunities(acu);
    
    switch (analysisResult.recommendedStrategy) {
      case CompressionStrategy.DELTA_COMPRESSION:
        return await this.applyDeltaCompression(acu, analysisResult.baseACU);
      
      case CompressionStrategy.PATTERN_COMPRESSION:
        return await this.applyPatternCompression(acu, analysisResult.patterns);
      
      case CompressionStrategy.SEMANTIC_COMPRESSION:
        return await this.applySemanticCompression(acu);
      
      default:
        return await this.applyGenericCompression(acu);
    }
  }
  
  private async applyDeltaCompression(acu: AtomicChangeUnit, baseACU: AtomicChangeUnit): Promise<CompressedACU> {
    const delta = this.computeACUDelta(baseACU, acu);
    
    return {
      type: CompressionStrategy.DELTA_COMPRESSION,
      baseACUId: baseACU.id,
      delta: delta,
      compressionRatio: this.calculateCompressionRatio(acu, delta)
    };
  }
}
```

### 3.3 Advanced Indexing Strategies

#### 3.3.1 Locality-Sensitive Hashing for Semantic Similarity

```typescript
class SemanticLSHIndex {
  private hashFunctions: LSHFunction[];
  private buckets: Map<string, ACUId[]>;
  private threshold: number;
  
  constructor(dimensions: number, bands: number, rows: number) {
    this.hashFunctions = this.generateLSHFunctions(dimensions, bands, rows);
    this.buckets = new Map();
    this.threshold = this.calculateThreshold(bands, rows);
  }
  
  async insert(semanticVector: number[], acuId: ACUId): Promise<void> {
    const hashes = this.hashFunctions.map(fn => fn(semanticVector));
    
    for (const hash of hashes) {
      const bucket = this.buckets.get(hash) || [];
      bucket.push(acuId);
      this.buckets.set(hash, bucket);
    }
  }
  
  async findSimilar(queryVector: number[], threshold: number): Promise<ACUId[]> {
    const candidateSet = new Set<ACUId>();
    const queryHashes = this.hashFunctions.map(fn => fn(queryVector));
    
    // Collect candidates from matching buckets
    for (const hash of queryHashes) {
      const bucket = this.buckets.get(hash);
      if (bucket) {
        bucket.forEach(id => candidateSet.add(id));
      }
    }
    
    // Verify candidates with exact similarity computation
    const results: ACUId[] = [];
    for (const candidateId of candidateSet) {
      const candidateVector = await this.getSemanticVector(candidateId);
      if (this.cosineSimilarity(queryVector, candidateVector) >= threshold) {
        results.push(candidateId);
      }
    }
    
    return results;
  }
}
```

#### 3.3.2 Dependency Graph Storage

```typescript
class DependencyGraphStore {
  private adjacencyList: Map<ACUId, Set<ACUId>>;
  private reverseAdjacencyList: Map<ACUId, Set<ACUId>>;
  private topologicalCache: Map<string, ACUId[]>;
  
  addDependency(dependent: ACUId, dependency: ACUId): void {
    // Check for cycles before adding
    if (this.wouldCreateCycle(dependent, dependency)) {
      throw new CyclicDependencyError(`Adding dependency ${dependency} to ${dependent} would create a cycle`);
    }
    
    // Add forward edge
    const dependents = this.adjacencyList.get(dependency) || new Set();
    dependents.add(dependent);
    this.adjacencyList.set(dependency, dependents);
    
    // Add reverse edge
    const dependencies = this.reverseAdjacencyList.get(dependent) || new Set();
    dependencies.add(dependency);
    this.reverseAdjacencyList.set(dependent, dependencies);
    
    // Invalidate topological sort cache
    this.invalidateTopologicalCache();
  }
  
  getTopologicalOrder(acuSet: Set<ACUId>): ACUId[] {
    const cacheKey = this.computeCacheKey(acuSet);
    let cached = this.topologicalCache.get(cacheKey);
    
    if (cached) {
      return cached;
    }
    
    // Compute topological order using Kahn's algorithm
    const result = this.computeTopologicalOrder(acuSet);
    this.topologicalCache.set(cacheKey, result);
    
    return result;
  }
  
  private computeTopologicalOrder(acuSet: Set<ACUId>): ACUId[] {
    const inDegree = new Map<ACUId, number>();
    const queue: ACUId[] = [];
    const result: ACUId[] = [];
    
    // Initialize in-degrees
    for (const acu of acuSet) {
      const dependencies = this.reverseAdjacencyList.get(acu) || new Set();
      const relevantDependencies = new Set([...dependencies].filter(dep => acuSet.has(dep)));
      inDegree.set(acu, relevantDependencies.size);
      
      if (relevantDependencies.size === 0) {
        queue.push(acu);
      }
    }
    
    // Process queue
    while (queue.length > 0) {
      const current = queue.shift()!;
      result.push(current);
      
      const dependents = this.adjacencyList.get(current) || new Set();
      for (const dependent of dependents) {
        if (acuSet.has(dependent)) {
          const newInDegree = inDegree.get(dependent)! - 1;
          inDegree.set(dependent, newInDegree);
          
          if (newInDegree === 0) {
            queue.push(dependent);
          }
        }
      }
    }
    
    return result;
  }
}
```

## 4. Event Sourcing Architecture

### 4.1 Event Stream Design

#### 4.1.1 Event Structure and Types

```typescript
interface ACUEvent {
  // Event identity
  eventId: EventId;                   // Unique event identifier
  streamId: StreamId;                 // Event stream identifier
  sequenceNumber: number;             // Monotonic sequence within stream
  
  // Event metadata
  timestamp: Timestamp;               // Event occurrence time
  eventType: ACUEventType;           // Type of event
  eventVersion: number;              // Event schema version
  
  // Event payload
  acuId: ACUId;                      // Subject ACU
  payload: EventPayload;             // Event-specific data
  
  // Causality and correlation
  causationId?: EventId;             // Direct cause event
  correlationId?: CorrelationId;     // Related events group
  
  // Integrity and authentication
  checksum: CryptographicHash;       // Event integrity
  signature: DigitalSignature;       // Event authenticity
}

enum ACUEventType {
  ACU_CREATED = 'acu.created',
  ACU_APPLIED = 'acu.applied',
  ACU_REVERTED = 'acu.reverted',
  ACU_MODIFIED = 'acu.modified',
  CONFLICT_DETECTED = 'conflict.detected',
  CONFLICT_RESOLVED = 'conflict.resolved',
  SNAPSHOT_CREATED = 'snapshot.created'
}

interface EventPayload {
  [key: string]: any;
}

// Specific payload types
interface ACUCreatedPayload extends EventPayload {
  acu: AtomicChangeUnit;
  context: CreationContext;
}

interface ACUAppliedPayload extends EventPayload {
  targetBranch: BranchId;
  applicationResult: ApplicationResult;
  beforeSnapshot?: SnapshotId;
  afterSnapshot?: SnapshotId;
}
```

#### 4.1.2 Event Stream Storage

```typescript
class EventStreamStore {
  private streams: Map<StreamId, EventStream>;
  private globalSequence: AtomicCounter;
  private eventIndex: BTreeIndex<EventId, EventPointer>;
  private streamIndex: Map<StreamId, EventStream>;
  private typeIndex: InvertedIndex<ACUEventType, EventId[]>;
  private timestampIndex: TimeSeriesIndex<Timestamp, EventId>;
  
  async appendEvent(streamId: StreamId, event: ACUEvent): Promise<void> {
    // Validate event integrity
    await this.validateEvent(event);
    
    // Assign global sequence number
    event.globalSequenceNumber = await this.globalSequence.incrementAndGet();
    
    // Get or create event stream
    let stream = this.streams.get(streamId);
    if (!stream) {
      stream = new EventStream(streamId);
      this.streams.set(streamId, stream);
    }
    
    // Append to stream with optimistic concurrency control
    await stream.appendEvent(event);
    
    // Update indices
    await this.updateIndices(event);
    
    // Trigger event processing
    await this.triggerEventProcessing(event);
  }
  
  async getEvents(streamId: StreamId, fromSequence?: number, toSequence?: number): Promise<ACUEvent[]> {
    const stream = this.streams.get(streamId);
    if (!stream) {
      return [];
    }
    
    return await stream.getEvents(fromSequence, toSequence);
  }
  
  async getEventsByType(eventType: ACUEventType, limit?: number): Promise<ACUEvent[]> {
    const eventIds = await this.typeIndex.lookup(eventType);
    
    if (limit) {
      eventIds.splice(limit);
    }
    
    return await Promise.all(
      eventIds.map(id => this.getEventById(id))
    );
  }
}
```

### 4.2 Deterministic Replay Engine

#### 4.2.1 Replay State Management

```typescript
interface ReplayState {
  currentSequence: number;
  appliedEvents: Set<EventId>;
  branchState: BranchState;
  pendingConflicts: ConflictResolution[];
  snapshotReferences: SnapshotReference[];
}

interface BranchState {
  files: Map<FilePath, FileContent>;
  metadata: BranchMetadata;
  appliedACUs: Set<ACUId>;
  dependencies: DependencyGraph;
}

class DeterministicReplayEngine {
  private stateCheckpoints: Map<number, ReplayState>;
  private eventProcessors: Map<ACUEventType, EventProcessor>;
  
  constructor() {
    this.initializeEventProcessors();
  }
  
  async replayFromEvents(events: ACUEvent[]): Promise<BranchState> {
    // Sort events by global sequence to ensure deterministic ordering
    const sortedEvents = events.sort((a, b) => a.globalSequenceNumber - b.globalSequenceNumber);
    
    // Find optimal checkpoint
    const checkpoint = this.findOptimalCheckpoint(sortedEvents);
    let state = checkpoint ? this.loadCheckpoint(checkpoint) : this.createInitialState();
    
    // Replay events from checkpoint
    const eventsToReplay = sortedEvents.filter(e => e.globalSequenceNumber > state.currentSequence);
    
    for (const event of eventsToReplay) {
      state = await this.processEvent(state, event);
      
      // Create checkpoint at regular intervals
      if (this.shouldCreateCheckpoint(event.globalSequenceNumber)) {
        await this.createCheckpoint(state);
      }
    }
    
    return state.branchState;
  }
  
  private async processEvent(state: ReplayState, event: ACUEvent): Promise<ReplayState> {
    // Verify event hasn't been processed
    if (state.appliedEvents.has(event.eventId)) {
      return state;
    }
    
    // Get appropriate processor
    const processor = this.eventProcessors.get(event.eventType);
    if (!processor) {
      throw new UnknownEventTypeError(`No processor for event type: ${event.eventType}`);
    }
    
    // Process event
    const newState = await processor.process(state, event);
    
    // Mark event as processed
    newState.appliedEvents.add(event.eventId);
    newState.currentSequence = event.globalSequenceNumber;
    
    return newState;
  }
}
```

#### 4.2.2 Event Processors

```typescript
abstract class EventProcessor {
  abstract process(state: ReplayState, event: ACUEvent): Promise<ReplayState>;
}

class ACUAppliedProcessor extends EventProcessor {
  async process(state: ReplayState, event: ACUEvent): Promise<ReplayState> {
    const payload = event.payload as ACUAppliedPayload;
    const acu = await this.loadACU(event.acuId);
    
    // Validate ACU dependencies are satisfied
    await this.validateDependencies(acu, state);
    
    // Apply ACU to branch state
    const newBranchState = await this.applyACUToBranchState(state.branchState, acu);
    
    return {
      ...state,
      branchState: newBranchState,
      appliedACUs: new Set([...state.branchState.appliedACUs, acu.id])
    };
  }
  
  private async applyACUToBranchState(branchState: BranchState, acu: AtomicChangeUnit): Promise<BranchState> {
    const newFiles = new Map(branchState.files);
    
    for (const operation of acu.changeset) {
      switch (operation.type) {
        case OperationType.CREATE:
          newFiles.set(operation.path, operation.content!);
          break;
          
        case OperationType.MODIFY:
          const currentContent = newFiles.get(operation.path) || '';
          const newContent = this.applyDiff(currentContent, operation.diff!);
          newFiles.set(operation.path, newContent);
          break;
          
        case OperationType.DELETE:
          newFiles.delete(operation.path);
          break;
          
        case OperationType.MOVE:
          const content = newFiles.get(operation.path);
          if (content) {
            newFiles.delete(operation.path);
            newFiles.set(operation.newPath!, content);
          }
          break;
      }
    }
    
    return {
      ...branchState,
      files: newFiles,
      appliedACUs: new Set([...branchState.appliedACUs, acu.id])
    };
  }
}
```

### 4.3 Snapshot Management

#### 4.3.1 Snapshot Creation Strategy

```typescript
interface Snapshot {
  snapshotId: SnapshotId;
  sequenceNumber: number;
  timestamp: Timestamp;
  branchState: BranchState;
  compressionType: CompressionType;
  size: number;
  checksum: CryptographicHash;
}

class SnapshotManager {
  private snapshots: Map<SnapshotId, Snapshot>;
  private snapshotIndex: BTreeIndex<number, SnapshotId>;
  private compressionEngine: CompressionEngine;
  
  async createSnapshot(state: ReplayState, trigger: SnapshotTrigger): Promise<SnapshotId> {
    const snapshotId = this.generateSnapshotId();
    
    // Compress state for storage efficiency
    const compressedState = await this.compressionEngine.compressState(state.branchState);
    
    const snapshot: Snapshot = {
      snapshotId,
      sequenceNumber: state.currentSequence,
      timestamp: Date.now(),
      branchState: compressedState,
      compressionType: CompressionType.ZSTD,
      size: this.calculateSize(compressedState),
      checksum: this.calculateChecksum(compressedState)
    };
    
    // Store snapshot
    await this.storeSnapshot(snapshot);
    
    // Update index
    await this.snapshotIndex.insert(snapshot.sequenceNumber, snapshotId);
    
    // Schedule cleanup of old snapshots
    await this.scheduleSnapshotCleanup();
    
    return snapshotId;
  }
  
  async findOptimalSnapshot(targetSequence: number): Promise<Snapshot | null> {
    // Find latest snapshot before target sequence
    const candidates = await this.snapshotIndex.findLessThanOrEqual(targetSequence);
    
    if (candidates.length === 0) {
      return null;
    }
    
    // Return the most recent snapshot
    const snapshotId = candidates[candidates.length - 1];
    return await this.loadSnapshot(snapshotId);
  }
  
  shouldCreateSnapshot(sequenceNumber: number, lastSnapshotSequence: number): boolean {
    const sequenceDelta = sequenceNumber - lastSnapshotSequence;
    
    // Create snapshot based on multiple criteria
    return (
      sequenceDelta >= SNAPSHOT_SEQUENCE_INTERVAL ||
      this.getTimeSinceLastSnapshot() >= SNAPSHOT_TIME_INTERVAL ||
      this.getMemoryPressure() > MEMORY_PRESSURE_THRESHOLD
    );
  }
}
```

## 5. Layer Caching System

### 5.1 Docker-Inspired Layer Architecture

#### 5.1.1 Layer Structure and Management

```typescript
interface Layer {
  layerId: LayerId;
  parentLayerId?: LayerId;           // Previous layer in chain
  acuId: ACUId;                     // ACU that created this layer
  
  // Content information
  changes: FileSystemDelta;         // What changed in this layer
  size: number;                     // Layer size in bytes
  checksum: CryptographicHash;      // Content integrity hash
  
  // Metadata
  createdAt: Timestamp;
  createdBy: AuthorId;
  compressionType: CompressionType;
  
  // Performance optimization
  accessCount: number;              // Usage frequency tracking
  lastAccessed: Timestamp;         // LRU cache management
  
  // Storage location
  storagePointer: StoragePointer;   // Physical storage reference
}

interface FileSystemDelta {
  added: Map<FilePath, FileContent>;
  modified: Map<FilePath, FileDiff>;
  deleted: Set<FilePath>;
  moved: Map<FilePath, FilePath>;   // oldPath -> newPath
}

class LayerManager {
  private layers: Map<LayerId, Layer>;
  private layerChains: Map<string, LayerId[]>;    // Chain hash -> layer sequence
  private lruCache: LRUCache<LayerId, Layer>;
  private storageBackend: LayerStorageBackend;
  
  async createLayer(parentLayerId: LayerId | null, acu: AtomicChangeUnit): Promise<Layer> {
    const layerId = this.generateLayerId(parentLayerId, acu);
    
    // Check if layer already exists (deduplication)
    if (this.layers.has(layerId)) {
      return this.layers.get(layerId)!;
    }
    
    // Compute file system delta from ACU
    const changes = await this.computeFileSystemDelta(acu);
    
    const layer: Layer = {
      layerId,
      parentLayerId,
      acuId: acu.id,
      changes,
      size: this.calculateLayerSize(changes),
      checksum: this.calculateLayerChecksum(changes),
      createdAt: Date.now(),
      createdBy: acu.author,
      compressionType: CompressionType.ZSTD,
      accessCount: 0,
      lastAccessed: Date.now(),
      storagePointer: await this.storeLayer(changes)
    };
    
    // Store layer
    this.layers.set(layerId, layer);
    this.lruCache.put(layerId, layer);
    
    return layer;
  }
  
  async materializeLayerChain(layerIds: LayerId[]): Promise<FileSystemState> {
    const layers = await Promise.all(
      layerIds.map(id => this.getLayer(id))
    );
    
    // Apply layers in sequence to build final state
    let state = new Map<FilePath, FileContent>();
    
    for (const layer of layers) {
      state = this.applyLayerToState(state, layer);
      
      // Update access statistics
      layer.accessCount++;
      layer.lastAccessed = Date.now();
    }
    
    return new FileSystemState(state);
  }
  
  private applyLayerToState(state: Map<FilePath, FileContent>, layer: Layer): Map<FilePath, FileContent> {
    const newState = new Map(state);
    
    // Apply changes from this layer
    for (const [path, content] of layer.changes.added) {
      newState.set(path, content);
    }
    
    for (const [path, diff] of layer.changes.modified) {
      const currentContent = newState.get(path) || '';
      const newContent = this.applyDiff(currentContent, diff);
      newState.set(path, newContent);
    }
    
    for (const path of layer.changes.deleted) {
      newState.delete(path);
    }
    
    for (const [oldPath, newPath] of layer.changes.moved) {
      const content = newState.get(oldPath);
      if (content) {
        newState.delete(oldPath);
        newState.set(newPath, content);
      }
    }
    
    return newState;
  }
}
```

#### 5.1.2 Layer Deduplication and Garbage Collection

```typescript
class LayerDeduplicationEngine {
  private contentHashIndex: HashIndex<ContentHash, LayerId[]>;
  private referenceCounter: Map<LayerId, number>;
  
  async findDuplicateLayers(layer: Layer): Promise<LayerId[]> {
    const contentHash = this.calculateContentHash(layer.changes);
    return await this.contentHashIndex.lookup(contentHash) || [];
  }
  
  async deduplicateLayer(layer: Layer): Promise<LayerId> {
    const duplicates = await this.findDuplicateLayers(layer);
    
    if (duplicates.length > 0) {
      // Use existing layer instead of creating new one
      const existingLayerId = duplicates[0];
      this.incrementReference(existingLayerId);
      return existingLayerId;
    }
    
    // No duplicates found, store new layer
    const contentHash = this.calculateContentHash(layer.changes);
    await this.contentHashIndex.insert(contentHash, [layer.layerId]);
    this.incrementReference(layer.layerId);
    
    return layer.layerId;
  }
  
  async garbageCollectLayers(): Promise<void> {
    const unreferencedLayers: LayerId[] = [];
    
    for (const [layerId, refCount] of this.referenceCounter) {
      if (refCount === 0) {
        unreferencedLayers.push(layerId);
      }
    }
    
    // Remove unreferenced layers
    for (const layerId of unreferencedLayers) {
      await this.deleteLayer(layerId);
      this.referenceCounter.delete(layerId);
    }
  }
}
```

### 5.2 Copy-on-Write Implementation

#### 5.2.1 Efficient Layer Branching

```typescript
class CopyOnWriteManager {
  private sharedLayers: Map<LayerId, SharedLayerInfo>;
  private branchSpecificLayers: Map<BranchId, Set<LayerId>>;
  
  async branchFromLayer(sourceLayerId: LayerId, targetBranchId: BranchId): Promise<LayerId> {
    // Initially, the new branch shares the same layer
    this.addBranchReference(sourceLayerId, targetBranchId);
    
    return sourceLayerId;
  }
  
  async modifyLayer(layerId: LayerId, branchId: BranchId, modifications: FileSystemDelta): Promise<LayerId> {
    const sharedInfo = this.sharedLayers.get(layerId);
    
    if (!sharedInfo || sharedInfo.branchReferences.size === 1) {
      // Layer is not shared or only used by this branch, modify in place
      return await this.modifyLayerInPlace(layerId, modifications);
    }
    
    // Layer is shared, create copy-on-write fork
    return await this.createCopyOnWriteFork(layerId, branchId, modifications);
  }
  
  private async createCopyOnWriteFork(sourceLayerId: LayerId, branchId: BranchId, modifications: FileSystemDelta): Promise<LayerId> {
    const sourceLayer = await this.getLayer(sourceLayerId);
    
    // Create new layer with combined changes
    const combinedChanges = this.combineChanges(sourceLayer.changes, modifications);
    
    const forkedLayer: Layer = {
      ...sourceLayer,
      layerId: this.generateLayerId(sourceLayer.parentLayerId, modifications),
      changes: combinedChanges,
      createdAt: Date.now(),
      checksum: this.calculateLayerChecksum(combinedChanges)
    };
    
    // Store forked layer
    await this.storeLayer(forkedLayer);
    
    // Update branch references
    this.removeBranchReference(sourceLayerId, branchId);
    this.addBranchReference(forkedLayer.layerId, branchId);
    
    return forkedLayer.layerId;
  }
}
```

### 5.3 Performance Optimization Strategies

#### 5.3.1 Layer Composition Caching

```typescript
class LayerCompositionCache {
  private compositionCache: LRUCache<string, FileSystemState>;
  private hotPathDetector: HotPathDetector;
  
  async getCachedComposition(layerChain: LayerId[]): Promise<FileSystemState | null> {
    const cacheKey = this.computeCompositionKey(layerChain);
    
    const cached = this.compositionCache.get(cacheKey);
    if (cached) {
      // Update hot path statistics
      this.hotPathDetector.recordHit(layerChain);
      return cached;
    }
    
    return null;
  }
  
  async cacheComposition(layerChain: LayerId[], state: FileSystemState): Promise<void> {
    const cacheKey = this.computeCompositionKey(layerChain);
    
    // Only cache if this composition is likely to be reused
    const hotPathScore = this.hotPathDetector.calculateHotPathScore(layerChain);
    
    if (hotPathScore > HOT_PATH_THRESHOLD) {
      this.compositionCache.put(cacheKey, state);
    }
  }
  
  private computeCompositionKey(layerChain: LayerId[]): string {
    // Create deterministic key from layer chain
    return layerChain
      .map(id => id.toString())
      .join('::');
  }
}

class HotPathDetector {
  private accessPatterns: Map<string, AccessPattern>;
  
  recordHit(layerChain: LayerId[]): void {
    const key = this.computePatternKey(layerChain);
    const pattern = this.accessPatterns.get(key) || new AccessPattern();
    
    pattern.hitCount++;
    pattern.lastAccess = Date.now();
    
    this.accessPatterns.set(key, pattern);
  }
  
  calculateHotPathScore(layerChain: LayerId[]): number {
    const key = this.computePatternKey(layerChain);
    const pattern = this.accessPatterns.get(key);
    
    if (!pattern) {
      return 0;
    }
    
    // Score based on frequency and recency
    const frequencyScore = Math.log(pattern.hitCount + 1);
    const recencyScore = this.calculateRecencyScore(pattern.lastAccess);
    
    return frequencyScore * recencyScore;
  }
}
```

## 6. Composition Engine Architecture

### 6.1 Real-time Branch Materialization

#### 6.1.1 Composition Request Processing

```typescript
interface CompositionRequest {
  requestId: RequestId;
  branchId: BranchId;
  acuRecipe: ACUId[];               // Ordered list of ACUs to compose
  materializationTarget: MaterializationTarget;
  priority: Priority;
  requesterContext: RequesterContext;
}

enum MaterializationTarget {
  FULL_MATERIALIZATION = 'full',        // Complete file system state
  INCREMENTAL_DIFF = 'incremental',      // Only changes from base
  CONFLICT_ANALYSIS = 'conflicts',       // Just conflict detection
  PREVIEW_MODE = 'preview'              // Read-only preview
}

class CompositionEngine {
  private requestQueue: PriorityQueue<CompositionRequest>;
  private workerPool: WorkerPool<CompositionWorker>;
  private compositionCache: LayerCompositionCache;
  private conflictDetector: ConflictDetector;
  
  async requestComposition(request: CompositionRequest): Promise<CompositionResult> {
    // Validate request
    await this.validateCompositionRequest(request);
    
    // Check for cached result
    const cachedResult = await this.getCachedComposition(request);
    if (cachedResult) {
      return cachedResult;
    }
    
    // Queue request for processing
    await this.requestQueue.enqueue(request, request.priority);
    
    // Wait for completion or timeout
    return await this.waitForComposition(request.requestId);
  }
  
  private async processCompositionRequest(request: CompositionRequest): Promise<CompositionResult> {
    // Analyze ACU dependencies and conflicts
    const analysis = await this.analyzeComposition(request.acuRecipe);
    
    if (analysis.hasConflicts && request.materializationTarget !== MaterializationTarget.CONFLICT_ANALYSIS) {
      return new CompositionResult({
        status: CompositionStatus.CONFLICTS_DETECTED,
        conflicts: analysis.conflicts,
        conflictResolutionRequired: true
      });
    }
    
    // Determine optimal composition strategy
    const strategy = await this.selectCompositionStrategy(request, analysis);
    
    // Execute composition
    return await this.executeComposition(request, strategy);
  }
  
  private async executeComposition(request: CompositionRequest, strategy: CompositionStrategy): Promise<CompositionResult> {
    switch (strategy.type) {
      case CompositionStrategyType.LAYERED_COMPOSITION:
        return await this.executeLayeredComposition(request, strategy);
      
      case CompositionStrategyType.INCREMENTAL_COMPOSITION:
        return await this.executeIncrementalComposition(request, strategy);
      
      case CompositionStrategyType.CACHED_COMPOSITION:
        return await this.executeCachedComposition(request, strategy);
      
      default:
        throw new UnsupportedCompositionStrategyError(strategy.type);
    }
  }
}
```

#### 6.1.2 Conflict Detection During Composition

```typescript
class ConflictDetector {
  private semanticAnalyzer: SemanticAnalyzer;
  private conflictResolutionCache: Map<string, ConflictResolution>;
  
  async detectConflicts(acuRecipe: ACUId[]): Promise<ConflictAnalysis> {
    const conflicts: ConflictDescription[] = [];
    const acus = await Promise.all(acuRecipe.map(id => this.loadACU(id)));
    
    // Check pairwise conflicts
    for (let i = 0; i < acus.length; i++) {
      for (let j = i + 1; j < acus.length; j++) {
        const conflict = await this.detectPairwiseConflict(acus[i], acus[j]);
        if (conflict) {
          conflicts.push(conflict);
        }
      }
    }
    
    // Check semantic conflicts
    const semanticConflicts = await this.detectSemanticConflicts(acus);
    conflicts.push(...semanticConflicts);
    
    return new ConflictAnalysis({
      hasConflicts: conflicts.length > 0,
      conflicts,
      resolutionComplexity: this.calculateResolutionComplexity(conflicts),
      suggestedResolutions: await this.suggestResolutions(conflicts)
    });
  }
  
  private async detectPairwiseConflict(acu1: AtomicChangeUnit, acu2: AtomicChangeUnit): Promise<ConflictDescription | null> {
    // Check for file-level conflicts
    const fileConflicts = this.detectFileConflicts(acu1.changeset, acu2.changeset);
    
    // Check for semantic conflicts
    const semanticConflicts = await this.semanticAnalyzer.detectSemanticConflicts(acu1, acu2);
    
    // Check for dependency conflicts
    const dependencyConflicts = this.detectDependencyConflicts(acu1, acu2);
    
    if (fileConflicts.length === 0 && semanticConflicts.length === 0 && dependencyConflicts.length === 0) {
      return null;
    }
    
    return new ConflictDescription({
      conflictingACUs: [acu1.id, acu2.id],
      conflictType: this.categorizeConflictType(fileConflicts, semanticConflicts, dependencyConflicts),
      severity: this.calculateConflictSeverity(fileConflicts, semanticConflicts, dependencyConflicts),
      affectedFiles: this.extractAffectedFiles(fileConflicts),
      description: this.generateConflictDescription(acu1, acu2, fileConflicts, semanticConflicts)
    });
  }
}
```

### 6.2 Optimization Strategies

#### 6.2.1 Parallel Composition Processing

```typescript
class ParallelCompositionProcessor {
  private dependencyGraph: DependencyGraph;
  private workerPool: WorkerPool<CompositionWorker>;
  
  async processCompositionInParallel(acuRecipe: ACUId[]): Promise<CompositionResult> {
    // Build dependency graph for ACUs
    const dependencyGraph = await this.buildDependencyGraph(acuRecipe);
    
    // Find independent ACU groups that can be processed in parallel
    const parallelGroups = this.findParallelGroups(dependencyGraph);
    
    // Process each group in parallel
    const groupResults = await Promise.all(
      parallelGroups.map(group => this.processACUGroup(group))
    );
    
    // Merge results in dependency order
    return await this.mergeGroupResults(groupResults, dependencyGraph);
  }
  
  private findParallelGroups(dependencyGraph: DependencyGraph): ACUId[][] {
    const groups: ACUId[][] = [];
    const processed = new Set<ACUId>();
    const topologicalOrder = dependencyGraph.getTopologicalOrder();
    
    let currentLevel = 0;
    while (processed.size < topologicalOrder.length) {
      const currentGroup: ACUId[] = [];
      
      for (const acuId of topologicalOrder) {
        if (processed.has(acuId)) continue;
        
        // Check if all dependencies are already processed
        const dependencies = dependencyGraph.getDependencies(acuId);
        const allDependenciesProcessed = dependencies.every(dep => processed.has(dep));
        
        if (allDependenciesProcessed) {
          currentGroup.push(acuId);
        }
      }
      
      if (currentGroup.length === 0) {
        throw new CircularDependencyError("Circular dependency detected in ACU recipe");
      }
      
      groups.push(currentGroup);
      currentGroup.forEach(acuId => processed.add(acuId));
      currentLevel++;
    }
    
    return groups;
  }
}
```

#### 6.2.2 Speculative Execution

```typescript
class SpeculativeExecutionEngine {
  private speculativeCache: Map<string, SpeculativeResult>;
  private probabilityEstimator: CompositionProbabilityEstimator;
  
  async speculativelyExecuteComposition(baseRecipe: ACUId[], candidateACUs: ACUId[]): Promise<void> {
    // Estimate probability of each candidate ACU being included
    const probabilities = await this.probabilityEstimator.estimateProbabilities(baseRecipe, candidateACUs);
    
    // Select high-probability candidates for speculative execution
    const speculativeCandidates = candidateACUs.filter(
      (acu, index) => probabilities[index] > SPECULATIVE_EXECUTION_THRESHOLD
    );
    
    // Execute speculative compositions in background
    const speculativePromises = speculativeCandidates.map(candidateACU => {
      const speculativeRecipe = [...baseRecipe, candidateACU];
      return this.executeSpeculativeComposition(speculativeRecipe);
    });
    
    // Store results for potential future use
    await Promise.allSettled(speculativePromises);
  }
  
  private async executeSpeculativeComposition(recipe: ACUId[]): Promise<void> {
    const cacheKey = this.computeSpeculativeCacheKey(recipe);
    
    // Check if already computed
    if (this.speculativeCache.has(cacheKey)) {
      return;
    }
    
    try {
      // Execute composition with lower priority to avoid impacting real requests
      const result = await this.compositionEngine.executeComposition(recipe, {
        priority: Priority.LOW,
        speculative: true
      });
      
      // Cache result for potential future use
      this.speculativeCache.set(cacheKey, {
        result,
        computedAt: Date.now(),
        accessCount: 0
      });
      
    } catch (error) {
      // Speculative execution failures are non-critical
      console.warn(`Speculative execution failed for recipe ${recipe.join(',')}:`, error);
    }
  }
}
```

## 7. Integration and Query Systems

### 7.1 Multi-dimensional Indexing

#### 7.1.1 Composite Index Architecture

```typescript
interface CompositeIndex {
  primaryDimension: IndexDimension;
  secondaryIndices: Map<IndexDimension, SecondaryIndex>;
  crossReferences: Map<string, CrossReference>;
}

enum IndexDimension {
  ACU_ID = 'id',
  CONTENT_HASH = 'content_hash',
  SEMANTIC_SIGNATURE = 'semantic_signature',
  INTENT = 'intent',
  AUTHOR = 'author',
  TIMESTAMP = 'timestamp',
  DEPENDENCIES = 'dependencies',
  FILE_PATHS = 'file_paths'
}

class MultiDimensionalIndexManager {
  private indices: Map<IndexDimension, IndexStructure>;
  private queryOptimizer: QueryOptimizer;
  
  constructor() {
    this.initializeIndices();
  }
  
  private initializeIndices(): void {
    this.indices.set(IndexDimension.ACU_ID, new BTreeIndex<ACUId, ACUPointer>());
    this.indices.set(IndexDimension.CONTENT_HASH, new HashIndex<ContentHash, ACUId[]>());
    this.indices.set(IndexDimension.SEMANTIC_SIGNATURE, new LSHIndex<SemanticHash, ACUId[]>());
    this.indices.set(IndexDimension.INTENT, new InvertedIndex<Intent, ACUId[]>());
    this.indices.set(IndexDimension.AUTHOR, new HashIndex<AuthorId, ACUId[]>());
    this.indices.set(IndexDimension.TIMESTAMP, new TimeSeriesIndex<Timestamp, ACUId>());
    this.indices.set(IndexDimension.DEPENDENCIES, new GraphIndex<ACUId, DependencyType>());
    this.indices.set(IndexDimension.FILE_PATHS, new InvertedIndex<FilePath, ACUId[]>());
  }
  
  async query(querySpec: QuerySpecification): Promise<QueryResult> {
    // Optimize query execution plan
    const executionPlan = await this.queryOptimizer.optimizeQuery(querySpec);
    
    // Execute query using optimal index combination
    return await this.executeQuery(executionPlan);
  }
  
  private async executeQuery(plan: QueryExecutionPlan): Promise<QueryResult> {
    let candidateSet: Set<ACUId> | null = null;
    
    // Execute each query step
    for (const step of plan.steps) {
      const stepResults = await this.executeQueryStep(step);
      
      if (candidateSet === null) {
        candidateSet = new Set(stepResults);
      } else {
        // Apply set operation (intersection, union, difference)
        candidateSet = this.applySetOperation(candidateSet, stepResults, step.operation);
      }
    }
    
    // Apply final filters and sorting
    const filteredResults = await this.applyFilters(candidateSet, plan.filters);
    const sortedResults = await this.applySorting(filteredResults, plan.sorting);
    
    return new QueryResult({
      results: sortedResults,
      totalCount: candidateSet?.size || 0,
      executionTime: plan.executionTime,
      indexesUsed: plan.indexesUsed
    });
  }
}
```

#### 7.1.2 Query Optimization

```typescript
class QueryOptimizer {
  private indexStatistics: Map<IndexDimension, IndexStatistics>;
  private queryHistory: QueryHistory;
  
  async optimizeQuery(querySpec: QuerySpecification): Promise<QueryExecutionPlan> {
    // Analyze query predicates
    const predicates = this.extractPredicates(querySpec);
    
    // Estimate selectivity for each predicate
    const selectivities = await Promise.all(
      predicates.map(p => this.estimateSelectivity(p))
    );
    
    // Choose optimal index access path
    const accessPath = this.chooseOptimalAccessPath(predicates, selectivities);
    
    // Generate execution plan
    return this.generateExecutionPlan(querySpec, accessPath);
  }
  
  private async estimateSelectivity(predicate: QueryPredicate): Promise<number> {
    const dimension = predicate.dimension;
    const stats = this.indexStatistics.get(dimension);
    
    if (!stats) {
      return 1.0; // Conservative estimate
    }
    
    switch (predicate.operator) {
      case QueryOperator.EQUALS:
        return 1.0 / stats.uniqueValues;
      
      case QueryOperator.IN:
        return predicate.values.length / stats.uniqueValues;
      
      case QueryOperator.RANGE:
        return this.estimateRangeSelectivity(predicate, stats);
      
      case QueryOperator.SIMILARITY:
        return this.estimateSimilaritySelectivity(predicate, stats);
      
      default:
        return 0.1; // Default conservative estimate
    }
  }
  
  private chooseOptimalAccessPath(predicates: QueryPredicate[], selectivities: number[]): AccessPath {
    // Sort predicates by selectivity (most selective first)
    const sortedPredicates = predicates
      .map((p, i) => ({ predicate: p, selectivity: selectivities[i] }))
      .sort((a, b) => a.selectivity - b.selectivity);
    
    // Build access path starting with most selective predicate
    const accessPath = new AccessPath();
    
    for (const { predicate, selectivity } of sortedPredicates) {
      const indexType = this.getOptimalIndexType(predicate);
      accessPath.addStep(new AccessStep(predicate, indexType, selectivity));
    }
    
    return accessPath;
  }
}
```

### 7.2 Complex Query Support

#### 7.2.1 Semantic Query Language

```typescript
interface SemanticQuery {
  findACUs: FindACUsClause;
  where?: WhereClause[];
  similarTo?: SimilarityClause;
  relatedTo?: RelationshipClause;
  orderBy?: OrderByClause[];
  limit?: number;
  offset?: number;
}

interface FindACUsClause {
  withIntent?: Intent | Intent[];
  byAuthor?: AuthorId | AuthorId[];
  inTimeRange?: TimeRange;
  affectingFiles?: FilePath | FilePath[];
  withDependencies?: ACUId | ACUId[];
}

interface SimilarityClause {
  toACU?: ACUId;
  toSemanticSignature?: SemanticHash;
  threshold?: number;
  algorithm?: SimilarityAlgorithm;
}

class SemanticQueryEngine {
  private parser: SemanticQueryParser;
  private executor: QueryExecutor;
  
  async executeSemanticQuery(query: SemanticQuery): Promise<SemanticQueryResult> {
    // Parse and validate query
    const parsedQuery = await this.parser.parseQuery(query);
    
    // Convert to internal query representation
    const internalQuery = this.convertToInternalQuery(parsedQuery);
    
    // Execute query
    const results = await this.executor.executeQuery(internalQuery);
    
    // Post-process results for semantic query specific formatting
    return this.formatSemanticResults(results);
  }
  
  async findSimilarACUs(referenceACU: ACUId, threshold: number): Promise<SimilarACUResult[]> {
    const reference = await this.loadACU(referenceACU);
    
    // Find semantically similar ACUs
    const semanticallySimilar = await this.semanticIndex.findSimilar(
      reference.semanticSignature,
      threshold
    );
    
    // Find ACUs with similar intents
    const intentSimilar = await this.intentIndex.lookup(reference.intent);
    
    // Combine and rank results
    const combined = this.combineAndRankSimilarity(
      semanticallySimilar,
      intentSimilar,
      reference
    );
    
    return combined;
  }
  
  async findACUsByPattern(pattern: ChangePattern): Promise<ACUId[]> {
    // Convert pattern to query predicates
    const predicates = this.patternToPredicates(pattern);
    
    // Execute pattern-based search
    const candidates = await this.executePatternQuery(predicates);
    
    // Verify pattern match with detailed analysis
    const verified = await this.verifyPatternMatches(candidates, pattern);
    
    return verified;
  }
}
```

## 8. Performance Analysis and Optimization

### 8.1 Complexity Analysis

#### 8.1.1 Space Complexity

**ACU Storage**: `O(n × s)` where `n` is number of ACUs and `s` is average ACU size
- Primary storage: `O(n × s)`
- Index overhead: `O(n × log n)` for primary indices
- Secondary indices: `O(n × k)` where `k` is average number of index entries per ACU

**Layer Storage**: `O(l × d)` where `l` is number of layers and `d` is average delta size
- With deduplication: `O(u × d)` where `u` is number of unique layers
- Compression can reduce by factor of 2-10 depending on content

**Event Stream Storage**: `O(e × m)` where `e` is number of events and `m` is average event size
- Snapshot overhead: `O(s × n)` where `s` is number of snapshots
- Total with snapshots: `O(e × m + s × n)`

#### 8.1.2 Time Complexity

**ACU Operations**:
- Store ACU: `O(log n + k)` where `k` is number of index updates
- Retrieve ACU: `O(log n)` for primary key lookup
- Find similar ACUs: `O(log n + r)` where `r` is number of results
- Complex queries: `O(log^d n + r)` where `d` is number of query dimensions

**Composition Operations**:
- Layer composition: `O(l × f)` where `l` is number of layers and `f` is average files per layer
- Conflict detection: `O(a^2 × c)` where `a` is ACUs in recipe and `c` is conflict check cost
- Branch materialization: `O(l × f + a^2 × c)` combining layer composition and conflict detection

**Event Sourcing Operations**:
- Event append: `O(1)` amortized
- Event replay: `O(e × p)` where `e` is events to replay and `p` is processing cost per event
- Snapshot creation: `O(s)` where `s` is state size

### 8.2 Performance Optimization Techniques

#### 8.2.1 Caching Strategies

```typescript
class MultiLevelCacheManager {
  private l1Cache: LRUCache<string, any>;        // In-memory hot data
  private l2Cache: MemoryMappedCache<string, any>; // Memory-mapped warm data
  private l3Cache: DiskCache<string, any>;       // Disk-based cold data
  
  constructor() {
    this.l1Cache = new LRUCache(L1_CACHE_SIZE);
    this.l2Cache = new MemoryMappedCache(L2_CACHE_SIZE);
    this.l3Cache = new DiskCache(L3_CACHE_SIZE);
  }
  
  async get<T>(key: string): Promise<T | null> {
    // Try L1 cache first
    let value = this.l1Cache.get(key);
    if (value) {
      return value as T;
    }
    
    // Try L2 cache
    value = await this.l2Cache.get(key);
    if (value) {
      // Promote to L1
      this.l1Cache.put(key, value);
      return value as T;
    }
    
    // Try L3 cache
    value = await this.l3Cache.get(key);
    if (value) {
      // Promote to L2 and L1
      await this.l2Cache.put(key, value);
      this.l1Cache.put(key, value);
      return value as T;
    }
    
    return null;
  }
  
  async put<T>(key: string, value: T, hint: CacheHint = CacheHint.NORMAL): Promise<void> {
    // Always store in L1
    this.l1Cache.put(key, value);
    
    // Conditionally store in lower levels based on hint
    if (hint >= CacheHint.WARM) {
      await this.l2Cache.put(key, value);
    }
    
    if (hint >= CacheHint.PERSISTENT) {
      await this.l3Cache.put(key, value);
    }
  }
}
```

#### 8.2.2 Parallel Processing Architecture

```typescript
class ParallelProcessingCoordinator {
  private cpuBoundPool: WorkerPool<CPUBoundWorker>;
  private ioBoundPool: WorkerPool<IOBoundWorker>;
  private taskScheduler: TaskScheduler;
  
  async processACUBatch(acus: AtomicChangeUnit[]): Promise<ProcessingResult[]> {
    // Classify tasks by resource requirements
    const cpuTasks = acus.filter(acu => this.isCPUBound(acu));
    const ioTasks = acus.filter(acu => this.isIOBound(acu));
    
    // Process in parallel using appropriate worker pools
    const [cpuResults, ioResults] = await Promise.all([
      this.processCPUBoundTasks(cpuTasks),
      this.processIOBoundTasks(ioTasks)
    ]);
    
    // Merge and order results
    return this.mergeResults(cpuResults, ioResults, acus);
  }
  
  private async processCPUBoundTasks(acus: AtomicChangeUnit[]): Promise<ProcessingResult[]> {
    // Divide work among CPU workers
    const chunks = this.chunkArray(acus, this.cpuBoundPool.size);
    
    const chunkPromises = chunks.map((chunk, index) => {
      return this.cpuBoundPool.execute(index % this.cpuBoundPool.size, {
        task: 'process_acu_chunk',
        data: chunk
      });
    });
    
    const chunkResults = await Promise.all(chunkPromises);
    return chunkResults.flat();
  }
  
  private isCPUBound(acu: AtomicChangeUnit): boolean {
    // Heuristic to determine if ACU processing is CPU-bound
    return (
      acu.changeset.length > CPU_BOUND_THRESHOLD ||
      acu.intent === Intent.REFACTOR ||
      acu.semanticSignature.complexity > COMPLEXITY_THRESHOLD
    );
  }
}
```

#### 8.2.3 Memory Management

```typescript
class MemoryManager {
  private memoryPool: MemoryPool;
  private memoryMonitor: MemoryMonitor;
  private gcScheduler: GarbageCollectionScheduler;
  
  constructor() {
    this.memoryPool = new MemoryPool(TOTAL_MEMORY_LIMIT);
    this.memoryMonitor = new MemoryMonitor();
    this.gcScheduler = new GarbageCollectionScheduler();
    
    this.setupMemoryPressureHandling();
  }
  
  private setupMemoryPressureHandling(): void {
    this.memoryMonitor.onHighPressure(() => {
      this.handleHighMemoryPressure();
    });
    
    this.memoryMonitor.onCriticalPressure(() => {
      this.handleCriticalMemoryPressure();
    });
  }
  
  private async handleHighMemoryPressure(): Promise<void> {
    // Aggressive cache eviction
    await this.cacheManager.evictLRUEntries(0.3); // Evict 30% of cache
    
    // Trigger incremental garbage collection
    await this.gcScheduler.scheduleIncrementalGC();
    
    // Compact memory pools
    await this.memoryPool.compact();
  }
  
  private async handleCriticalMemoryPressure(): Promise<void> {
    // Emergency cache clearing
    await this.cacheManager.clearAll();
    
    // Force full garbage collection
    await this.gcScheduler.scheduleFullGC();
    
    // Temporarily reduce worker pool sizes
    this.reduceWorkerPoolSizes();
    
    // Consider swapping some data to disk
    await this.initiateSwapToDisk();
  }
}
```

## 9. Integration Points and APIs

### 9.1 External Integration Interfaces

#### 9.1.1 REST API Design

```typescript
// ACU Management API
interface ACUManagementAPI {
  // Basic CRUD operations
  createACU(acu: CreateACURequest): Promise<ACUResponse>;
  getACU(id: ACUId): Promise<ACUResponse>;
  updateACU(id: ACUId, updates: UpdateACURequest): Promise<ACUResponse>;
  deleteACU(id: ACUId): Promise<void>;
  
  // Advanced operations
  findSimilarACUs(id: ACUId, threshold: number): Promise<SimilarACUResponse>;
  getACUDependencies(id: ACUId): Promise<DependencyResponse>;
  detectConflicts(acu1: ACUId, acu2: ACUId): Promise<ConflictResponse>;
}

// Composition API
interface CompositionAPI {
  // Branch operations
  createBranch(recipe: ACUId[]): Promise<BranchResponse>;
  materializeBranch(id: BranchId): Promise<MaterializationResponse>;
  previewComposition(recipe: ACUId[]): Promise<PreviewResponse>;
  
  // Conflict management
  detectCompositionConflicts(recipe: ACUId[]): Promise<ConflictAnalysisResponse>;
  resolveConflicts(conflicts: ConflictResolution[]): Promise<ResolutionResponse>;
  
  // Performance monitoring
  getBranchStatistics(id: BranchId): Promise<BranchStatisticsResponse>;
  getCompositionMetrics(): Promise<CompositionMetricsResponse>;
}

// Event Sourcing API
interface EventSourcingAPI {
  // Event stream operations
  getEventStream(streamId: StreamId, fromSequence?: number): Promise<EventStreamResponse>;
  replayEvents(streamId: StreamId, toSequence: number): Promise<ReplayResponse>;
  
  // Snapshot management
  createSnapshot(streamId: StreamId): Promise<SnapshotResponse>;
  getSnapshots(streamId: StreamId): Promise<SnapshotListResponse>;
  restoreFromSnapshot(snapshotId: SnapshotId): Promise<RestoreResponse>;
}
```

#### 9.1.2 WebSocket Real-time API

```typescript
interface RealtimeAPI {
  // Connection management
  connect(credentials: AuthCredentials): Promise<WebSocketConnection>;
  disconnect(): Promise<void>;
  
  // Event subscriptions
  subscribeToACUEvents(filter: ACUEventFilter): Promise<Subscription>;
  subscribeToCompositionEvents(branchId: BranchId): Promise<Subscription>;
  subscribeToConflictEvents(): Promise<Subscription>;
  
  // Real-time notifications
  onACUCreated(callback: (event: ACUCreatedEvent) => void): void;
  onCompositionUpdated(callback: (event: CompositionUpdatedEvent) => void): void;
  onConflictDetected(callback: (event: ConflictDetectedEvent) => void): void;
}

class RealtimeEventBroadcaster {
  private connections: Map<ConnectionId, WebSocketConnection>;
  private subscriptions: Map<SubscriptionId, EventSubscription>;
  
  async broadcastACUEvent(event: ACUEvent): Promise<void> {
    const relevantSubscriptions = this.findRelevantSubscriptions(event);
    
    const broadcastPromises = relevantSubscriptions.map(subscription => {
      const connection = this.connections.get(subscription.connectionId);
      if (connection && connection.isActive()) {
        return this.sendEventToConnection(connection, event);
      }
      return Promise.resolve();
    });
    
    await Promise.allSettled(broadcastPromises);
  }
  
  private findRelevantSubscriptions(event: ACUEvent): EventSubscription[] {
    return Array.from(this.subscriptions.values()).filter(subscription => {
      return this.eventMatchesFilter(event, subscription.filter);
    });
  }
}
```

### 9.2 Plugin Architecture

#### 9.2.1 Extension Points

```typescript
interface PluginInterface {
  name: string;
  version: string;
  initialize(context: PluginContext): Promise<void>;
  shutdown(): Promise<void>;
}

interface SemanticAnalysisPlugin extends PluginInterface {
  analyzeSemanticChange(change: SemanticChange): Promise<SemanticAnalysisResult>;
  extractIntent(change: SemanticChange): Promise<Intent>;
  detectPatterns(changes: SemanticChange[]): Promise<ChangePattern[]>;
}

interface ConflictResolutionPlugin extends PluginInterface {
  detectConflicts(acu1: AtomicChangeUnit, acu2: AtomicChangeUnit): Promise<ConflictDescription[]>;
  resolveConflicts(conflicts: ConflictDescription[]): Promise<ConflictResolution[]>;
  suggestResolutions(conflicts: ConflictDescription[]): Promise<ResolutionSuggestion[]>;
}

class PluginManager {
  private plugins: Map<string, PluginInterface>;
  private extensionPoints: Map<ExtensionPoint, PluginInterface[]>;
  
  async loadPlugin(pluginPath: string): Promise<void> {
    const plugin = await this.loadPluginFromPath(pluginPath);
    
    // Validate plugin interface
    await this.validatePlugin(plugin);
    
    // Initialize plugin
    const context = this.createPluginContext(plugin);
    await plugin.initialize(context);
    
    // Register plugin for relevant extension points
    this.registerPluginForExtensionPoints(plugin);
    
    this.plugins.set(plugin.name, plugin);
  }
  
  async executeExtensionPoint<T>(
    extensionPoint: ExtensionPoint,
    ...args: any[]
  ): Promise<T[]> {
    const relevantPlugins = this.extensionPoints.get(extensionPoint) || [];
    
    const executionPromises = relevantPlugins.map(plugin => {
      return this.executePluginMethod(plugin, extensionPoint, ...args);
    });
    
    const results = await Promise.allSettled(executionPromises);
    
    // Filter successful results
    return results
      .filter(result => result.status === 'fulfilled')
      .map(result => (result as PromiseFulfilledResult<T>).value);
  }
}
```

## 10. Testing and Validation Framework

### 10.1 Unit Testing Architecture

#### 10.1.1 ACU Storage Testing

```typescript
describe('ACU Storage Layer', () => {
  let storageLayer: ACUStorageLayer;
  let testDatabase: TestDatabase;
  
  beforeEach(async () => {
    testDatabase = await TestDatabase.create();
    storageLayer = new ACUStorageLayer(testDatabase);
  });
  
  describe('storeACU', () => {
    it('should store ACU and update all indices', async () => {
      const testACU = TestDataGenerator.createACU({
        intent: Intent.REFACTOR,
        changesetSize: 5
      });
      
      await storageLayer.storeACU(testACU);
      
      // Verify primary storage
      const retrieved = await storageLayer.retrieveACU(testACU.id);
      expect(retrieved).toEqual(testACU);
      
      // Verify index updates
      const byIntent = await storageLayer.findByIntent(Intent.REFACTOR);
      expect(byIntent).toContain(testACU.id);
      
      const byAuthor = await storageLayer.findByAuthor(testACU.author);
      expect(byAuthor).toContain(testACU.id);
    });
    
    it('should handle concurrent storage operations', async () => {
      const testACUs = TestDataGenerator.createACUBatch(100);
      
      // Store ACUs concurrently
      const storePromises = testACUs.map(acu => storageLayer.storeACU(acu));
      await Promise.all(storePromises);
      
      // Verify all ACUs were stored correctly
      const retrievalPromises = testACUs.map(acu => storageLayer.retrieveACU(acu.id));
      const retrieved = await Promise.all(retrievalPromises);
      
      expect(retrieved.length).toBe(testACUs.length);
      expect(retrieved.every(acu => acu !== null)).toBe(true);
    });
  });
  
  describe('findSimilarACUs', () => {
    it('should find semantically similar ACUs', async () => {
      const baseACU = TestDataGenerator.createACU({
        semanticSignature: 'signature_a',
        intent: Intent.BUG_FIX
      });
      
      const similarACU = TestDataGenerator.createACU({
        semanticSignature: 'signature_a_variant',
        intent: Intent.BUG_FIX
      });
      
      const dissimilarACU = TestDataGenerator.createACU({
        semanticSignature: 'completely_different',
        intent: Intent.FEATURE_ADD
      });
      
      await Promise.all([
        storageLayer.storeACU(baseACU),
        storageLayer.storeACU(similarACU),
        storageLayer.storeACU(dissimilarACU)
      ]);
      
      const similar = await storageLayer.findSimilarACUs(baseACU);
      
      expect(similar.map(s => s.acuId)).toContain(similarACU.id);
      expect(similar.map(s => s.acuId)).not.toContain(dissimilarACU.id);
    });
  });
});
```

#### 10.1.2 Event Sourcing Testing

```typescript
describe('Event Sourcing Engine', () => {
  let eventStore: EventStreamStore;
  let replayEngine: DeterministicReplayEngine;
  
  describe('deterministic replay', () => {
    it('should produce identical results for same event sequence', async () => {
      const events = TestDataGenerator.createEventSequence(50);
      
      // Replay events multiple times
      const result1 = await replayEngine.replayFromEvents(events);
      const result2 = await replayEngine.replayFromEvents(events);
      const result3 = await replayEngine.replayFromEvents(events);
      
      // Results should be identical
      expect(result1).toEqual(result2);
      expect(result2).toEqual(result3);
    });
    
    it('should handle event reordering consistently', async () => {
      const events = TestDataGenerator.createEventSequence(20);
      
      // Shuffle events (maintaining causal dependencies)
      const shuffledEvents = this.shufflePreservingDependencies(events);
      
      const originalResult = await replayEngine.replayFromEvents(events);
      const shuffledResult = await replayEngine.replayFromEvents(shuffledEvents);
      
      // Results should be identical regardless of ordering
      expect(originalResult).toEqual(shuffledResult);
    });
  });
  
  describe('snapshot optimization', () => {
    it('should produce same result with and without snapshots', async () => {
      const events = TestDataGenerator.createEventSequence(100);
      
      // Replay without snapshots
      const directResult = await replayEngine.replayFromEvents(events);
      
      // Create snapshot at midpoint
      const midpoint = Math.floor(events.length / 2);
      const snapshotResult = await replayEngine.replayFromEvents(events.slice(0, midpoint));
      const snapshot = await this.snapshotManager.createSnapshot(snapshotResult);
      
      // Replay from snapshot
      const snapshotedResult = await replayEngine.replayFromSnapshot(
        snapshot,
        events.slice(midpoint)
      );
      
      expect(directResult).toEqual(snapshotedResult);
    });
  });
});
```

### 10.2 Integration Testing

#### 10.2.1 Full System Testing

```typescript
describe('Full System Integration', () => {
  let system: CompositionalSourceControlSystem;
  
  beforeEach(async () => {
    system = await CompositionalSourceControlSystem.createTestInstance();
  });
  
  it('should handle complete ACU lifecycle', async () => {
    // Create ACU
    const acu = TestDataGenerator.createACU();
    const stored = await system.storeACU(acu);
    
    // Create branch with ACU
    const branch = await system.createBranch([stored.id]);
    
    // Materialize branch
    const materialized = await system.materializeBranch(branch.id);
    
    // Verify materialization
    expect(materialized.status).toBe(MaterializationStatus.SUCCESS);
    expect(materialized.appliedACUs).toContain(stored.id);
  });
  
  it('should handle complex composition scenarios', async () => {
    // Create interdependent ACUs
    const acuA = TestDataGenerator.createACU({ id: 'acu-a' });
    const acuB = TestDataGenerator.createACU({ 
      id: 'acu-b',
      dependencies: ['acu-a']
    });
    const acuC = TestDataGenerator.createACU({
      id: 'acu-c',
      dependencies: ['acu-a'],
      conflicts: ['acu-b']
    });
    
    await Promise.all([
      system.storeACU(acuA),
      system.storeACU(acuB),
      system.storeACU(acuC)
    ]);
    
    // Test valid composition
    const validBranch = await system.createBranch(['acu-a', 'acu-b']);
    const validResult = await system.materializeBranch(validBranch.id);
    expect(validResult.status).toBe(MaterializationStatus.SUCCESS);
    
    // Test conflicting composition
    const conflictBranch = await system.createBranch(['acu-a', 'acu-b', 'acu-c']);
    const conflictResult = await system.materializeBranch(conflictBranch.id);
    expect(conflictResult.status).toBe(MaterializationStatus.CONFLICTS_DETECTED);
    expect(conflictResult.conflicts.length).toBeGreaterThan(0);
  });
});
```

### 10.3 Performance Testing

#### 10.3.1 Load Testing

```typescript
describe('Performance and Load Testing', () => {
  let system: CompositionalSourceControlSystem;
  let performanceMonitor: PerformanceMonitor;
  
  it('should handle high ACU creation rate', async () => {
    const targetRate = 1000; // ACUs per second
    const duration = 30; // seconds
    const totalACUs = targetRate * duration;
    
    performanceMonitor.startMonitoring();
    
    const startTime = Date.now();
    const createPromises: Promise<any>[] = [];
    
    for (let i = 0; i < totalACUs; i++) {
      const acu = TestDataGenerator.createACU();
      createPromises.push(system.storeACU(acu));
      
      // Maintain target rate
      if (i % targetRate === 0) {
        await this.waitForInterval(1000); // Wait 1 second
      }
    }
    
    await Promise.all(createPromises);
    const endTime = Date.now();
    
    const actualRate = totalACUs / ((endTime - startTime) / 1000);
    const metrics = performanceMonitor.getMetrics();
    
    expect(actualRate).toBeGreaterThanOrEqual(targetRate * 0.9); // 90% of target
    expect(metrics.averageResponseTime).toBeLessThan(100); // < 100ms
    expect(metrics.errorRate).toBeLessThan(0.01); // < 1% errors
  });
  
  it('should scale materialization with composition complexity', async () => {
    const complexitLevels = [10, 50, 100, 500, 1000];
    const results: PerformanceResult[] = [];
    
    for (const complexity of complexitLevels) {
      // Create composition recipe of given complexity
      const recipe = TestDataGenerator.createComplexRecipe(complexity);
      
      const startTime = Date.now();
      const branch = await system.createBranch(recipe);
      const materialized = await system.materializeBranch(branch.id);
      const endTime = Date.now();
      
      results.push({
        complexity,
        materializationTime: endTime - startTime,
        success: materialized.status === MaterializationStatus.SUCCESS
      });
    }
    
    // Verify scalability characteristics
    this.verifyScalabilityProfile(results);
  });
});
```

## 11. Conclusion

This comprehensive exploration of core data structures and algorithms provides the technical foundation necessary for implementing compositional source control systems. The proposed architecture addresses the fundamental challenges of AI-accelerated development through:

### 11.1 Key Technical Contributions

1. **Optimized ACU Storage**: Multi-dimensional indexing and semantic similarity search capabilities that enable efficient storage and retrieval of atomic change units at scale.

2. **Deterministic Event Sourcing**: Immutable event streams with guaranteed deterministic replay, providing the foundation for reliable compositional operations.

3. **Layered Caching Architecture**: Docker-inspired copy-on-write layering system that achieves significant storage efficiency while maintaining fast access times.

4. **Real-time Composition Engine**: Sophisticated conflict detection and branch materialization capabilities that enable real-time collaborative development.

5. **Performance Optimization Framework**: Comprehensive caching, parallel processing, and memory management strategies that ensure system scalability.

### 11.2 Implementation Readiness

The detailed specifications, algorithms, and code examples provided in this research enable direct implementation of the compositional source control system. Key implementation-ready components include:

- Complete data structure definitions with TypeScript interfaces
- Optimized algorithms with complexity analysis
- Comprehensive testing frameworks
- Performance optimization strategies
- Integration interfaces and APIs

### 11.3 Integration with Broader Research

This core infrastructure directly enables the subsequent research phases:

- **AI-Assisted Conflict Resolution**: Leverages semantic indexing and conflict detection infrastructure
- **Collaborative Intelligence**: Builds upon event sourcing and composition engine capabilities  
- **Performance Architecture**: Extends the optimization frameworks presented here
- **Security Architecture**: Integrates with the cryptographic integrity mechanisms

### 11.4 Future Scalability

The proposed architecture is designed to scale beyond current requirements:

- **Horizontal Scaling**: Event sourcing and layered storage naturally distribute across multiple servers
- **Elastic Performance**: Caching and parallel processing adapt to varying load patterns
- **Extensibility**: Plugin architecture enables future capability expansion
- **Evolution Support**: Versioned data structures accommodate system evolution

This foundational infrastructure establishes the technical bedrock upon which the complete compositional source control paradigm can be built, tested, and deployed at enterprise scale.

---

*This research document provides the complete technical specification for implementing the core data structures and algorithms required for compositional source control systems. The comprehensive coverage of storage optimization, event sourcing, layer caching, and performance architecture ensures that subsequent research phases have a solid technical foundation to build upon.*