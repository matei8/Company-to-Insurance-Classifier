# Insurance Classification System

A scalable system for classifying companies into insurance categories using NLP and deep learning techniques.

## Overview

This system combines multiple approaches to classify companies into insurance categories:

- Sentence Transformers for semantic understanding
- Cosine similarity for flexible matching
- TF-IDF for traditional text analysis

## Data Processing Pipeline

1. **Data Loading and Initial Processing**
   - Load company data from CSV
   - Combine multiple text fields (description, business_tags, category, niche)
   - Clean and normalize text (lowercase, remove numbers/punctuation)
   - Handle missing values

2. **Feature Extraction**
   - Generate sentence embeddings using all-MiniLM-L6-v2
   - Compute TF-IDF features for n-grams (bigrams and trigrams)
   - Select top-k features based on importance
   - Normalize features (sample-wise or feature-wise)

3. **Model Processing**
   - Load pre-trained sentence transformer
   - Pre-compute taxonomy label embeddings
   - Set similarity threshold (0.85)

4. **Classification**
   - Compute cosine similarity between company and taxonomy embeddings
   - Apply threshold to filter matches
   - Use fallback mechanism (top-3) if no matches meet threshold
   - Assign multiple labels when appropriate

5. **Output and Evaluation**
   - Save classifications to CSV
   - Track processing progress

## Strengths

1. **Flexible Classification**
   - Multi-label support for complex business categorization
   - Threshold-based matching with fallback mechanism
   - Handles ambiguous cases through top-k matching

2. **Robust Text Processing**
   - Combines multiple text fields (description, tags, category, niche)
   - Handles missing data gracefully
   - Normalization and cleaning for consistent input

## Weaknesses

**Current Limitations**

   - Relies heavily on pre-trained embeddings
   - No active learning or feedback loop
   - Fixed similarity threshold
   - Sequential processing of companies

## Scalability

### Current Implementation
- Processes data sequentially
- Memory usage scales with dataset size
- Pre-computed embeddings reduce computation time

## Future Improvements

1. **Parallel Processing**
   - Implement Map-Reduce paradigm
   - Distributed computing with Dask/Spark
   - Batch processing optimization

2. **Model Enhancements**
   - Active learning for continuous improvement
   - Dynamic threshold adjustment
   - Confidence scoring for predictions
   - Industry-specific fine-tuning

## Design Decisions and Problem-Solving Process

### Architecture Choices

1. **Sentence Transformers:**
   - Considered alternatives:
     - Pure BERT: Too resource-intensive for large-scale processing
     - Traditional ML (SVM, Random Forest): Limited semantic understanding
     - CNN: Higher parameter count, less efficient


2. **Why Cosine Similarity?**
   - Alternatives considered:
     - Euclidean distance: Less effective for text similarity
   - Chose cosine similarity because:
     - Proven effectiveness for text similarity
     - Computationally efficient

### Implementation Decisions


2. **Text Processing Strategy**
   - Considered alternatives:
     - More aggressive cleaning: Risk of losing important information
     - Less cleaning: More noise in embeddings
     - Custom cleaning rules: Hard to maintain
   - Chose current approach because:
     - Preserves important business terminology
     - Removes common noise (numbers, punctuation)
     - Maintains readability for debugging

3. **Memory Management**
   - Considered alternatives:
     - Process all data at once: Memory constraints
     - Very small chunks: Too much overhead
     - Complex caching: Implementation complexity
   - Chose current approach because:
     - Pre-computing taxonomy embeddings reduces runtime memory
     - Sequential processing with progress tracking
     - Clear memory usage patterns


Bibliography:

- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
- https://developers.google.com/machine-learning/guides/text-classification
- https://www.youtube.com/watch?v=ATK6fm3cYfI (NLP)
- https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a
- https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/
- https://levity.ai/blog/text-classifiers-in-machine-learning-a-practical-guide