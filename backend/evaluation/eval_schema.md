Ah yes — you mean **one example/schema for each evaluation type**.

### 1. Multi-source

Tests whether the system retrieves evidence from **multiple episodes**.

```json
{
  "id": "case_01",
  "category": "multi_source",
  "question": "How does Shannon's concept of information relate to the way information is stored in DNA?",
  "required_sources": [
    "Great Papers 03 - The Double Helix, Watson and Crick 1953.mp3",
    "Great Papers 04 - Shannon and the Birth of Information, 1948.mp3"
  ],
  "expected_evidence": [
    {
      "source": "Great Papers 03 - The Double Helix, Watson and Crick 1953.mp3",
      "timestamp_start": 145.58,
      "timestamp_end": 162.3,
      "concept": "DNA stores information in its physical structure."
    },
    {
      "source": "Great Papers 04 - Shannon and the Birth of Information, 1948.mp3",
      "timestamp_start": 314.64,
      "timestamp_end": 340.0,
      "concept": "Information can be understood as the resolution of uncertainty."
    }
  ],
  "minimum_evidence": {
    "required_sources": 2,
    "required_concepts": 2
  }
}
```

### 2. Deep single-source

Tests whether the system can retrieve **multiple pieces of evidence from one episode**.

```json
{
  "id": "case_06",
  "category": "deep_single_source",
  "question": "How did the Transformer address the limitations of recurrent neural networks, and what architectural choices enabled this?",
  "required_sources": [
    "Great Papers 05 - Attention Is All You Need, 2017.mp3"
  ],
  "expected_evidence": [
    {
      "source": "Great Papers 05 - Attention Is All You Need, 2017.mp3",
      "timestamp_start": 223.96,
      "timestamp_end": 256.42,
      "concept": "Recurrent neural networks process sequences sequentially."
    },
    {
      "source": "Great Papers 05 - Attention Is All You Need, 2017.mp3",
      "timestamp_start": 296.94,
      "timestamp_end": 321.52,
      "concept": "Sequential processing makes long-range dependencies difficult."
    },
    {
      "source": "Great Papers 05 - Attention Is All You Need, 2017.mp3",
      "timestamp_start": 507.58,
      "timestamp_end": 521.58,
      "concept": "The Transformer uses attention without recurrence."
    },
    {
      "source": "Great Papers 05 - Attention Is All You Need, 2017.mp3",
      "timestamp_start": 711.9,
      "timestamp_end": 743.48,
      "concept": "Attention creates direct connections between words."
    }
  ],
  "minimum_evidence": {
    "required_sources": 1,
    "required_concepts": 3
  }
}
```

### 3. Precise / needle-in-a-haystack

Tests whether the system can find **one specific buried fact**.

```json
{
  "id": "case_11",
  "category": "precise",
  "question": "What four-letter alphabet is used to represent information in DNA?",
  "required_sources": [
    "Great Papers 03 - The Double Helix, Watson and Crick 1953.mp3"
  ],
  "expected_evidence": [
    {
      "source": "Great Papers 03 - The Double Helix, Watson and Crick 1953.mp3",
      "timestamp_start": 1965.56,
      "timestamp_end": 2013.34,
      "concept": "DNA represents information using a four-letter nucleotide alphabet."
    }
  ],
  "minimum_evidence": {
    "required_sources": 1,
    "required_concepts": 1
  }
}
```


**Multi-source → multiple episodes**
**Deep single-source → multiple evidence spans within one episode**
**Precise → one specific evidence span**
