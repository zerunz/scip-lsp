package partialloader

import (
	"hash/maphash"
	"math"
)

// BloomFilter is a simple in-memory Bloom filter for strings.
// It supports Add() and MightContain() membership checks (with false positives).
//
// This is intentionally minimal: it's used only as a fast pre-filter to reduce the
// set of index files to scan when resolving relationships like implementations.
type BloomFilter struct {
	mBits uint64 // number of bits
	k     uint64 // number of hash functions
	words []uint64

	seed1 maphash.Seed
	seed2 maphash.Seed
}

func NewBloomFilter(estimatedN int, falsePositiveRate float64) *BloomFilter {
	// Clamp inputs to safe ranges.
	if estimatedN < 1 {
		estimatedN = 1
	}
	if falsePositiveRate <= 0 || falsePositiveRate >= 1 {
		falsePositiveRate = 0.01
	}

	// Standard Bloom filter parameterization:
	// m = -n ln(p) / (ln 2)^2
	// k = (m/n) ln 2
	n := float64(estimatedN)
	ln2 := math.Ln2
	m := uint64(math.Ceil(-n * math.Log(falsePositiveRate) / (ln2 * ln2)))
	if m < 1024 {
		m = 1024
	}
	k := uint64(math.Ceil((float64(m) / n) * ln2))
	if k < 1 {
		k = 1
	}
	if k > 16 {
		// Avoid too many hash rounds.
		k = 16
	}

	words := make([]uint64, (m+63)/64)
	return &BloomFilter{
		mBits: m,
		k:     k,
		words: words,
		seed1: maphash.MakeSeed(),
		seed2: maphash.MakeSeed(),
	}
}

func (b *BloomFilter) Add(s string) {
	if b == nil {
		return
	}
	h1 := b.hash64(b.seed1, s)
	h2 := b.hash64(b.seed2, s)
	if h2 == 0 {
		h2 = 0x9e3779b97f4a7c15 // avoid pathological 0 step
	}

	for i := uint64(0); i < b.k; i++ {
		pos := (h1 + i*h2) % b.mBits
		b.words[pos>>6] |= 1 << (pos & 63)
	}
}

func (b *BloomFilter) MightContain(s string) bool {
	if b == nil {
		return true
	}
	h1 := b.hash64(b.seed1, s)
	h2 := b.hash64(b.seed2, s)
	if h2 == 0 {
		h2 = 0x9e3779b97f4a7c15
	}

	for i := uint64(0); i < b.k; i++ {
		pos := (h1 + i*h2) % b.mBits
		if (b.words[pos>>6] & (1 << (pos & 63))) == 0 {
			return false
		}
	}
	return true
}

func (b *BloomFilter) hash64(seed maphash.Seed, s string) uint64 {
	var h maphash.Hash
	h.SetSeed(seed)
	_, _ = h.WriteString(s)
	return h.Sum64()
}
