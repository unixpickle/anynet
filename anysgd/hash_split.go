package anysgd

import (
	"encoding/binary"
	"io"
	"math"
	"sort"
)

// A Hasher is a SampleList with the added capability to
// produce a hash for a given sample.
type Hasher interface {
	SampleList
	Hash(i int) []byte
}

// HashSplit partitions a Hasher.
// It can be used to deterministically split data up into
// separate validation and training samples.
//
// The Hasher h will be re-ordered as needed for internal
// computations.
//
// The leftRatio argument specifies the expected fraction
// of samples that should end up on the left partition.
func HashSplit(h Hasher, leftRatio float64) (left, right SampleList) {
	if leftRatio == 0 {
		return h.Slice(0, 0), h
	} else if leftRatio == 1 {
		return h, h.Slice(0, 0)
	}
	cutoff := hashCutoff(leftRatio)
	insertIdx := 0
	for i := 0; i < h.Len(); i++ {
		hash := h.Hash(i)
		if compareHashes(hash, cutoff) < 0 {
			h.Swap(insertIdx, i)
			insertIdx++
		}
	}
	splitIdx := sort.Search(h.Len(), func(i int) bool {
		return compareHashes(h.Hash(i), cutoff) >= 0
	})
	return h.Slice(0, splitIdx), h.Slice(splitIdx, h.Len())
}

// Most of the following code was taken from my old sgd package.
// See https://github.com/unixpickle/sgd/blob/0e3d4c9d317b1095d02febdaedf802f6d1dbd5b1/hash_split.go.

func hashCutoff(ratio float64) []byte {
	res := make([]byte, 8)
	for i := range res {
		ratio *= 256
		value := int(ratio)
		ratio -= float64(value)
		if value == 256 {
			value = 255
		}
		res[i] = byte(value)
	}
	return res
}

func compareHashes(h1, h2 []byte) int {
	max := len(h1)
	if len(h2) > max {
		max = len(h2)
	}
	for i := 0; i < max; i++ {
		var h1Val, h2Val byte
		if i < len(h1) {
			h1Val = h1[i]
		}
		if i < len(h2) {
			h2Val = h2[i]
		}
		if h1Val < h2Val {
			return -1
		} else if h1Val > h2Val {
			return 1
		}
	}
	return 0
}

func writeFloatBits(w io.Writer, temp []byte, val float64) {
	binary.BigEndian.PutUint64(temp, math.Float64bits(val))
	w.Write(temp)
}
