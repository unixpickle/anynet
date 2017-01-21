package anyrnn

import "github.com/unixpickle/anyvec"

// A VecState is a State and/or StateGrad that can be
// expressed as a vector.
type VecState struct {
	Vector     anyvec.Vector
	PresentMap PresentMap
}

// Present returns the PresentMap.
func (v *VecState) Present() PresentMap {
	return v.PresentMap
}

// Reduce generates a new *VecState with a subset of the
// chunks in v.
func (v *VecState) Reduce(p PresentMap) *VecState {
	n := v.PresentMap.NumPresent()
	inc := v.Vector.Len() / n

	var chunks []anyvec.Vector
	var chunkStart, chunkSize int
	for i, pres := range p {
		if pres {
			if !v.PresentMap[i] {
				panic("argument to Reduce must be a subset")
			}
			chunkSize += inc
		} else if v.PresentMap[i] {
			if chunkSize > 0 {
				chunks = append(chunks, v.Vector.Slice(chunkStart, chunkStart+chunkSize))
				chunkStart += chunkSize
				chunkSize = 0
			}
			chunkStart += inc
		}
	}
	if chunkSize > 0 {
		chunks = append(chunks, v.Vector.Slice(chunkStart, chunkStart+chunkSize))
	}

	return &VecState{
		Vector:     v.Vector.Creator().Concat(chunks...),
		PresentMap: p,
	}
}

// Expand expands the *VecState by inserting zero chunks
// where necessary.
func (v *VecState) Expand(p PresentMap) *VecState {
	n := v.PresentMap.NumPresent()
	inc := v.Vector.Len() / n
	filler := v.Vector.Creator().MakeVector(inc)

	var chunks []anyvec.Vector
	var chunkStart, chunkSize int

	for i, pres := range p {
		if v.PresentMap[i] {
			if !pres {
				panic("argument to Expand must be a superset")
			}
			chunkSize += inc
		} else if pres {
			if chunkSize > 0 {
				chunks = append(chunks, v.Vector.Slice(chunkStart, chunkStart+chunkSize))
				chunkStart += chunkSize
				chunkSize = 0
			}
			chunks = append(chunks, filler)
		}
	}
	if chunkSize > 0 {
		chunks = append(chunks, v.Vector.Slice(chunkStart, chunkSize+chunkStart))
	}

	return &VecState{
		Vector:     v.Vector.Creator().Concat(chunks...),
		PresentMap: p,
	}
}
