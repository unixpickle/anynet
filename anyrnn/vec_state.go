package anyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

// A VecState is a State and/or StateGrad that can be
// expressed as a vector.
type VecState struct {
	Vector     anyvec.Vector
	PresentMap PresentMap
}

// NewVecState generates a VecState with the vector
// repeated n times.
func NewVecState(v anyvec.Vector, n int) *VecState {
	rep := v.Creator().MakeVector(v.Len() * n)
	anyvec.AddRepeated(rep, v)
	p := make([]bool, n)
	for i := range p {
		p[i] = true
	}
	return &VecState{
		Vector:     rep,
		PresentMap: p,
	}
}

// Present returns the PresentMap.
func (v *VecState) Present() PresentMap {
	return v.PresentMap
}

// Reduce generates a new *VecState with a subset of the
// chunks in v.
func (v *VecState) Reduce(p PresentMap) State {
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
// where necessary, producing a new *VecState.
func (v *VecState) Expand(p PresentMap) StateGrad {
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

// PropagateStart propagates the contents of the vector,
// treated as a batched upstream gradient, through the
// variable.
//
// All sequences must be present.
func (v *VecState) PropagateStart(va *anydiff.Var, g anydiff.Grad) {
	for _, x := range v.PresentMap {
		if !x {
			panic("all sequences must be present")
		}
	}
	if dest, ok := g[va]; ok {
		dest.Add(anyvec.SumRows(v.Vector, len(v.PresentMap)))
	}
}
