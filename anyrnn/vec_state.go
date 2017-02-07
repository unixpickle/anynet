package anyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
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
	b := &anyseq.Batch{Packed: v.Vector, Present: v.PresentMap}
	res := anyseq.ReduceBatch(b, p)
	return &VecState{Vector: res.Packed, PresentMap: p}
}

// Expand expands the *VecState by inserting zero chunks
// where necessary, producing a new *VecState.
func (v *VecState) Expand(p PresentMap) StateGrad {
	b := &anyseq.Batch{Packed: v.Vector, Present: v.PresentMap}
	res := anyseq.ExpandBatch(b, p)
	return &VecState{Vector: res.Packed, PresentMap: p}
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
		dest.Add(anyvec.SumRows(v.Vector, v.Vector.Len()/len(v.PresentMap)))
	}
}
