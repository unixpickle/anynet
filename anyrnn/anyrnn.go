// Package anyrnn implements recurrent neural networks.
package anyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

// A PresentMap is used to indicate which sequences are
// present in a State and which ones are not.
// A true value indicates present.
//
// See State for more details on how PresentMap is used.
type PresentMap []bool

// NumPresent counts the present sequences.
func (p PresentMap) NumPresent() int {
	var i int
	for _, x := range p {
		if x {
			i++
		}
	}
	return i
}

// A State stores a batch of internal Block states.
//
// Since RNNs are typically used to evaluate sequences, a
// state has the idea of present and absent sequences.
// A present sequence is one which has not yet terminated,
// thus we need the state of the RNN for that sequence.
// An absent sequence has already finished, so we don't
// need to track its state.
//
// If an RNN is being evaluated on sequences of differing
// lengths, then the idea of present/active is essential.
type State interface {
	// Present provides information about which sequences
	// have states in the batch.
	Present() PresentMap

	// Reduce creates a copy of the State with a new
	// PresentMap.
	// It is intended to be used to remove states from the
	// batch when some sequences end during RNN evaluation.
	//
	// The PresentMap must be a subset of Present(), meaning
	// that every true value in it must be true in Present().
	// The method is called "Reduce" because it can only be
	// used to remove some states from the batch; it cannot
	// add states to the batch.
	Reduce(PresentMap) State
}

// A StateGrad is an upstream gradient for a State.
// It is used while back-propagating through a Block.
type StateGrad interface {
	// Present provides information about which sequences
	// have upstream gradients in the batch.
	Present() PresentMap

	// Expand inserts zero gradients as necessary to expand
	// the present map.
	// The resulting StateGrad will include all of the
	// sequences from Present() and all of the sequences from
	// the passed PresentMap.
	//
	// Expand is the inverse of State.Reduce().
	Expand(PresentMap) StateGrad
}

// A Block is a differentiable unit in an RNN.
// It receives an input/state batch and produces a batch
// of outputs and new states.
type Block interface {
	// Start produces the start state with a batch size of n.
	Start(n int) State

	// PropagateStart back-propagates through the start
	// state.
	// After this is called, s should not be used again.
	PropagateStart(s StateGrad, g anydiff.Grad)

	// Step applies the block for a single timestep.
	Step(s State, in anyvec.Vector) Res
}

// A Res represents the output of a Block and is used to
// back-propagate through a Block.
type Res interface {
	// State returns the output state batch.
	State() State

	// Output returns the Block outputs.
	Output() anyvec.Vector

	// Vars returns the variables upon which the output
	// depends, including variables from previous states.
	Vars() anydiff.VarSet

	// Propagate propagates the gradient for one timestep.
	// It takes an upstream vector u for the output, an
	// upstream StateGrad s for the output state, and the
	// output gradient to which partials should be added.
	//
	// It returns a downstream vector for the input and a
	// StateGrad for the previous timestep.
	//
	// The upstream state s may be nil, indicating a zero
	// upstream.
	// This is useful for the final timestep, whose state is
	// never used for anything.
	//
	// All upstream objects may be modified.
	// A call to Propagate may change u and s, meaning that s
	// in particular should not be used again.
	//
	// The downstream input vector may be modified by the
	// caller (e.g. as scratch space).
	// Modifying said vector should not affect the returned
	// downstream StateGrad.
	Propagate(u anyvec.Vector, s StateGrad, g anydiff.Grad) (anyvec.Vector, StateGrad)
}
