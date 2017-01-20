// Package anysgd provides tools for Stochastic Gradient
// Descent.
// It is intended to be used for Machine Learning, but it
// can be applied to other things as well.
package anysgd

import "github.com/unixpickle/anydiff"

// A Transformer transforms gradients, perhaps by applying
// some form of pre-conditioning.
//
// After its first call, a Transformer expects to see
// gradients of the same form (i.e. containing the same
// variables).
// However, a Transformer is not allowed to assume
// owernship of any gradients it takes as input, as those
// gradients may be re-used as inputs at some later time.
type Transformer interface {
	Transform(g anydiff.Grad) anydiff.Grad
}

// A Gradienter computes a gradient for a list of samples.
//
// The same gradient instance may be re-used by successive
// calls to Gradient.
type Gradienter interface {
	Gradient(s SampleList) anydiff.Grad
}

// A SampleList represents a list of training samples.
type SampleList interface {
	// Len returns the number of samples.
	Len() int

	// Copy creates a shallow copy of the list.
	Copy() SampleList

	// Swap swaps two samples.
	Swap(i, j int)

	// Slice generates a shallow copy of a subset of the
	// list.
	Slice(i, j int) SampleList
}
