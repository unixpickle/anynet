package anysgd

import "github.com/unixpickle/anydiff"

// A Transformer transforms gradients.
// For example, pre-conditioning could be implemented as a
// transformer.
//
// After its first call, a Transformer expects to see
// gradients of the same form (i.e. containing the same
// variables).
//
// A Transformer may modify its own input and return the
// same gradient as an output.
// However, a Transformer should not modify its input
// after Transform returns.
// In other words, the input still belongs to the caller,
// and the transformer should not retain a reference to
// the input.
// If a Transformer needs to cache things relating to its
// inputs, it must allocate a separate gradient.
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

// A Stopper indicates that gradient descent should stop
// by returning true from its Done method.
type Stopper interface {
	Done() bool
}

// A Rater determines the learning rate given the epoch
// number.
// An "epoch" is a full pass over the training set, so
// fractional epochs are possible.
type Rater interface {
	Rate(epoch float64) float64
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

// PostShuffler is used to notify a SampleList that it has
// been shuffled, allowing it to perform any sample
// re-ordering it likes.
//
// For example, you might use a PostShuffler to make sure
// that "compatible" samples are close to each other so
// they end up in the same mini-batch.
type PostShuffler interface {
	PostShuffle()
}
