package anysgd

import (
	"encoding"

	"github.com/unixpickle/anydiff"
)

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
//
// A Transformer's output is only guaranteed to be valid
// until the next time Transform is called.
type Transformer interface {
	Transform(g anydiff.Grad) anydiff.Grad
}

// TransformMarshaler is a Transformer with support for
// binary marshalling and unmarshalling.
type TransformMarshaler interface {
	Transformer
	encoding.BinaryMarshaler
	encoding.BinaryUnmarshaler
}

// A Batch is an immutable list of samples.
//
// In contrast to a SampleList, a Batch is not assumed to
// use lazy evaluation.
// This means that Batches should only be created when
// they are about to be used.
//
// Batches are obtained using a Fetcher and then used as
// arguments to a Gradienter.
type Batch interface{}

// A Fetcher is responsible for fetching Batches for
// SampleLists.
//
// Typically, a Fetcher will be used concurrently with
// SGD, making it possible to have a new Batch available
// exactly when the previous one is done being used.
type Fetcher interface {
	Fetch(s SampleList) (Batch, error)
}

// A Gradienter computes a gradient for a Batch.
//
// The same gradient instance may be re-used by successive
// calls to Gradient.
type Gradienter interface {
	Gradient(b Batch) anydiff.Grad
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

// A Coster computes differentiable costs for a Batch.
// The resulting cost vectors should have one component.
type Coster interface {
	TotalCost(b Batch) anydiff.Res
}
