package anys2v

import (
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
)

// A Sample is a training sequence with a corresponding
// desired output vector.
//
// It is invalid for the input sequence to be empty.
// All input sequences must be non-empty.
type Sample struct {
	Input  []anyvec.Vector
	Output anyvec.Vector
}

// A SampleList is an anysgd.SampleList that produces
// sequence-to-vector samples.
type SampleList interface {
	anysgd.SampleList

	GetSample(idx int) (*Sample, error)
}
