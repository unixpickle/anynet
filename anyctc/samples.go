package anyctc

import (
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
)

// A Sample is a training sequence paired with its
// corresponding label.
type Sample struct {
	Input []anyvec.Vector
	Label []int
}

// A SampleList is an anysgd.SampleList that produces
// CTC samples.
type SampleList interface {
	anysgd.SampleList

	GetSample(idx int) (*Sample, error)
	Creator() anyvec.Creator
}
