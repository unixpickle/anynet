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

	GetSample(idx int) *Sample
	Creator() anyvec.Creator
}

// A SliceSampleList is a concrete SampleList with
// predetermined samples.
type SliceSampleList []*Sample

// Len returns the number of samples.
func (s SliceSampleList) Len() int {
	return len(s)
}

// Swap swaps two samples.
func (s SliceSampleList) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// Slice copies a sub-slice of the list.
func (s SliceSampleList) Slice(i, j int) anysgd.SampleList {
	return append(SliceSampleList{}, s[i:j]...)
}

// GetSample returns the sample at the index.
func (s SliceSampleList) GetSample(idx int) *Sample {
	return s[idx]
}
