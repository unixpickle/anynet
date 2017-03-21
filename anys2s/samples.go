package anys2s

import (
	"sort"

	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
)

// A Sample is a training sequence with a corresponding
// desired output sequence.
type Sample struct {
	Input  []anyvec.Vector
	Output []anyvec.Vector
}

// A SampleList is an anysgd.SampleList that produces
// sequence-to-sequence samples.
type SampleList interface {
	anysgd.SampleList

	GetSample(idx int) (*Sample, error)
	Creator() anyvec.Creator
}

// A SortableSampleList is a SampleList with an extra
// LenAt method for efficiently getting the length of an
// input sequence.
type SortableSampleList interface {
	SampleList

	LenAt(idx int) int
}

// A SortSampleList wraps a SampleList and ensures that
// samples will be sorted within reasonably small chunks.
// This is often beneficial for RNNs on a GPU, since it
// helps to keep batch sizes stable across timesteps.
type SortSampleList struct {
	SortableSampleList

	// BatchSize is the size of the chunks that should be
	// sorted.
	BatchSize int
}

// Slice produces a subset of the SortSampleList.
func (s *SortSampleList) Slice(i, j int) anysgd.SampleList {
	sliced := s.SortableSampleList.Slice(i, j)
	return &SortSampleList{
		SortableSampleList: sliced.(SortableSampleList),
		BatchSize:          s.BatchSize,
	}
}

// PostShuffle sorts batches of sequences.
func (s *SortSampleList) PostShuffle() {
	for i := 0; i < s.Len(); i += s.BatchSize {
		bs := s.BatchSize
		if bs > s.Len()-i {
			bs = s.Len() - i
		}
		s := &sorter{S: s.SortableSampleList, Start: i, End: i + bs}
		sort.Sort(s)
	}
}

type sorter struct {
	S     SortableSampleList
	Start int
	End   int
}

func (s *sorter) Len() int {
	return s.End - s.Start
}

func (s *sorter) Swap(i, j int) {
	s.S.Swap(i+s.Start, j+s.Start)
}

func (s *sorter) Less(i, j int) bool {
	return s.S.LenAt(i+s.Start) < s.S.LenAt(j+s.Start)
}
