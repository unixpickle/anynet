package anysgd

import "math/rand"

// Shuffle shuffles a list of samples.
// If the list implements PostShuffler, then PostShuffle
// is called after the shuffle completes.
func Shuffle(s SampleList) {
	for i := 0; i < s.Len(); i++ {
		j := i + rand.Intn(s.Len()-i)
		s.Swap(i, j)
	}
	if p, ok := s.(PostShuffler); ok {
		p.PostShuffle()
	}
}

// A ConstRater is a Rater which always returns the same
// constant learning rate.
type ConstRater float64

// Rate returns float64(c).
func (c ConstRater) Rate(epoch float64) float64 {
	return float64(c)
}
