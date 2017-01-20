// Package anysgd provides tools for Stochastic Gradient
// Descent.
// It is intended to be used for Machine Learning, but it
// can be applied to other areas as well.
package anysgd

import "github.com/unixpickle/anydiff"

// SGD performs stochastic gradient descent.
type SGD struct {
	// Gradienter is used to compute initial, untransformed
	// gradients for each mini-batch.
	Gradienter Gradienter

	// Transformer, if non-nil, is used to transform each
	// gradient before the step.
	Transformer Transformer

	// Samples is the list of training samples to use for
	// training.
	// It will be shuffled and re-shuffled as needed.
	//
	// The list may not be empty.
	Samples SampleList

	// Rater determines the learning rate for each step.
	Rater Rater

	// StatusFunc, if non-nil, is called before every
	// iteration with the next mini-batch.
	StatusFunc func(batch SampleList)

	// BatchSize is the mini-batch size.
	// If it is 0, then the entire sample list is used at
	// every iteration.
	BatchSize int

	// NumProcessed keeps track of the number of samples that
	// have been passed to Gradienter so far.
	// It is used to compute the epoch for Rater.
	// Most of the time, this should be initialized to 0.
	NumProcessed int
}

// Run runs SGD until s indicates to stop.
func (s *SGD) Run(stopper Stopper) {
	if s.Samples.Len() == 0 {
		panic("cannot run SGD with empty sample list")
	}
	idx := s.Samples.Len()
	for !stopper.Done() {
		remaining := s.Samples.Len() - idx
		if remaining == 0 {
			Shuffle(s.Samples)
			idx = 0
			remaining = s.Samples.Len()
		}
		batchSize := s.batchSize(remaining)
		batch := s.Samples.Slice(idx, idx+batchSize)
		idx += batchSize

		if s.StatusFunc != nil {
			s.StatusFunc(batch)
			if stopper.Done() {
				break
			}
		}

		grad := s.Gradienter.Gradient(batch)
		if s.Transformer != nil {
			grad = s.Transformer.Transform(grad)
		}

		epoch := float64(s.NumProcessed) / float64(s.Samples.Len())
		scaleGradient(grad, -s.Rater.Rate(epoch))
		grad.AddToVars()

		s.NumProcessed += batchSize
	}
}

func (s *SGD) batchSize(remaining int) int {
	if s.BatchSize == 0 || s.BatchSize > remaining {
		return remaining
	} else {
		return s.BatchSize
	}
}

func scaleGradient(g anydiff.Grad, s float64) {
	for _, v := range g {
		g.Scale(v.Creator().MakeNumeric(s))
		return
	}
}
