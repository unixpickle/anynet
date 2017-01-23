// Package anysgd provides tools for Stochastic Gradient
// Descent.
// It is intended to be used for Machine Learning, but it
// can be applied to other areas as well.
package anysgd

// SGD performs stochastic gradient descent.
type SGD struct {
	// Fetcher is used to obtain Batches for mini-batch
	// slices of the sample list.
	Fetcher Fetcher

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
	StatusFunc func(batch Batch)

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

// Run runs SGD until doneChan is closed or the fetcher
// returns an error.
//
// Run is not thread-safe, and you should never modify the
// struct's fields while Run is active.
// However, you may safely read from s.NumProcessed during
// calls to s.StatusFunc.
func (s *SGD) Run(doneChan <-chan struct{}) error {
	if s.Samples.Len() == 0 {
		panic("cannot run SGD with empty sample list")
	}

	errChan := make(chan error, 1)
	batchChan := make(chan *batchInfo)

	go func() {
		idx := s.Samples.Len()
		for {
			select {
			case <-doneChan:
				return
			default:
			}
			remaining := s.Samples.Len() - idx
			if remaining == 0 {
				Shuffle(s.Samples)
				idx = 0
				remaining = s.Samples.Len()
			}
			batchSize := s.batchSize(remaining)
			batchSlice := s.Samples.Slice(idx, idx+batchSize)
			idx += batchSize
			batch, err := s.Fetcher.Fetch(batchSlice)
			if err != nil {
				errChan <- err
				return
			}
			select {
			case batchChan <- &batchInfo{batch, batchSize}:
			case <-doneChan:
				return
			}
		}
	}()

	for {
		select {
		case <-doneChan:
			return nil
		default:
		}

		var info *batchInfo
		select {
		case info = <-batchChan:
		case err := <-errChan:
			return err
		case <-doneChan:
			return nil
		}

		if s.StatusFunc != nil {
			s.StatusFunc(info.Batch)
			select {
			case <-doneChan:
				return nil
			default:
			}
		}

		grad := s.Gradienter.Gradient(info.Batch)
		if s.Transformer != nil {
			grad = s.Transformer.Transform(grad)
		}

		epoch := float64(s.NumProcessed) / float64(s.Samples.Len())
		scaleGrad(grad, -s.Rater.Rate(epoch))
		grad.AddToVars()

		s.NumProcessed += info.Size
	}
}

func (s *SGD) batchSize(remaining int) int {
	if s.BatchSize == 0 || s.BatchSize > remaining {
		return remaining
	} else {
		return s.BatchSize
	}
}

type batchInfo struct {
	Batch Batch
	Size  int
}
