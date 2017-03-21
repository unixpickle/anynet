package anyff

import (
	"errors"
	"runtime"
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

// A Batch stores an input and output batch in a packed
// format.
type Batch struct {
	Inputs  *anydiff.Const
	Outputs *anydiff.Const
	Num     int
}

// A Trainer can construct batches, compute gradients, and
// tally up costs for feed-forward neural networks.
type Trainer struct {
	Net    anynet.Layer
	Cost   anynet.Cost
	Params []*anydiff.Var

	// Average indicates whether or not the total cost should
	// be averaged before computing gradients.
	// This affects gradients, LastCost, and the output of
	// TotalCost().
	Average bool

	// After every gradient computation, LastCost is set to
	// the cost from the batch.
	LastCost anyvec.Numeric

	// MaxGos specifies the maximum goroutines to use
	// simultaneously for fetching samples.
	// If it is 0, GOMAXPROCS is used.
	MaxGos int
}

// Fetch produces a *Batch for the subset of samples.
// The s argument must implement SampleList.
// The batch may not be empty.
func (t *Trainer) Fetch(s anysgd.SampleList) (anysgd.Batch, error) {
	if s.Len() == 0 {
		return nil, errors.New("fetch batch: empty batch")
	}

	l := s.(SampleList)
	ins := make([]anyvec.Vector, l.Len())
	outs := make([]anyvec.Vector, l.Len())

	idxChan := make(chan int, l.Len())
	for i := 0; i < l.Len(); i++ {
		idxChan <- i
	}
	close(idxChan)

	maxGos := t.MaxGos
	if maxGos == 0 {
		maxGos = runtime.GOMAXPROCS(0)
	}

	wg := sync.WaitGroup{}
	errChan := make(chan error, maxGos)
	for i := 0; i < maxGos; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range idxChan {
				sample, err := l.GetSample(i)
				if err != nil {
					errChan <- essentials.AddCtx("fetch batch", err)
					return
				}
				ins[i] = sample.Input
				outs[i] = sample.Output
			}
		}()
	}

	wg.Wait()
	close(errChan)

	if err := <-errChan; err != nil {
		return nil, err
	}

	joinedIns := ins[0].Creator().Concat(ins...)
	joinedOuts := outs[0].Creator().Concat(outs...)

	return &Batch{
		Inputs:  anydiff.NewConst(joinedIns),
		Outputs: anydiff.NewConst(joinedOuts),
		Num:     l.Len(),
	}, nil
}

// TotalCost computes the total cost for the *Batch.
func (t *Trainer) TotalCost(batch anysgd.Batch) anydiff.Res {
	b := batch.(*Batch)
	outRes := t.Net.Apply(b.Inputs, b.Num)
	cost := t.Cost.Cost(b.Outputs, outRes, b.Num)
	total := anydiff.Sum(cost)
	if t.Average {
		divisor := 1 / float64(cost.Output().Len())
		return anydiff.Scale(total, total.Output().Creator().MakeNumeric(divisor))
	} else {
		return total
	}
}

// Gradient computes the gradient for the batch's cost.
// It also sets t.LastCost to the numerical value of the
// total cost.
//
// The b argument must be a *Batch.
func (t *Trainer) Gradient(b anysgd.Batch) anydiff.Grad {
	grad, lc := anysgd.CosterGrad(t, b, t.Params)
	t.LastCost = lc
	return grad
}
