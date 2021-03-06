package anys2s

import (
	"errors"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

// A Batch stores an input and output batch in a packed
// format.
type Batch struct {
	Inputs  anyseq.Seq
	Outputs anyseq.Seq
}

// A Trainer creates batches, computes gradients, and adds
// up costs for a sequence-to-sequence mapping.
type Trainer struct {
	Func   func(anyseq.Seq) anyseq.Seq
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
}

// Fetch produces a *Batch for the subset of samples.
// The s argument must implement SampleList.
// The batch may not be empty.
func (t *Trainer) Fetch(s anysgd.SampleList) (anysgd.Batch, error) {
	if s.Len() == 0 {
		return nil, errors.New("fetch batch: empty batch")
	}
	l := s.(SampleList)
	ins := make([][]anyvec.Vector, l.Len())
	outs := make([][]anyvec.Vector, l.Len())
	for i := 0; i < l.Len(); i++ {
		sample, err := l.GetSample(i)
		if err != nil {
			return nil, essentials.AddCtx("fetch batch", err)
		}
		ins[i] = sample.Input
		outs[i] = sample.Output
	}
	return &Batch{
		Inputs:  anyseq.ConstSeqList(l.Creator(), ins),
		Outputs: anyseq.ConstSeqList(l.Creator(), outs),
	}, nil
}

// TotalCost computes the total cost for the *Batch.
func (t *Trainer) TotalCost(batch anysgd.Batch) anydiff.Res {
	b := batch.(*Batch)
	actual := t.Func(b.Inputs)

	if len(actual.Output()) != len(b.Outputs.Output()) {
		panic("mismatching actual and desired sequence shapes")
	}

	var idx int
	var costCount int
	allCosts := anyseq.Map(actual, func(a anydiff.Res, n int) anydiff.Res {
		batch := b.Outputs.Output()[idx]
		if batch.NumPresent() != n {
			panic("mismatching actual and desired sequence shapes")
		}
		costCount += n
		idx++
		c := t.Cost.Cost(anydiff.NewConst(batch.Packed), a, n)
		return c
	})

	sum := anydiff.Sum(anyseq.Sum(allCosts))
	if t.Average {
		scaler := sum.Output().Creator().MakeNumeric(1 / float64(costCount))
		return anydiff.Scale(sum, scaler)
	} else {
		return sum
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
